from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat
from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_from_riccati,
    xi_to_y,
)


DEFAULT_CANDIDATES = {
    "hybrid_8pt": Path("archive/repo_cleanup_2026-04-24/model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics/hybrid_8pt"),
    "riccati_multibranch": Path("archive/repo_cleanup_2026-04-24/kh_roots/kh_subsonic_fixed_mach_M05_riccati_multibranch"),
    "modefocus_lowalpha": Path("archive/repo_cleanup_2026-04-24/model_saved/kh_subsonic_fixed_mach_M05_modefocus_lowalpha"),
}


@dataclass
class Candidate:
    name: str
    run_dir: Path
    config: pd.Series
    model: torch.nn.Module
    mach: float


def normalize_full_mode(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u = u * phase
        v = v * phase
        p = p * phase
        rho = rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u = -u
        v = -v
        p = -p
        rho = -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return {
        "y": np.asarray(y, dtype=float),
        "u": u / scale,
        "v": v / scale,
        "p": p / scale,
        "rho": rho / scale,
    }


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def compute_visible_xlim(y: np.ndarray, fields: list[np.ndarray], *, threshold_ratio: float = 0.02, min_half_width: float = 8.0) -> tuple[float, float]:
    envelope = np.zeros_like(y, dtype=float)
    for field in fields:
        envelope = np.maximum(envelope, np.abs(np.real(field)))
        envelope = np.maximum(envelope, np.abs(np.imag(field)))
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return float(y[0]), float(y[-1])
    mask = envelope >= threshold_ratio * peak
    if not np.any(mask):
        return float(y[0]), float(y[-1])
    y_vis = y[mask]
    half_width = max(float(np.max(np.abs(y_vis))), float(min_half_width))
    return -half_width, half_width


def load_candidate(name: str, run_dir: Path, device: torch.device) -> Candidate:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    load_fixed_mach_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()
    return Candidate(
        name=name,
        run_dir=run_dir,
        config=config,
        model=model,
        mach=float(config["mach"]),
    )


def load_classic_full_mode(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
    solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    k_left = np.asarray(sol_left.y[0])
    q_left = np.asarray(sol_left.y[1])
    ln_p_left = np.asarray(sol_left.y[2])
    phi_left = np.asarray(sol_left.y[3])
    k_right = np.asarray(sol_right.y[0])
    q_right = np.asarray(sol_right.y[1])
    ln_p_right = np.asarray(sol_right.y[2])
    phi_right = np.asarray(sol_right.y[3])

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    mask_left = y_left < 0.0
    y = np.concatenate([y_left[mask_left], y_right[::-1]])
    p = np.concatenate([p_left[mask_left], p_right[::-1]])
    gamma = np.concatenate([gamma_left[mask_left], gamma_right[::-1]])

    p_y = gamma * p
    c = -1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)
    return normalize_full_mode(y, u, v, p, rho), float(result.ci)


def load_pinn_full_mode(candidate: Candidate, *, alpha: float, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    xi.requires_grad_(True)
    alpha_tensor = torch.full_like(xi, float(alpha))

    mode_representation = str(candidate.config.get("mode_representation", "cartesian"))
    if mode_representation == "riccati":
        pr, pi, y_t = reconstruct_pressure_from_riccati(candidate.model, xi, alpha_tensor, anchor_xi=0.0)
    else:
        pred = candidate.model(xi, alpha_tensor)
        pr = pred[:, 0:1]
        pi = pred[:, 1:2]
        y_t = xi_to_y(xi, candidate.model.get_mapping_scale().detach())

    p = torch.complex(pr, pi)
    p_r_xi = torch.autograd.grad(pr, xi, grad_outputs=torch.ones_like(pr), create_graph=False, retain_graph=True)[0]
    p_i_xi = torch.autograd.grad(pi, xi, grad_outputs=torch.ones_like(pi), create_graph=False, retain_graph=True)[0]
    p_xi = torch.complex(p_r_xi, p_i_xi)
    y_xi = dy_dxi(xi, candidate.model.get_mapping_scale().detach())
    p_y = p_xi / y_xi

    ci = float(candidate.model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    c = -1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (candidate.mach**2)

    return normalize_full_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    ), ci


def compute_mode_metrics(classic: dict[str, np.ndarray], pinn: dict[str, np.ndarray], *, y_common: np.ndarray, phase_threshold: float = 0.02) -> dict[str, float]:
    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))

    p_c = interp_complex(classic["y"], classic["p"], y_common)
    p_p = interp_complex(pinn["y"], pinn["p"], y_common)
    rho_c = interp_complex(classic["y"], classic["rho"], y_common)
    rho_p = interp_complex(pinn["y"], pinn["rho"], y_common)
    u_c = interp_complex(classic["y"], classic["u"], y_common)
    u_p = interp_complex(pinn["y"], pinn["u"], y_common)
    v_c = interp_complex(classic["y"], classic["v"], y_common)
    v_p = interp_complex(pinn["y"], pinn["v"], y_common)

    amp_c = np.abs(p_c)
    amp_p = np.abs(p_p)
    phase_c = np.unwrap(np.angle(p_c))
    phase_p = np.unwrap(np.angle(p_p))
    phase_c -= phase_c[np.argmax(amp_c)]
    phase_p -= phase_p[np.argmax(amp_p)]
    mask = np.maximum(amp_c, amp_p) > float(phase_threshold)
    if np.any(mask):
        phase_diff = np.angle(np.exp(1j * (phase_p[mask] - phase_c[mask])))
        phase_rmse = float(np.sqrt(np.mean(phase_diff**2)))
    else:
        phase_rmse = float("nan")

    return {
        "p_rel": rel(p_c, p_p),
        "rho_rel": rel(rho_c, rho_p),
        "u_rel": rel(u_c, u_p),
        "v_rel": rel(v_c, v_p),
        "amp_rel": float(np.linalg.norm(amp_p - amp_c) / max(np.linalg.norm(amp_c), 1e-12)),
        "phase_rmse": phase_rmse,
    }


def plot_ci_outputs(output_dir: Path, alpha_values: np.ndarray, ci_ref: np.ndarray, ci_pred: np.ndarray) -> tuple[float, float]:
    err = np.abs(ci_pred - ci_ref)
    df = pd.DataFrame({"alpha": alpha_values, "ci_reference": ci_ref, "ci_pinn": ci_pred, "ci_abs_err": err})
    df.to_csv(output_dir / "ci_curve_vs_reference.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(alpha_values, ci_ref, label="Classique", linewidth=2.0)
    axes[0].plot(alpha_values, ci_pred, "--", label="PINN", linewidth=2.0)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(r"Courbe $c_i(\alpha)$")

    axes[1].plot(alpha_values, err, color="tab:red")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$|c_i^{PINN}-c_i^{ref}|$")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(r"Erreur absolue sur $c_i$")
    fig.tight_layout()
    fig.savefig(output_dir / "ci_curve_vs_reference.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_curve)
    ax_curve.plot(alpha_values, ci_ref, label="Classique", linewidth=2.0)
    ax_curve.plot(alpha_values, ci_pred, "--", label="PINN", linewidth=2.0)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.set_title(r"Comparaison $c_i(\alpha)$ + bande d'erreur")
    heat = np.tile(err[None, :], (6, 1))
    im = ax_heat.imshow(heat, aspect="auto", origin="lower", extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0], cmap="magma")
    ax_heat.set_yticks([])
    ax_heat.set_xlabel(r"$\alpha$")
    ax_heat.set_title(r"Bande heatmap de l'erreur")
    fig.colorbar(im, ax=ax_heat, pad=0.02, label=r"$|c_i^{PINN}-c_i^{ref}|$")
    fig.savefig(output_dir / "ci_error_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(output_dir / "ci_error_heatmap.csv", index=False)
    return float(err.mean()), float(err.max())


def plot_overlay_pdf(output_dir: Path, candidate: Candidate, overlay_alphas: list[float], classic_cache: dict[float, tuple[dict[str, np.ndarray], float]], *, n_y_pinn: int, device: torch.device) -> pd.DataFrame:
    rows = []
    titles = [
        ("rho", r"Density Perturbation $\hat{\rho}$"),
        ("u", r"Streamwise Velocity $\hat{u}$"),
        ("v", r"Vertical Velocity $\hat{v}$"),
        ("p", r"Pressure Perturbation $\hat{p}$"),
    ]
    pdf_path = output_dir / "classic_vs_pinn_modes_overlay.pdf"
    csv_path = output_dir / "classic_vs_pinn_modes_overlay.csv"

    with PdfPages(pdf_path) as pdf:
        for alpha in overlay_alphas:
            classic, ci_classic = classic_cache[float(alpha)]
            pinn, ci_pinn = load_pinn_full_mode(candidate, alpha=float(alpha), n_y=n_y_pinn, device=device)
            y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
            y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
            y_common = np.linspace(y_min, y_max, 1200, dtype=float)
            metrics = compute_mode_metrics(classic, pinn, y_common=y_common)
            rows.append({"alpha": float(alpha), "mach": candidate.mach, "ci_classic": ci_classic, "ci_pinn": ci_pinn, "ci_abs_err": abs(ci_pinn - ci_classic), **metrics})

            x_limits_classic = compute_visible_xlim(classic["y"], [classic["rho"], classic["u"], classic["v"], classic["p"]])
            x_limits_pinn = compute_visible_xlim(pinn["y"], [pinn["rho"], pinn["u"], pinn["v"], pinn["p"]])
            x_limits = (min(x_limits_classic[0], x_limits_pinn[0]), max(x_limits_classic[1], x_limits_pinn[1]))

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            for ax, (field_name, title) in zip(axes.flat, titles):
                field_c = classic[field_name]
                field_p = pinn[field_name]
                ax.plot(classic["y"], np.real(field_c), color="tab:blue", linewidth=1.6, label="Classic Re")
                ax.plot(classic["y"], np.imag(field_c), color="tab:orange", linewidth=1.6, label="Classic Im")
                ax.plot(pinn["y"], np.real(field_p), "--", color="tab:blue", linewidth=1.6, label="PINN Re")
                ax.plot(pinn["y"], np.imag(field_p), "--", color="tab:orange", linewidth=1.6, label="PINN Im")
                ax.set_title(title)
                ax.set_xlim(*x_limits)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7)

            fig.suptitle(
                fr"{candidate.name} | $\alpha={alpha:.3f}$, $M={candidate.mach:.3f}$"
                "\n"
                fr"$c_i^{{classic}}={ci_classic:.5f}$ | $c_i^{{PINN}}={ci_pinn:.5f}$ | "
                fr"$p_{{rel}}={metrics['p_rel']:.3e}$ | phase RMSE={metrics['phase_rmse']:.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def plot_mode_heatmaps(output_dir: Path, candidate: Candidate, alpha_values: np.ndarray, classic_cache: dict[float, tuple[dict[str, np.ndarray], float]], *, y_max: float, y_common_n: int, n_y_pinn: int, device: torch.device) -> pd.DataFrame:
    y_common = np.linspace(-float(y_max), float(y_max), int(y_common_n), dtype=float)
    fields = ("rho", "u", "v", "p")
    error_maps = {name: np.full((len(alpha_values), len(y_common)), np.nan, dtype=float) for name in fields}
    summary_rows = []
    long_rows = []

    for i, alpha in enumerate(alpha_values):
        classic, ci_classic = classic_cache[float(alpha)]
        pinn, ci_pinn = load_pinn_full_mode(candidate, alpha=float(alpha), n_y=n_y_pinn, device=device)
        row = {"alpha": float(alpha), "mach": candidate.mach, "ci_classic": ci_classic, "ci_pinn": ci_pinn, "ci_abs_err": abs(ci_pinn - ci_classic)}
        metrics = compute_mode_metrics(classic, pinn, y_common=y_common)
        row.update(metrics)
        for field in fields:
            classic_i = interp_complex(classic["y"], classic[field], y_common)
            pinn_i = interp_complex(pinn["y"], pinn[field], y_common)
            abs_err = np.abs(pinn_i - classic_i)
            error_maps[field][i, :] = abs_err
            for y_val, err_val in zip(y_common, abs_err):
                long_rows.append({"alpha": float(alpha), "mach": candidate.mach, "field": field, "y": float(y_val), "abs_err": float(err_val)})
        summary_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)
    long_df.to_csv(output_dir / "mode_field_error_heatmaps.csv", index=False)
    summary_df.to_csv(output_dir / "mode_field_error_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True, sharex=True, sharey=True)
    titles = {
        "rho": r"Erreur $|\hat{\rho}_{PINN}-\hat{\rho}_{classic}|$",
        "u": r"Erreur $|\hat{u}_{PINN}-\hat{u}_{classic}|$",
        "v": r"Erreur $|\hat{v}_{PINN}-\hat{v}_{classic}|$",
        "p": r"Erreur $|\hat{p}_{PINN}-\hat{p}_{classic}|$",
    }
    for ax, field in zip(axes.flat, fields):
        pcm = ax.pcolormesh(alpha_values, y_common, error_maps[field].T, shading="auto", cmap="magma")
        ax.set_title(titles[field])
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("y")
        fig.colorbar(pcm, ax=ax)
    fig.suptitle(fr"{candidate.name} | heatmaps d'erreur modale vs classique ($M={candidate.mach:.2f}$)")
    fig.savefig(output_dir / "mode_field_error_heatmaps.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rapidement les candidats PINN 1D modaux a Mach fixe.")
    parser.add_argument("--output-root", type=Path, default=Path("assets/pinn_subsonic/mach_fixed_candidates"))
    parser.add_argument("--candidates", type=str, nargs="*", default=list(DEFAULT_CANDIDATES.keys()))
    parser.add_argument("--num-alpha-ci", type=int, default=41)
    parser.add_argument("--num-alpha-modes", type=int, default=21)
    parser.add_argument("--overlay-alphas", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-common", type=int, default=801)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidate_names = args.candidates
    candidates = [load_candidate(name, DEFAULT_CANDIDATES[name], device) for name in candidate_names]
    mach_values = {cand.mach for cand in candidates}
    if len(mach_values) != 1:
        raise RuntimeError(f"Expected same Mach for all candidates, got {mach_values}")
    mach = next(iter(mach_values))
    alpha_min = max(float(c.config["alpha_min"]) for c in candidates)
    alpha_max = min(float(c.config["alpha_max"]) for c in candidates)

    alpha_values_ci = np.linspace(alpha_min, alpha_max, int(args.num_alpha_ci), dtype=float)
    alpha_values_modes = np.linspace(alpha_min, alpha_max, int(args.num_alpha_modes), dtype=float)
    overlay_alphas = [float(alpha) for alpha in args.overlay_alphas]

    print(f"Compute classical ci reference for Mach={mach:.3f} on {len(alpha_values_ci)} alphas.")
    ci_ref = np.array(
        [RobustSubsonicShootingSolver(alpha=float(alpha), Mach=mach).solve().ci for alpha in alpha_values_ci],
        dtype=float,
    )

    print(f"Compute classical modal reference for Mach={mach:.3f} on {len(alpha_values_modes)} alphas.")
    classic_mode_cache = {float(alpha): load_classic_full_mode(float(alpha), mach) for alpha in alpha_values_modes}
    for alpha in overlay_alphas:
        if float(alpha) not in classic_mode_cache:
            classic_mode_cache[float(alpha)] = load_classic_full_mode(float(alpha), mach)

    summary_rows = []
    for cand in candidates:
        print(f"\n=== Candidate: {cand.name} ===")
        out_dir = args.output_root / cand.name
        ci_dir = out_dir / "ci"
        modes_dir = out_dir / "modes"
        ci_dir.mkdir(parents=True, exist_ok=True)
        modes_dir.mkdir(parents=True, exist_ok=True)

        alpha_tensor = torch.tensor(alpha_values_ci, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            ci_pred = cand.model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
        ci_mae, ci_max = plot_ci_outputs(ci_dir, alpha_values_ci, ci_ref, ci_pred)

        overlay_df = plot_overlay_pdf(
            modes_dir,
            cand,
            overlay_alphas,
            classic_mode_cache,
            n_y_pinn=int(args.n_y_pinn),
            device=device,
        )
        mode_summary_df = plot_mode_heatmaps(
            modes_dir,
            cand,
            alpha_values_modes,
            classic_mode_cache,
            y_max=float(args.y_max),
            y_common_n=int(args.y_common),
            n_y_pinn=int(args.n_y_pinn),
            device=device,
        )

        summary_rows.append(
            {
                "candidate": cand.name,
                "mach": cand.mach,
                "ci_mae": ci_mae,
                "ci_max_abs": ci_max,
                "mode_p_rel_mean": float(mode_summary_df["p_rel"].mean()),
                "mode_p_rel_max": float(mode_summary_df["p_rel"].max()),
                "mode_u_rel_mean": float(mode_summary_df["u_rel"].mean()),
                "mode_u_rel_max": float(mode_summary_df["u_rel"].max()),
                "mode_v_rel_mean": float(mode_summary_df["v_rel"].mean()),
                "mode_v_rel_max": float(mode_summary_df["v_rel"].max()),
                "mode_phase_rmse_mean": float(mode_summary_df["phase_rmse"].mean()),
                "overlay_p_rel_mean": float(overlay_df["p_rel"].mean()),
                "overlay_phase_rmse_mean": float(overlay_df["phase_rmse"].mean()),
            }
        )
        print(
            f"{cand.name}: ci_mae={ci_mae:.3e} ci_max={ci_max:.3e} "
            f"mode_p_rel_mean={mode_summary_df['p_rel'].mean():.3e} "
            f"mode_u_rel_mean={mode_summary_df['u_rel'].mean():.3e}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["mode_p_rel_mean", "ci_mae"], kind="stable")
    summary_path = args.output_root / "candidate_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
