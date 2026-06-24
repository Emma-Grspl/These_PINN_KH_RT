from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.data.kh_subsonic_sampling_2d import build_reference_surface, resolve_reference_cache_path
from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    reconstruct_pressure_p_y_from_riccati_2d,
    xi_to_y,
)
from src.training.kh_subsonic_trainer_2d_stage1 import load_stage0_model_from_checkpoint


def _parse_float_list(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        import json

        return [float(item) for item in json.loads(text)]
    return [float(item) for item in text.replace(",", " ").split()]


def _load_stage1ter_model(stage1ter_dir: Path, *, device: torch.device):
    checkpoint_path = stage1ter_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported Stage 1ter checkpoint format in {checkpoint_path}.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        config_df = pd.read_csv(stage1ter_dir / "config.csv")
        model_config = config_df.iloc[0].to_dict()
    from src.models.kh_subsonic_pinn_2d import build_kh_subsonic_pinn_2d_from_config

    model = build_kh_subsonic_pinn_2d_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    config_df = pd.read_csv(stage1ter_dir / "config.csv")
    config = config_df.iloc[0].to_dict()
    return model, config, checkpoint


def _build_prediction_surface(model, alpha_values: np.ndarray, mach_values: np.ndarray, *, device: torch.device) -> pd.DataFrame:
    aa, mm = np.meshgrid(alpha_values, mach_values)
    alpha_t = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_t = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_t, mach_t).detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({"alpha": aa.reshape(-1), "Mach": mm.reshape(-1), "ci_pinn": ci_pred})


def plot_ci_curves_by_mach(
    grid_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    *,
    mach_values: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(mach_values), 1, figsize=(8.0, 3.0 * len(mach_values)), sharex=True)
    if len(mach_values) == 1:
        axes = [axes]
    for ax, mach in zip(axes, mach_values):
        sub = grid_df[np.isclose(grid_df["Mach"], float(mach), atol=1e-12)].sort_values("alpha")
        anc = anchors_df[np.isclose(anchors_df["Mach"], float(mach), atol=1e-12)].sort_values("alpha")
        ax.plot(sub["alpha"], sub["ci_reference"], color="black", linewidth=1.6, label="Classique")
        ax.plot(sub["alpha"], sub["ci_pinn"], color="tab:orange", linewidth=1.6, linestyle="--", label="PINN Stage 1ter")
        ax.scatter(anc["alpha"], anc["ci_reference"], color="black", s=28, zorder=3, label="Anchors")
        ax.scatter(anc["alpha"], anc["ci_pred_stage1"], color="tab:orange", s=28, zorder=3)
        ax.set_ylabel(r"$c_i$")
        ax.set_title(f"Mach = {mach:.2f}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$\alpha$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ci_error_heatmap(grid_df: pd.DataFrame, output_path: Path) -> None:
    work = grid_df.copy()
    ci_ref = work["ci_reference"].to_numpy(dtype=float)
    ci_abs_err = work["ci_abs_err"].to_numpy(dtype=float)
    work["ci_error_display"] = np.where(
        np.abs(ci_ref) > 1e-4,
        ci_abs_err / np.maximum(np.abs(ci_ref), 1.0e-12),
        ci_abs_err,
    )
    pivot = work.pivot(index="Mach", columns="alpha", values="ci_error_display").sort_index().sort_index(axis=1)
    alpha_values = pivot.columns.to_numpy(dtype=float)
    mach_values = pivot.index.to_numpy(dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pcm = ax.pcolormesh(aa, mm, pivot.to_numpy(dtype=float), shading="auto", cmap="magma")
    ax.set_title(r"Erreur $c_i$: relative instable / absolue neutre")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, alpha=0.20)
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), constrained_layout=True)
    left_keys = [
        "loss_total",
        "loss_pde",
        "loss_bc",
        "loss_match",
        "loss_center_pde",
        "loss_norm",
        "loss_phase",
    ]
    for key in left_keys:
        if key in history.columns:
            axes[0].plot(history["epoch"], history[key], label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss history")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    right_keys = [
        "w_match_effective",
        "ci_anchor_max_abs",
        "ci_anchor_max_rel_unstable",
        "ci_neutral_max_abs",
        "gamma_left_right_abs_mean",
        "gamma_left_right_abs_max",
    ]
    for key in right_keys:
        if key in history.columns:
            axes[1].plot(history["epoch"], history[key], label=key)
    axes[1].set_yscale("log")
    axes[1].set_title("Matching and spectral lock")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def _normalize_extended_mode(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
    gamma: np.ndarray,
    p_y: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho, p_y = u * phase, v * phase, p * phase, rho * phase, p_y * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho, p_y = -u, -v, -p, -rho, -p_y

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return {
        "y": np.asarray(y, dtype=float),
        "u": np.asarray(u / scale, dtype=np.complex128),
        "v": np.asarray(v / scale, dtype=np.complex128),
        "p": np.asarray(p / scale, dtype=np.complex128),
        "rho": np.asarray(rho / scale, dtype=np.complex128),
        "p_y": np.asarray(p_y / scale, dtype=np.complex128),
        "gamma": np.asarray(gamma, dtype=np.complex128),
    }


def load_classic_full_mode_extended(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
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
    c = 1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)

    return _normalize_extended_mode(y, u, v, p, rho, gamma, p_y), float(result.ci)


def load_pinn_full_mode_extended(
    model,
    *,
    alpha: float,
    mach: float,
    n_y: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    mach_tensor = torch.full_like(xi, float(mach))
    xi.requires_grad_(True)

    pr, pi, p_y, gamma, y_t = reconstruct_pressure_p_y_from_riccati_2d(
        model,
        xi,
        alpha_tensor,
        mach_tensor,
        anchor_xi=0.0,
    )
    p = torch.complex(pr, pi)
    ci = float(
        model.get_ci(
            torch.tensor([[alpha]], dtype=torch.float32, device=device),
            torch.tensor([[mach]], dtype=torch.float32, device=device),
        ).item()
    )
    c = 1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (float(mach) ** 2)

    fields = _normalize_extended_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
        gamma[:, 0].detach().cpu().numpy(),
        p_y[:, 0].detach().cpu().numpy(),
    )
    return fields, ci


def compute_extended_mode_metrics(
    classic: dict[str, np.ndarray],
    pinn: dict[str, np.ndarray],
    *,
    n_common: int = 1200,
    phase_threshold: float = 1e-2,
) -> dict[str, float]:
    y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
    y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
    y_common = np.linspace(y_min, y_max, int(n_common), dtype=float)

    fields = {}
    for name in ("p", "rho", "u", "v", "gamma", "p_y"):
        fields[f"{name}_c"] = _interp_complex(classic["y"], classic[name], y_common)
        fields[f"{name}_p"] = _interp_complex(pinn["y"], pinn[name], y_common)

    def rel(field_a: np.ndarray, field_b: np.ndarray) -> float:
        return float(np.linalg.norm(field_b - field_a) / max(np.linalg.norm(field_a), 1e-12))

    amp_c = np.abs(fields["p_c"])
    amp_p = np.abs(fields["p_p"])
    amp_rel = float(np.linalg.norm(amp_p - amp_c) / max(np.linalg.norm(amp_c), 1e-12))

    phase_c = np.unwrap(np.angle(fields["p_c"]))
    phase_p = np.unwrap(np.angle(fields["p_p"]))
    phase_c -= phase_c[np.argmax(amp_c)]
    phase_p -= phase_p[np.argmax(amp_p)]
    mask = np.maximum(amp_c, amp_p) > float(phase_threshold)
    if np.any(mask):
        phase_diff = np.angle(np.exp(1j * (phase_p[mask] - phase_c[mask])))
        phase_rmse = float(np.sqrt(np.mean(phase_diff**2)))
    else:
        phase_rmse = float("nan")

    return {
        "p_rel": rel(fields["p_c"], fields["p_p"]),
        "rho_rel": rel(fields["rho_c"], fields["rho_p"]),
        "u_rel": rel(fields["u_c"], fields["u_p"]),
        "v_rel": rel(fields["v_c"], fields["v_p"]),
        "gamma_rel": rel(fields["gamma_c"], fields["gamma_p"]),
        "p_y_rel": rel(fields["p_y_c"], fields["p_y_p"]),
        "amp_rel": amp_rel,
        "phase_rmse": phase_rmse,
    }


def _plot_mode_fields(
    classic_fields: dict[str, np.ndarray],
    pinn_fields: dict[str, np.ndarray],
    *,
    alpha: float,
    mach: float,
    output_path: Path,
) -> None:
    titles = [("rho", r"$\hat{\rho}$"), ("u", r"$\hat{u}$"), ("v", r"$\hat{v}$"), ("p", r"$\hat{p}$")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for ax, (field_name, title) in zip(axes.flat, titles):
        ax.plot(classic_fields["y"], np.real(classic_fields[field_name]), color="black", linewidth=1.4, label="Classic Re")
        ax.plot(classic_fields["y"], np.imag(classic_fields[field_name]), color="black", linewidth=1.0, linestyle=":", label="Classic Im")
        ax.plot(pinn_fields["y"], np.real(pinn_fields[field_name]), color="tab:orange", linewidth=1.4, label="PINN Re")
        ax.plot(pinn_fields["y"], np.imag(pinn_fields[field_name]), color="tab:orange", linewidth=1.0, linestyle=":", label="PINN Im")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 1ter mode diagnostic | alpha={alpha:.2f}, Mach={mach:.2f}")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gamma_py_fields(
    classic_fields: dict[str, np.ndarray],
    pinn_fields: dict[str, np.ndarray],
    *,
    alpha: float,
    mach: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), constrained_layout=True)
    panels = [
        ("gamma", "real", r"$\Re(\gamma)$"),
        ("gamma", "imag", r"$\Im(\gamma)$"),
        ("p_y", "real", r"$\Re(p_y)$"),
        ("p_y", "imag", r"$\Im(p_y)$"),
    ]
    for ax, (field_name, component, title) in zip(axes.flat, panels):
        extractor = np.real if component == "real" else np.imag
        ax.plot(classic_fields["y"], extractor(classic_fields[field_name]), color="black", linewidth=1.4, label="Classic")
        ax.plot(pinn_fields["y"], extractor(pinn_fields[field_name]), color="tab:orange", linewidth=1.4, label="PINN")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 1ter gamma / p_y diagnostic | alpha={alpha:.2f}, Mach={mach:.2f}")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_readme(output_dir: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Stage 1ter Diagnostics",
        "",
        "Ce dossier contient des diagnostics **post-traitement** pour le Stage 1ter 2D subsonique.",
        "",
        "Important :",
        "",
        "- le classique est utilise ici uniquement pour comparer apres entrainement",
        "- aucun mode classique n'est injecte dans la loss d'entrainement",
        "- l'entrainement Stage 1ter utilise uniquement PDE, BC Riccati, matching Riccati et contraintes spectrales eventuelles sur `c_i`",
        "",
        "Fichiers principaux :",
        "",
        "- `01_stage1ter_ci_vs_alpha_by_mach.png`",
        "- `02_stage1ter_ci_error_heatmap.png`",
        "- `03_stage1ter_loss_history.png`",
        "- `04/06/08_stage1ter_modes_*.png` pour `rho,u,v,p`",
        "- `05/07/09_stage1ter_gamma_py_*.png` pour `gamma` et `p_y`",
        "- `diagnostics_summary.csv`",
        "- `stage1ter_ci_surface_table.csv`",
        "",
        f"Nombre de points modaux resumes : {len(summary_df)}",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render diagnostics for the 2D Stage 1ter Riccati matching run.")
    parser.add_argument("--stage1ter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/baseline_2D_Stage1ter_matching"),
    )
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--grid-alpha", type=int, default=61)
    parser.add_argument("--grid-mach", type=int, default=31)
    parser.add_argument("--mode-alpha-values", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    parser.add_argument("--mode-mach", type=float, default=0.50)
    parser.add_argument("--mode-n-y", type=int, default=1001)
    parser.add_argument("--mode-n-common", type=int, default=1200)
    parser.add_argument("--phase-threshold", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")

    model, config, checkpoint = _load_stage1ter_model(args.stage1ter_dir, device=device)
    history = pd.read_csv(args.stage1ter_dir / "history.csv")
    anchors_df = pd.read_csv(args.stage1ter_dir / "anchor_predictions_stage1ter.csv")

    mach_values = _parse_float_list(config.get("mach_values_json") or config.get("mach_values"))
    if not mach_values:
        mach_values = sorted(anchors_df["Mach"].drop_duplicates().to_list())

    alpha_grid = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.grid_alpha), dtype=float)
    mach_grid = np.linspace(min(mach_values), max(mach_values), int(args.grid_mach), dtype=float)

    reference_cache = args.reference_cache if args.reference_cache is not None else resolve_reference_cache_path()
    reference_df = build_reference_surface(alpha_values=alpha_grid, mach_values=mach_grid, reference_cache=reference_cache)
    pred_df = _build_prediction_surface(model, alpha_grid, mach_grid, device=device)
    grid_df = reference_df.merge(pred_df, on=["alpha", "Mach"], how="inner")
    grid_df["ci_abs_err"] = np.abs(grid_df["ci_pinn"] - grid_df["ci_reference"])
    grid_df.to_csv(args.output_dir / "stage1ter_ci_surface_table.csv", index=False)

    plot_ci_curves_by_mach(
        grid_df,
        anchors_df,
        mach_values=mach_values,
        output_path=args.output_dir / "01_stage1ter_ci_vs_alpha_by_mach.png",
    )
    plot_ci_error_heatmap(grid_df, args.output_dir / "02_stage1ter_ci_error_heatmap.png")
    plot_loss_history(history, args.output_dir / "03_stage1ter_loss_history.png")

    summary_rows: list[dict[str, float]] = []
    for index, alpha in enumerate(args.mode_alpha_values, start=1):
        classic_fields, ci_classic = load_classic_full_mode_extended(alpha=float(alpha), mach=float(args.mode_mach))
        pinn_fields, ci_pinn = load_pinn_full_mode_extended(
            model,
            alpha=float(alpha),
            mach=float(args.mode_mach),
            n_y=int(args.mode_n_y),
            device=device,
        )
        metrics = compute_extended_mode_metrics(
            classic_fields,
            pinn_fields,
            n_common=int(args.mode_n_common),
            phase_threshold=float(args.phase_threshold),
        )
        summary_rows.append(
            {
                "alpha": float(alpha),
                "mach": float(args.mode_mach),
                "ci_classic": float(ci_classic),
                "ci_pinn": float(ci_pinn),
                "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                **metrics,
            }
        )

        mode_path = args.output_dir / f"{2 * index + 2:02d}_stage1ter_modes_M05_alpha{int(round(alpha * 1000)):03d}_2x2.png"
        gamma_path = args.output_dir / f"{2 * index + 3:02d}_stage1ter_gamma_py_M05_alpha{int(round(alpha * 1000)):03d}.png"
        _plot_mode_fields(classic_fields, pinn_fields, alpha=float(alpha), mach=float(args.mode_mach), output_path=mode_path)
        _plot_gamma_py_fields(classic_fields, pinn_fields, alpha=float(alpha), mach=float(args.mode_mach), output_path=gamma_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "diagnostics_summary.csv", index=False)
    _write_readme(args.output_dir, summary_df)
    print(f"Stage 1ter diagnostics written to {args.output_dir}")


if __name__ == "__main__":
    main()
