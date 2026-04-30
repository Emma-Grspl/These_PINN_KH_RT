from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati


def load_model(run_dir: Path, checkpoint: Path | None, device: torch.device):
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    history = pd.read_csv(run_dir / "history.csv")
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(checkpoint or (run_dir / "model_best.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, history


def solve_reference_curve(mach: float, alpha_min: float, alpha_max: float, num_alpha: int) -> pd.DataFrame:
    rows = []
    for alpha in np.linspace(alpha_min, alpha_max, num_alpha):
        result = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve()
        rows.append({"alpha": float(alpha), "ci_reference": float(result.ci)})
    return pd.DataFrame(rows)


def load_reference_curve_from_cache(alpha_values: np.ndarray) -> pd.DataFrame | None:
    candidates = [
        ROOT_DIR / "model_saved" / "kh_subsonic_fixed_mach_M05" / "ci_curve_vs_reference.csv",
        ROOT_DIR / "model_saved" / "kh_subsonic_fixed_mach_M05_adaptive_jz" / "ci_curve_vs_reference.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "alpha" not in df.columns or "ci_reference" not in df.columns:
            continue
        ref = np.interp(alpha_values, df["alpha"].to_numpy(dtype=float), df["ci_reference"].to_numpy(dtype=float))
        return pd.DataFrame({"alpha": alpha_values, "ci_reference": ref})
    return None


def load_ci_curve(model, alpha_values: np.ndarray, device: torch.device) -> np.ndarray:
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        return model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)


def normalize_pressure(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
    if np.max(np.real(p)) < abs(np.min(np.real(p))):
        p = -p
    scale = max(np.max(np.abs(p)), 1e-12)
    return y, p / scale


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def load_classic_mode(alpha: float, mach: float) -> tuple[np.ndarray, np.ndarray, float]:
    solver = Mstab17SubsonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    ln_p_left, phi_left = np.asarray(sol_left.y[2]), np.asarray(sol_left.y[3])
    ln_p_right, phi_right = np.asarray(sol_right.y[2]), np.asarray(sol_right.y[3])

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    y = np.concatenate([y_left[y_left < 0.0], y_right[::-1]])
    p = np.concatenate([p_left[y_left < 0.0], p_right[::-1]])
    return normalize_pressure(y, p)[0], normalize_pressure(y, p)[1], float(result.ci)


def load_pinn_mode(model, alpha: float, device: torch.device, n_y: int = 1001) -> tuple[np.ndarray, np.ndarray, float]:
    xi = torch.linspace(-0.98, 0.98, n_y, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    with torch.no_grad():
        pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
        y = y_t.cpu().numpy().reshape(-1)
        p = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    return normalize_pressure(y, p)[0], normalize_pressure(y, p)[1], ci


def copy_run_artifacts(run_dir: Path, dst_dir: Path, prefix: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "config.csv", dst_dir / f"{prefix}_config.csv")
    shutil.copy2(run_dir / "history.csv", dst_dir / f"{prefix}_history.csv")


def plot_ci_comparison(
    run_a,
    run_b,
    label_a: str,
    label_b: str,
    output_png: Path,
    output_csv: Path,
    device: torch.device,
    num_alpha: int,
) -> None:
    model_a, config_a, _ = load_model(run_a, None, device)
    model_b, _, _ = load_model(run_b, None, device)
    alpha_values = np.linspace(float(config_a["alpha_min"]), float(config_a["alpha_max"]), num_alpha)
    ci_ref_df = load_reference_curve_from_cache(alpha_values)
    if ci_ref_df is None:
        ci_ref_df = solve_reference_curve(
            float(config_a["mach"]),
            float(config_a["alpha_min"]),
            float(config_a["alpha_max"]),
            num_alpha,
        )
    ci_ref = ci_ref_df["ci_reference"].to_numpy()
    ci_a = load_ci_curve(model_a, alpha_values, device)
    ci_b = load_ci_curve(model_b, alpha_values, device)
    err_a = np.abs(ci_a - ci_ref)
    err_b = np.abs(ci_b - ci_ref)

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_err_a = fig.add_subplot(gs[1, 0], sharex=ax_curve)
    ax_err_b = fig.add_subplot(gs[2, 0], sharex=ax_curve)

    ax_curve.plot(alpha_values, ci_ref, color="black", linewidth=2.2, label="Classique")
    ax_curve.plot(alpha_values, ci_a, "--", linewidth=2.0, label=label_a)
    ax_curve.plot(alpha_values, ci_b, "-.", linewidth=2.0, label=label_b)
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.set_title(r"Comparaison 1D de $c_i(\alpha)$ a $M=0.5$")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    im_a = ax_err_a.imshow(np.tile(err_a[None, :], (6, 1)), aspect="auto", origin="lower",
                           extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0], cmap="magma")
    ax_err_a.set_yticks([])
    ax_err_a.set_title(fr"Erreur absolue {label_a}")
    fig.colorbar(im_a, ax=ax_err_a, pad=0.02).set_label(r"$|c_i-c_i^{ref}|$")

    im_b = ax_err_b.imshow(np.tile(err_b[None, :], (6, 1)), aspect="auto", origin="lower",
                           extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0], cmap="magma")
    ax_err_b.set_yticks([])
    ax_err_b.set_xlabel(r"$\alpha$")
    ax_err_b.set_title(fr"Erreur absolue {label_b}")
    fig.colorbar(im_b, ax=ax_err_b, pad=0.02).set_label(r"$|c_i-c_i^{ref}|$")

    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "alpha": alpha_values,
        "ci_reference": ci_ref,
        "ci_experts": ci_a,
        "ci_light_anchor": ci_b,
        "err_experts": err_a,
        "err_light_anchor": err_b,
    }).to_csv(output_csv, index=False)


def plot_history_comparison(hist_a: pd.DataFrame, hist_b: pd.DataFrame, label_a: str, label_b: str, output_png: Path) -> None:
    aud_a = hist_a.dropna(subset=["audit_checkpoint_metric"]).copy()
    aud_b = hist_b.dropna(subset=["audit_checkpoint_metric"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("audit_ci_mae", r"Audit $c_i$ MAE"),
        ("audit_p_rel_l2_mean", r"Erreur mode $L^2$"),
        ("audit_env_rel_mean", r"Erreur enveloppe"),
        ("audit_phase_rel_mean", r"Erreur phase"),
    ]
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(aud_a["epoch"], aud_a[col], label=label_a, linewidth=2.0)
        ax.plot(aud_b["epoch"], aud_b[col], label=label_b, linewidth=2.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    fig.suptitle("Historique des audits 1D")
    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mode_heatmap(hist_a: pd.DataFrame, hist_b: pd.DataFrame, label_a: str, label_b: str, output_png: Path, output_csv: Path) -> None:
    def best_and_final(df: pd.DataFrame) -> pd.DataFrame:
        audited = df.dropna(subset=["audit_checkpoint_metric"]).copy()
        best = audited.sort_values("audit_checkpoint_metric").iloc[0]
        final = audited.iloc[-1]
        return pd.DataFrame([best, final], index=["best", "final"])

    a = best_and_final(hist_a)
    b = best_and_final(hist_b)
    metrics = [
        ("audit_ci_mae", "ci_mae"),
        ("audit_p_rel_l2_mean", "p_rel"),
        ("audit_env_rel_mean", "env"),
        ("audit_phase_rel_mean", "phase"),
        ("audit_peak_shift_mean", "peak"),
        ("audit_checkpoint_metric", "metric"),
    ]
    rows = []
    for label, df in [(label_a, a), (label_b, b)]:
        for state in ["best", "final"]:
            row = {"run": label, "state": state}
            for key, short in metrics:
                row[short] = float(df.loc[state, key])
            rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    mat = out_df[[m[1] for m in metrics]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_yticks(np.arange(len(out_df)))
    ax.set_yticklabels([f"{r.run} | {r.state}" for r in out_df.itertuples()])
    ax.set_title("Synthese des metriques de mode 1D")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mode_comparison(run_a: Path, run_b: Path, label_a: str, label_b: str, output_png: Path, alpha: float, device: torch.device) -> None:
    model_a, _, _ = load_model(run_a, None, device)
    model_b, _, _ = load_model(run_b, None, device)
    y_c, p_c, ci_c = load_classic_mode(alpha, 0.5)
    y_a, p_a, ci_a = load_pinn_mode(model_a, alpha, device)
    y_b, p_b, ci_b = load_pinn_mode(model_b, alpha, device)
    y_common = np.linspace(max(y_c.min(), y_a.min(), y_b.min()), min(y_c.max(), y_a.max(), y_b.max()), 1200)
    p_c_i = interp_complex(y_c, p_c, y_common)
    p_a_i = interp_complex(y_a, p_a, y_common)
    p_b_i = interp_complex(y_b, p_b, y_common)
    amp_c, amp_a, amp_b = np.abs(p_c_i), np.abs(p_a_i), np.abs(p_b_i)
    phase_c = np.unwrap(np.angle(p_c_i)); phase_c -= phase_c[np.argmax(amp_c)]
    phase_a = np.unwrap(np.angle(p_a_i)); phase_a -= phase_a[np.argmax(amp_a)]
    phase_b = np.unwrap(np.angle(p_b_i)); phase_b -= phase_b[np.argmax(amp_b)]
    env = np.maximum.reduce([amp_c, amp_a, amp_b])
    mask = env >= 0.02 * float(np.max(env))
    x_min = float(np.min(y_common[mask])); x_max = float(np.max(y_common[mask]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    axes[0].plot(y_common, amp_c, color="black", linewidth=2.2, label="Classique")
    axes[0].plot(y_common, amp_a, "--", linewidth=2.0, label=label_a)
    axes[0].plot(y_common, amp_b, "-.", linewidth=2.0, label=label_b)
    axes[0].set_title(r"Amplitude $|\hat{p}|$")
    axes[0].set_ylabel("Amplitude normalisee")
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_common, phase_c, color="black", linewidth=2.2, label="Classique")
    axes[1].plot(y_common, phase_a, "--", linewidth=2.0, label=label_a)
    axes[1].plot(y_common, phase_b, "-.", linewidth=2.0, label=label_b)
    axes[1].set_title(r"Phase $\arg(\hat{p})$")
    axes[1].set_xlim(x_min, x_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        fr"Mode de pression a $\alpha={alpha:.2f}$, $M=0.5$"
        "\n"
        fr"Classique: $c_i={ci_c:.5f}$ | {label_a}: $c_i={ci_a:.5f}$ | {label_b}: $c_i={ci_b:.5f}$"
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_three_modes(run_dir: Path, label: str, output_png: Path, device: torch.device, alphas: list[float]) -> None:
    model, _, _ = load_model(run_dir, None, device)
    fig, axes = plt.subplots(len(alphas), 2, figsize=(12.5, 10.0), sharex=False)
    if len(alphas) == 1:
        axes = np.asarray([axes])
    for row_idx, alpha in enumerate(alphas):
        y_c, p_c, ci_c = load_classic_mode(alpha, 0.5)
        y_p, p_p, ci_p = load_pinn_mode(model, alpha, device)
        y_common = np.linspace(max(y_c.min(), y_p.min()), min(y_c.max(), y_p.max()), 1200)
        p_c_i = interp_complex(y_c, p_c, y_common)
        p_p_i = interp_complex(y_p, p_p, y_common)
        amp_c, amp_p = np.abs(p_c_i), np.abs(p_p_i)
        phase_c = np.unwrap(np.angle(p_c_i)); phase_c -= phase_c[np.argmax(amp_c)]
        phase_p = np.unwrap(np.angle(p_p_i)); phase_p -= phase_p[np.argmax(amp_p)]
        axes[row_idx, 0].plot(y_common, amp_c, color="black", linewidth=2.0, label="Classique")
        axes[row_idx, 0].plot(y_common, amp_p, "--", color="tab:blue", linewidth=2.0, label=label)
        axes[row_idx, 0].set_title(fr"Amplitude | $\alpha={alpha:.2f}$")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].legend()
        axes[row_idx, 1].plot(y_common, phase_c, color="black", linewidth=2.0, label="Classique")
        axes[row_idx, 1].plot(y_common, phase_p, "--", color="tab:orange", linewidth=2.0, label=label)
        axes[row_idx, 1].set_title(fr"Phase | $c_i^{{classic}}={ci_c:.4f}$, $c_i^{{PINN}}={ci_p:.4f}$")
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].legend()
    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle(f"Subsonique 1D : {label} vs classique sur trois modes")
    fig.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_summary(run_a: Path, run_b: Path, label_a: str, label_b: str, output_txt: Path) -> None:
    rows = []
    for run_dir, label in [(run_a, label_a), (run_b, label_b)]:
        hist = pd.read_csv(run_dir / "history.csv")
        audited = hist.dropna(subset=["audit_checkpoint_metric"]).copy()
        best = audited.sort_values("audit_checkpoint_metric").iloc[0]
        final = audited.iloc[-1]
        rows.append(f"{label}")
        rows.append(f"best_epoch={int(best['epoch'])}")
        rows.append(
            "best_metrics: "
            f"ci_mae={best['audit_ci_mae']:.6f}, "
            f"p_rel={best['audit_p_rel_l2_mean']:.6f}, "
            f"env={best['audit_env_rel_mean']:.6f}, "
            f"phase={best['audit_phase_rel_mean']:.6f}, "
            f"peak={best['audit_peak_shift_mean']:.6f}, "
            f"metric={best['audit_checkpoint_metric']:.6f}"
        )
        rows.append(
            "final_metrics: "
            f"ci_mae={final['audit_ci_mae']:.6f}, "
            f"p_rel={final['audit_p_rel_l2_mean']:.6f}, "
            f"env={final['audit_env_rel_mean']:.6f}, "
            f"phase={final['audit_phase_rel_mean']:.6f}, "
            f"peak={final['audit_peak_shift_mean']:.6f}, "
            f"metric={final['audit_checkpoint_metric']:.6f}"
        )
        rows.append("")
    output_txt.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le dossier de presentation 1D pour deux runs PINN subsoniques.")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="Experts")
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-b", type=str, default="Experts + light anchor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-alpha", type=int, default=41)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    copy_run_artifacts(args.run_a, out, "experts")
    copy_run_artifacts(args.run_b, out, "experts_light_anchor")

    _, _, hist_a = load_model(args.run_a, None, device)
    _, _, hist_b = load_model(args.run_b, None, device)

    plot_ci_comparison(
        args.run_a, args.run_b, args.label_a, args.label_b,
        out / "01_ci_curve_two_runs_vs_reference.png",
        out / "ci_curve_two_runs_vs_reference.csv",
        device,
        args.num_alpha,
    )
    plot_history_comparison(
        hist_a, hist_b, args.label_a, args.label_b,
        out / "02_history_diagnostics_1d_experts_vs_light_anchor.png",
    )
    plot_mode_comparison(
        args.run_a, args.run_b, args.label_a, args.label_b,
        out / "03_mode_comparison_a050_m050.png",
        alpha=0.50,
        device=device,
    )
    plot_three_modes(
        args.run_a, args.label_a,
        out / "04_three_modes_experts_m050.png",
        device=device,
        alphas=[0.30, 0.50, 0.70],
    )
    plot_three_modes(
        args.run_b, args.label_b,
        out / "05_three_modes_experts_light_anchor_m050.png",
        device=device,
        alphas=[0.30, 0.50, 0.70],
    )
    plot_mode_heatmap(
        hist_a, hist_b, args.label_a, args.label_b,
        out / "06_mode_error_heatmaps_1d.png",
        out / "mode_error_heatmaps_1d.csv",
    )
    write_summary(
        args.run_a, args.run_b, args.label_a, args.label_b,
        out / "best_checkpoint_summary.txt",
    )

    print(out)


if __name__ == "__main__":
    main()
