from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config  # noqa: E402
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, xi_to_y  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plots compares: supervision legere de c_i vs physique pure."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot_presentation/subsonic_pinn/ci_supervision_vs_physics"),
    )
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--mode-alphas", type=float, nargs="+", default=[0.20, 0.50, 0.80])
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_model(run_dir: Path, device: torch.device):
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return config, model


def predict_ci_curve(run_dir: Path, alpha_values: np.ndarray, device: torch.device) -> tuple[pd.Series, np.ndarray]:
    config, model = build_model(run_dir, device)
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_values = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    return config, ci_values


def normalize_pressure(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=complex).copy()
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
    if np.max(np.real(p)) < abs(np.min(np.real(p))):
        p = -p
    amp = max(float(np.max(np.abs(p))), 1e-12)
    return y, p / amp


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def load_mode(run_dir: Path, alpha: float, n_y_pinn: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    config, model = build_model(run_dir, device)
    xi = torch.linspace(-0.98, 0.98, n_y_pinn, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    with torch.no_grad():
        if str(config["mode_representation"]) == "riccati":
            pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
            y_pred = y_t.cpu().numpy().reshape(-1)
            p_pred = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor)
            y_pred = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p_pred = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
        ci_pred = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    return *normalize_pressure(y_pred, p_pred), ci_pred


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def save_ci_curve_and_heatmaps(
    alpha_values: np.ndarray,
    ci_ref: np.ndarray,
    ci_hybrid: np.ndarray,
    ci_physics: np.ndarray,
    output_dir: Path,
    mach: float,
) -> tuple[Path, Path]:
    err_h = np.abs(ci_hybrid - ci_ref)
    err_p = np.abs(ci_physics - ci_ref)

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_err_h = fig.add_subplot(gs[1, 0], sharex=ax_curve)
    ax_err_p = fig.add_subplot(gs[2, 0], sharex=ax_curve)

    ax_curve.plot(alpha_values, ci_ref, color="black", linewidth=2.2, label="Classique")
    ax_curve.plot(alpha_values, ci_hybrid, "--", linewidth=2.0, color="#0b6e4f", label="PINN hybride, 8 points")
    ax_curve.plot(alpha_values, ci_physics, "-.", linewidth=2.0, color="#c84c09", label="PINN physique pure")
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.set_title(fr"Comparison of $c_i(\alpha)$ at fixed Mach $M={mach:.3f}$")
    ax_curve.legend()

    heat_h = np.tile(err_h[None, :], (6, 1))
    im_h = ax_err_h.imshow(
        heat_h,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0],
        cmap="magma",
    )
    ax_err_h.set_yticks([])
    ax_err_h.set_title(r"Absolute error heatmap, PINN hybride (8 points)")
    cbar_h = fig.colorbar(im_h, ax=ax_err_h, pad=0.02)
    cbar_h.set_label(r"$|c_i - c_i^{ref}|$")

    heat_p = np.tile(err_p[None, :], (6, 1))
    im_p = ax_err_p.imshow(
        heat_p,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0],
        cmap="magma",
    )
    ax_err_p.set_yticks([])
    ax_err_p.set_xlabel(r"$\alpha$")
    ax_err_p.set_title(r"Absolute error heatmap, PINN physique pure")
    cbar_p = fig.colorbar(im_p, ax=ax_err_p, pad=0.02)
    cbar_p.set_label(r"$|c_i - c_i^{ref}|$")

    fig_path = output_dir / "ci_curve_hybrid8_vs_physics_only.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "ci_curve_hybrid8_vs_physics_only.csv"
    pd.DataFrame(
        {
            "alpha": alpha_values,
            "ci_reference": ci_ref,
            "ci_hybrid_8pt": ci_hybrid,
            "ci_physics_only": ci_physics,
            "err_hybrid_8pt": err_h,
            "err_physics_only": err_p,
        }
    ).to_csv(csv_path, index=False)
    return fig_path, csv_path


def save_mode_comparison(
    *,
    alpha: float,
    mach: float,
    hybrid_run: Path,
    physics_run: Path,
    n_y_pinn: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, dict[str, float]]:
    solver = NotebookStyleDenseGEPSolver(
        alpha=float(alpha),
        Mach=float(mach),
        n_points=561,
        mapping_scale=3.0,
        xi_max=0.99,
    )
    mode, _, _ = solver.get_selected_mode()
    if mode is None:
        raise RuntimeError(f"No classical mode found for alpha={alpha:.3f}, M={mach:.3f}.")
    y_classic, p_classic = normalize_pressure(solver.y, mode["vector"][2 * solver.n_points : 3 * solver.n_points])

    y_h, p_h, ci_h = load_mode(hybrid_run, alpha, n_y_pinn, device)
    y_p, p_p, ci_p = load_mode(physics_run, alpha, n_y_pinn, device)
    ci_ref = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve().ci

    y_min = max(float(np.min(y_classic)), float(np.min(y_h)), float(np.min(y_p)))
    y_max = min(float(np.max(y_classic)), float(np.max(y_h)), float(np.max(y_p)))
    y_common = np.linspace(y_min, y_max, 1200, dtype=float)

    p_classic_i = interp_complex(y_classic, p_classic, y_common)
    p_h_i = interp_complex(y_h, p_h, y_common)
    p_p_i = interp_complex(y_p, p_p, y_common)

    amp_classic = np.abs(p_classic_i)
    amp_h = np.abs(p_h_i)
    amp_p = np.abs(p_p_i)
    phase_classic = np.unwrap(np.angle(p_classic_i))
    phase_h = np.unwrap(np.angle(p_h_i))
    phase_p = np.unwrap(np.angle(p_p_i))
    phase_classic -= phase_classic[np.argmax(amp_classic)]
    phase_h -= phase_h[np.argmax(amp_h)]
    phase_p -= phase_p[np.argmax(amp_p)]

    env = np.maximum.reduce([amp_classic, amp_h, amp_p])
    mask = env >= 0.02 * float(np.max(env))
    x_min = float(np.min(y_common[mask])) if np.any(mask) else float(y_common[0])
    x_max = float(np.max(y_common[mask])) if np.any(mask) else float(y_common[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    axes[0].plot(y_common, amp_classic, color="black", linewidth=2.2, label="Classique")
    axes[0].plot(y_common, amp_h, "--", color="#0b6e4f", linewidth=2.0, label="PINN hybride, 8 points")
    axes[0].plot(y_common, amp_p, "-.", color="#c84c09", linewidth=2.0, label="PINN physique pure")
    axes[0].set_title(r"Amplitude $|\hat{p}|$")
    axes[0].set_ylabel("Normalised amplitude")
    axes[0].set_xlim(x_min, x_max)
    axes[0].legend()

    axes[1].plot(y_common, phase_classic, color="black", linewidth=2.2, label="Classique")
    axes[1].plot(y_common, phase_h, "--", color="#0b6e4f", linewidth=2.0, label="PINN hybride, 8 points")
    axes[1].plot(y_common, phase_p, "-.", color="#c84c09", linewidth=2.0, label="PINN physique pure")
    axes[1].set_title(r"Phase $\arg(\hat{p})$")
    axes[1].set_xlim(x_min, x_max)
    axes[1].legend()

    fig.suptitle(
        fr"Mode reconstruction at $\alpha={alpha:.3f}$, $M={mach:.3f}$"
        "\n"
        fr"Classic $c_i={mode['ci']:.5f}$ | Shooting $c_i={ci_ref:.5f}$ | Hybrid $c_i={ci_h:.5f}$ | Physics-only $c_i={ci_p:.5f}$"
    )
    fig.tight_layout()

    fig_path = output_dir / f"mode_reconstruction_alpha_{alpha:.3f}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    amp_rel_h = float(np.linalg.norm(amp_h - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12))
    amp_rel_p = float(np.linalg.norm(amp_p - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12))
    phase_rmse_h = float(np.sqrt(np.mean((phase_h[mask] - phase_classic[mask]) ** 2))) if np.any(mask) else np.nan
    phase_rmse_p = float(np.sqrt(np.mean((phase_p[mask] - phase_classic[mask]) ** 2))) if np.any(mask) else np.nan

    metrics = {
        "alpha": float(alpha),
        "ci_reference": float(ci_ref),
        "ci_classic_gep": float(mode["ci"]),
        "ci_hybrid_8pt": float(ci_h),
        "ci_physics_only": float(ci_p),
        "amp_rel_hybrid_8pt": amp_rel_h,
        "amp_rel_physics_only": amp_rel_p,
        "phase_rmse_hybrid_8pt": phase_rmse_h,
        "phase_rmse_physics_only": phase_rmse_p,
    }
    return fig_path, metrics


def save_summary_barplot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(summary_df))
    width = 0.36
    ax.bar(x - width / 2, summary_df["err_hybrid_8pt"], width=width, color="#0b6e4f", label="PINN hybride, 8 points")
    ax.bar(x + width / 2, summary_df["err_physics_only"], width=width, color="#c84c09", label="PINN physique pure")
    ax.set_xticks(x, [str(v) for v in summary_df["n_alpha_supervision"]])
    ax.set_xlabel("Number of classical supervision points")
    ax.set_ylabel("Best audit $c_i$ MAE")
    ax.set_title("Why a small amount of supervision is needed")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "ci_supervision_needed_barplot.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    setup_matplotlib()

    hybrid_run = args.base_dir / "hybrid_8pt"
    physics_run = args.base_dir / "physics_only"
    if not hybrid_run.exists() or not physics_run.exists():
        raise FileNotFoundError(
            "Expected both run directories: "
            f"{hybrid_run} and {physics_run}."
        )

    config, ci_hybrid = predict_ci_curve(hybrid_run, np.linspace(0.0, 1.0, 2), device)
    mach = float(config["mach"])
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.num_alpha))
    _, ci_hybrid = predict_ci_curve(hybrid_run, alpha_values, device)
    _, ci_physics = predict_ci_curve(physics_run, alpha_values, device)
    ci_ref = np.array(
        [RobustSubsonicShootingSolver(alpha=float(alpha), Mach=mach).solve().ci for alpha in alpha_values],
        dtype=float,
    )

    curve_fig, curve_csv = save_ci_curve_and_heatmaps(
        alpha_values,
        ci_ref,
        ci_hybrid,
        ci_physics,
        args.output_dir,
        mach,
    )

    mode_rows: list[dict[str, float]] = []
    mode_figs: list[Path] = []
    for alpha in args.mode_alphas:
        fig_path, metrics = save_mode_comparison(
            alpha=float(alpha),
            mach=mach,
            hybrid_run=hybrid_run,
            physics_run=physics_run,
            n_y_pinn=int(args.n_y_pinn),
            output_dir=args.output_dir,
            device=device,
        )
        mode_rows.append(metrics)
        mode_figs.append(fig_path)

    pd.DataFrame(mode_rows).to_csv(args.output_dir / "mode_reconstruction_metrics.csv", index=False)

    budget_summary = pd.DataFrame(
        {
            "n_alpha_supervision": [0, 8],
            "err_hybrid_8pt": [np.nan, float(np.mean(np.abs(ci_hybrid - ci_ref)))],
            "err_physics_only": [float(np.mean(np.abs(ci_physics - ci_ref))), np.nan],
        }
    )
    budget_summary = pd.DataFrame(
        {
            "n_alpha_supervision": [0, 8],
            "best_audit_ci_mae": [np.nan, np.nan],
        }
    )
    comparison_summary = pd.read_csv(args.base_dir / "comparison_summary.csv")
    comparison_summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    bar_df = pd.DataFrame(
        {
            "n_alpha_supervision": comparison_summary["n_alpha_supervision"],
            "err_hybrid_8pt": [
                np.nan if label != "hybrid_8pt" else val
                for label, val in zip(comparison_summary["label"], comparison_summary["best_audit_ci_mae"])
            ],
            "err_physics_only": [
                np.nan if label != "physics_only" else val
                for label, val in zip(comparison_summary["label"], comparison_summary["best_audit_ci_mae"])
            ],
        }
    )
    save_summary_barplot(bar_df, args.output_dir)

    print(curve_fig)
    print(curve_csv)
    for fig_path in mode_figs:
        print(fig_path)
    print(args.output_dir / "mode_reconstruction_metrics.csv")
    print(args.output_dir / "comparison_summary.csv")


if __name__ == "__main__":
    main()
