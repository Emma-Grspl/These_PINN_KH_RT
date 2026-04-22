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
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config  # noqa: E402
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, xi_to_y  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace l'erreur de reconstruction du mode en fonction de alpha.")
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
    parser.add_argument("--num-alpha", type=int, default=17)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-a-dir", type=Path, default=None)
    parser.add_argument("--run-b-dir", type=Path, default=None)
    parser.add_argument("--run-a-label", type=str, default="PINN hybride, 8 points")
    parser.add_argument("--run-b-label", type=str, default="PINN physique pure")
    parser.add_argument("--output-stem", type=str, default="mode_error_vs_alpha")
    return parser


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


def build_model(run_dir: Path, device: torch.device):
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return config, model


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


def compute_mode_metrics_at_alpha(
    *,
    alpha: float,
    mach: float,
    run_a: Path,
    run_b: Path,
    n_y_pinn: int,
    device: torch.device,
) -> dict[str, float]:
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

    y_a, p_a, ci_a = load_mode(run_a, alpha, n_y_pinn, device)
    y_b, p_b, ci_b = load_mode(run_b, alpha, n_y_pinn, device)

    y_min = max(float(np.min(y_classic)), float(np.min(y_a)), float(np.min(y_b)))
    y_max = min(float(np.max(y_classic)), float(np.max(y_a)), float(np.max(y_b)))
    y_common = np.linspace(y_min, y_max, 1200, dtype=float)

    p_classic_i = interp_complex(y_classic, p_classic, y_common)
    p_a_i = interp_complex(y_a, p_a, y_common)
    p_b_i = interp_complex(y_b, p_b, y_common)

    amp_classic = np.abs(p_classic_i)
    amp_a = np.abs(p_a_i)
    amp_b = np.abs(p_b_i)
    phase_classic = np.unwrap(np.angle(p_classic_i))
    phase_a = np.unwrap(np.angle(p_a_i))
    phase_b = np.unwrap(np.angle(p_b_i))
    phase_classic -= phase_classic[np.argmax(amp_classic)]
    phase_a -= phase_a[np.argmax(amp_a)]
    phase_b -= phase_b[np.argmax(amp_b)]

    env = np.maximum.reduce([amp_classic, amp_a, amp_b])
    mask = env >= 0.02 * float(np.max(env))
    amp_rel_a = float(np.linalg.norm(amp_a - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12))
    amp_rel_b = float(np.linalg.norm(amp_b - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12))
    phase_rmse_a = float(np.sqrt(np.mean((phase_a[mask] - phase_classic[mask]) ** 2))) if np.any(mask) else np.nan
    phase_rmse_b = float(np.sqrt(np.mean((phase_b[mask] - phase_classic[mask]) ** 2))) if np.any(mask) else np.nan

    return {
        "alpha": float(alpha),
        "ci_classic_gep": float(mode["ci"]),
        "ci_run_a": float(ci_a),
        "ci_run_b": float(ci_b),
        "amp_rel_run_a": amp_rel_a,
        "amp_rel_run_b": amp_rel_b,
        "phase_rmse_run_a": phase_rmse_a,
        "phase_rmse_run_b": phase_rmse_b,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    setup_matplotlib()

    run_a = args.run_a_dir or (args.base_dir / "hybrid_8pt")
    run_b = args.run_b_dir or (args.base_dir / "physics_only")
    config = pd.read_csv(run_a / "config.csv").iloc[0]
    mach = float(config["mach"])
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.num_alpha))

    rows = [
        compute_mode_metrics_at_alpha(
            alpha=float(alpha),
            mach=mach,
            run_a=run_a,
            run_b=run_b,
            n_y_pinn=int(args.n_y_pinn),
            device=device,
        )
        for alpha in alpha_values
    ]
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True)
    axes[0].plot(df["alpha"], df["amp_rel_run_a"], "--", color="#0b6e4f", linewidth=2.0, label=args.run_a_label)
    axes[0].plot(df["alpha"], df["amp_rel_run_b"], "-.", color="#c84c09", linewidth=2.0, label=args.run_b_label)
    axes[0].set_ylabel("Relative amplitude error")
    axes[0].set_title(r"Mode reconstruction error vs $\alpha$ at fixed Mach")
    axes[0].legend()

    axes[1].plot(df["alpha"], df["phase_rmse_run_a"], "--", color="#0b6e4f", linewidth=2.0, label=args.run_a_label)
    axes[1].plot(df["alpha"], df["phase_rmse_run_b"], "-.", color="#c84c09", linewidth=2.0, label=args.run_b_label)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Phase RMSE")
    axes[1].legend()

    fig.tight_layout()
    fig_path = args.output_dir / f"{args.output_stem}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output_dir / f"{args.output_stem}.csv"
    df.to_csv(csv_path, index=False)

    print(fig_path)
    print(csv_path)


if __name__ == "__main__":
    main()
