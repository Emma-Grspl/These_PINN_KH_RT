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

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, xi_to_y


def build_pinn_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return KHSubsonicFixedMachPINN(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        hidden_dim=int(config["hidden_dim"]),
        mode_depth=int(config["mode_depth"]),
        ci_depth=int(config["ci_depth"]),
        activation=str(config["activation"]),
        fourier_features=int(config["fourier_features"]),
        fourier_scale=float(config["fourier_scale"]),
        initial_ci=float(config["initial_ci"]),
        mapping_scale=float(config["mapping_scale"]),
        trainable_mapping_scale=bool(config["trainable_mapping_scale"]),
        enforce_mode_symmetry=bool(config["enforce_mode_symmetry"]) if "enforce_mode_symmetry" in config.index else False,
        mode_representation=str(config["mode_representation"]) if "mode_representation" in config.index else "cartesian",
    )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare le mode de pression classique vs PINN en amplitude/phase.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points-classic", type=int, default=561)
    parser.add_argument("--mapping-scale-classic", type=float, default=3.0)
    parser.add_argument("--xi-max-classic", type=float, default=0.99)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(args.alpha),
        Mach=float(args.mach),
        n_points=int(args.n_points_classic),
        mapping_scale=float(args.mapping_scale_classic),
        xi_max=float(args.xi_max_classic),
    )
    mode, selection_source, n_modes = solver.get_selected_mode()
    if mode is None:
        raise RuntimeError("Aucun mode classique selectionne.")
    p_classic = mode["vector"][2 * solver.n_points : 3 * solver.n_points]
    y_classic, p_classic = normalize_pressure(solver.y, p_classic)

    config = pd.read_csv(args.run_dir / "config.csv").iloc[0]
    model = build_pinn_model(config)
    checkpoint_path = args.checkpoint if args.checkpoint is not None else args.run_dir / "model_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, args.n_y_pinn, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(args.alpha))
    with torch.no_grad():
        if model.mode_representation == "riccati":
            pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
            y_pinn = y_t.cpu().numpy().reshape(-1)
            p_pinn = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor)
            y_pinn = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p_pinn = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
        ci_pinn = float(model.get_ci(torch.tensor([[args.alpha]], dtype=torch.float32, device=device)).item())
    y_pinn, p_pinn = normalize_pressure(y_pinn, p_pinn)

    ci_ref = RobustSubsonicShootingSolver(alpha=float(args.alpha), Mach=float(args.mach)).solve().ci

    y_min = max(float(np.min(y_classic)), float(np.min(y_pinn)))
    y_max = min(float(np.max(y_classic)), float(np.max(y_pinn)))
    y_common = np.linspace(y_min, y_max, 1200, dtype=float)
    p_classic_i = interp_complex(y_classic, p_classic, y_common)
    p_pinn_i = interp_complex(y_pinn, p_pinn, y_common)

    amp_classic = np.abs(p_classic_i)
    amp_pinn = np.abs(p_pinn_i)
    phase_classic = np.unwrap(np.angle(p_classic_i))
    phase_pinn = np.unwrap(np.angle(p_pinn_i))
    phase_classic -= phase_classic[np.argmax(amp_classic)]
    phase_pinn -= phase_pinn[np.argmax(amp_pinn)]

    env = np.maximum(amp_classic, amp_pinn)
    mask = env >= 0.02 * float(np.max(env))
    if np.any(mask):
        x_min = float(np.min(y_common[mask]))
        x_max = float(np.max(y_common[mask]))
    else:
        x_min, x_max = float(y_common[0]), float(y_common[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)

    axes[0].plot(y_common, amp_classic, label="Classique", linewidth=2.0)
    axes[0].plot(y_common, amp_pinn, "--", label="PINN", linewidth=2.0)
    axes[0].set_title(r"Amplitude $|\hat{p}|$")
    axes[0].set_ylabel("Amplitude normalisee")
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_common, phase_classic, label="Classique", linewidth=2.0)
    axes[1].plot(y_common, phase_pinn, "--", label="PINN", linewidth=2.0)
    axes[1].set_title(r"Phase $\arg(\hat{p})$")
    axes[1].set_xlim(x_min, x_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        fr"Mode de pression subsonique en amplitude/phase pour $\alpha={float(args.alpha):.3f}$ et $M={float(args.mach):.3f}$"
        "\n"
        fr"Classic GEP: $c_i={mode['ci']:.5f}$ | Shooting ref: $c_i={ci_ref:.5f}$ | PINN: $c_i={ci_pinn:.5f}$"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    phase_err = np.sqrt(np.mean((phase_pinn[mask] - phase_classic[mask]) ** 2)) if np.any(mask) else np.nan
    amp_rel = np.linalg.norm(amp_pinn - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12)

    print(
        f"classic_ci={mode['ci']:.6e} shooting_ci={ci_ref:.6e} pinn_ci={ci_pinn:.6e} "
        f"amp_rel={amp_rel:.6e} phase_rmse={phase_err:.6e} "
        f"selection_source={selection_source} n_finite_modes={n_modes}"
    )
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
