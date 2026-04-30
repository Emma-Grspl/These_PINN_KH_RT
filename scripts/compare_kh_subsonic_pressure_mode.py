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
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN, build_fixed_mach_model_from_config
from src.physics.kh_subsonic_residual import xi_to_y


def build_pinn_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return build_fixed_mach_model_from_config(config)


def normalize_pressure(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
    if np.max(np.real(p)) < abs(np.min(np.real(p))):
        p = -p
    scale = max(np.max(np.abs(np.real(p))), np.max(np.abs(np.imag(p))), 1e-12)
    return y, p / scale


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
    half_width = max(float(np.max(np.abs(y_vis))), min_half_width)
    return -half_width, half_width


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare seulement la pression modale classique vs PINN en subsonique.")
    parser.add_argument("--run-dir", type=Path, required=True)
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
    state_dict = torch.load(args.run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, args.n_y_pinn, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(args.alpha))
    with torch.no_grad():
        pred = model(xi, alpha_tensor)
        ci_pinn = float(model.get_ci(torch.tensor([[args.alpha]], dtype=torch.float32, device=device)).item())
        mapping_scale = model.get_mapping_scale().detach()
        y_pinn = xi_to_y(xi, mapping_scale).cpu().numpy().reshape(-1)
        p_pinn = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
    y_pinn, p_pinn = normalize_pressure(y_pinn, p_pinn)

    ci_ref = RobustSubsonicShootingSolver(alpha=float(args.alpha), Mach=float(args.mach)).solve().ci
    x_limits_classic = compute_visible_xlim(y_classic, [p_classic])
    x_limits_pinn = compute_visible_xlim(y_pinn, [p_pinn])
    x_limits = (min(x_limits_classic[0], x_limits_pinn[0]), max(x_limits_classic[1], x_limits_pinn[1]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    axes[0].plot(y_classic, np.real(p_classic), label="Classic", linewidth=2.0)
    axes[0].plot(y_pinn, np.real(p_pinn), "--", label="PINN", linewidth=2.0)
    axes[0].set_title(r"Real Pressure Mode $\Re(\hat{p})$")
    axes[0].set_xlim(*x_limits)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_classic, np.imag(p_classic), label="Classic", linewidth=2.0)
    axes[1].plot(y_pinn, np.imag(p_pinn), "--", label="PINN", linewidth=2.0)
    axes[1].set_title(r"Imag Pressure Mode $\Im(\hat{p})$")
    axes[1].set_xlim(*x_limits)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        fr"Subsonic pressure mode comparison for $\alpha={float(args.alpha):.3f}$ and $M={float(args.mach):.3f}$"
        "\n"
        fr"Classic GEP: $c_i={mode['ci']:.5f}$ | Shooting ref: $c_i={ci_ref:.5f}$ | PINN: $c_i={ci_pinn:.5f}$"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(
        f"classic_ci={mode['ci']:.6e} shooting_ci={ci_ref:.6e} pinn_ci={ci_pinn:.6e} "
        f"selection_source={selection_source} n_finite_modes={n_modes}"
    )
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
