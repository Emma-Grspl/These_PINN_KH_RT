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
from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative, dy_dxi, reconstruct_pressure_p_y_from_riccati, xi_to_y


def normalize_classic_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = vector[0:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2

    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def build_pinn_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return build_fixed_mach_model_from_config(config)


def reconstruct_pinn_fields(
    model: KHSubsonicFixedMachPINN,
    *,
    xi: torch.Tensor,
    alpha: float,
    mach: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    xi = xi.clone().detach().to(device)
    xi.requires_grad_(True)
    alpha_tensor = torch.full_like(xi, float(alpha))

    if model.mode_representation == "riccati":
        p_r, p_i, p_y, _, y = reconstruct_pressure_p_y_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
        p = p_r + 1j * p_i
    else:
        pred = model(xi, alpha_tensor)
        p_r = pred[:, 0:1]
        p_i = pred[:, 1:2]
        p = p_r + 1j * p_i

        mapping_scale = model.get_mapping_scale().detach()
        y = xi_to_y(xi, mapping_scale)
        p_r_xi = torch.autograd.grad(p_r, xi, grad_outputs=torch.ones_like(p_r), create_graph=False, retain_graph=True)[0]
        p_i_xi = torch.autograd.grad(p_i, xi, grad_outputs=torch.ones_like(p_i), create_graph=False, retain_graph=True)[0]
        p_xi = p_r_xi + 1j * p_i_xi
        y_xi = dy_dxi(xi, mapping_scale)
        p_y = p_xi / y_xi

    ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    c = 1j * ci
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)

    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)

    u_np = u.detach().cpu().numpy().reshape(-1)
    v_np = v.detach().cpu().numpy().reshape(-1)
    p_np = p.detach().cpu().numpy().reshape(-1)
    rho_np = rho.detach().cpu().numpy().reshape(-1)
    y_np = y.detach().cpu().numpy().reshape(-1)

    idx = int(np.argmax(np.abs(rho_np)))
    if np.abs(rho_np[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho_np[idx]))
        u_np, v_np, p_np, rho_np = u_np * phase, v_np * phase, p_np * phase, rho_np * phase

    if np.max(np.real(rho_np)) < abs(np.min(np.real(rho_np))):
        u_np, v_np, p_np, rho_np = -u_np, -v_np, -p_np, -rho_np

    scale = max(np.max(np.abs(np.real(rho_np))), np.max(np.abs(np.imag(rho_np))), 1e-12)
    return y_np, u_np / scale, v_np / scale, p_np / scale, rho_np / scale, ci


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
    parser = argparse.ArgumentParser(description="Compare un mode classique subsonique et un mode PINN sur la meme figure.")
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
    u_c, v_c, p_c, rho_c = normalize_classic_mode(mode["vector"], solver.n_points, solver.Mach)

    config = pd.read_csv(args.run_dir / "config.csv").iloc[0]
    model = build_pinn_model(config)
    state_dict = torch.load(args.run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, args.n_y_pinn, device=device).view(-1, 1)
    y_p, u_p, v_p, p_p, rho_p, ci_pinn = reconstruct_pinn_fields(
        model,
        xi=xi,
        alpha=float(args.alpha),
        mach=float(args.mach),
        device=device,
    )

    ci_ref = RobustSubsonicShootingSolver(alpha=float(args.alpha), Mach=float(args.mach)).solve().ci

    x_limits_classic = compute_visible_xlim(solver.y, [rho_c, u_c, v_c, p_c])
    x_limits_pinn = compute_visible_xlim(y_p, [rho_p, u_p, v_p, p_p])
    x_limits = (
        min(x_limits_classic[0], x_limits_pinn[0]),
        max(x_limits_classic[1], x_limits_pinn[1]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    classics = [rho_c, u_c, v_c, p_c]
    pinns = [rho_p, u_p, v_p, p_p]
    titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]

    for ax, field_c, field_p, title in zip(axes.flat, classics, pinns, titles):
        ax.plot(solver.y, np.real(field_c), color="tab:blue", label="Classic Real")
        ax.plot(solver.y, np.imag(field_c), color="tab:orange", label="Classic Imag")
        ax.plot(y_p, np.real(field_p), "--", color="tab:blue", label="PINN Real")
        ax.plot(y_p, np.imag(field_p), "--", color="tab:orange", label="PINN Imag")
        ax.set_title(title)
        ax.set_xlim(*x_limits)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        fr"Classical vs PINN most unstable eigenmode for $\alpha={float(args.alpha):.3f}$ and $M={float(args.mach):.3f}$"
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
