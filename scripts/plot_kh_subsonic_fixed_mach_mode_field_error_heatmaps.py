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

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat
from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_from_riccati,
    xi_to_y,
)


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


def load_pinn_full_mode(run_dir: Path, *, alpha: float, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float, float]:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    load_fixed_mach_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    xi.requires_grad_(True)
    alpha_tensor = torch.full_like(xi, float(alpha))

    if str(config.get("mode_representation", "cartesian")) == "riccati":
        pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
    else:
        pred = model(xi, alpha_tensor)
        pr = pred[:, 0:1]
        pi = pred[:, 1:2]
        y_t = xi_to_y(xi, model.get_mapping_scale().detach())

    p = torch.complex(pr, pi)
    p_r_xi = torch.autograd.grad(pr, xi, grad_outputs=torch.ones_like(pr), create_graph=False, retain_graph=True)[0]
    p_i_xi = torch.autograd.grad(pi, xi, grad_outputs=torch.ones_like(pi), create_graph=False, retain_graph=True)[0]
    p_xi = torch.complex(p_r_xi, p_i_xi)
    y_xi = dy_dxi(xi, model.get_mapping_scale().detach())
    p_y = p_xi / y_xi

    ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    mach = float(config["mach"])
    c = -1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (mach**2)

    fields = normalize_full_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    )
    return fields, ci, mach


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heatmaps alpha-y des erreurs modales PINN vs classique a Mach fixe.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-alpha", type=int, default=41)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--n-y-common", type=int, default=801)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    config = pd.read_csv(args.run_dir / "config.csv").iloc[0]
    mach = float(config["mach"])
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.num_alpha), dtype=float)
    y_common = np.linspace(-float(args.y_max), float(args.y_max), int(args.n_y_common), dtype=float)

    fields = ("rho", "u", "v", "p")
    error_maps = {name: np.full((len(alpha_values), len(y_common)), np.nan, dtype=float) for name in fields}
    summary_rows: list[dict[str, float]] = []
    long_rows: list[dict[str, float | str]] = []

    for i, alpha in enumerate(alpha_values):
        classic, ci_classic = load_classic_full_mode(float(alpha), mach)
        pinn, ci_pinn, _ = load_pinn_full_mode(args.run_dir, alpha=float(alpha), n_y=args.n_y_pinn, device=device)

        row = {
            "alpha": float(alpha),
            "mach": float(mach),
            "ci_classic": float(ci_classic),
            "ci_pinn": float(ci_pinn),
            "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
        }

        for field in fields:
            classic_i = interp_complex(classic["y"], classic[field], y_common)
            pinn_i = interp_complex(pinn["y"], pinn[field], y_common)
            abs_err = np.abs(pinn_i - classic_i)
            error_maps[field][i, :] = abs_err
            row[f"{field}_rel"] = float(np.linalg.norm(pinn_i - classic_i) / max(np.linalg.norm(classic_i), 1e-12))

            for y_val, err_val in zip(y_common, abs_err):
                long_rows.append(
                    {
                        "alpha": float(alpha),
                        "mach": float(mach),
                        "y": float(y_val),
                        "field": field,
                        "abs_err": float(err_val),
                    }
                )

        summary_rows.append(row)
        print(
            f"[{i+1}/{len(alpha_values)}] alpha={alpha:.4f} "
            f"ci_err={row['ci_abs_err']:.3e} "
            f"p_rel={row['p_rel']:.3e} rho_rel={row['rho_rel']:.3e} "
            f"u_rel={row['u_rel']:.3e} v_rel={row['v_rel']:.3e}"
        )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(long_rows).to_csv(args.output_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)

    aa, yy = np.meshgrid(alpha_values, y_common, indexing="ij")
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

    fig.suptitle(fr"PINN subsonique a Mach fixe : heatmaps d'erreur modale vs classique ($M={mach:.2f}$)")
    fig.savefig(args.output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(args.output_png)
    print(args.output_csv)
    print(args.summary_csv)


if __name__ == "__main__":
    main()
