from __future__ import annotations

import argparse
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
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat
from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_from_riccati,
    reconstruct_pressure_p_y_from_riccati,
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
        pr, pi, p_y, _, y_t = reconstruct_pressure_p_y_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
    else:
        pred = model(xi, alpha_tensor)
        pr = pred[:, 0:1]
        pi = pred[:, 1:2]
        y_t = xi_to_y(xi, model.get_mapping_scale().detach())
        p_r_xi = torch.autograd.grad(pr, xi, grad_outputs=torch.ones_like(pr), create_graph=False, retain_graph=True)[0]
        p_i_xi = torch.autograd.grad(pi, xi, grad_outputs=torch.ones_like(pi), create_graph=False, retain_graph=True)[0]
        p_xi = torch.complex(p_r_xi, p_i_xi)
        y_xi = dy_dxi(xi, model.get_mapping_scale().detach())
        p_y = p_xi / y_xi

    p = torch.complex(pr, pi)

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


def compute_mode_metrics(classic: dict[str, np.ndarray], pinn: dict[str, np.ndarray], *, n_common: int = 1200, phase_threshold: float = 0.02) -> dict[str, float]:
    y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
    y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
    y_common = np.linspace(y_min, y_max, int(n_common), dtype=float)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare les modes classiques et PINN en Mach fixe.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--n-y", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    titles = [
        ("rho", r"Density Perturbation $\hat{\rho}$"),
        ("u", r"Streamwise Velocity $\hat{u}$"),
        ("v", r"Vertical Velocity $\hat{v}$"),
        ("p", r"Pressure Perturbation $\hat{p}$"),
    ]

    with PdfPages(args.output_pdf) as pdf:
        for alpha in args.alphas:
            classic, ci_classic = load_classic_full_mode(float(alpha), mach=0.5)
            pinn, ci_pinn, mach = load_pinn_full_mode(args.run_dir, alpha=float(alpha), n_y=args.n_y, device=device)
            metrics = compute_mode_metrics(classic, pinn)
            rows.append(
                {
                    "alpha": float(alpha),
                    "mach": float(mach),
                    "ci_classic": float(ci_classic),
                    "ci_pinn": float(ci_pinn),
                    "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                    **metrics,
                }
            )

            x_limits_classic = compute_visible_xlim(classic["y"], [classic["rho"], classic["u"], classic["v"], classic["p"]])
            x_limits_pinn = compute_visible_xlim(pinn["y"], [pinn["rho"], pinn["u"], pinn["v"], pinn["p"]])
            x_limits = (
                min(x_limits_classic[0], x_limits_pinn[0]),
                max(x_limits_classic[1], x_limits_pinn[1]),
            )

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=False)
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
                fr"$\alpha={float(alpha):.3f}$, $M={mach:.3f}$"
                "\n"
                fr"$c_i^{{classic}}={ci_classic:.5f}$ | $c_i^{{PINN}}={ci_pinn:.5f}$ | "
                fr"$p_{{rel}}={metrics['p_rel']:.3e}$ | phase RMSE={metrics['phase_rmse']:.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(args.output_pdf)
    print(args.output_csv)


if __name__ == "__main__":
    main()
