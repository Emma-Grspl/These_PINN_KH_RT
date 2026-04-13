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
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN, KHSubsonicMultiMachPINN
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, reconstruct_pressure_from_riccati_2d, xi_to_y


def build_fixed_mach_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    mode_hidden_dim = None if "mode_hidden_dim" not in config.index or pd.isna(config["mode_hidden_dim"]) else int(config["mode_hidden_dim"])
    ci_hidden_dim = None if "ci_hidden_dim" not in config.index or pd.isna(config["ci_hidden_dim"]) else int(config["ci_hidden_dim"])
    return KHSubsonicFixedMachPINN(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        hidden_dim=int(config["hidden_dim"]),
        mode_hidden_dim=mode_hidden_dim,
        ci_hidden_dim=ci_hidden_dim,
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


def build_multi_mach_model(config: pd.Series) -> KHSubsonicMultiMachPINN:
    mode_hidden_dim = None if "mode_hidden_dim" not in config.index or pd.isna(config["mode_hidden_dim"]) else int(config["mode_hidden_dim"])
    ci_hidden_dim = None if "ci_hidden_dim" not in config.index or pd.isna(config["ci_hidden_dim"]) else int(config["ci_hidden_dim"])
    return KHSubsonicMultiMachPINN(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        mach_min=float(config["mach_min"]),
        mach_max=float(config["mach_max"]),
        hidden_dim=int(config["hidden_dim"]),
        mode_hidden_dim=mode_hidden_dim,
        ci_hidden_dim=ci_hidden_dim,
        mode_depth=int(config["mode_depth"]),
        ci_depth=int(config["ci_depth"]),
        activation=str(config["activation"]),
        fourier_features=int(config["fourier_features"]),
        fourier_scale=float(config["fourier_scale"]),
        initial_ci=float(config["initial_ci"]),
        mapping_scale=float(config["mapping_scale"]),
        trainable_mapping_scale=bool(config["trainable_mapping_scale"]),
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


def interp_real(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, f_src)


def load_classic_mode(alpha: float, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    solver = Mstab17SubsonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    q_left = np.asarray(sol_left.y[1])
    q_right = np.asarray(sol_right.y[1])
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
    q = np.concatenate([q_left[y_left < 0.0], q_right[::-1]])
    y, p = normalize_pressure(y, p)
    return y, p, q, float(result.ci)


def load_pinn_mode(
    run_dir: Path,
    checkpoint: Path,
    alpha: float,
    mach: float,
    *,
    n_y: int = 1001,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dev = torch.device(device)
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    is_multi_mach = "mach_min" in config.index and "mach_max" in config.index and not pd.isna(config["mach_min"]) and not pd.isna(config["mach_max"])
    model = build_multi_mach_model(config) if is_multi_mach else build_fixed_mach_model(config)
    state_dict = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, n_y, device=dev).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    mach_tensor = torch.full_like(xi, float(mach))
    with torch.no_grad():
        if is_multi_mach:
            pred = model(xi, alpha_tensor, mach_tensor)
            if model.mode_representation == "riccati":
                q = pred[:, 1].cpu().numpy().reshape(-1)
                pr, pi, y_t = reconstruct_pressure_from_riccati_2d(model, xi, alpha_tensor, mach_tensor, anchor_xi=0.0)
                y = y_t.cpu().numpy().reshape(-1)
                p = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
            else:
                y = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
                p = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
                phase = np.unwrap(np.angle(p))
                q = np.gradient(phase, y, edge_order=1)
            ci = float(
                model.get_ci(
                    torch.tensor([[alpha]], dtype=torch.float32, device=dev),
                    torch.tensor([[mach]], dtype=torch.float32, device=dev),
                ).item()
            )
        else:
            pred = model(xi, alpha_tensor)
            if model.mode_representation == "riccati":
                q = pred[:, 1].cpu().numpy().reshape(-1)
                pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
                y = y_t.cpu().numpy().reshape(-1)
                p = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
            else:
                y = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
                p = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
                phase = np.unwrap(np.angle(p))
                q = np.gradient(phase, y, edge_order=1)
            ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=dev)).item())
    y, p = normalize_pressure(y, p)
    return y, p, q, ci


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trois modes subsoniques: amplitude, phase masquee, q.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    parser.add_argument("--phase-threshold", type=float, default=1e-2)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fig, axes = plt.subplots(len(args.alphas), 3, figsize=(16, 10.5), sharex=False)
    if len(args.alphas) == 1:
        axes = np.asarray([axes])

    for row_idx, alpha in enumerate(args.alphas):
        y_c, p_c, q_c, ci_c = load_classic_mode(alpha, args.mach)
        y_p, p_p, q_p, ci_p = load_pinn_mode(
            args.run_dir,
            args.checkpoint,
            alpha,
            args.mach,
            n_y=args.n_y_pinn,
            device=args.device,
        )

        y_min = max(float(np.min(y_c)), float(np.min(y_p)))
        y_max = min(float(np.max(y_c)), float(np.max(y_p)))
        y_common = np.linspace(y_min, y_max, 1200, dtype=float)

        p_c_i = interp_complex(y_c, p_c, y_common)
        p_p_i = interp_complex(y_p, p_p, y_common)
        q_c_i = interp_real(y_c, q_c, y_common)
        q_p_i = interp_real(y_p, q_p, y_common)

        amp_c = np.abs(p_c_i)
        amp_p = np.abs(p_p_i)

        phase_c = np.unwrap(np.angle(p_c_i))
        phase_p = np.unwrap(np.angle(p_p_i))
        phase_c -= phase_c[np.argmax(amp_c)]
        phase_p -= phase_p[np.argmax(amp_p)]

        phase_mask = np.maximum(amp_c, amp_p) > float(args.phase_threshold)

        ax_amp = axes[row_idx, 0]
        ax_phase = axes[row_idx, 1]
        ax_q = axes[row_idx, 2]

        ax_amp.plot(y_common, amp_c, color="black", linewidth=2.0, label="Classique")
        ax_amp.plot(y_common, amp_p, "--", color="tab:blue", linewidth=2.0, label="PINN")
        ax_amp.set_title(fr"Amplitude | $\alpha={alpha:.2f}$, $M={args.mach:.2f}$")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.legend()

        if np.any(phase_mask):
            ax_phase.plot(y_common[phase_mask], phase_c[phase_mask], color="black", linewidth=2.0, label="Classique")
            ax_phase.plot(y_common[phase_mask], phase_p[phase_mask], "--", color="tab:orange", linewidth=2.0, label="PINN")
        ax_phase.set_title(
            fr"Phase masquee | $|p|>{args.phase_threshold:.0e}$"
            "\n"
            fr"$c_i^{{classic}}={ci_c:.4f}$, $c_i^{{PINN}}={ci_p:.4f}$"
        )
        ax_phase.set_ylabel("Phase")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend()

        ax_q.plot(y_common, q_c_i, color="black", linewidth=2.0, label="Classique")
        ax_q.plot(y_common, q_p_i, "--", color="tab:green", linewidth=2.0, label="PINN")
        ax_q.set_title(r"Riccati imaginaire $q = \partial_y \phi$")
        ax_q.set_ylabel("q")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend()

    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    axes[-1, 2].set_xlabel("y")
    fig.suptitle("Subsonique PINN vs classique : amplitude, phase masquee et q")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
