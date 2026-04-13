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
from src.models.kh_subsonic_pinn import KHSubsonicMultiMachPINN
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati_2d, xi_to_y


def build_model_from_config(config: pd.Series) -> KHSubsonicMultiMachPINN:
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
    y, p = normalize_pressure(y, p)
    return y, p, float(result.ci)


def load_pinn_mode(
    model: KHSubsonicMultiMachPINN,
    *,
    alpha: float,
    mach: float,
    n_y: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    xi = torch.linspace(-0.98, 0.98, n_y, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    mach_tensor = torch.full_like(xi, float(mach))
    with torch.no_grad():
        if getattr(model, "mode_representation", "cartesian") == "riccati":
            pr, pi, y_t = reconstruct_pressure_from_riccati_2d(model, xi, alpha_tensor, mach_tensor, anchor_xi=0.0)
            y = y_t.cpu().numpy().reshape(-1)
            p = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor, mach_tensor)
            y = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
        ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device), torch.tensor([[mach]], dtype=torch.float32, device=device)).item())
    y, p = normalize_pressure(y, p)
    return y, p, ci


def compute_mode_errors(
    y_classic: np.ndarray,
    p_classic: np.ndarray,
    y_pinn: np.ndarray,
    p_pinn: np.ndarray,
    *,
    n_common: int,
    phase_threshold: float,
) -> tuple[float, float]:
    y_min = max(float(np.min(y_classic)), float(np.min(y_pinn)))
    y_max = min(float(np.max(y_classic)), float(np.max(y_pinn)))
    y_common = np.linspace(y_min, y_max, n_common, dtype=float)

    p_c = interp_complex(y_classic, p_classic, y_common)
    p_p = interp_complex(y_pinn, p_pinn, y_common)

    amp_c = np.abs(p_c)
    amp_p = np.abs(p_p)
    amp_rel = float(np.linalg.norm(amp_p - amp_c) / max(np.linalg.norm(amp_c), 1e-12))

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
    return amp_rel, phase_rmse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heatmaps d'erreur de mode 2D pour le PINN subsonique.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-alpha", type=int, default=10)
    parser.add_argument("--num-mach", type=int, default=5)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--n-y-common", type=int, default=1200)
    parser.add_argument("--phase-threshold", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    config = pd.read_csv(args.run_dir / "config.csv").iloc[0]
    model = build_model_from_config(config)
    checkpoint_path = args.checkpoint if args.checkpoint is not None else args.run_dir / "model_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), args.num_alpha, dtype=float)
    mach_values = np.linspace(float(config["mach_min"]), float(config["mach_max"]), args.num_mach, dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)

    rows: list[dict[str, float]] = []
    amp_field = np.full_like(aa, np.nan, dtype=float)
    phase_field = np.full_like(aa, np.nan, dtype=float)

    total = aa.size
    counter = 0
    for j in range(mm.shape[0]):
        for i in range(aa.shape[1]):
            alpha = float(aa[j, i])
            mach = float(mm[j, i])
            y_c, p_c, ci_c = load_classic_mode(alpha, mach)
            y_p, p_p, ci_p = load_pinn_mode(model, alpha=alpha, mach=mach, n_y=args.n_y_pinn, device=device)
            amp_rel, phase_rmse = compute_mode_errors(
                y_c,
                p_c,
                y_p,
                p_p,
                n_common=args.n_y_common,
                phase_threshold=args.phase_threshold,
            )
            amp_field[j, i] = amp_rel
            phase_field[j, i] = phase_rmse
            rows.append(
                {
                    "alpha": alpha,
                    "mach": mach,
                    "ci_classic": ci_c,
                    "ci_pinn": ci_p,
                    "amp_rel": amp_rel,
                    "phase_masked_rmse": phase_rmse,
                }
            )
            counter += 1
            print(f"[{counter}/{total}] alpha={alpha:.4f} mach={mach:.4f} amp_rel={amp_rel:.3e} phase_rmse={phase_rmse:.3e}")

    df = pd.DataFrame(rows)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_output, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    panels = [
        (amp_field, r"Erreur relative d'amplitude", "magma"),
        (phase_field, rf"RMSE phase masquee ($|p|>{args.phase_threshold:.0e}$)", "viridis"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        pcm = ax.pcolormesh(aa, mm, field, shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(pcm, ax=ax)

    fig.suptitle("Subsonique PINN 2D : erreurs de mode vs classique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {args.output}")
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
