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
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, xi_to_y


def build_pinn_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return build_fixed_mach_model_from_config(config)


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


def load_mode(
    run_dir: Path,
    checkpoint: Path | None,
    alpha: float,
    n_y_pinn: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_pinn_model(config)
    checkpoint_path = checkpoint if checkpoint is not None else run_dir / "model_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, n_y_pinn, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    with torch.no_grad():
        if model.mode_representation == "riccati":
            pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
            y_pred = y_t.cpu().numpy().reshape(-1)
            p_pred = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor)
            y_pred = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p_pred = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
        ci_pred = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    return normalize_pressure(y_pred, p_pred)[0], normalize_pressure(y_pred, p_pred)[1], ci_pred


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare deux runs PINN au classique sur le mode de pression.")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--label-a", type=str, default="Run A")
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument("--label-b", type=str, default="Run B")
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

    y_a, p_a, ci_a = load_mode(args.run_a, args.checkpoint_a, float(args.alpha), int(args.n_y_pinn), device)
    y_b, p_b, ci_b = load_mode(args.run_b, args.checkpoint_b, float(args.alpha), int(args.n_y_pinn), device)
    ci_ref = RobustSubsonicShootingSolver(alpha=float(args.alpha), Mach=float(args.mach)).solve().ci

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
    if np.any(mask):
        x_min = float(np.min(y_common[mask]))
        x_max = float(np.max(y_common[mask]))
    else:
        x_min, x_max = float(y_common[0]), float(y_common[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

    axes[0].plot(y_common, amp_classic, color="black", linewidth=2.2, label="Classique")
    axes[0].plot(y_common, amp_a, "--", linewidth=2.0, label=args.label_a)
    axes[0].plot(y_common, amp_b, "-.", linewidth=2.0, label=args.label_b)
    axes[0].set_title(r"Amplitude $|\hat{p}|$")
    axes[0].set_ylabel("Amplitude normalisee")
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_common, phase_classic, color="black", linewidth=2.2, label="Classique")
    axes[1].plot(y_common, phase_a, "--", linewidth=2.0, label=args.label_a)
    axes[1].plot(y_common, phase_b, "-.", linewidth=2.0, label=args.label_b)
    axes[1].set_title(r"Phase $\arg(\hat{p})$")
    axes[1].set_xlim(x_min, x_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        fr"Comparaison des modes de pression pour $\alpha={float(args.alpha):.3f}$ et $M={float(args.mach):.3f}$"
        "\n"
        fr"Classic GEP: $c_i={mode['ci']:.5f}$ | Shooting ref: $c_i={ci_ref:.5f}$ | "
        fr"{args.label_a}: $c_i={ci_a:.5f}$ | {args.label_b}: $c_i={ci_b:.5f}$"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    amp_rel_a = np.linalg.norm(amp_a - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12)
    amp_rel_b = np.linalg.norm(amp_b - amp_classic) / max(np.linalg.norm(amp_classic), 1e-12)
    phase_err_a = np.sqrt(np.mean((phase_a[mask] - phase_classic[mask]) ** 2)) if np.any(mask) else np.nan
    phase_err_b = np.sqrt(np.mean((phase_b[mask] - phase_classic[mask]) ** 2)) if np.any(mask) else np.nan

    print(
        f"{args.label_a}: ci={ci_a:.6e} amp_rel={amp_rel_a:.6e} phase_rmse={phase_err_a:.6e} | "
        f"{args.label_b}: ci={ci_b:.6e} amp_rel={amp_rel_b:.6e} phase_rmse={phase_err_b:.6e} | "
        f"selection_source={selection_source} n_finite_modes={n_modes}"
    )
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
