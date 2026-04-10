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

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN


def build_model_from_config(config: pd.Series) -> KHSubsonicFixedMachPINN:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace c_i classique vs PINN avec bande heatmap d'erreur.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
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

    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.num_alpha))
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pinn = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)

    ci_ref = np.array(
        [
            RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(config["mach"])).solve().ci
            for alpha in alpha_values
        ],
        dtype=float,
    )
    ci_abs_err = np.abs(ci_pinn - ci_ref)

    df = pd.DataFrame(
        {
            "alpha": alpha_values,
            "ci_reference": ci_ref,
            "ci_pinn": ci_pinn,
            "ci_abs_err": ci_abs_err,
        }
    )

    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_curve)

    ax_curve.plot(alpha_values, ci_ref, label="Classique", linewidth=2.0)
    ax_curve.plot(alpha_values, ci_pinn, "--", label="PINN", linewidth=2.0)
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.set_title(fr"Comparaison $c_i(\alpha)$ pour $M={float(config['mach']):.3f}$")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    heat = np.tile(ci_abs_err[None, :], (6, 1))
    im = ax_heat.imshow(
        heat,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0],
        cmap="magma",
    )
    ax_heat.set_yticks([])
    ax_heat.set_xlabel(r"$\alpha$")
    ax_heat.set_title(r"Bande heatmap de l'erreur $|c_i^{PINN} - c_i^{ref}|$")
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02)
    cbar.set_label(r"$|c_i^{PINN} - c_i^{ref}|$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    print(f"mae={ci_abs_err.mean():.6e} max_err={ci_abs_err.max():.6e}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
