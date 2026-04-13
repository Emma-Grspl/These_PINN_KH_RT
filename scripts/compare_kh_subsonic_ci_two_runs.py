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
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN, build_fixed_mach_model_from_config


def build_model_from_config(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return build_fixed_mach_model_from_config(config)


def load_ci_curve(run_dir: Path, checkpoint: Path | None, alpha_values: np.ndarray, device: torch.device) -> tuple[pd.Series, np.ndarray]:
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_model_from_config(config)
    checkpoint_path = checkpoint if checkpoint is not None else run_dir / "model_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_values = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    return config, ci_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare c_i classique avec deux runs PINN.")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--label-a", type=str, default="Run A")
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument("--label-b", type=str, default="Run B")
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    config_a = pd.read_csv(args.run_a / "config.csv").iloc[0]
    mach = float(config_a["mach"])
    alpha_min = float(config_a["alpha_min"])
    alpha_max = float(config_a["alpha_max"])
    alpha_values = np.linspace(alpha_min, alpha_max, int(args.num_alpha))

    _, ci_a = load_ci_curve(args.run_a, args.checkpoint_a, alpha_values, device)
    _, ci_b = load_ci_curve(args.run_b, args.checkpoint_b, alpha_values, device)
    ci_ref = np.array(
        [RobustSubsonicShootingSolver(alpha=float(alpha), Mach=mach).solve().ci for alpha in alpha_values],
        dtype=float,
    )

    err_a = np.abs(ci_a - ci_ref)
    err_b = np.abs(ci_b - ci_ref)

    fig = plt.figure(figsize=(11.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_err_a = fig.add_subplot(gs[1, 0], sharex=ax_curve)
    ax_err_b = fig.add_subplot(gs[2, 0], sharex=ax_curve)

    ax_curve.plot(alpha_values, ci_ref, color="black", linewidth=2.2, label="Classique")
    ax_curve.plot(alpha_values, ci_a, "--", linewidth=2.0, label=args.label_a)
    ax_curve.plot(alpha_values, ci_b, "-.", linewidth=2.0, label=args.label_b)
    ax_curve.set_ylabel(r"$c_i$")
    ax_curve.set_title(fr"Comparaison $c_i(\alpha)$ pour $M={mach:.3f}$")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    heat_a = np.tile(err_a[None, :], (6, 1))
    im_a = ax_err_a.imshow(
        heat_a,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0],
        cmap="magma",
    )
    ax_err_a.set_yticks([])
    ax_err_a.set_title(fr"Erreur absolue {args.label_a}")
    cbar_a = fig.colorbar(im_a, ax=ax_err_a, pad=0.02)
    cbar_a.set_label(r"$|c_i - c_i^{ref}|$")

    heat_b = np.tile(err_b[None, :], (6, 1))
    im_b = ax_err_b.imshow(
        heat_b,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], 0.0, 1.0],
        cmap="magma",
    )
    ax_err_b.set_yticks([])
    ax_err_b.set_xlabel(r"$\alpha$")
    ax_err_b.set_title(fr"Erreur absolue {args.label_b}")
    cbar_b = fig.colorbar(im_b, ax=ax_err_b, pad=0.02)
    cbar_b.set_label(r"$|c_i - c_i^{ref}|$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    pd.DataFrame(
        {
            "alpha": alpha_values,
            "ci_reference": ci_ref,
            "ci_a": ci_a,
            "ci_b": ci_b,
            "err_a": err_a,
            "err_b": err_b,
        }
    ).to_csv(csv_path, index=False)

    print(f"{args.label_a}: mae={err_a.mean():.6e} max_err={err_a.max():.6e}")
    print(f"{args.label_b}: mae={err_b.mean():.6e} max_err={err_b.max():.6e}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
