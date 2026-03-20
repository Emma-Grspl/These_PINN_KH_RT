from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INPUT_CSV = ROOT_DIR / "assets" / "blumen_shooting" / "subsonic_shooting_growth_map.csv"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting"


def nearest_value(values: np.ndarray, target: float) -> float:
    return float(values[np.argmin(np.abs(values - target))])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation de c_i sur la carte subsonique.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "subsonic_ci_visualization.png")
    parser.add_argument("--mach-cut", type=float, default=0.40, help="Coupe c_i(alpha) au Mach le plus proche.")
    parser.add_argument("--alpha-cut", type=float, default=0.30, help="Coupe c_i(M) au alpha le plus proche.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)

    pivot = df.pivot(index="alpha", columns="Mach", values="ci").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    ci_values = pivot.to_numpy(dtype=float)

    mach_grid, alpha_grid = np.meshgrid(machs, alphas)
    mach_cut = nearest_value(machs, args.mach_cut)
    alpha_cut = nearest_value(alphas, args.alpha_cut)

    alpha_slice = df[np.isclose(df["Mach"], mach_cut)].sort_values("alpha")
    mach_slice = df[np.isclose(df["alpha"], alpha_cut)].sort_values("Mach")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    heatmap = axes[0].pcolormesh(
        mach_grid,
        alpha_grid,
        ci_values,
        shading="auto",
        cmap="viridis",
    )
    axes[0].axvline(mach_cut, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    axes[0].axhline(alpha_cut, color="white", linestyle=":", linewidth=1.2, alpha=0.9)
    axes[0].set_title(r"Heatmap de $c_i(M,\alpha)$")
    axes[0].set_xlabel("Nombre de Mach (M)")
    axes[0].set_ylabel(r"Nombre d'onde ($\alpha$)")
    axes[0].grid(False)
    fig.colorbar(heatmap, ax=axes[0], label=r"$c_i$")

    axes[1].plot(alpha_slice["alpha"], alpha_slice["ci"], marker="o", linewidth=1.8)
    axes[1].set_title(fr"$c_i$ en fonction de $\alpha$ pour $M \approx {mach_cut:.2f}$")
    axes[1].set_xlabel(r"Nombre d'onde ($\alpha$)")
    axes[1].set_ylabel(r"$c_i$")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(mach_slice["Mach"], mach_slice["ci"], marker="o", linewidth=1.8)
    axes[2].set_title(fr"$c_i$ en fonction de $M$ pour $\alpha \approx {alpha_cut:.2f}$")
    axes[2].set_xlabel("Nombre de Mach (M)")
    axes[2].set_ylabel(r"$c_i$")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(r"Visualisation subsonique de $c_i$", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistree dans {args.output}")


if __name__ == "__main__":
    main()
