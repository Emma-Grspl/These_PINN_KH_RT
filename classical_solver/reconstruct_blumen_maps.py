from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path("/Users/emma.grospellier/Thèse/These_PINN_KH_RT")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.compressible_rayleigh import sample_growth_map

DATA_DIR = ROOT_DIR / "KH_RT_Blumen"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_classical"


def parse_reference_level(csv_path: str) -> tuple[float | None, str]:
    stem = Path(csv_path).stem.strip().replace("_", ".").replace(",", ".")
    lower = stem.lower()

    if lower.startswith("ci"):
        value = float(lower[2:]) / 100.0
        return value, fr"$c_r = 0,\; c_i = {value:.2f}$"

    if lower.startswith("cr"):
        value = float(lower[2:] or "0")
        return value, fr"$c_i = 0,\; c_r = {value:.2f}$"

    numeric_part = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    if not numeric_part:
        return None, stem

    value = float(numeric_part)
    if "supersonic" in csv_path:
        return value, fr"$c_i = {value:.2f}$"
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return value, fr"$\omega_i = {formatted}$"


def load_digitized_curves(regime: str) -> list[dict]:
    csv_files = sorted(glob.glob(str(DATA_DIR / regime / "*.csv")))
    curves = []
    for csv_file in csv_files:
        level, label = parse_reference_level(csv_file)
        df = pd.read_csv(
            csv_file,
            header=None,
            names=["Mach", "alpha"],
            sep=";",
            decimal=",",
            engine="python",
        ).apply(pd.to_numeric, errors="coerce").dropna()

        curves.append(
            {
                "path": csv_file,
                "level": level,
                "label": label,
                "data": df.reset_index(drop=True),
            }
        )
    return curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruit les cartes de croissance de Blumen avec le solveur matriciel.",
    )
    parser.add_argument("--subsonic-mach-min", type=float, default=0.0)
    parser.add_argument("--subsonic-mach-max", type=float, default=0.98)
    parser.add_argument("--subsonic-alpha-min", type=float, default=0.02)
    parser.add_argument("--subsonic-alpha-max", type=float, default=1.00)
    parser.add_argument("--supersonic-mach-min", type=float, default=1.0)
    parser.add_argument("--supersonic-mach-max", type=float, default=2.0)
    parser.add_argument("--supersonic-alpha-min", type=float, default=0.02)
    parser.add_argument("--supersonic-alpha-max", type=float, default=0.55)
    parser.add_argument("--num-mach", type=int, default=21)
    parser.add_argument("--num-alpha", type=int, default=21)
    parser.add_argument("--grid-points", type=int, default=160)
    parser.add_argument("--domain-size", type=float, default=15.0)
    parser.add_argument("--n-eig", type=int, default=10)
    return parser


def pivot_field(df: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="alpha", columns="Mach", values=value_column).sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)
    return mach_grid, alpha_grid, values


def plot_subsonic_map(df: pd.DataFrame, curves: list[dict], output_path: Path) -> None:
    mach_grid, alpha_grid, omega_grid = pivot_field(df, "omega_i")
    levels = sorted({curve["level"] for curve in curves if curve["level"] is not None})

    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contour(
        mach_grid,
        alpha_grid,
        omega_grid,
        levels=levels,
        cmap="viridis",
        linewidths=1.8,
    )
    ax.clabel(contour, fmt="%0.3f", fontsize=8)

    mach_line = np.linspace(0.0, 1.0, 500)
    alpha_line = np.sqrt(np.clip(1.0 - mach_line**2, 0.0, None))
    ax.plot(mach_line, alpha_line, "--", color="black", linewidth=1.5, label=r"$\alpha^2 + M^2 = 1$")

    for curve in curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=10,
            alpha=0.65,
            label=f"Blumen {curve['label']}",
        )

    ax.set_title("Blumen subsonique : solveur classique vs. points digitalisés")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(float(np.min(mach_grid)), float(np.max(mach_grid)))
    ax.set_ylim(float(np.min(alpha_grid)), float(np.max(alpha_grid)))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_supersonic_map(df: pd.DataFrame, curves: list[dict], output_path: Path) -> None:
    mach_grid, alpha_grid, ci_grid = pivot_field(df, "ci")
    levels = sorted(
        {
            curve["level"]
            for curve in curves
            if curve["level"] is not None and "c_i =" in curve["label"]
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contour(
        mach_grid,
        alpha_grid,
        ci_grid,
        levels=levels,
        cmap="plasma",
        linewidths=1.8,
    )
    ax.clabel(contour, fmt="%0.2f", fontsize=8)

    for curve in curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=10,
            alpha=0.7,
            label=f"Blumen {curve['label']}",
        )

    ax.set_title("Blumen supersonique : solveur classique vs. points digitalisés")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(float(np.min(mach_grid)), float(np.max(mach_grid)))
    ax.set_ylim(float(np.min(alpha_grid)), float(np.max(alpha_grid)))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sub_curves = load_digitized_curves("subsonic")
    sup_curves = load_digitized_curves("supersonic")

    sub_alphas = np.linspace(args.subsonic_alpha_min, args.subsonic_alpha_max, args.num_alpha)
    sub_machs = np.linspace(args.subsonic_mach_min, args.subsonic_mach_max, args.num_mach)
    sup_alphas = np.linspace(args.supersonic_alpha_min, args.supersonic_alpha_max, args.num_alpha)
    sup_machs = np.linspace(args.supersonic_mach_min, args.supersonic_mach_max, args.num_mach)

    print("Echantillonnage subsonique...")
    sub_df = sample_growth_map(
        sub_alphas,
        sub_machs,
        N=args.grid_points,
        L=args.domain_size,
        stretched=True,
        n_eig=args.n_eig,
    )
    sub_df.to_csv(OUTPUT_DIR / "subsonic_growth_map.csv", index=False)

    print("Echantillonnage supersonique...")
    sup_df = sample_growth_map(
        sup_alphas,
        sup_machs,
        N=args.grid_points,
        L=args.domain_size,
        stretched=True,
        n_eig=args.n_eig,
    )
    sup_df.to_csv(OUTPUT_DIR / "supersonic_growth_map.csv", index=False)

    plot_subsonic_map(sub_df, sub_curves, OUTPUT_DIR / "subsonic_classical_vs_blumen.png")
    plot_supersonic_map(sup_df, sup_curves, OUTPUT_DIR / "supersonic_classical_vs_blumen.png")

    print(f"Resultats enregistres dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
