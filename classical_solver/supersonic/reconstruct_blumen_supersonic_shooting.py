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

from classical_solver.supersonic.shooting_supersonic import sample_supersonic_growth_map


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting_supersonic"


def parse_reference_level(csv_path: str) -> tuple[float | None, str, str]:
    stem = Path(csv_path).stem.strip().replace("_", ".").replace(",", ".")
    lower = stem.lower()
    if lower.startswith("ci"):
        value = float(lower[2:]) / 100.0
        return value, fr"$c_r = 0,\; c_i = {value:.2f}$", "ci_special"
    if lower.startswith("cr"):
        value = float(lower[2:] or "0")
        return value, fr"$c_i = 0,\; c_r = {value:.2f}$", "cr_special"
    numeric = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    if not numeric:
        return None, stem, "unknown"
    value = float(numeric)
    return value, fr"$c_i = {value:.2f}$", "ci_level"


def load_digitized_curves() -> list[dict]:
    curves = []
    for csv_file in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        level, label, family = parse_reference_level(csv_file)
        df = pd.read_csv(
            csv_file,
            header=None,
            names=["Mach", "alpha"],
            sep=";",
            decimal=",",
            engine="python",
        ).apply(pd.to_numeric, errors="coerce").dropna()
        curves.append({"level": level, "label": label, "family": family, "data": df})
    return curves


def build_anchor_points(curves: list[dict]) -> list[dict]:
    anchors: list[dict] = []
    for curve in curves:
        if curve["level"] is None:
            continue
        for _, row in curve["data"].iterrows():
            mach = float(row["Mach"])
            alpha = float(row["alpha"])
            if mach < 1.0 or alpha < 0.0:
                continue
            if curve["family"] == "ci_level":
                cr_seed = 0.03
                ci_seed = float(curve["level"])
            elif curve["family"] == "ci_special":
                cr_seed = 0.0
                ci_seed = float(curve["level"])
            elif curve["family"] == "cr_special":
                cr_seed = float(curve["level"])
                ci_seed = 0.03
            else:
                cr_seed = 0.03
                ci_seed = 0.03
            anchors.append(
                {
                    "Mach": mach,
                    "alpha": alpha,
                    "cr_seed": cr_seed,
                    "ci_seed": ci_seed,
                }
            )
    return anchors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruction Blumen 1975 par méthode du tir supersonique.")
    parser.add_argument("--mach-min", type=float, default=1.0)
    parser.add_argument("--mach-max", type=float, default=2.0)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.55)
    parser.add_argument("--num-mach", type=int, default=13)
    parser.add_argument("--num-alpha", type=int, default=13)
    parser.add_argument("--ci-max", type=float, default=0.12)
    parser.add_argument("--cr-max", type=float, default=0.20)
    parser.add_argument("--tracking-weight", type=float, default=2e-2)
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting")
    return parser


def plot_map(df: pd.DataFrame, curves: list[dict], output_path: Path) -> None:
    pivot = df.pivot(index="alpha", columns="Mach", values="ci").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    ci_values = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    ci_levels = sorted(
        curve["level"]
        for curve in curves
        if curve["family"] == "ci_level" and curve["level"] is not None
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    if ci_levels:
        contour = ax.contour(
            mach_grid,
            alpha_grid,
            ci_values,
            levels=ci_levels,
            cmap="plasma",
            linewidths=1.8,
        )
        ax.clabel(contour, fmt="%0.2f", fontsize=8)

    markers = {"ci_level": "o", "ci_special": "s", "cr_special": "^", "unknown": "x"}
    for curve in curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=18,
            alpha=0.75,
            marker=markers.get(curve["family"], "o"),
            label=f"Blumen {curve['label']}",
        )

    ax.set_title("Blumen 1975 : reconstruction supersonique par méthode du tir")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)

    curves = load_digitized_curves()
    anchors = build_anchor_points(curves)
    print("Echantillonnage supersonique par méthode du tir...")
    df = sample_supersonic_growth_map(
        alphas,
        machs,
        ci_max=args.ci_max,
        cr_max=args.cr_max,
        anchor_points=anchors,
        tracking_weight=args.tracking_weight,
    )
    csv_path = OUTPUT_DIR / f"{args.output_stem}_growth_map.csv"
    fig_path = OUTPUT_DIR / f"{args.output_stem}_vs_blumen.png"
    df.to_csv(csv_path, index=False)
    plot_map(df, curves, fig_path)
    print(f"Resultats enregistres dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
