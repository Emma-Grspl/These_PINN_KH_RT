from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def parse_reference_level(csv_path: Path) -> tuple[float | None, str, str]:
    stem = csv_path.stem.strip().replace("_", ".").replace(",", ".")
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
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        level, label, family = parse_reference_level(csv_file)
        df = (
            pd.read_csv(
                csv_file,
                header=None,
                names=["Mach", "alpha"],
                sep=";",
                decimal=",",
                engine="python",
            )
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        curves.append({"level": level, "label": label, "family": family, "data": df})
    return curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Superposition GEP supersonique vs Blumen.")
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.surface_csv)
    curves = load_digitized_curves()

    pivot = df.pivot(index="alpha", columns="Mach", values="gep_ci").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    ci = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    ci_min = float(np.nanmin(ci))
    ci_max = float(np.nanmax(ci))
    positive_levels = sorted(
        curve["level"]
        for curve in curves
        if curve["family"] in {"ci_level", "ci_special"} and curve["level"] is not None and ci_min <= curve["level"] <= ci_max
    )
    if not positive_levels:
        positive_levels = list(np.linspace(ci_min, ci_max, 4))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    contour = ax.contour(
        mach_grid,
        alpha_grid,
        ci,
        levels=positive_levels,
        cmap="viridis",
        linewidths=2.0,
    )
    ax.clabel(contour, fmt="%0.3f", fontsize=9)

    markers = {"ci_level": "o", "ci_special": "s", "cr_special": "^", "unknown": "x"}
    for curve in curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=16,
            alpha=0.8,
            marker=markers.get(curve["family"], "o"),
            color="black",
            label=f"Blumen {curve['label']}",
        )

    ax.set_title("Supersonique : isolignes GEP et points Blumen")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(float(df["Mach"].min()), float(df["Mach"].max()))
    ax.set_ylim(float(df["alpha"].min()), float(df["alpha"].max()))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
