from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def load_curve(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    return (
        pd.read_csv(path, header=None, names=["Mach", "alpha"], sep=";", decimal=",", engine="python")
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )


def nearest_point(df: pd.DataFrame, mach_target: float) -> tuple[float, float]:
    idx = (df["Mach"] - mach_target).abs().idxmin()
    row = df.loc[idx]
    return float(row["Mach"]), float(row["alpha"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduction propre de la figure supersonique de Blumen.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
        }
    )

    curves = {
        "0.01": load_curve("0.01.csv"),
        "0.03": load_curve("0.03.csv"),
        "0.05": load_curve("0.05 (1).csv"),
        "0.07": load_curve("0.07.csv"),
        "0.1": load_curve("0.1.csv"),
        "ci01": load_curve("ci01.csv"),
        "ci02": load_curve("ci02.csv"),
        "cr0": load_curve("cr0.csv"),
    }

    fig, ax = plt.subplots(figsize=(5.7, 3.9))

    # Main Blumen ci curves.
    for key in ["0.01", "0.03", "0.05", "0.07", "0.1"]:
        df = curves[key]
        ax.plot(df["Mach"], df["alpha"], color="black", linewidth=1.2)

    # Special curves visible in the original plate.
    ax.plot(curves["ci02"]["Mach"], curves["ci02"]["alpha"], color="black", linewidth=1.2)
    ax.plot(curves["ci01"]["Mach"], curves["ci01"]["alpha"], color="black", linewidth=0.95)
    ax.plot(curves["cr0"]["Mach"], curves["cr0"]["alpha"], color="black", linewidth=0.95, linestyle=(0, (5, 3)))

    # Axes style close to scanned-paper figure.
    ax.set_xlim(0.9, 2.1)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel(r"$M$", labelpad=6)
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Manual labels to match the original visual hierarchy.
    x, y = nearest_point(curves["0.01"], 1.88)
    ax.text(x + 0.02, y - 0.012, "0·01", fontsize=9)
    x, y = nearest_point(curves["0.03"], 1.63)
    ax.text(x + 0.03, y - 0.002, "0·03", fontsize=9)
    x, y = nearest_point(curves["0.05"], 1.48)
    ax.text(x + 0.02, y - 0.001, "0·05", fontsize=9)
    x, y = nearest_point(curves["0.07"], 1.36)
    ax.text(x + 0.015, y - 0.001, "0·07", fontsize=9)
    x, y = nearest_point(curves["0.1"], 1.16)
    ax.text(x - 0.01, y - 0.014, "0·1", fontsize=9)

    x, y = nearest_point(curves["ci02"], 1.02)
    ax.text(x - 0.015, y + 0.018, r"$c_i = 0$", fontsize=9)
    x, y = nearest_point(curves["cr0"], 1.04)
    ax.text(x + 0.01, y - 0.018, r"$c_r = 0$", fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
