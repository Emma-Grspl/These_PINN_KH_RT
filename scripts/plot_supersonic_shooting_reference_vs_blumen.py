from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BLUMEN_CI_POINTS_CSV = (
    ROOT_DIR / "assets" / "classic_supersonic" / "blumen_reference" / "supersonic_ci_digitized_points.csv"
)
DEFAULT_SPECTRAL_CSV = (
    ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_spectral.csv"
)
DEFAULT_MODAL_CSV = (
    ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_modal.csv"
)
DEFAULT_OUTPUT = (
    ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_ci_vs_blumen.png"
)


def load_curves() -> dict[str, pd.DataFrame]:
    points = pd.read_csv(BLUMEN_CI_POINTS_CSV)
    curves: dict[str, pd.DataFrame] = {}
    for level, df in points.groupby("level", sort=False):
        curves[str(level)] = df[["Mach", "alpha"]].sort_values("Mach").reset_index(drop=True)
    return curves


def nearest_point(df: pd.DataFrame, mach_target: float) -> tuple[float, float]:
    idx = (df["Mach"] - mach_target).abs().idxmin()
    row = df.loc[idx]
    return float(row["Mach"]), float(row["alpha"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay des points de reference shooting acceptes sur les isolignes supersoniques de Blumen en c_i."
    )
    parser.add_argument("--spectral-csv", type=Path, default=DEFAULT_SPECTRAL_CSV)
    parser.add_argument("--modal-csv", type=Path, default=DEFAULT_MODAL_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spectral_df = pd.read_csv(args.spectral_csv)
    modal_df = pd.read_csv(args.modal_csv)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
        }
    )

    curves = load_curves()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for key in ["0.01", "0.03", "0.05", "0.07", "0.1"]:
        df = curves[key]
        ax.scatter(df["Mach"], df["alpha"], s=13, marker="o", facecolor="black", edgecolor="none", alpha=0.85, zorder=1)

    ax.scatter(
        curves["ci=0"]["Mach"],
        curves["ci=0"]["alpha"],
        s=13,
        marker="o",
        facecolor="black",
        edgecolor="none",
        alpha=0.85,
        zorder=1,
    )
    ax.scatter(
        curves["ci_sup=0"]["Mach"],
        curves["ci_sup=0"]["alpha"],
        s=15,
        marker="s",
        facecolor="black",
        edgecolor="none",
        alpha=0.85,
        zorder=1,
    )
    ax.scatter(
        curves["cr=0"]["Mach"],
        curves["cr=0"]["alpha"],
        s=15,
        marker="^",
        facecolor="black",
        edgecolor="none",
        alpha=0.85,
        zorder=1,
    )

    ax.scatter(
        spectral_df["Mach"],
        spectral_df["alpha"],
        s=55,
        marker="o",
        facecolor="#D97706",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
        label=r"Shooting accepte (spectral)",
        zorder=3,
    )
    ax.scatter(
        modal_df["Mach"],
        modal_df["alpha"],
        s=95,
        marker="*",
        facecolor="#0F766E",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.98,
        label=r"Shooting valide (modal)",
        zorder=4,
    )

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

    x, y = nearest_point(curves["ci=0"], 1.02)
    ax.text(x - 0.015, y + 0.018, r"$c_i = 0$", fontsize=9)
    x, y = nearest_point(curves["cr=0"], 1.04)
    ax.text(x + 0.01, y - 0.018, r"$c_r = 0$", fontsize=9)

    ax.set_xlim(0.9, 2.1)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel(r"$M$", labelpad=6)
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=10)
    ax.set_title(r"Shooting accepte sur les points de Blumen ($c_i$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, linestyle=":", alpha=0.25)
    blumen_handle = plt.Line2D([], [], linestyle="none", marker="o", color="black", markersize=4, label="Points Blumen")
    spectral_handle = plt.Line2D(
        [], [], linestyle="none", marker="o", markerfacecolor="#D97706", markeredgecolor="black", markersize=6, label=r"Shooting accepte (spectral)"
    )
    modal_handle = plt.Line2D(
        [], [], linestyle="none", marker="*", markerfacecolor="#0F766E", markeredgecolor="black", markersize=10, label=r"Shooting valide (modal)"
    )
    ax.legend(handles=[blumen_handle, spectral_handle, modal_handle], loc="upper right", frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
