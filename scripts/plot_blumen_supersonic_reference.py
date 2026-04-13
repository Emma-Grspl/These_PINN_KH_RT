from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def parse_label(csv_path: Path) -> tuple[float, str]:
    stem = csv_path.stem.strip().replace("_", ".").replace(",", ".")
    lower = stem.lower()
    if lower.startswith("ci"):
        value = float(lower[2:] or "0") / 100.0
        return 100.0 + value, fr"$c_r = 0,\; c_i = {value:.2f}$"
    if lower.startswith("cr"):
        value = float(lower[2:] or "0")
        return 200.0 + value, fr"$c_i = 0,\; c_r = {value:.2f}$"
    numeric = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    value = float(numeric)
    txt = f"{value:.3f}".rstrip("0").rstrip(".")
    return value, fr"$c_i = {txt}$"


def load_curve(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path, header=None, names=["Mach", "alpha"], sep=";", decimal=",", engine="python")
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace propre des courbes supersoniques de Blumen.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zoom-output", type=Path, default=None)
    return parser


def plot_reference(
    curves: list[tuple[float, str, pd.DataFrame]],
    output: Path,
    *,
    xlim=None,
    ylim=None,
    title: str,
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    for _, label, df in curves:
        ax.plot(df["Mach"], df["alpha"], color="black", linewidth=1.8)
        if annotate:
            end = df.iloc[-1]
            ax.text(float(end["Mach"]) + 0.015, float(end["alpha"]), label, fontsize=9, va="center", clip_on=True)

    ax.set_title(title)
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"Nombre d'onde $\alpha$")
    ax.grid(True, linestyle=":", alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    curves = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        order, label = parse_label(path)
        curves.append((order, label, load_curve(path)))
    curves.sort(key=lambda item: item[0])

    plot_reference(
        curves,
        args.output,
        title="Courbes supersoniques de Blumen",
    )
    if args.zoom_output is not None:
        plot_reference(
            curves,
            args.zoom_output,
            xlim=(1.65, 1.82),
            ylim=(0.17, 0.22),
            title="Courbes supersoniques de Blumen : zoom local",
            annotate=False,
        )


if __name__ == "__main__":
    main()
