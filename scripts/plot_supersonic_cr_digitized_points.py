from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_wide_digitization(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    levels = raw.iloc[0].tolist()
    coords = raw.iloc[1].tolist()
    data = raw.iloc[2:].reset_index(drop=True)
    rows: list[dict] = []
    for i in range(0, len(levels) - 1, 2):
        level = str(levels[i]).strip()
        x_coord = str(coords[i]).strip().upper()
        y_coord = str(coords[i + 1]).strip().upper()
        if not level or level.lower() == "nan" or x_coord != "X" or y_coord != "Y":
            continue
        x = pd.to_numeric(data.iloc[:, i], errors="coerce")
        y = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        mask = x.notna() & y.notna()
        for mach, alpha in zip(x[mask], y[mask]):
            rows.append({"level": str(level), "Mach": float(mach), "alpha": float(alpha)})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace les points digitalises c_r/c_i de Blumen.")
    parser.add_argument("--quantity", choices=["cr", "ci"], default="cr")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or ROOT_DIR / "KH_RT_Blumen" / "supersonic" / f"{args.quantity}_datasets.csv"
    output_path = args.output or ROOT_DIR / "assets" / "blumen" / f"supersonic_{args.quantity}_digitized_points.png"
    long_csv_path = args.long_csv or ROOT_DIR / "assets" / "blumen" / f"supersonic_{args.quantity}_digitized_points.csv"
    df = load_wide_digitization(input_path)
    if df.empty:
        raise RuntimeError(f"Aucun point lisible dans {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(long_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for level, sub in df.groupby("level", sort=False):
        marker = "x" if level in {"ci=0", "cr=0", "ci_sup=0"} else "o"
        alpha = 0.65 if marker == "x" else 0.85
        ax.scatter(sub["Mach"], sub["alpha"], s=18, alpha=alpha, marker=marker, label=level)

    figure = "3" if args.quantity == "cr" else "4"
    ax.set_title(rf"Blumen 1975, figure {figure} : points digitalises de $c_{args.quantity[-1]}$")
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"Nombre d'onde $\alpha$")
    ax.set_xlim(0.88, 2.12)
    ax.set_ylim(0.0, 0.52)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(output_path)
    print(long_csv_path)


if __name__ == "__main__":
    main()
