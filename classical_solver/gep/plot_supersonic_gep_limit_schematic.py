from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


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
    parser = argparse.ArgumentParser(description="Schema de la limite actuelle du GEP supersonique.")
    parser.add_argument(
        "--frontier-csv",
        type=Path,
        default=OUTPUT_DIR / "scan_gep_validity_frontier_pin3_xi099_m110_200_frontier.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "supersonic_gep_limit_schematic.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    curves = load_digitized_curves()
    frontier = pd.read_csv(args.frontier_csv)

    fig, ax = plt.subplots(figsize=(11, 7))

    x_min, x_max = 0.95, 2.05
    y_min, y_max = 0.0, 0.52
    frontier = frontier.sort_values("Mach").reset_index(drop=True)

    # Use the Blumen point cloud itself as the background visual support.
    all_points = []
    for curve in curves:
        df = curve["data"].copy()
        df["family"] = curve["family"]
        df["level"] = curve["level"]
        all_points.append(df)
    blumen = pd.concat(all_points, ignore_index=True)

    ci_levels = blumen[blumen["family"] == "ci_level"]
    ci_special = blumen[blumen["family"] == "ci_special"]
    cr_special = blumen[blumen["family"] == "cr_special"]

    machs = frontier["Mach"].to_numpy(dtype=float)
    if len(machs) >= 2:
        edges = np.empty(len(machs) + 1, dtype=float)
        edges[1:-1] = 0.5 * (machs[:-1] + machs[1:])
        edges[0] = x_min
        edges[-1] = x_max
    else:
        edges = np.array([x_min, x_max], dtype=float)

    legend_done = {"valid": False, "drift": False, "invalid": False}
    for idx, row in frontier.iterrows():
        left = float(edges[idx])
        right = float(edges[idx + 1])
        alpha_max_valid = row["alpha_max_valid"]
        if pd.notna(alpha_max_valid):
            alpha_max_valid = float(alpha_max_valid)
            ax.fill_between(
                [left, right],
                y_min,
                alpha_max_valid,
                color="#5fbf63",
                alpha=0.26,
                label=None if legend_done["valid"] else "Zone ou le GEP reconstruit correctement le mode dominant",
            )
            legend_done["valid"] = True
            ax.fill_between(
                [left, right],
                alpha_max_valid,
                y_max,
                color="#f28e2b",
                alpha=0.26,
                label=None if legend_done["drift"] else "Zone ou la reconstruction commence a decrocher",
            )
            legend_done["drift"] = True
        else:
            ax.axvspan(
                left,
                right,
                color="#b22222",
                alpha=0.22,
                label=None if legend_done["invalid"] else "Zone non validee avec la recette actuelle",
            )
            legend_done["invalid"] = True

    if not ci_levels.empty:
        ax.scatter(
            ci_levels["Mach"],
            ci_levels["alpha"],
            s=16,
            color="black",
            alpha=0.55,
            edgecolors="none",
            label="Points Blumen",
        )
    if not ci_special.empty:
        ax.scatter(
            ci_special["Mach"],
            ci_special["alpha"],
            s=20,
            marker="s",
            color="black",
            alpha=0.65,
            edgecolors="none",
        )
    if not cr_special.empty:
        ax.scatter(
            cr_special["Mach"],
            cr_special["alpha"],
            s=22,
            marker="^",
            color="black",
            alpha=0.65,
            edgecolors="none",
        )

    valid = frontier.dropna(subset=["alpha_max_valid"]).copy()
    if not valid.empty:
        ax.plot(
            valid["Mach"],
            valid["alpha_max_valid"],
            color="#d62728",
            linewidth=2.8,
            marker="o",
            label=r"Frontiere de validite actuelle du GEP",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Mach $M$")
    ax.set_ylabel(r"Nombre d'onde $\alpha$")
    ax.set_title("Limite actuelle de reconstruction du GEP supersonique")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistree dans {args.output}")


if __name__ == "__main__":
    main()
