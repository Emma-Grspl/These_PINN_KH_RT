from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classical_solver.supersonic.reconstruct_blumen_supersonic_shooting import parse_reference_level


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting_supersonic"


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
        curves.append(
            {
                "path": Path(csv_file),
                "level": level,
                "label": label,
                "family": family,
                "data": df,
            }
        )
    return curves


def nearest_curve_info(mach: float, alpha: float, ci: float, curves: list[dict]) -> dict | None:
    ci_level_curves = [curve for curve in curves if curve["family"] == "ci_level" and curve["level"] is not None]
    if not ci_level_curves:
        return None

    best_curve = min(ci_level_curves, key=lambda curve: abs(curve["level"] - ci))
    points = best_curve["data"][["Mach", "alpha"]].to_numpy(dtype=float)
    distances = np.hypot(points[:, 0] - mach, points[:, 1] - alpha)
    index = int(np.argmin(distances))
    nearest = points[index]
    return {
        "curve_level": float(best_curve["level"]),
        "curve_label": best_curve["label"],
        "nearest_mach": float(nearest[0]),
        "nearest_alpha": float(nearest[1]),
        "distance": float(distances[index]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare quelques points mstab17 supersoniques aux points de Blumen.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "mstab17_supersonic_point_checks.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_points = [
        {"alpha": 0.18, "Mach": 1.10, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.24, "Mach": 1.10, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.30, "Mach": 1.10, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.18, "Mach": 1.20, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.24, "Mach": 1.20, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.18, "Mach": 1.30, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
    ]

    curves = load_digitized_curves()
    rows = []
    for item in test_points:
        solver = Mstab17SupersonicSolver(alpha=item["alpha"], Mach=item["Mach"])
        result = solver.solve(
            cr_min=item["cr_min"],
            cr_max=item["cr_max"],
            ci_min=item["ci_min"],
            ci_max=item["ci_max"],
            max_iter=item["max_iter"],
        )
        nearest = nearest_curve_info(result.Mach, result.alpha, result.ci, curves)
        rows.append(
            {
                "alpha": result.alpha,
                "Mach": result.Mach,
                "cr": result.cr,
                "ci": result.ci,
                "omega_i": result.omega_i,
                "stage1_mismatch": result.stage1_mismatch,
                "stage2_mismatch": result.stage2_mismatch,
                "success": result.success,
                **(nearest or {}),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "mstab17_supersonic_point_checks.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ci_levels = sorted(
        curve["level"]
        for curve in curves
        if curve["family"] == "ci_level" and curve["level"] is not None
    )

    color_map = plt.get_cmap("plasma")
    for idx, curve in enumerate(curves):
        if curve["family"] != "ci_level" or curve["level"] is None:
            continue
        color = color_map(idx / max(len(ci_levels), 1))
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=14,
            alpha=0.45,
            color=color,
            label=f"Blumen {curve['label']}",
        )

    success_df = df[df["success"]]
    failed_df = df[~df["success"]]
    ax.scatter(success_df["Mach"], success_df["alpha"], s=90, color="black", marker="x", label="mstab17 success")
    if not failed_df.empty:
        ax.scatter(failed_df["Mach"], failed_df["alpha"], s=90, color="red", marker="x", label="mstab17 stage2 fail")

    for _, row in df.iterrows():
        ax.annotate(
            f"ci={row['ci']:.3f}",
            (row["Mach"], row["alpha"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
        if not pd.isna(row.get("nearest_mach", np.nan)):
            ax.plot(
                [row["Mach"], row["nearest_mach"]],
                [row["alpha"], row["nearest_alpha"]],
                linestyle="--",
                linewidth=0.8,
                color="gray",
                alpha=0.6,
            )

    ax.set_title("Points supersoniques mstab17 compares aux points de Blumen")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nCSV enregistre dans {csv_path}")
    print(f"Figure enregistree dans {args.output}")


if __name__ == "__main__":
    main()
