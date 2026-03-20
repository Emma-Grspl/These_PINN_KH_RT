from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from classical_solver.subsonic.shooting_subsonic import SubsonicShootingSolver


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "subsonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting"


def parse_level(path: Path) -> float:
    stem = path.stem.strip().replace("_", ".").replace(",", ".")
    return float(stem)


def representative_points(df: pd.DataFrame, n_points: int = 3) -> pd.DataFrame:
    if len(df) <= n_points:
        return df.copy()
    indices = np.linspace(0, len(df) - 1, n_points, dtype=int)
    return df.iloc[indices].copy()


def load_blumen_samples(n_points_per_level: int = 3) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        level = parse_level(csv_path)
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["Mach", "alpha"],
            sep=";",
            decimal=",",
            engine="python",
        ).apply(pd.to_numeric, errors="coerce").dropna()
        sample_df = representative_points(df, n_points=n_points_per_level)
        for _, row in sample_df.iterrows():
            rows.append({"level": level, "Mach": float(row["Mach"]), "alpha": float(row["alpha"])})
    return pd.DataFrame(rows)


def run_current_solver(alpha: float, mach: float) -> dict:
    solver = SubsonicShootingSolver(alpha=alpha, Mach=mach)
    result = solver.solve_ci(ci_max=1.0, n_scan=81)
    return {
        "ci": result.ci,
        "omega_i": result.omega_i,
        "mismatch": result.mismatch,
        "success": result.success,
    }


def run_mstab_solver(alpha: float, mach: float) -> dict:
    solver = Mstab17SubsonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve(ci_min=1e-3, ci_max=1.0, n_scan=61)
    return {
        "ci": result.ci,
        "omega_i": result.omega_i,
        "stage1_mismatch": result.stage1_mismatch,
        "stage2_mismatch": result.stage2_mismatch,
        "success": result.success,
    }


def compare_on_blumen_points(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in samples.iterrows():
        alpha = float(row["alpha"])
        mach = float(row["Mach"])
        level = float(row["level"])

        current = run_current_solver(alpha, mach)
        mstab = run_mstab_solver(alpha, mach)

        rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "omega_target": level,
                "current_ci": current["ci"],
                "current_omega_i": current["omega_i"],
                "current_abs_omega_error": abs(current["omega_i"] - level),
                "current_success": current["success"],
                "mstab_ci": mstab["ci"],
                "mstab_omega_i": mstab["omega_i"],
                "mstab_abs_omega_error": abs(mstab["omega_i"] - level),
                "mstab_success": mstab["success"],
                "ci_abs_diff": abs(current["ci"] - mstab["ci"]),
                "omega_abs_diff": abs(current["omega_i"] - mstab["omega_i"]),
            }
        )
    return pd.DataFrame(rows)


def compare_on_grid() -> pd.DataFrame:
    grid = [
        (0.30, 0.00),
        (0.50, 0.00),
        (0.50, 0.40),
        (0.30, 0.60),
        (0.20, 0.80),
    ]
    rows = []
    for alpha, mach in grid:
        current = run_current_solver(alpha, mach)
        mstab = run_mstab_solver(alpha, mach)
        rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "current_ci": current["ci"],
                "mstab_ci": mstab["ci"],
                "ci_abs_diff": abs(current["ci"] - mstab["ci"]),
                "current_omega_i": current["omega_i"],
                "mstab_omega_i": mstab["omega_i"],
                "omega_abs_diff": abs(current["omega_i"] - mstab["omega_i"]),
                "current_success": current["success"],
                "mstab_success": mstab["success"],
            }
        )
    return pd.DataFrame(rows)


def summarize(blumen_df: pd.DataFrame, grid_df: pd.DataFrame) -> dict:
    return {
        "n_blumen_points": int(len(blumen_df)),
        "current_vs_blumen_mae_omega": float(blumen_df["current_abs_omega_error"].mean()),
        "mstab_vs_blumen_mae_omega": float(blumen_df["mstab_abs_omega_error"].mean()),
        "current_vs_mstab_mae_ci_grid": float(grid_df["ci_abs_diff"].mean()),
        "current_vs_mstab_mae_omega_grid": float(grid_df["omega_abs_diff"].mean()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blumen_samples = load_blumen_samples(n_points_per_level=3)
    blumen_comparison = compare_on_blumen_points(blumen_samples)
    grid_comparison = compare_on_grid()
    summary = summarize(blumen_comparison, grid_comparison)

    blumen_csv = OUTPUT_DIR / "subsonic_solver_comparison_blumen_points.csv"
    grid_csv = OUTPUT_DIR / "subsonic_solver_comparison_grid.csv"
    summary_json = OUTPUT_DIR / "subsonic_solver_comparison_summary.json"

    blumen_comparison.to_csv(blumen_csv, index=False)
    grid_comparison.to_csv(grid_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {blumen_csv}")
    print(f"Saved: {grid_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
