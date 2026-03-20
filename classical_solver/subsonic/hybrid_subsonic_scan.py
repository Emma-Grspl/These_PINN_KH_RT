from __future__ import annotations

"""
Balayage subsonique hybride.

Idee :
- evaluation rapide sur toute la grille avec le solveur principal ;
- correction locale par le solveur mstab17 uniquement dans une bande proche de la
  frontiere neutre ou si le point principal parait numeriquement suspect.
"""

import argparse
import glob
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from classical_solver.subsonic.shooting_subsonic import SubsonicShootingSolver


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "subsonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting_hybrid"


def parse_level(csv_path: str) -> tuple[float, str]:
    stem = Path(csv_path).stem.strip().replace("_", ".").replace(",", ".")
    value = float(stem)
    label = f"{value:.3f}".rstrip("0").rstrip(".")
    return value, fr"$\omega_i = {label}$"


def load_digitized_curves() -> list[dict]:
    curves = []
    for csv_file in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        level, label = parse_level(csv_file)
        df = pd.read_csv(
            csv_file,
            header=None,
            names=["Mach", "alpha"],
            sep=";",
            decimal=",",
            engine="python",
        ).apply(pd.to_numeric, errors="coerce").dropna()
        curves.append({"level": level, "label": label, "data": df})
    return curves


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if np.isclose(denom, 0.0):
        return float(np.linalg.norm(point - a))
    t = np.dot(point - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    return min(point_to_segment_distance(point, polyline[idx], polyline[idx + 1]) for idx in range(len(polyline) - 1))


def build_solver_contours(mach_grid: np.ndarray, alpha_grid: np.ndarray, omega: np.ndarray, levels: list[float]) -> dict[float, list[np.ndarray]]:
    fig, ax = plt.subplots()
    contour = ax.contour(mach_grid, alpha_grid, omega, levels=levels)
    contour_map: dict[float, list[np.ndarray]] = {}
    for level, segments in zip(contour.levels, contour.allsegs):
        contour_map[float(level)] = [np.asarray(segment, dtype=float) for segment in segments if len(segment) >= 2]
    plt.close(fig)
    return contour_map


def compute_error_report(df: pd.DataFrame, curves: list[dict]) -> tuple[pd.DataFrame, dict]:
    pivot = df.pivot(index="alpha", columns="Mach", values="omega_i").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    omega = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    levels = sorted(curve["level"] for curve in curves if curve["level"] > 0.0)
    interpolator = RegularGridInterpolator((alphas, machs), omega, bounds_error=False, fill_value=np.nan)
    contour_map = build_solver_contours(mach_grid, alpha_grid, omega, levels)

    level_rows = []
    point_rows = []
    for curve in curves:
        level = float(curve["level"])
        points = curve["data"][["Mach", "alpha"]].to_numpy(dtype=float)
        query_points = curve["data"][["alpha", "Mach"]].to_numpy(dtype=float)
        omega_pred = interpolator(query_points)
        abs_level_residual = np.abs(omega_pred - level)

        polylines = contour_map.get(level, [])
        distances = []
        for point in points:
            if polylines:
                distances.append(min(point_to_polyline_distance(point, polyline) for polyline in polylines))
            else:
                distances.append(np.nan)
        distances = np.asarray(distances, dtype=float)

        for idx, point in enumerate(points):
            point_rows.append(
                {
                    "level": level,
                    "Mach": float(point[0]),
                    "alpha": float(point[1]),
                    "abs_omega_residual": float(abs_level_residual[idx]),
                    "distance_to_solver_isoline": float(distances[idx]) if np.isfinite(distances[idx]) else np.nan,
                }
            )

        finite_residuals = abs_level_residual[np.isfinite(abs_level_residual)]
        finite_distances = distances[np.isfinite(distances)]
        level_rows.append(
            {
                "level": level,
                "n_points": len(points),
                "mae_omega": float(np.mean(finite_residuals)) if len(finite_residuals) else np.nan,
                "median_distance": float(np.median(finite_distances)) if len(finite_distances) else np.nan,
                "p90_distance": float(np.percentile(finite_distances, 90)) if len(finite_distances) else np.nan,
            }
        )

    level_df = pd.DataFrame(level_rows).sort_values("level").reset_index(drop=True)
    point_df = pd.DataFrame(point_rows).sort_values(["level", "Mach", "alpha"]).reset_index(drop=True)
    summary = {
        "global_mae_omega": float(level_df["mae_omega"].dropna().mean()) if not level_df.empty else None,
        "global_median_distance": float(level_df["median_distance"].dropna().mean()) if not level_df.empty else None,
        "source_counts": df["source"].value_counts(dropna=False).to_dict(),
    }
    return level_df, {"summary": summary, "points": point_df}


def plot_map(df: pd.DataFrame, curves: list[dict], output_path: Path) -> None:
    pivot = df.pivot(index="alpha", columns="Mach", values="omega_i").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    omega = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    positive_levels = sorted(curve["level"] for curve in curves if curve["level"] > 0.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    if positive_levels:
        contour = ax.contour(mach_grid, alpha_grid, omega, levels=positive_levels, cmap="viridis", linewidths=1.8)
        ax.clabel(contour, fmt="%0.3f", fontsize=8)

    for curve in curves:
        ax.scatter(curve["data"]["Mach"], curve["data"]["alpha"], s=12, alpha=0.7, label=f"Blumen {curve['label']}")

    mach_line = np.linspace(0.0, 1.0, 500)
    alpha_line = np.sqrt(np.clip(1.0 - mach_line**2, 0.0, None))
    ax.plot(mach_line, alpha_line, "--", color="black", linewidth=1.8, label=r"Frontière neutre $\alpha^2 + M^2 = 1$")

    ax.set_title("Blumen 1970 : reconstruction subsonique hybride")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Balayage subsonique hybride.")
    parser.add_argument("--mach-min", type=float, default=0.0)
    parser.add_argument("--mach-max", type=float, default=0.98)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--num-mach", type=int, default=41)
    parser.add_argument("--num-alpha", type=int, default=41)
    parser.add_argument("--ci-max", type=float, default=1.0)
    parser.add_argument("--n-scan", type=int, default=81)
    parser.add_argument("--neutral-ratio-threshold", type=float, default=0.85)
    parser.add_argument("--mismatch-threshold", type=float, default=5e-4)
    return parser


def sample_hybrid_subsonic_growth_map(
    alphas: np.ndarray,
    machs: np.ndarray,
    *,
    ci_max: float,
    n_scan: int,
    neutral_ratio_threshold: float,
    mismatch_threshold: float,
) -> pd.DataFrame:
    rows = []
    for mach in machs:
        alpha_cut = np.sqrt(max(0.0, 1.0 - float(mach) ** 2))
        unstable_alphas = np.sort(alphas[alphas < alpha_cut - 1e-12])[::-1]
        stable_alphas = np.sort(alphas[alphas >= alpha_cut - 1e-12])
        previous_ci = None

        for alpha in unstable_alphas:
            primary_solver = SubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            primary = primary_solver.solve_ci(ci_max=ci_max, n_scan=n_scan, previous_ci=previous_ci)

            source = "primary"
            final_ci = primary.ci
            final_omega = primary.omega_i
            secondary_ci = None
            secondary_omega = None
            secondary_stage1 = None
            secondary_stage2 = None

            alpha_ratio = float(alpha / alpha_cut) if alpha_cut > 1e-12 else 1.0
            need_secondary = (alpha_ratio >= neutral_ratio_threshold) or (primary.mismatch >= mismatch_threshold) or (not primary.success)

            if need_secondary:
                secondary_solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
                secondary = secondary_solver.solve(ci_min=1e-3, ci_max=ci_max, n_scan=41)
                secondary_ci = secondary.ci
                secondary_omega = secondary.omega_i
                secondary_stage1 = secondary.stage1_mismatch
                secondary_stage2 = secondary.stage2_mismatch

                if secondary.success:
                    final_ci = secondary.ci
                    final_omega = secondary.omega_i
                    source = "secondary"

            if primary.success:
                previous_ci = final_ci

            rows.append(
                {
                    "alpha": float(alpha),
                    "Mach": float(mach),
                    "ci": final_ci,
                    "omega_i": final_omega,
                    "source": source,
                    "primary_ci": primary.ci,
                    "primary_omega_i": primary.omega_i,
                    "primary_mismatch": primary.mismatch,
                    "primary_success": primary.success,
                    "secondary_ci": secondary_ci,
                    "secondary_omega_i": secondary_omega,
                    "secondary_stage1_mismatch": secondary_stage1,
                    "secondary_stage2_mismatch": secondary_stage2,
                }
            )

        for alpha in stable_alphas:
            rows.append(
                {
                    "alpha": float(alpha),
                    "Mach": float(mach),
                    "ci": 0.0,
                    "omega_i": 0.0,
                    "source": "stable",
                    "primary_ci": 0.0,
                    "primary_omega_i": 0.0,
                    "primary_mismatch": 0.0,
                    "primary_success": True,
                    "secondary_ci": None,
                    "secondary_omega_i": None,
                    "secondary_stage1_mismatch": None,
                    "secondary_stage2_mismatch": None,
                }
            )

    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)

    print("Echantillonnage subsonique hybride...")
    df = sample_hybrid_subsonic_growth_map(
        alphas,
        machs,
        ci_max=args.ci_max,
        n_scan=args.n_scan,
        neutral_ratio_threshold=args.neutral_ratio_threshold,
        mismatch_threshold=args.mismatch_threshold,
    )

    csv_path = OUTPUT_DIR / "subsonic_hybrid_growth_map.csv"
    fig_path = OUTPUT_DIR / "subsonic_hybrid_vs_blumen.png"
    error_level_path = OUTPUT_DIR / "subsonic_hybrid_error_by_level.csv"
    error_point_path = OUTPUT_DIR / "subsonic_hybrid_error_by_point.csv"
    error_summary_path = OUTPUT_DIR / "subsonic_hybrid_error_summary.json"

    df.to_csv(csv_path, index=False)
    curves = load_digitized_curves()
    plot_map(df, curves, fig_path)
    error_by_level, error_payload = compute_error_report(df, curves)
    error_by_level.to_csv(error_level_path, index=False)
    error_payload["points"].to_csv(error_point_path, index=False)
    with open(error_summary_path, "w", encoding="utf-8") as stream:
        json.dump(error_payload["summary"], stream, indent=2)

    if not error_by_level.empty:
        print("\nErreur moyenne par niveau:")
        print(error_by_level.to_string(index=False))
        print("\nResume global:")
        print(json.dumps(error_payload["summary"], indent=2))

    print(f"Resultats enregistres dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
