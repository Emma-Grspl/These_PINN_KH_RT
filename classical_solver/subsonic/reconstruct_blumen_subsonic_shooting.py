from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT_DIR = Path("/Users/emma.grospellier/Thèse/These_PINN_KH_RT")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.shooting_subsonic import sample_subsonic_growth_map


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "subsonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruction Blumen 1970 par méthode du tir subsonique.")
    parser.add_argument("--mach-min", type=float, default=0.0)
    parser.add_argument("--mach-max", type=float, default=0.98)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--num-mach", type=int, default=25)
    parser.add_argument("--num-alpha", type=int, default=25)
    parser.add_argument("--ci-max", type=float, default=1.0)
    parser.add_argument("--n-scan", type=int, default=81)
    return parser


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
    return min(
        point_to_segment_distance(point, polyline[idx], polyline[idx + 1])
        for idx in range(len(polyline) - 1)
    )


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
    interpolator = RegularGridInterpolator(
        (alphas, machs),
        omega,
        bounds_error=False,
        fill_value=np.nan,
    )
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
                "rmse_omega": float(np.sqrt(np.mean(finite_residuals**2))) if len(finite_residuals) else np.nan,
                "median_omega": float(np.median(finite_residuals)) if len(finite_residuals) else np.nan,
                "median_distance": float(np.median(finite_distances)) if len(finite_distances) else np.nan,
                "p90_distance": float(np.percentile(finite_distances, 90)) if len(finite_distances) else np.nan,
            }
        )

    level_df = pd.DataFrame(level_rows).sort_values("level").reset_index(drop=True)
    point_df = pd.DataFrame(point_rows).sort_values(["level", "Mach", "alpha"]).reset_index(drop=True)

    finite_level_mae = level_df["mae_omega"].dropna().to_numpy(dtype=float)
    finite_level_dist = level_df["median_distance"].dropna().to_numpy(dtype=float)
    summary = {
        "global_mae_omega": float(np.mean(finite_level_mae)) if len(finite_level_mae) else None,
        "global_median_distance": float(np.mean(finite_level_dist)) if len(finite_level_dist) else None,
        "notes": [
            "mae_omega mesure l'ecart entre omega_i du solveur et le niveau Blumen aux points digitalises.",
            "median_distance mesure la distance geometrique point -> isoligne du solveur du meme niveau dans le plan (M, alpha).",
            "Ces metriques sont plus robustes qu'un appariement exact point a point et tolerent le bruit de digitalisation manuelle.",
        ],
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
        contour = ax.contour(
            mach_grid,
            alpha_grid,
            omega,
            levels=positive_levels,
            cmap="viridis",
            linewidths=1.8,
        )
        ax.clabel(contour, fmt="%0.3f", fontsize=8)

    for curve in curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=12,
            alpha=0.7,
            label=f"Blumen {curve['label']}",
        )

    mach_line = np.linspace(0.0, 1.0, 500)
    alpha_line = np.sqrt(np.clip(1.0 - mach_line**2, 0.0, None))
    ax.plot(mach_line, alpha_line, "--", color="black", linewidth=1.8, label=r"Frontière neutre $\alpha^2 + M^2 = 1$")

    ax.set_title("Blumen 1970 : reconstruction subsonique par méthode du tir")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
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

    print("Echantillonnage subsonique par méthode du tir...")
    df = sample_subsonic_growth_map(
        alphas,
        machs,
        ci_max=args.ci_max,
        n_scan=args.n_scan,
    )

    csv_path = OUTPUT_DIR / "subsonic_shooting_growth_map.csv"
    fig_path = OUTPUT_DIR / "subsonic_shooting_vs_blumen.png"
    error_level_path = OUTPUT_DIR / "subsonic_shooting_error_by_level.csv"
    error_point_path = OUTPUT_DIR / "subsonic_shooting_error_by_point.csv"
    error_summary_path = OUTPUT_DIR / "subsonic_shooting_error_summary.json"
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
