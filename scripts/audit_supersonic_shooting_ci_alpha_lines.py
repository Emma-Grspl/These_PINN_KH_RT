from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_supersonic_shooting_point_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    build_parser as build_point_parser,
    evaluate_point,
)


def build_parser() -> argparse.ArgumentParser:
    point_defaults = build_point_parser()
    parser = argparse.ArgumentParser(
        description=(
            "Balayage supersonique au shooting pour tracer des courbes locales c_i(alpha) a Mach fixe, "
            "avec comparaison a Blumen la ou c_i est disponible."
        )
    )
    parser.add_argument("--machs", type=float, nargs="+", required=True, help="Liste des Mach fixes a auditer.")
    parser.add_argument("--alphas", type=float, nargs="*", default=None, help="Liste explicite des alpha a utiliser.")
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.30)
    parser.add_argument("--num-alpha", type=int, default=9)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--match-y", type=float, default=point_defaults.get_default("match_y"))
    parser.add_argument("--use-mapping", action="store_true", default=point_defaults.get_default("use_mapping"))
    parser.add_argument("--mapping-scale", type=float, default=point_defaults.get_default("mapping_scale"))
    parser.add_argument("--min-y-limit", type=float, default=point_defaults.get_default("min_y_limit"))
    parser.add_argument("--max-y-limit", type=float, default=point_defaults.get_default("max_y_limit"))
    parser.add_argument("--y-limit-factor", type=float, default=point_defaults.get_default("y_limit_factor"))
    parser.add_argument("--amp-lower-bound", type=float, default=point_defaults.get_default("amp_lower_bound"))
    parser.add_argument("--amp-upper-bound", type=float, default=point_defaults.get_default("amp_upper_bound"))
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=point_defaults.get_default("cr_half_windows"))
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=point_defaults.get_default("ci_half_windows"))
    parser.add_argument("--retry-growth", type=float, default=point_defaults.get_default("retry_growth"))
    parser.add_argument("--max-retries", type=int, default=point_defaults.get_default("max_retries"))
    parser.add_argument("--max-iter", type=int, default=point_defaults.get_default("max_iter"))
    parser.add_argument("--grid-size", type=int, default=point_defaults.get_default("grid_size"))
    parser.add_argument("--ci-weight", type=float, default=point_defaults.get_default("ci_weight"))
    parser.add_argument("--cr-weight", type=float, default=point_defaults.get_default("cr_weight"))
    parser.add_argument("--continuity-weight", type=float, default=point_defaults.get_default("continuity_weight"))
    parser.add_argument("--edge-amp-threshold", type=float, default=point_defaults.get_default("edge_amp_threshold"))
    parser.add_argument("--no-generic-seeds", action="store_false", dest="include_generic_seeds")
    parser.set_defaults(include_generic_seeds=True)
    parser.add_argument("--cr-points", type=Path, default=point_defaults.get_default("cr_points"))
    parser.add_argument("--ci-points", type=Path, default=point_defaults.get_default("ci_points"))
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def alpha_grid_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.alphas:
        return np.array(sorted({float(value) for value in args.alphas}), dtype=float)
    return np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.num_alpha), dtype=float)


def build_cfg(args: argparse.Namespace) -> dict[str, object]:
    return {
        "match_y": float(args.match_y),
        "use_mapping": bool(args.use_mapping),
        "mapping_scale": float(args.mapping_scale),
        "min_y_limit": float(args.min_y_limit),
        "max_y_limit": float(args.max_y_limit),
        "y_limit_factor": float(args.y_limit_factor),
        "amp_lower_bound": float(args.amp_lower_bound),
        "amp_upper_bound": float(args.amp_upper_bound),
        "cr_half_windows": [float(value) for value in args.cr_half_windows],
        "ci_half_windows": [float(value) for value in args.ci_half_windows],
        "retry_growth": float(args.retry_growth),
        "max_retries": int(args.max_retries),
        "max_iter": int(args.max_iter),
        "grid_size": int(args.grid_size),
        "ci_weight": float(args.ci_weight),
        "cr_weight": float(args.cr_weight),
        "continuity_weight": float(args.continuity_weight),
        "include_generic_seeds": bool(args.include_generic_seeds),
        "edge_amp_threshold": float(args.edge_amp_threshold),
        "cr_points": str(args.cr_points),
        "ci_points": str(args.ci_points),
    }


def build_points(alphas: np.ndarray, machs: list[float]) -> list[tuple[float, float]]:
    return [(float(alpha), float(mach)) for mach in machs for alpha in alphas]


def plot_ci_alpha_lines(summary_df: pd.DataFrame, machs: list[float], output_path: Path) -> None:
    ncols = 2 if len(machs) > 1 else 1
    nrows = int(np.ceil(len(machs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.2 * nrows), squeeze=False, sharex=True)
    status_colors = {
        "validated": "#15803D",
        "spectral_only": "#D97706",
        "mode_only": "#7C3AED",
        "failed": "#DC2626",
        "exception": "#111827",
    }

    for ax, mach in zip(axes.ravel(), machs):
        sub = summary_df[np.isclose(summary_df["Mach"].to_numpy(dtype=float), float(mach))].sort_values("alpha")
        if sub.empty:
            ax.set_visible(False)
            continue
        alpha = sub["alpha"].to_numpy(dtype=float)
        ci_shoot = sub["best_shooting_ci"].to_numpy(dtype=float)
        ci_blumen = sub["blumen_ci"].to_numpy(dtype=float)
        ax.plot(alpha, ci_shoot, color="#111827", linewidth=1.6, label="shooting")
        finite_ci = np.isfinite(ci_blumen)
        if np.any(finite_ci):
            ax.plot(alpha[finite_ci], ci_blumen[finite_ci], color="#2563EB", linestyle="--", linewidth=1.4, label="Blumen $c_i$")
        for status, sub_status in sub.groupby("best_status", sort=False):
            ax.scatter(
                sub_status["alpha"],
                sub_status["best_shooting_ci"],
                s=42,
                color=status_colors.get(str(status), "#4B5563"),
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
                label=str(status) if str(status) not in ax.get_legend_handles_labels()[1] else None,
            )
        ax.set_title(f"Mach = {float(mach):.2f}")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$c_i$")
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes.ravel()[len(machs) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=True)
    fig.suptitle(r"Supersonic shooting: local $c_i(\alpha)$ curves at fixed Mach", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ci_alpha_errors(summary_df: pd.DataFrame, machs: list[float], output_path: Path) -> None:
    ncols = 2 if len(machs) > 1 else 1
    nrows = int(np.ceil(len(machs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.2 * nrows), squeeze=False, sharex=True)
    for ax, mach in zip(axes.ravel(), machs):
        sub = summary_df[np.isclose(summary_df["Mach"].to_numpy(dtype=float), float(mach))].sort_values("alpha")
        if sub.empty:
            ax.set_visible(False)
            continue
        alpha = sub["alpha"].to_numpy(dtype=float)
        err_abs = sub["best_err_ci_abs"].to_numpy(dtype=float)
        finite = np.isfinite(err_abs)
        if np.any(finite):
            ax.plot(alpha[finite], err_abs[finite], color="#B45309", marker="o", linewidth=1.5, markersize=4.5)
        else:
            ax.text(0.5, 0.5, "No Blumen $c_i$ on this line", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Mach = {float(mach):.2f}")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$|c_i^{shoot} - c_i^{Blumen}|$")
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes.ravel()[len(machs) :]:
        ax.set_visible(False)
    fig.suptitle(r"Supersonic shooting: absolute $c_i$ error against Blumen", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = alpha_grid_from_args(args)
    machs = [float(value) for value in args.machs]
    points = build_points(alphas, machs)
    cfg = build_cfg(args)

    print("Supersonic shooting ci(alpha) local line audit")
    print(f"machs: {' '.join(f'{mach:.3f}' for mach in machs)}")
    print(f"alphas: {' '.join(f'{alpha:.3f}' for alpha in alphas)}")
    print(f"workers={int(args.workers)}")
    print(
        f"box: min={float(args.min_y_limit):.1f} max={float(args.max_y_limit):.1f} "
        f"factor={float(args.y_limit_factor):.2f} amp=[{float(args.amp_lower_bound):.1f},{float(args.amp_upper_bound):.1f}]"
    )

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = {executor.submit(evaluate_point, point, cfg): point for point in points}
        for future in as_completed(futures):
            alpha, mach = futures[future]
            summary_row, point_candidates, _point_fields = future.result()
            summary_rows.append(summary_row)
            candidate_rows.extend(point_candidates)
            print(
                f"[line] alpha={alpha:.3f} Mach={mach:.3f} "
                f"status={summary_row['best_status']} "
                f"ci={float(summary_row['best_shooting_ci']) if np.isfinite(summary_row['best_shooting_ci']) else np.nan:.5f} "
                f"err_ci={float(summary_row['best_err_ci_abs']) if np.isfinite(summary_row['best_err_ci_abs']) else np.nan:.3e} "
                f"stage1={float(summary_row['best_stage1_mismatch']) if np.isfinite(summary_row['best_stage1_mismatch']) else np.nan:.3e} "
                f"box_any={bool(summary_row['box_truncation_suspect_any_field'])}"
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["Mach", "alpha", "success", "selection_metric"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    curve_path = output_dir / f"{args.output_stem}_ci_alpha_lines.png"
    error_path = output_dir / f"{args.output_stem}_ci_alpha_errors.png"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    plot_ci_alpha_lines(summary_df, machs, curve_path)
    plot_ci_alpha_errors(summary_df, machs, error_path)

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))

    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {curve_path}")
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
