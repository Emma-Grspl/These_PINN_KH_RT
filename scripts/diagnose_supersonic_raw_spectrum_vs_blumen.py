from __future__ import annotations

import argparse
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

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    build_blumen_targets,
    load_digitized_long,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic du spectre GEP brut contre la cible Blumen.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--n-points-values", type=int, nargs="+", default=[561, 801, 1001])
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--max-abs-c", type=float, default=10.0)
    parser.add_argument("--positive-ci-only", action="store_true", default=True)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--cr-box-tol", type=float, default=0.03)
    parser.add_argument("--ci-box-tol", type=float, default=0.015)
    parser.add_argument("--top-k-nearest", type=int, default=12)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def extract_raw_modes(
    solver: NotebookStyleDenseGEPSolver,
    *,
    max_abs_c: float,
    positive_ci_only: bool,
) -> list[dict]:
    vals, _ = solver.solve_all()
    finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
    vals = vals[finite]
    vals = vals[np.abs(vals) < max_abs_c]

    rows: list[dict] = []
    for val in vals:
        cr = float(np.real(val))
        ci = float(np.imag(val))
        if positive_ci_only and ci <= 0.0:
            continue
        rows.append(
            {
                "cr": cr,
                "ci": ci,
                "omega_i": float(solver.alpha * ci),
                "abs_c": float(abs(val)),
            }
        )
    return rows


def plot_raw_spectrum(
    modes_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    output_path: Path,
    *,
    cr_box_tol: float,
    ci_box_tol: float,
) -> None:
    mach_values = sorted(targets_df["Mach"].unique())
    ncols = 2
    nrows = int(np.ceil(len(mach_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_values = sorted(modes_df["n_points"].dropna().unique())
    color_map = {n_value: color_cycle[idx % len(color_cycle)] for idx, n_value in enumerate(n_values)}

    for ax, mach in zip(axes_flat, mach_values):
        sub = modes_df[modes_df["Mach"] == mach].copy()
        target = targets_df[targets_df["Mach"] == mach].iloc[0]

        for n_points in n_values:
            group = sub[sub["n_points"] == n_points].copy()
            if group.empty:
                continue
            ax.scatter(
                group["cr"],
                group["ci"],
                s=16,
                alpha=0.65,
                color=color_map[n_points],
                label=f"N={int(n_points)}",
            )
            nearest = group.nsmallest(1, "distance_to_blumen")
            if not nearest.empty:
                ax.scatter(
                    nearest["cr"],
                    nearest["ci"],
                    s=72,
                    marker="x",
                    linewidths=2.0,
                    color=color_map[n_points],
                )

        rect = plt.Rectangle(
            (target["blumen_cr"] - cr_box_tol, target["blumen_ci"] - ci_box_tol),
            2.0 * cr_box_tol,
            2.0 * ci_box_tol,
            facecolor="tab:red",
            edgecolor="tab:red",
            alpha=0.12,
        )
        ax.add_patch(rect)
        ax.scatter(
            [target["blumen_cr"]],
            [target["blumen_ci"]],
            s=100,
            marker="*",
            color="black",
            label="Blumen",
            zorder=5,
        )

        ax.set_title(
            f"alpha={target['alpha']:.3f}, M={mach:.3f}\n"
            f"Blumen=({target['blumen_cr']:.3f},{target['blumen_ci']:.3f})"
        )
        ax.set_xlabel("c_r")
        ax.set_ylabel("c_i")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(mach_values):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    targets_df = build_blumen_targets(args.mach_values, args.alpha, cr_points, ci_points)

    mode_rows: list[dict] = []
    summary_rows: list[dict] = []

    for mach in args.mach_values:
        target = targets_df[targets_df["Mach"] == float(mach)].iloc[0]
        target_guess = (float(target["blumen_cr"]), float(target["blumen_ci"]))
        for n_points in args.n_points_values:
            solver = NotebookStyleDenseGEPSolver(
                alpha=float(args.alpha),
                Mach=float(mach),
                n_points=int(n_points),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
            )
            raw_modes = extract_raw_modes(
                solver,
                max_abs_c=float(args.max_abs_c),
                positive_ci_only=bool(args.positive_ci_only),
            )

            for raw_mode in raw_modes:
                distance = float(np.hypot(raw_mode["cr"] - target_guess[0], raw_mode["ci"] - target_guess[1]))
                mode_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "cr": float(raw_mode["cr"]),
                        "ci": float(raw_mode["ci"]),
                        "omega_i": float(raw_mode["omega_i"]),
                        "abs_c": float(raw_mode["abs_c"]),
                        "blumen_cr": target_guess[0],
                        "blumen_ci": target_guess[1],
                        "err_cr": abs(float(raw_mode["cr"]) - target_guess[0]),
                        "err_ci": abs(float(raw_mode["ci"]) - target_guess[1]),
                        "distance_to_blumen": distance,
                        "inside_blumen_box": bool(
                            abs(float(raw_mode["cr"]) - target_guess[0]) <= float(args.cr_box_tol)
                            and abs(float(raw_mode["ci"]) - target_guess[1]) <= float(args.ci_box_tol)
                        ),
                    }
                )

            if raw_modes:
                sub = pd.DataFrame(
                    [
                        {
                            "cr": row["cr"],
                            "ci": row["ci"],
                            "distance_to_blumen": float(np.hypot(row["cr"] - target_guess[0], row["ci"] - target_guess[1])),
                            "inside_blumen_box": bool(
                                abs(row["cr"] - target_guess[0]) <= float(args.cr_box_tol)
                                and abs(row["ci"] - target_guess[1]) <= float(args.ci_box_tol)
                            ),
                        }
                        for row in raw_modes
                    ]
                )
                nearest = sub.nsmallest(max(int(args.top_k_nearest), 1), "distance_to_blumen")
                best = nearest.iloc[0]
                summary_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "n_raw_modes": int(len(raw_modes)),
                        "n_inside_blumen_box": int(sub["inside_blumen_box"].sum()),
                        "nearest_cr": float(best["cr"]),
                        "nearest_ci": float(best["ci"]),
                        "nearest_distance_to_blumen": float(best["distance_to_blumen"]),
                        "min_err_cr": float((sub["cr"] - target_guess[0]).abs().min()),
                        "min_err_ci": float((sub["ci"] - target_guess[1]).abs().min()),
                        "blumen_cr": target_guess[0],
                        "blumen_ci": target_guess[1],
                    }
                )
            else:
                summary_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "n_raw_modes": 0,
                        "n_inside_blumen_box": 0,
                        "nearest_cr": np.nan,
                        "nearest_ci": np.nan,
                        "nearest_distance_to_blumen": np.nan,
                        "min_err_cr": np.nan,
                        "min_err_ci": np.nan,
                        "blumen_cr": target_guess[0],
                        "blumen_ci": target_guess[1],
                    }
                )

    modes_df = pd.DataFrame(mode_rows).sort_values(["Mach", "n_points", "distance_to_blumen", "cr", "ci"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "n_points"]).reset_index(drop=True)

    nearest_frames = []
    for _, sub in modes_df.groupby(["Mach", "n_points"], sort=True):
        nearest_frames.append(sub.nsmallest(max(int(args.top_k_nearest), 1), "distance_to_blumen"))
    nearest_df = pd.concat(nearest_frames, ignore_index=True) if nearest_frames else pd.DataFrame(columns=modes_df.columns)

    modes_path = OUTPUT_DIR / f"{args.output_stem}_raw_modes.csv"
    nearest_path = OUTPUT_DIR / f"{args.output_stem}_nearest.csv"
    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}.png"

    modes_df.to_csv(modes_path, index=False)
    nearest_df.to_csv(nearest_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_raw_spectrum(
        modes_df,
        targets_df,
        png_path,
        cr_box_tol=float(args.cr_box_tol),
        ci_box_tol=float(args.ci_box_tol),
    )

    print("Blumen targets:")
    print(targets_df.to_string(index=False))
    print("\nRaw spectrum summary:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {modes_path}")
    print(f"Wrote {nearest_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
