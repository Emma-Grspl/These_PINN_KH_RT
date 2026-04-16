from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic de spectre brut local en supersonique.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--n-points-values", type=int, nargs="+", default=[561, 801])
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def solve_shooting(alpha: float, mach: float) -> tuple[float, float, float, bool]:
    shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=10,
    )
    return shooting.cr, shooting.ci, shooting.omega_i, shooting.spectral_success


def plot_local_spectrum(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    mach_values = sorted(summary_df["Mach"].unique())
    ncols = 2
    nrows = int(np.ceil(len(mach_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, mach in zip(axes_flat, mach_values):
        sub = df[df["Mach"] == mach].copy()
        sub_summary = summary_df[summary_df["Mach"] == mach].sort_values("n_points")
        for n_points, group in sub.groupby("n_points"):
            ax.scatter(
                group["cand_cr"],
                group["cand_ci"],
                s=18,
                alpha=0.7,
                label=f"N={int(n_points)}",
            )
            nearest = group.nsmallest(1, "distance_to_shooting")
            if not nearest.empty:
                ax.scatter(
                    nearest["cand_cr"],
                    nearest["cand_ci"],
                    s=80,
                    marker="x",
                    linewidths=2.0,
                )
        if not sub_summary.empty:
            ax.scatter(
                sub_summary["shooting_cr"].iloc[0],
                sub_summary["shooting_ci"].iloc[0],
                s=90,
                marker="*",
                color="black",
                label="shooting",
            )
        ax.set_title(f"alpha={sub_summary['alpha'].iloc[0]:.3f}, M={mach:.3f}")
        ax.set_xlabel("c_r")
        ax.set_ylabel("c_i")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes_flat[len(mach_values) :]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict] = []
    summary_rows: list[dict] = []

    for mach in args.mach_values:
        shooting_cr, shooting_ci, shooting_omega_i, shooting_ok = solve_shooting(args.alpha, mach)
        shooting_guess = (shooting_cr, shooting_ci)
        for n_points in args.n_points_values:
            solver = NotebookStyleDenseGEPSolver(
                alpha=args.alpha,
                Mach=float(mach),
                n_points=int(n_points),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
            )
            modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
            modes_sorted = sorted(
                modes,
                key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight),
            )[: max(int(args.top_k), 1)]

            for rank, mode in enumerate(modes_sorted, start=1):
                mode_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "rank": int(rank),
                        "cand_cr": float(mode["cr"]),
                        "cand_ci": float(mode["ci"]),
                        "cand_omega_i": float(mode["omega_i"]),
                        "distance_to_shooting": float(
                            solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight)
                        ),
                        "shooting_cr": float(shooting_cr),
                        "shooting_ci": float(shooting_ci),
                        "shooting_omega_i": float(shooting_omega_i),
                        "shooting_spectral_success": bool(shooting_ok),
                    }
                )

            if modes_sorted:
                best = modes_sorted[0]
                high_family = [mode for mode in modes_sorted if mode["cr"] > 0.6]
                best_high = min(
                    high_family,
                    key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight),
                ) if high_family else None
                summary_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "n_finite_modes": int(len(modes)),
                        "nearest_cr": float(best["cr"]),
                        "nearest_ci": float(best["ci"]),
                        "nearest_distance_to_shooting": float(
                            solver.spectral_distance(best, shooting_guess, ci_weight=args.ci_weight)
                        ),
                        "best_high_cr": np.nan if best_high is None else float(best_high["cr"]),
                        "best_high_ci": np.nan if best_high is None else float(best_high["ci"]),
                        "best_high_distance_to_shooting": (
                            np.nan
                            if best_high is None
                            else float(solver.spectral_distance(best_high, shooting_guess, ci_weight=args.ci_weight))
                        ),
                        "shooting_cr": float(shooting_cr),
                        "shooting_ci": float(shooting_ci),
                        "shooting_omega_i": float(shooting_omega_i),
                        "shooting_spectral_success": bool(shooting_ok),
                    }
                )
            else:
                summary_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "n_finite_modes": 0,
                        "nearest_cr": np.nan,
                        "nearest_ci": np.nan,
                        "nearest_distance_to_shooting": np.nan,
                        "best_high_cr": np.nan,
                        "best_high_ci": np.nan,
                        "best_high_distance_to_shooting": np.nan,
                        "shooting_cr": float(shooting_cr),
                        "shooting_ci": float(shooting_ci),
                        "shooting_omega_i": float(shooting_omega_i),
                        "shooting_spectral_success": bool(shooting_ok),
                    }
                )

    modes_df = pd.DataFrame(mode_rows).sort_values(["Mach", "n_points", "rank"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "n_points"]).reset_index(drop=True)

    modes_path = OUTPUT_DIR / f"{args.output_stem}_modes.csv"
    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}.png"

    modes_df.to_csv(modes_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_local_spectrum(modes_df, summary_df, png_path)

    print(f"Wrote {modes_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
