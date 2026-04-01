from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare un sweep en alpha entre GEP dense et tir supersonique.")
    parser.add_argument("--mach", type=float, default=1.10)
    parser.add_argument("--alpha-min", type=float, default=0.18)
    parser.add_argument("--alpha-max", type=float, default=0.22)
    parser.add_argument("--num-points", type=int, default=7)
    parser.add_argument("--gep-csv", type=str, default=None, help="CSV GEP existant a reutiliser au lieu de recalculer le sweep.")
    parser.add_argument("--n-points", type=int, default=241)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--cr-window", type=float, default=0.2)
    parser.add_argument("--output-stem", type=str, default="compare_sweep_gep_vs_shooting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.gep_csv is not None:
        gep_df = pd.read_csv(args.gep_csv)
        alpha_values = gep_df["alpha"].to_numpy(dtype=float)
    else:
        gep_df = None
        alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.num_points)
    rows = []
    previous_gep_guess: tuple[float, float] | None = None

    for idx, alpha in enumerate(alpha_values):
        shooting = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(args.mach)).solve(
            cr_min=0.03,
            cr_max=0.35,
            ci_min=0.01,
            ci_max=0.12,
            max_iter=8,
        )

        if gep_df is not None:
            gep_row = gep_df.iloc[idx]
            gep_cr = float(gep_row["cr"])
            gep_ci = float(gep_row["ci"])
            gep_omega_i = float(gep_row.get("omega_i", alpha * gep_ci))
            gep_success = bool(gep_row.get("success", True))
            gep_selection_source = str(gep_row.get("selection_source", "from_csv"))
            target_guess = None
            target_source = "gep_csv"
        else:
            if previous_gep_guess is not None:
                target_guess = previous_gep_guess
                target_source = "gep_continuation"
            elif shooting.spectral_success:
                target_guess = (shooting.cr, shooting.ci)
                target_source = "shooting_anchor"
            else:
                target_guess = None
                target_source = "none"

            gep_solver = NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=float(args.mach),
                n_points=args.n_points,
                mapping_scale=args.mapping_scale,
                xi_max=args.xi_max,
            )
            gep = gep_solver.solve_most_unstable(
                target_guess=target_guess,
                cr_window=args.cr_window,
            )
            gep_cr = gep.cr
            gep_ci = gep.ci
            gep_omega_i = gep.omega_i
            gep_success = gep.success
            gep_selection_source = gep.selection_source
            if gep.success:
                previous_gep_guess = (gep.cr, gep.ci)

        distance = None
        if shooting.spectral_success and gep_success:
            distance = float(((gep_cr - shooting.cr) ** 2 + 4.0 * (gep_ci - shooting.ci) ** 2) ** 0.5)

        rows.append(
            {
                "Mach": float(args.mach),
                "alpha": float(alpha),
                "shooting_cr": shooting.cr,
                "shooting_ci": shooting.ci,
                "shooting_omega_i": shooting.omega_i,
                "shooting_spectral_success": shooting.spectral_success,
                "shooting_mode_success": shooting.mode_success,
                "gep_cr": gep_cr,
                "gep_ci": gep_ci,
                "gep_omega_i": gep_omega_i,
                "gep_selection_source": gep_selection_source,
                "gep_success": gep_success,
                "target_source": target_source,
                "target_cr": None if target_guess is None else target_guess[0],
                "target_ci": None if target_guess is None else target_guess[1],
                "distance_gep_to_shooting": distance,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{args.output_stem}_mach_{args.mach:.3f}.csv"
    fig_path = OUTPUT_DIR / f"{args.output_stem}_mach_{args.mach:.3f}.png"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_ci, ax_cr = axes

    ax_ci.plot(df["alpha"], df["shooting_ci"], "o-", label="Shooting $c_i$")
    ax_ci.plot(df["alpha"], df["gep_ci"], "s--", label="GEP $c_i$")
    ax_ci.set_title(r"Comparison of $c_i$")
    ax_ci.set_xlabel(r"$\alpha$")
    ax_ci.set_ylabel(r"$c_i$")
    ax_ci.grid(True, linestyle="--", alpha=0.35)
    ax_ci.legend()

    ax_cr.plot(df["alpha"], df["shooting_cr"], "o-", label="Shooting $c_r$")
    ax_cr.plot(df["alpha"], df["gep_cr"], "s--", label="GEP $c_r$")
    ax_cr.set_title(r"Comparison of $c_r$")
    ax_cr.set_xlabel(r"$\alpha$")
    ax_cr.set_ylabel(r"$c_r$")
    ax_cr.grid(True, linestyle="--", alpha=0.35)
    ax_cr.legend()

    fig.suptitle(f"Sweep comparison at M={args.mach:.3f}")
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
