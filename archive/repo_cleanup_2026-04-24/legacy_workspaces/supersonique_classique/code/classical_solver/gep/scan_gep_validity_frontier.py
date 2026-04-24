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

from classical_solver.gep.adaptive_continuation_sweep_gep import run_point
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scanne la frontiere de validite du GEP supersonique par rapport au tir.")
    parser.add_argument("--mach-min", type=float, required=True)
    parser.add_argument("--mach-max", type=float, required=True)
    parser.add_argument("--num-mach", type=int, default=4)
    parser.add_argument("--alpha-min", type=float, required=True)
    parser.add_argument("--alpha-max", type=float, required=True)
    parser.add_argument("--num-alpha", type=int, default=5)
    parser.add_argument("--n-values", type=int, nargs="+", default=[401, 481, 561, 641])
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.99)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--previous-weight", type=float, default=0.3)
    parser.add_argument("--output-stem", type=str, default="scan_gep_validity_frontier")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)

    summary_rows: list[dict] = []
    frontier_rows: list[dict] = []

    for mach in machs:
        previous_gep: tuple[float, float] | None = None
        previous_signature = None
        alpha_max_valid = None

        for idx, alpha in enumerate(alphas):
            shooting = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(mach)).solve(
                cr_min=0.03,
                cr_max=0.45,
                ci_min=0.005,
                ci_max=0.12,
                max_iter=8,
            )
            shooting_guess = (shooting.cr, shooting.ci)

            if idx == 0 or previous_gep is None:
                target_guess = shooting_guess
                target_source = "shooting_anchor"
            else:
                target_guess = (
                    args.previous_weight * previous_gep[0] + (1.0 - args.previous_weight) * shooting_guess[0],
                    args.previous_weight * previous_gep[1] + (1.0 - args.previous_weight) * shooting_guess[1],
                )
                target_source = "blended_continuation"

            chosen, chosen_signature, attempts = run_point(
                alpha=float(alpha),
                mach=float(mach),
                target_guess=target_guess,
                shooting_guess=shooting_guess,
                previous_signature=previous_signature,
                n_values=list(args.n_values),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
                distance_tol=args.distance_tol,
                ci_weight=args.ci_weight,
            )

            row = dict(chosen)
            row["target_cr"] = target_guess[0]
            row["target_ci"] = target_guess[1]
            row["target_source"] = target_source
            row["shooting_cr"] = shooting.cr
            row["shooting_ci"] = shooting.ci
            row["shooting_omega_i"] = shooting.omega_i
            row["shooting_spectral_success"] = shooting.spectral_success
            summary_rows.append(row)

            if chosen["success"]:
                previous_gep = (chosen["gep_cr"], chosen["gep_ci"])
                previous_signature = chosen_signature

            if bool(chosen["accepted"]):
                alpha_max_valid = float(alpha)

        frontier_rows.append(
            {
                "Mach": float(mach),
                "alpha_max_valid": alpha_max_valid,
                "distance_tol": args.distance_tol,
                "mapping_kind": args.mapping_kind,
                "mapping_scale": args.mapping_scale,
                "cubic_delta": args.cubic_delta,
                "xi_max": args.xi_max,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    frontier_df = pd.DataFrame(frontier_rows)

    summary_csv = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    frontier_csv = OUTPUT_DIR / f"{args.output_stem}_frontier.csv"
    fig_path = OUTPUT_DIR / f"{args.output_stem}_frontier.png"

    summary_df.to_csv(summary_csv, index=False)
    frontier_df.to_csv(frontier_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = frontier_df.dropna(subset=["alpha_max_valid"])
    if not valid.empty:
        ax.plot(valid["Mach"], valid["alpha_max_valid"], "o-", label=r"$\alpha_{\max,\mathrm{valid}}(M)$")
    ax.set_xlabel("Mach")
    ax.set_ylabel(r"Maximum accepted $\alpha$")
    ax.set_title("GEP validity frontier against shooting")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(frontier_df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Frontier CSV: {frontier_csv}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
