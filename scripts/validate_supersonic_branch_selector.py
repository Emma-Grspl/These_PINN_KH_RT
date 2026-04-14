from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rejoue les points critiques avec le nouveau score de branche.")
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--branch-overlap-weight", type=float, default=0.35)
    parser.add_argument("--branch-jump-cr-weight", type=float, default=0.45)
    parser.add_argument("--branch-jump-ci-weight", type=float, default=0.20)
    parser.add_argument("--branch-overlap-top-k", type=int, default=5)
    return parser


def row_key(row: pd.Series) -> tuple[float, float]:
    return round(float(row["Mach"]), 12), round(float(row["alpha"]), 12)


def main() -> None:
    args = build_parser().parse_args()
    surface_df = pd.read_csv(args.surface_csv).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    diagnostics_df = pd.read_csv(args.diagnostics_csv)

    selected = (
        diagnostics_df[["Mach", "alpha"]]
        .drop_duplicates()
        .head(args.top_k)
        .merge(surface_df, on=["Mach", "alpha"], how="left")
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )

    previous_map = {row_key(row): row for _, row in surface_df.iterrows()}
    rows: list[dict] = []

    for _, row in selected.iterrows():
        mach = float(row["Mach"])
        alpha = float(row["alpha"])

        prev_candidates = surface_df[
            (surface_df["Mach"] == mach) & (surface_df["alpha"] < alpha)
        ].sort_values("alpha")
        previous_guess = None
        previous_signature = None
        if not prev_candidates.empty:
            prev_row = prev_candidates.iloc[-1]
            previous_guess = (float(prev_row["gep_cr"]), float(prev_row["gep_ci"]))
            prev_solver = NotebookStyleDenseGEPSolver(
                alpha=float(prev_row["alpha"]),
                Mach=mach,
                n_points=args.n_points,
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
            )
            prev_mode, _, _ = prev_solver.get_branch_mode(
                target_guess=(float(prev_row["target_cr"]), float(prev_row["target_ci"])),
                previous_guess=None,
                previous_signature=None,
                prefer_positive_cr=True,
                ci_weight=args.ci_weight,
            )
            if prev_mode is not None:
                previous_signature = prev_mode.get("signature")

        shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
            cr_min=0.03,
            cr_max=min(0.7, max(0.35, 0.5 * mach)),
            ci_min=0.001,
            ci_max=0.12,
            max_iter=10,
        )
        shooting_guess = (shooting.cr, shooting.ci)

        if previous_guess is None:
            target_guess = shooting_guess
        else:
            target_guess = (float(row["target_cr"]), float(row["target_ci"]))

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=args.n_points,
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        baseline_mode, baseline_source, _ = solver.get_branch_mode(
            target_guess=target_guess,
            previous_guess=None,
            previous_signature=previous_signature,
            prefer_positive_cr=True,
            ci_weight=args.ci_weight,
        )
        composite_mode, composite_source, _ = solver.get_branch_mode(
            target_guess=target_guess,
            previous_guess=previous_guess,
            previous_signature=previous_signature,
            prefer_positive_cr=True,
            ci_weight=args.ci_weight,
            overlap_top_k=args.branch_overlap_top_k,
            overlap_weight=args.branch_overlap_weight,
            jump_cr_weight=args.branch_jump_cr_weight,
            jump_ci_weight=args.branch_jump_ci_weight,
        )

        out = {
            "Mach": mach,
            "alpha": alpha,
            "old_gep_cr": float(row["gep_cr"]),
            "old_gep_ci": float(row["gep_ci"]),
            "old_selection_source": str(row.get("selection_source", "")),
            "shooting_cr": shooting.cr,
            "shooting_ci": shooting.ci,
            "target_cr": target_guess[0],
            "target_ci": target_guess[1],
        }

        if baseline_mode is not None:
            out["baseline_cr"] = baseline_mode["cr"]
            out["baseline_ci"] = baseline_mode["ci"]
            out["baseline_distance_to_shooting"] = solver.spectral_distance(
                baseline_mode,
                shooting_guess,
                ci_weight=args.ci_weight,
            )
            out["baseline_source"] = baseline_source
        else:
            out["baseline_cr"] = np.nan
            out["baseline_ci"] = np.nan
            out["baseline_distance_to_shooting"] = np.nan
            out["baseline_source"] = baseline_source

        if composite_mode is not None:
            out["composite_cr"] = composite_mode["cr"]
            out["composite_ci"] = composite_mode["ci"]
            out["composite_distance_to_shooting"] = solver.spectral_distance(
                composite_mode,
                shooting_guess,
                ci_weight=args.ci_weight,
            )
            out["composite_source"] = composite_source
        else:
            out["composite_cr"] = np.nan
            out["composite_ci"] = np.nan
            out["composite_distance_to_shooting"] = np.nan
            out["composite_source"] = composite_source

        rows.append(out)

    output_df = pd.DataFrame(rows)
    output_df["delta_dist_vs_old"] = output_df["composite_distance_to_shooting"] - np.hypot(
        output_df["old_gep_cr"] - output_df["shooting_cr"],
        args.ci_weight * (output_df["old_gep_ci"] - output_df["shooting_ci"]),
    )
    output_df["delta_dist_vs_baseline"] = (
        output_df["composite_distance_to_shooting"] - output_df["baseline_distance_to_shooting"]
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
