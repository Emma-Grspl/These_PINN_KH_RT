from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR, plot_isolines
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic de tracking de branche Mach supersonique.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-anchor", type=float, required=True)
    parser.add_argument("--mach-min", type=float, required=True)
    parser.add_argument("--mach-max", type=float, required=True)
    parser.add_argument("--mach-step", type=float, default=0.025)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--previous-weight", type=float, default=0.6)
    parser.add_argument("--branch-overlap-weight", type=float, default=0.35)
    parser.add_argument("--branch-anchor-weight", type=float, default=0.60)
    parser.add_argument("--branch-jump-cr-weight", type=float, default=0.45)
    parser.add_argument("--branch-jump-ci-weight", type=float, default=0.20)
    parser.add_argument("--branch-overlap-top-k", type=int, default=5)
    parser.add_argument("--critical-mach-min", type=float, default=1.20)
    parser.add_argument("--critical-mach-max", type=float, default=1.35)
    parser.add_argument("--critical-top-k", type=int, default=6)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def make_mach_sequences(anchor: float, mach_min: float, mach_max: float, mach_step: float) -> tuple[np.ndarray, np.ndarray]:
    down = np.arange(anchor, mach_min - 0.5 * mach_step, -mach_step, dtype=float)
    up = np.arange(anchor, mach_max + 0.5 * mach_step, mach_step, dtype=float)
    return down, up


def select_branch_mode_with_anchor(
    solver: NotebookStyleDenseGEPSolver,
    *,
    target_guess: tuple[float, float],
    previous_guess: tuple[float, float] | None,
    previous_signature: np.ndarray | None,
    anchor_signature: np.ndarray,
    ci_weight: float,
    overlap_top_k: int,
    overlap_weight: float,
    anchor_weight: float,
    jump_cr_weight: float,
    jump_ci_weight: float,
) -> tuple[dict | None, str, int, list[dict]]:
    modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
    if not modes:
        return None, "no_mode", 0, []

    distances = np.array([solver.spectral_distance(mode, target_guess, ci_weight=ci_weight) for mode in modes], dtype=float)
    ranked = np.argsort(distances)
    candidates = [modes[idx] for idx in ranked[: max(1, overlap_top_k)]]

    distance_scale = max(float(np.max([solver.spectral_distance(mode, target_guess, ci_weight=ci_weight) for mode in candidates])), 1e-8)
    if previous_guess is None:
        jump_cr_scale = 1.0
        jump_ci_scale = 1.0
    else:
        jump_cr_scale = max(float(np.max([abs(mode["cr"] - previous_guess[0]) for mode in candidates])), 1e-8)
        jump_ci_scale = max(float(np.max([abs(mode["ci"] - previous_guess[1]) for mode in candidates])), 1e-8)

    scored: list[dict] = []
    for mode in candidates:
        dist = solver.spectral_distance(mode, target_guess, ci_weight=ci_weight)
        overlap_prev = 0.0 if previous_signature is None else solver.signature_overlap(mode, previous_signature)
        overlap_anchor = solver.signature_overlap(mode, anchor_signature)
        jump_cr_term = 0.0 if previous_guess is None else abs(mode["cr"] - previous_guess[0]) / jump_cr_scale
        jump_ci_term = 0.0 if previous_guess is None else abs(mode["ci"] - previous_guess[1]) / jump_ci_scale
        score = (
            dist / distance_scale
            + overlap_weight * (1.0 - overlap_prev)
            + anchor_weight * (1.0 - overlap_anchor)
            + jump_cr_weight * jump_cr_term
            + jump_ci_weight * jump_ci_term
        )
        scored.append(
            {
                **mode,
                "score": float(score),
                "distance_to_target": float(dist),
                "overlap_to_previous": float(overlap_prev),
                "overlap_to_anchor": float(overlap_anchor),
            }
        )

    chosen = min(scored, key=lambda row: row["score"])
    return chosen, "anchor_tracking_score", len(modes), scored


def solve_shooting(alpha: float, mach: float) -> tuple[float, float, float, bool]:
    shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=10,
    )
    return shooting.cr, shooting.ci, shooting.omega_i, shooting.spectral_success


def run_sequence(
    mach_values: np.ndarray,
    *,
    alpha: float,
    args: argparse.Namespace,
    anchor_signature: np.ndarray,
    initial_guess: tuple[float, float],
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []
    previous_guess: tuple[float, float] | None = initial_guess
    previous_signature: np.ndarray | None = anchor_signature

    for idx, mach in enumerate(mach_values):
        shooting_cr, shooting_ci, shooting_omega_i, shooting_ok = solve_shooting(alpha, float(mach))
        shooting_guess = (shooting_cr, shooting_ci)
        if idx == 0 and previous_guess is None:
            target_guess = shooting_guess
            target_source = "shooting_anchor"
        elif previous_guess is None:
            target_guess = shooting_guess
            target_source = "shooting_restart"
        else:
            target_guess = (
                args.previous_weight * previous_guess[0] + (1.0 - args.previous_weight) * shooting_guess[0],
                args.previous_weight * previous_guess[1] + (1.0 - args.previous_weight) * shooting_guess[1],
            )
            target_source = "blended_continuation"

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=float(mach),
            n_points=args.n_points,
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        chosen, selection_source, n_modes, scored_candidates = select_branch_mode_with_anchor(
            solver,
            target_guess=target_guess,
            previous_guess=previous_guess,
            previous_signature=previous_signature,
            anchor_signature=anchor_signature,
            ci_weight=args.ci_weight,
            overlap_top_k=args.branch_overlap_top_k,
            overlap_weight=args.branch_overlap_weight,
            anchor_weight=args.branch_anchor_weight,
            jump_cr_weight=args.branch_jump_cr_weight,
            jump_ci_weight=args.branch_jump_ci_weight,
        )

        if chosen is None:
            summary_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "N": np.nan,
                    "gep_cr": np.nan,
                    "gep_ci": np.nan,
                    "gep_omega_i": np.nan,
                    "distance_to_target": np.nan,
                    "distance_to_shooting": np.nan,
                    "overlap_to_previous": np.nan,
                    "overlap_to_anchor": np.nan,
                    "selection_source": selection_source,
                    "n_finite_modes": 0,
                    "success": False,
                    "accepted": False,
                    "target_cr": target_guess[0],
                    "target_ci": target_guess[1],
                    "target_source": target_source,
                    "shooting_cr": shooting_cr,
                    "shooting_ci": shooting_ci,
                    "shooting_omega_i": shooting_omega_i,
                    "shooting_spectral_success": shooting_ok,
                }
            )
            continue

        dist_shooting = solver.spectral_distance(chosen, shooting_guess, ci_weight=args.ci_weight)
        summary_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "N": args.n_points,
                "gep_cr": chosen["cr"],
                "gep_ci": chosen["ci"],
                "gep_omega_i": chosen["omega_i"],
                "distance_to_target": chosen["distance_to_target"],
                "distance_to_shooting": dist_shooting,
                "overlap_to_previous": chosen["overlap_to_previous"],
                "overlap_to_anchor": chosen["overlap_to_anchor"],
                "selection_source": selection_source,
                "n_finite_modes": n_modes,
                "success": True,
                "accepted": dist_shooting <= args.distance_tol,
                "target_cr": target_guess[0],
                "target_ci": target_guess[1],
                "target_source": target_source,
                "shooting_cr": shooting_cr,
                "shooting_ci": shooting_ci,
                "shooting_omega_i": shooting_omega_i,
                "shooting_spectral_success": shooting_ok,
            }
        )

        if args.critical_mach_min <= mach <= args.critical_mach_max:
            scored_candidates = sorted(scored_candidates, key=lambda row: row["score"])[: args.critical_top_k]
            for rank, cand in enumerate(scored_candidates, start=1):
                candidate_rows.append(
                    {
                        "alpha": alpha,
                        "Mach": mach,
                        "rank": rank,
                        "cand_cr": cand["cr"],
                        "cand_ci": cand["ci"],
                        "cand_omega_i": cand["omega_i"],
                        "distance_to_target": cand["distance_to_target"],
                        "distance_to_shooting": solver.spectral_distance(cand, shooting_guess, ci_weight=args.ci_weight),
                        "overlap_to_previous": cand["overlap_to_previous"],
                        "overlap_to_anchor": cand["overlap_to_anchor"],
                        "score": cand["score"],
                        "target_cr": target_guess[0],
                        "target_ci": target_guess[1],
                        "shooting_cr": shooting_cr,
                        "shooting_ci": shooting_ci,
                    }
                )

        previous_guess = (chosen["cr"], chosen["ci"])
        previous_signature = chosen["signature"]

    return summary_rows, candidate_rows


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    mach_anchor = float(args.mach_anchor)

    anchor_cr, anchor_ci, anchor_omega_i, anchor_ok = solve_shooting(alpha, mach_anchor)
    anchor_guess = (anchor_cr, anchor_ci)
    anchor_solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach_anchor,
        n_points=args.n_points,
        mapping_kind=args.mapping_kind,
        mapping_scale=args.mapping_scale,
        cubic_delta=args.cubic_delta,
        xi_max=args.xi_max,
    )
    anchor_mode, anchor_source, n_anchor_modes = anchor_solver.get_branch_mode(
        target_guess=anchor_guess,
        previous_guess=None,
        previous_signature=None,
        prefer_positive_cr=True,
        ci_weight=args.ci_weight,
        overlap_top_k=args.branch_overlap_top_k,
        overlap_weight=args.branch_overlap_weight,
        jump_cr_weight=args.branch_jump_cr_weight,
        jump_ci_weight=args.branch_jump_ci_weight,
    )
    if anchor_mode is None:
        raise RuntimeError("Aucune branche d'ancrage trouvee au Mach d'ancrage.")

    anchor_signature = anchor_mode["signature"]

    down, up = make_mach_sequences(mach_anchor, float(args.mach_min), float(args.mach_max), float(args.mach_step))
    down_rows, down_candidates = run_sequence(
        down,
        alpha=alpha,
        args=args,
        anchor_signature=anchor_signature,
        initial_guess=(anchor_mode["cr"], anchor_mode["ci"]),
    )
    up_rows, up_candidates = run_sequence(
        up[1:],
        alpha=alpha,
        args=args,
        anchor_signature=anchor_signature,
        initial_guess=(anchor_mode["cr"], anchor_mode["ci"]),
    )

    anchor_row = {
        "alpha": alpha,
        "Mach": mach_anchor,
        "N": args.n_points,
        "gep_cr": anchor_mode["cr"],
        "gep_ci": anchor_mode["ci"],
        "gep_omega_i": anchor_mode["omega_i"],
        "distance_to_target": anchor_solver.spectral_distance(anchor_mode, anchor_guess, ci_weight=args.ci_weight),
        "distance_to_shooting": anchor_solver.spectral_distance(anchor_mode, anchor_guess, ci_weight=args.ci_weight),
        "overlap_to_previous": 1.0,
        "overlap_to_anchor": 1.0,
        "selection_source": anchor_source,
        "n_finite_modes": n_anchor_modes,
        "success": True,
        "accepted": anchor_solver.spectral_distance(anchor_mode, anchor_guess, ci_weight=args.ci_weight) <= args.distance_tol,
        "target_cr": anchor_guess[0],
        "target_ci": anchor_guess[1],
        "target_source": "anchor_shooting",
        "shooting_cr": anchor_cr,
        "shooting_ci": anchor_ci,
        "shooting_omega_i": anchor_omega_i,
        "shooting_spectral_success": anchor_ok,
    }

    summary_df = pd.DataFrame(down_rows + [anchor_row] + up_rows).sort_values("Mach").drop_duplicates(subset=["Mach", "alpha"], keep="last").reset_index(drop=True)
    candidates_df = pd.DataFrame(down_candidates + up_candidates).sort_values(["Mach", "rank"]).reset_index(drop=True)

    surface_csv = OUTPUT_DIR / f"{args.output_stem}_surface.csv"
    candidates_csv = OUTPUT_DIR / f"{args.output_stem}_candidates.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}_isolines.png"

    summary_df.to_csv(surface_csv, index=False)
    candidates_df.to_csv(candidates_csv, index=False)
    plot_isolines(summary_df, png_path)

    print(summary_df.to_string(index=False))
    print(f"\nSurface CSV: {surface_csv}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Isoline figure: {png_path}")


if __name__ == "__main__":
    main()
