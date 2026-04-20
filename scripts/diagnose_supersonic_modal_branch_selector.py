from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR, plot_isolines  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selection de branche supersonique enrichie par signature modale.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--reference-mach", type=float, default=1.30)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--high-cr-threshold", type=float, default=0.6)
    parser.add_argument("--candidate-top-k", type=int, default=80)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--w-shooting", type=float, default=1.0)
    parser.add_argument("--w-reference-overlap", type=float, default=1.0)
    parser.add_argument("--w-previous-overlap", type=float, default=0.6)
    parser.add_argument("--w-centroid", type=float, default=0.5)
    parser.add_argument("--w-spread", type=float, default=0.3)
    parser.add_argument("--w-phase-span", type=float, default=0.3)
    parser.add_argument("--w-cr-jump", type=float, default=0.2)
    parser.add_argument("--w-ci-jump", type=float, default=0.1)
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


def normalize_pressure_profile(y: np.ndarray, vector: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(vector[2 * n_points : 3 * n_points], dtype=complex).copy()
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
        amp = float(np.max(np.abs(p)))
        if amp > 0.0:
            p = p / amp
    return y.copy(), p


def profile_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    amp = np.abs(p)
    amp_sum = float(np.sum(amp))
    peak_idx = int(np.argmax(amp))
    centroid = float(np.sum(y * amp) / amp_sum) if amp_sum > 0.0 else np.nan
    spread = float(np.sqrt(np.sum(((y - centroid) ** 2) * amp) / amp_sum)) if amp_sum > 0.0 else np.nan
    phase = np.unwrap(np.angle(p))
    return {
        "peak_y": float(y[peak_idx]),
        "peak_amp": float(amp[peak_idx]),
        "centroid_y": centroid,
        "spread_y": spread,
        "phase_span": float(np.max(phase) - np.min(phase)),
    }


def build_candidate_row(
    solver: NotebookStyleDenseGEPSolver,
    mode: dict,
    shooting_guess: tuple[float, float],
    *,
    ci_weight: float,
) -> dict:
    y, p = normalize_pressure_profile(solver.y, mode["vector"], solver.n_points)
    stats = profile_stats(y, p)
    return {
        "mode": mode,
        "cand_cr": float(mode["cr"]),
        "cand_ci": float(mode["ci"]),
        "cand_omega_i": float(mode["omega_i"]),
        "distance_to_shooting": float(solver.spectral_distance(mode, shooting_guess, ci_weight=ci_weight)),
        **stats,
    }


def select_reference_high_mode(
    solver: NotebookStyleDenseGEPSolver,
    shooting_guess: tuple[float, float],
    high_cr_threshold: float,
) -> dict:
    modes = [mode for mode in solver.finite_modes() if mode["cr"] >= high_cr_threshold]
    if not modes:
        raise RuntimeError("Aucune famille haute trouvee au Mach de reference.")
    return min(modes, key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=2.0))


def rerank_candidates(
    candidate_rows: list[dict],
    *,
    solver: NotebookStyleDenseGEPSolver,
    reference_mode: dict,
    reference_stats: dict[str, float],
    previous_mode: dict | None,
    previous_stats: dict[str, float] | None,
    args: argparse.Namespace,
) -> list[dict]:
    if not candidate_rows:
        return []

    def scale(values: list[float]) -> float:
        finite = [abs(v) for v in values if np.isfinite(v)]
        return max(max(finite, default=0.0), 1e-8)

    dist_scale = scale([row["distance_to_shooting"] for row in candidate_rows])
    centroid_scale = scale([row["centroid_y"] - reference_stats["centroid_y"] for row in candidate_rows])
    spread_scale = scale([row["spread_y"] - reference_stats["spread_y"] for row in candidate_rows])
    phase_scale = scale([row["phase_span"] - reference_stats["phase_span"] for row in candidate_rows])
    cr_jump_scale = scale(
        [] if previous_mode is None else [row["cand_cr"] - float(previous_mode["cr"]) for row in candidate_rows]
    )
    ci_jump_scale = scale(
        [] if previous_mode is None else [row["cand_ci"] - float(previous_mode["ci"]) for row in candidate_rows]
    )

    scored: list[dict] = []
    for row in candidate_rows:
        mode = row["mode"]
        overlap_ref = solver.signature_overlap(mode, reference_mode["signature"])
        overlap_prev = np.nan if previous_mode is None else solver.signature_overlap(mode, previous_mode["signature"])
        centroid_term = abs(row["centroid_y"] - reference_stats["centroid_y"]) / centroid_scale
        spread_term = abs(row["spread_y"] - reference_stats["spread_y"]) / spread_scale
        phase_term = abs(row["phase_span"] - reference_stats["phase_span"]) / phase_scale
        cr_jump_term = 0.0 if previous_mode is None else abs(row["cand_cr"] - float(previous_mode["cr"])) / cr_jump_scale
        ci_jump_term = 0.0 if previous_mode is None else abs(row["cand_ci"] - float(previous_mode["ci"])) / ci_jump_scale
        prev_overlap_term = 0.0 if previous_mode is None else (1.0 - float(overlap_prev))

        score = (
            args.w_shooting * (row["distance_to_shooting"] / dist_scale)
            + args.w_reference_overlap * (1.0 - float(overlap_ref))
            + args.w_previous_overlap * prev_overlap_term
            + args.w_centroid * centroid_term
            + args.w_spread * spread_term
            + args.w_phase_span * phase_term
            + args.w_cr_jump * cr_jump_term
            + args.w_ci_jump * ci_jump_term
        )
        scored.append(
            {
                **{k: v for k, v in row.items() if k != "mode"},
                "_mode": mode,
                "overlap_to_reference_high": float(overlap_ref),
                "overlap_to_previous": np.nan if previous_mode is None else float(overlap_prev),
                "score": float(score),
            }
        )

    return sorted(scored, key=lambda item: item["score"])


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mach_values = [float(m) for m in args.mach_values]
    alpha = float(args.alpha)

    ref_shooting_guess = solve_shooting(alpha, float(args.reference_mach))[:2]
    ref_solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=float(args.reference_mach),
        n_points=int(args.n_points),
        mapping_kind=args.mapping_kind,
        mapping_scale=args.mapping_scale,
        cubic_delta=args.cubic_delta,
        xi_max=args.xi_max,
    )
    reference_mode = select_reference_high_mode(
        ref_solver,
        ref_shooting_guess,
        float(args.high_cr_threshold),
    )
    _, ref_p = normalize_pressure_profile(ref_solver.y, reference_mode["vector"], ref_solver.n_points)
    reference_stats = profile_stats(ref_solver.y, ref_p)

    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []
    previous_mode: dict | None = reference_mode
    previous_stats: dict[str, float] | None = reference_stats

    for mach in mach_values:
        shooting_cr, shooting_ci, shooting_omega_i, shooting_ok = solve_shooting(alpha, mach)
        shooting_guess = (shooting_cr, shooting_ci)

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
        if not modes:
            summary_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "N": int(args.n_points),
                    "selection_source": "no_mode",
                    "success": False,
                    "accepted": False,
                    "shooting_cr": shooting_cr,
                    "shooting_ci": shooting_ci,
                    "shooting_omega_i": shooting_omega_i,
                    "shooting_spectral_success": shooting_ok,
                }
            )
            continue

        ranked_by_shooting = sorted(
            modes,
            key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight),
        )[: max(1, int(args.candidate_top_k))]
        candidates = [
            build_candidate_row(solver, mode, shooting_guess, ci_weight=args.ci_weight)
            for mode in ranked_by_shooting
        ]
        reranked = rerank_candidates(
            candidates,
            solver=solver,
            reference_mode=reference_mode,
            reference_stats=reference_stats,
            previous_mode=previous_mode,
            previous_stats=previous_stats,
            args=args,
        )

        chosen = reranked[0]
        summary_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "N": int(args.n_points),
                "gep_cr": chosen["cand_cr"],
                "gep_ci": chosen["cand_ci"],
                "gep_omega_i": chosen["cand_omega_i"],
                "distance_to_shooting": chosen["distance_to_shooting"],
                "overlap_to_reference_high": chosen["overlap_to_reference_high"],
                "overlap_to_previous": chosen["overlap_to_previous"],
                "peak_y": chosen["peak_y"],
                "centroid_y": chosen["centroid_y"],
                "spread_y": chosen["spread_y"],
                "phase_span": chosen["phase_span"],
                "score": chosen["score"],
                "selection_source": "modal_branch_score",
                "success": True,
                "accepted": chosen["distance_to_shooting"] <= args.distance_tol,
                "shooting_cr": shooting_cr,
                "shooting_ci": shooting_ci,
                "shooting_omega_i": shooting_omega_i,
                "shooting_spectral_success": shooting_ok,
                "reference_high_cr": float(reference_mode["cr"]),
                "reference_high_ci": float(reference_mode["ci"]),
                "reference_centroid_y": reference_stats["centroid_y"],
                "reference_spread_y": reference_stats["spread_y"],
                "reference_phase_span": reference_stats["phase_span"],
            }
        )

        for rank, row in enumerate(reranked, start=1):
            candidate_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "rank": rank,
                    **{k: v for k, v in row.items() if k != "_mode"},
                    "shooting_cr": shooting_cr,
                    "shooting_ci": shooting_ci,
                }
            )

        chosen_mode = chosen["_mode"]
        previous_mode = chosen_mode
        previous_stats = {
            "centroid_y": chosen["centroid_y"],
            "spread_y": chosen["spread_y"],
            "phase_span": chosen["phase_span"],
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("Mach").reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows).sort_values(["Mach", "rank"]).reset_index(drop=True)

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
