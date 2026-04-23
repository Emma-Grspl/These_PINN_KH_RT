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
from classical_solver.supersonic.blumen_reference import (  # noqa: E402
    estimate_blumen_ci,
    load_digitized_curves,
)


BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tracking fiable d'une branche supersonique par continuation en Mach avec audit Blumen sur c_i."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--reference-mach", type=float, default=1.30)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--candidate-top-k", type=int, default=40)
    parser.add_argument("--high-cr-threshold", type=float, default=0.60)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--w-prev-overlap", type=float, default=1.80)
    parser.add_argument("--w-ref-overlap", type=float, default=1.20)
    parser.add_argument("--w-prev-cr", type=float, default=0.80)
    parser.add_argument("--w-prev-ci", type=float, default=0.45)
    parser.add_argument("--w-centroid", type=float, default=0.55)
    parser.add_argument("--w-spread", type=float, default=0.35)
    parser.add_argument("--w-phase-span", type=float, default=0.35)
    parser.add_argument("--w-shooting", type=float, default=0.10)
    parser.add_argument("--w-blumen-ci", type=float, default=0.15)
    parser.add_argument("--w-cr-floor", type=float, default=0.25)
    parser.add_argument("--soft-cr-floor", type=float, default=0.55)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def solve_default_shooting(alpha: float, mach: float) -> tuple[float, float, float, bool]:
    shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=10,
    )
    return float(shooting.cr), float(shooting.ci), float(shooting.omega_i), bool(shooting.spectral_success)


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


def load_blumen_ci_curves() -> list[dict]:
    return [curve for curve in load_digitized_curves(BLUMEN_DIR) if curve["family"] == "ci_level"]


def build_candidate_row(
    solver: NotebookStyleDenseGEPSolver,
    mode: dict,
    previous_guess: tuple[float, float],
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
        "distance_to_previous": float(solver.spectral_distance(mode, previous_guess, ci_weight=ci_weight)),
        "distance_to_shooting": float(solver.spectral_distance(mode, shooting_guess, ci_weight=ci_weight)),
        **stats,
    }


def select_reference_mode(
    solver: NotebookStyleDenseGEPSolver,
    shooting_guess: tuple[float, float],
    high_cr_threshold: float,
    ci_weight: float,
) -> dict:
    modes = [mode for mode in solver.finite_modes() if mode["cr"] >= high_cr_threshold]
    if not modes:
        modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
    if not modes:
        raise RuntimeError("Aucun mode fini positif trouve au point d'ancrage.")
    return min(modes, key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=ci_weight))


def rerank_candidates(
    candidate_rows: list[dict],
    *,
    solver: NotebookStyleDenseGEPSolver,
    previous_mode: dict,
    previous_stats: dict[str, float],
    reference_mode: dict,
    blumen_ci: float,
    args: argparse.Namespace,
) -> list[dict]:
    if not candidate_rows:
        return []

    def scale(values: list[float]) -> float:
        finite = [abs(v) for v in values if np.isfinite(v)]
        return max(max(finite, default=0.0), 1e-8)

    prev_dist_scale = scale([row["distance_to_previous"] for row in candidate_rows])
    shooting_scale = scale([row["distance_to_shooting"] for row in candidate_rows])
    centroid_scale = scale([row["centroid_y"] - previous_stats["centroid_y"] for row in candidate_rows])
    spread_scale = scale([row["spread_y"] - previous_stats["spread_y"] for row in candidate_rows])
    phase_scale = scale([row["phase_span"] - previous_stats["phase_span"] for row in candidate_rows])
    cr_scale = scale([row["cand_cr"] - float(previous_mode["cr"]) for row in candidate_rows])
    ci_scale = scale([row["cand_ci"] - float(previous_mode["ci"]) for row in candidate_rows])
    blumen_scale = scale(
        []
        if not np.isfinite(blumen_ci)
        else [row["cand_ci"] - float(blumen_ci) for row in candidate_rows]
    )

    ranked: list[dict] = []
    for row in candidate_rows:
        mode = row["mode"]
        overlap_prev = solver.signature_overlap(mode, previous_mode["signature"])
        overlap_ref = solver.signature_overlap(mode, reference_mode["signature"])
        centroid_term = abs(row["centroid_y"] - previous_stats["centroid_y"]) / centroid_scale
        spread_term = abs(row["spread_y"] - previous_stats["spread_y"]) / spread_scale
        phase_term = abs(row["phase_span"] - previous_stats["phase_span"]) / phase_scale
        cr_jump_term = abs(row["cand_cr"] - float(previous_mode["cr"])) / cr_scale
        ci_jump_term = abs(row["cand_ci"] - float(previous_mode["ci"])) / ci_scale
        shooting_term = row["distance_to_shooting"] / shooting_scale
        blumen_ci_term = 0.0 if not np.isfinite(blumen_ci) else abs(row["cand_ci"] - float(blumen_ci)) / blumen_scale
        cr_floor_term = 0.0
        if np.isfinite(args.soft_cr_floor):
            cr_floor_term = max(float(args.soft_cr_floor) - row["cand_cr"], 0.0)

        score = (
            args.w_prev_overlap * (1.0 - float(overlap_prev))
            + args.w_ref_overlap * (1.0 - float(overlap_ref))
            + args.w_prev_cr * cr_jump_term
            + args.w_prev_ci * ci_jump_term
            + args.w_centroid * centroid_term
            + args.w_spread * spread_term
            + args.w_phase_span * phase_term
            + args.w_shooting * shooting_term
            + args.w_blumen_ci * blumen_ci_term
            + args.w_cr_floor * cr_floor_term
        )
        ranked.append(
            {
                **{k: v for k, v in row.items() if k != "mode"},
                "_mode": mode,
                "overlap_to_previous": float(overlap_prev),
                "overlap_to_reference": float(overlap_ref),
                "blumen_ci_target": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
                "blumen_ci_abs_err": abs(row["cand_ci"] - float(blumen_ci)) if np.isfinite(blumen_ci) else np.nan,
                "cr_floor_term": float(cr_floor_term),
                "score": float(score),
            }
        )

    return sorted(ranked, key=lambda item: item["score"])


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    mach_values = [float(m) for m in args.mach_values]
    if float(args.reference_mach) not in mach_values:
        mach_values = [float(args.reference_mach), *mach_values]
    mach_values = sorted(set(mach_values), reverse=True)

    blumen_curves = load_blumen_ci_curves()
    ref_shooting_guess = solve_default_shooting(alpha, float(args.reference_mach))[:2]
    ref_solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=float(args.reference_mach),
        n_points=int(args.n_points),
        mapping_kind=args.mapping_kind,
        mapping_scale=args.mapping_scale,
        cubic_delta=args.cubic_delta,
        xi_max=args.xi_max,
    )
    reference_mode = select_reference_mode(
        ref_solver,
        ref_shooting_guess,
        float(args.high_cr_threshold),
        float(args.ci_weight),
    )
    _, ref_p = normalize_pressure_profile(ref_solver.y, reference_mode["vector"], ref_solver.n_points)
    reference_stats = profile_stats(ref_solver.y, ref_p)

    previous_mode = reference_mode
    previous_stats = reference_stats
    previous_guess = (float(reference_mode["cr"]), float(reference_mode["ci"]))

    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for mach in mach_values:
        shooting_cr, shooting_ci, shooting_omega_i, shooting_ok = solve_default_shooting(alpha, mach)
        shooting_guess = (shooting_cr, shooting_ci)
        blumen_ci = estimate_blumen_ci(alpha, mach, blumen_curves)

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
                    "blumen_ci_target": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
                }
            )
            continue

        nearest = sorted(
            modes,
            key=lambda mode: solver.spectral_distance(mode, previous_guess, ci_weight=args.ci_weight),
        )[: max(1, int(args.candidate_top_k))]
        candidates = [
            build_candidate_row(
                solver,
                mode,
                previous_guess,
                shooting_guess,
                ci_weight=float(args.ci_weight),
            )
            for mode in nearest
        ]

        reranked = rerank_candidates(
            candidates,
            solver=solver,
            previous_mode=previous_mode,
            previous_stats=previous_stats,
            reference_mode=reference_mode,
            blumen_ci=blumen_ci,
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
                "distance_to_previous": chosen["distance_to_previous"],
                "distance_to_shooting": chosen["distance_to_shooting"],
                "overlap_to_previous": chosen["overlap_to_previous"],
                "overlap_to_reference": chosen["overlap_to_reference"],
                "peak_y": chosen["peak_y"],
                "centroid_y": chosen["centroid_y"],
                "spread_y": chosen["spread_y"],
                "phase_span": chosen["phase_span"],
                "blumen_ci_target": chosen["blumen_ci_target"],
                "blumen_ci_abs_err": chosen["blumen_ci_abs_err"],
                "cr_floor_term": chosen["cr_floor_term"],
                "score": chosen["score"],
                "selection_source": "continuation_score",
                "success": True,
                "accepted": chosen["distance_to_shooting"] <= args.distance_tol,
                "shooting_cr": shooting_cr,
                "shooting_ci": shooting_ci,
                "shooting_omega_i": shooting_omega_i,
                "shooting_spectral_success": shooting_ok,
                "reference_cr": float(reference_mode["cr"]),
                "reference_ci": float(reference_mode["ci"]),
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
                    "shooting_omega_i": shooting_omega_i,
                    "shooting_spectral_success": shooting_ok,
                }
            )

        chosen_mode = chosen["_mode"]
        previous_mode = chosen_mode
        previous_guess = (float(chosen["cand_cr"]), float(chosen["cand_ci"]))
        previous_stats = {
            "centroid_y": float(chosen["centroid_y"]),
            "spread_y": float(chosen["spread_y"]),
            "phase_span": float(chosen["phase_span"]),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("Mach", ascending=False).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows).sort_values(["Mach", "rank"], ascending=[False, True]).reset_index(
        drop=True
    )

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
