from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class PathState:
    path_id: int
    total_score: float
    modes: list[dict]
    rows: list[dict]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tracking supersonique multi-hypotheses par continuation en Mach avec beam search."
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
    parser.add_argument("--high-cr-threshold", type=float, default=0.60)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--branch-top-k", type=int, default=4)
    parser.add_argument("--candidate-top-k", type=int, default=40)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--w-prev-overlap", type=float, default=1.30)
    parser.add_argument("--w-ref-overlap", type=float, default=0.90)
    parser.add_argument("--w-prev-cr", type=float, default=0.70)
    parser.add_argument("--w-prev-ci", type=float, default=0.40)
    parser.add_argument("--w-centroid", type=float, default=0.40)
    parser.add_argument("--w-spread", type=float, default=0.25)
    parser.add_argument("--w-phase-span", type=float, default=0.25)
    parser.add_argument("--w-shooting", type=float, default=0.20)
    parser.add_argument("--w-blumen-ci", type=float, default=0.20)
    parser.add_argument("--w-cr-floor", type=float, default=0.20)
    parser.add_argument("--w-low-ci", type=float, default=1.20)
    parser.add_argument("--soft-cr-floor", type=float, default=0.55)
    parser.add_argument("--neutral-ci-threshold", type=float, default=0.0075)
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


def candidate_pool(
    solver: NotebookStyleDenseGEPSolver,
    previous_guess: tuple[float, float],
    *,
    ci_weight: float,
    candidate_top_k: int,
) -> list[dict]:
    modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
    if not modes:
        return []
    modes = sorted(
        modes,
        key=lambda mode: solver.spectral_distance(mode, previous_guess, ci_weight=ci_weight),
    )[: max(1, candidate_top_k)]
    rows: list[dict] = []
    for mode in modes:
        y, p = normalize_pressure_profile(solver.y, mode["vector"], solver.n_points)
        rows.append(
            {
                "mode": mode,
                "cand_cr": float(mode["cr"]),
                "cand_ci": float(mode["ci"]),
                "cand_omega_i": float(mode["omega_i"]),
                **profile_stats(y, p),
            }
        )
    return rows


def score_candidate(
    *,
    row: dict,
    solver: NotebookStyleDenseGEPSolver,
    previous_mode: dict,
    previous_stats: dict[str, float],
    reference_mode: dict,
    shooting_guess: tuple[float, float],
    blumen_ci: float,
    args: argparse.Namespace,
    scale_terms: dict[str, float],
) -> dict:
    mode = row["mode"]
    overlap_prev = solver.signature_overlap(mode, previous_mode["signature"])
    overlap_ref = solver.signature_overlap(mode, reference_mode["signature"])
    distance_to_previous = float(
        solver.spectral_distance(mode, (float(previous_mode["cr"]), float(previous_mode["ci"])), ci_weight=args.ci_weight)
    )
    distance_to_shooting = float(solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight))
    centroid_term = abs(row["centroid_y"] - previous_stats["centroid_y"]) / scale_terms["centroid"]
    spread_term = abs(row["spread_y"] - previous_stats["spread_y"]) / scale_terms["spread"]
    phase_term = abs(row["phase_span"] - previous_stats["phase_span"]) / scale_terms["phase"]
    cr_jump_term = abs(row["cand_cr"] - float(previous_mode["cr"])) / scale_terms["cr_jump"]
    ci_jump_term = abs(row["cand_ci"] - float(previous_mode["ci"])) / scale_terms["ci_jump"]
    shooting_term = distance_to_shooting / scale_terms["shooting"]
    blumen_ci_term = 0.0 if not np.isfinite(blumen_ci) else abs(row["cand_ci"] - float(blumen_ci)) / scale_terms["blumen"]
    cr_floor_term = max(float(args.soft_cr_floor) - row["cand_cr"], 0.0) if np.isfinite(args.soft_cr_floor) else 0.0
    low_ci_term = max(float(args.neutral_ci_threshold) - row["cand_ci"], 0.0) / max(float(args.neutral_ci_threshold), 1e-8)

    local_score = (
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
        + args.w_low_ci * low_ci_term
    )

    return {
        **{k: v for k, v in row.items() if k != "mode"},
        "_mode": mode,
        "distance_to_previous": distance_to_previous,
        "distance_to_shooting": distance_to_shooting,
        "overlap_to_previous": float(overlap_prev),
        "overlap_to_reference": float(overlap_ref),
        "blumen_ci_target": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
        "blumen_ci_abs_err": abs(row["cand_ci"] - float(blumen_ci)) if np.isfinite(blumen_ci) else np.nan,
        "cr_floor_term": float(cr_floor_term),
        "low_ci_term": float(low_ci_term),
        "local_score": float(local_score),
    }


def make_scale_terms(rows: list[dict], previous_mode: dict, previous_stats: dict[str, float], blumen_ci: float) -> dict[str, float]:
    def scale(values: list[float]) -> float:
        finite = [abs(v) for v in values if np.isfinite(v)]
        return max(max(finite, default=0.0), 1e-8)

    return {
        "centroid": scale([row["centroid_y"] - previous_stats["centroid_y"] for row in rows]),
        "spread": scale([row["spread_y"] - previous_stats["spread_y"] for row in rows]),
        "phase": scale([row["phase_span"] - previous_stats["phase_span"] for row in rows]),
        "cr_jump": scale([row["cand_cr"] - float(previous_mode["cr"]) for row in rows]),
        "ci_jump": scale([row["cand_ci"] - float(previous_mode["ci"]) for row in rows]),
        "shooting": 1.0,
        "blumen": scale([] if not np.isfinite(blumen_ci) else [row["cand_ci"] - float(blumen_ci) for row in rows]),
    }


def deduplicate_paths(paths: list[PathState]) -> list[PathState]:
    best_by_signature: dict[tuple[int, ...], PathState] = {}
    for path in paths:
        key = tuple(int(round(1e4 * row["gep_cr"])) for row in path.rows[1:])
        existing = best_by_signature.get(key)
        if existing is None or path.total_score < existing.total_score:
            best_by_signature[key] = path
    return list(best_by_signature.values())


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
    reference_mode = select_reference_mode(ref_solver, ref_shooting_guess, float(args.high_cr_threshold), float(args.ci_weight))
    _, ref_p = normalize_pressure_profile(ref_solver.y, reference_mode["vector"], ref_solver.n_points)
    reference_stats = profile_stats(ref_solver.y, ref_p)

    ref_row = {
        "alpha": alpha,
        "Mach": float(args.reference_mach),
        "N": int(args.n_points),
        "gep_cr": float(reference_mode["cr"]),
        "gep_ci": float(reference_mode["ci"]),
        "gep_omega_i": float(reference_mode["omega_i"]),
        "distance_to_previous": 0.0,
        "distance_to_shooting": float(ref_solver.spectral_distance(reference_mode, ref_shooting_guess, ci_weight=args.ci_weight)),
        "overlap_to_previous": 1.0,
        "overlap_to_reference": 1.0,
        **reference_stats,
        "blumen_ci_target": estimate_blumen_ci(alpha, float(args.reference_mach), blumen_curves),
        "blumen_ci_abs_err": np.nan,
        "cr_floor_term": 0.0,
        "low_ci_term": max(float(args.neutral_ci_threshold) - float(reference_mode["ci"]), 0.0)
        / max(float(args.neutral_ci_threshold), 1e-8),
        "score": 0.0,
        "selection_source": "reference_anchor",
        "success": True,
        "accepted": True,
        "shooting_cr": ref_shooting_guess[0],
        "shooting_ci": ref_shooting_guess[1],
        "shooting_omega_i": solve_default_shooting(alpha, float(args.reference_mach))[2],
        "shooting_spectral_success": solve_default_shooting(alpha, float(args.reference_mach))[3],
        "reference_cr": float(reference_mode["cr"]),
        "reference_ci": float(reference_mode["ci"]),
    }

    active_paths = [PathState(path_id=0, total_score=0.0, modes=[reference_mode], rows=[ref_row])]
    candidate_rows: list[dict] = []
    next_path_id = 1

    for mach in [m for m in mach_values if m != float(args.reference_mach)]:
        expanded_paths: list[PathState] = []
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

        for beam_rank, path in enumerate(active_paths, start=1):
            previous_mode = path.modes[-1]
            previous_row = path.rows[-1]
            previous_stats = {
                "centroid_y": float(previous_row["centroid_y"]),
                "spread_y": float(previous_row["spread_y"]),
                "phase_span": float(previous_row["phase_span"]),
            }
            rows = candidate_pool(
                solver,
                (float(previous_mode["cr"]), float(previous_mode["ci"])),
                ci_weight=float(args.ci_weight),
                candidate_top_k=int(args.candidate_top_k),
            )
            if not rows:
                continue
            scale_terms = make_scale_terms(rows, previous_mode, previous_stats, blumen_ci)
            scale_terms["shooting"] = max(
                max(
                    solver.spectral_distance(row["mode"], shooting_guess, ci_weight=args.ci_weight)
                    for row in rows
                ),
                1e-8,
            )
            scored = [
                score_candidate(
                    row=row,
                    solver=solver,
                    previous_mode=previous_mode,
                    previous_stats=previous_stats,
                    reference_mode=reference_mode,
                    shooting_guess=shooting_guess,
                    blumen_ci=blumen_ci,
                    args=args,
                    scale_terms=scale_terms,
                )
                for row in rows
            ]
            scored = sorted(scored, key=lambda item: item["local_score"])[: max(1, int(args.branch_top_k))]

            for local_rank, chosen in enumerate(scored, start=1):
                candidate_rows.append(
                    {
                        "alpha": alpha,
                        "Mach": mach,
                        "beam_parent_rank": beam_rank,
                        "beam_parent_id": path.path_id,
                        "local_rank": local_rank,
                        **{k: v for k, v in chosen.items() if k != "_mode"},
                        "shooting_cr": shooting_cr,
                        "shooting_ci": shooting_ci,
                        "shooting_omega_i": shooting_omega_i,
                        "shooting_spectral_success": shooting_ok,
                    }
                )
                new_row = {
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
                    "low_ci_term": chosen["low_ci_term"],
                    "score": chosen["local_score"],
                    "selection_source": "beam_search",
                    "success": True,
                    "accepted": chosen["distance_to_shooting"] <= args.distance_tol,
                    "shooting_cr": shooting_cr,
                    "shooting_ci": shooting_ci,
                    "shooting_omega_i": shooting_omega_i,
                    "shooting_spectral_success": shooting_ok,
                    "reference_cr": float(reference_mode["cr"]),
                    "reference_ci": float(reference_mode["ci"]),
                }
                expanded_paths.append(
                    PathState(
                        path_id=next_path_id,
                        total_score=path.total_score + float(chosen["local_score"]),
                        modes=[*path.modes, chosen["_mode"]],
                        rows=[*path.rows, new_row],
                    )
                )
                next_path_id += 1

        if not expanded_paths:
            raise RuntimeError(f"Aucune trajectoire candidate n'a survecu au Mach {mach:.3f}.")
        expanded_paths = deduplicate_paths(expanded_paths)
        expanded_paths = sorted(expanded_paths, key=lambda item: item.total_score)[: max(1, int(args.beam_width))]
        active_paths = expanded_paths

    best_path = min(active_paths, key=lambda item: item.total_score)
    surface_df = pd.DataFrame(best_path.rows).sort_values("Mach", ascending=False).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["Mach", "beam_parent_rank", "local_rank"], ascending=[False, True, True]
    ).reset_index(drop=True)

    path_rows: list[dict] = []
    for final_rank, path in enumerate(sorted(active_paths, key=lambda item: item.total_score), start=1):
        for row in path.rows:
            path_rows.append(
                {
                    "final_rank": final_rank,
                    "path_id": path.path_id,
                    "total_score": path.total_score,
                    **row,
                }
            )
    paths_df = pd.DataFrame(path_rows).sort_values(["final_rank", "Mach"], ascending=[True, False]).reset_index(drop=True)

    surface_csv = OUTPUT_DIR / f"{args.output_stem}_surface.csv"
    candidates_csv = OUTPUT_DIR / f"{args.output_stem}_candidates.csv"
    paths_csv = OUTPUT_DIR / f"{args.output_stem}_paths.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}_isolines.png"

    surface_df.to_csv(surface_csv, index=False)
    candidates_df.to_csv(candidates_csv, index=False)
    paths_df.to_csv(paths_csv, index=False)
    plot_isolines(surface_df, png_path)

    print(surface_df.to_string(index=False))
    print("\nTop trajectories:")
    print(
        paths_df[
            [
                "final_rank",
                "Mach",
                "gep_cr",
                "gep_ci",
                "distance_to_shooting",
                "blumen_ci_abs_err",
                "low_ci_term",
                "total_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSurface CSV: {surface_csv}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Paths CSV: {paths_csv}")
    print(f"Isoline figure: {png_path}")


if __name__ == "__main__":
    main()
