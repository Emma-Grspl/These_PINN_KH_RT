from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit du shooting supersonique contre les familles basse/haute du GEP."
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
    parser.add_argument("--high-cr-threshold", type=float, default=0.6)
    parser.add_argument("--shooting-max-iter", type=int, default=10)
    parser.add_argument("--shooting-grid-size", type=int, default=4)
    parser.add_argument("--shooting-box", type=float, nargs=4, action="append", default=None)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def default_shooting_boxes() -> list[tuple[str, float, float, float, float]]:
    return [
        ("default", 0.03, 0.70, 0.001, 0.12),
        ("tight_high", 0.45, 0.85, 0.001, 0.03),
        ("mid_high", 0.35, 0.75, 0.001, 0.06),
        ("wide", 0.03, 0.95, 0.001, 0.20),
    ]


def parse_shooting_boxes(
    raw_boxes: list[list[float]] | None,
) -> list[tuple[str, float, float, float, float]]:
    if not raw_boxes:
        return default_shooting_boxes()
    boxes: list[tuple[str, float, float, float, float]] = []
    for idx, values in enumerate(raw_boxes, start=1):
        cr_min, cr_max, ci_min, ci_max = [float(v) for v in values]
        boxes.append((f"box{idx:02d}", cr_min, cr_max, ci_min, ci_max))
    return boxes


def solve_shooting(
    *,
    alpha: float,
    mach: float,
    cr_min: float,
    cr_max: float,
    ci_min: float,
    ci_max: float,
    max_iter: int,
    grid_size: int,
) -> tuple[Mstab17SupersonicSolver, object]:
    solver = Mstab17SupersonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve(
        cr_min=cr_min,
        cr_max=cr_max,
        ci_min=ci_min,
        ci_max=ci_max,
        max_iter=max_iter,
        grid_size=grid_size,
    )
    return solver, result


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


def resample_profile(y: np.ndarray, p: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    pr = np.interp(y_ref, y, np.real(p))
    pi = np.interp(y_ref, y, np.imag(p))
    return pr + 1j * pi


def overlap_between_profiles(
    y: np.ndarray,
    p: np.ndarray,
    y_ref: np.ndarray,
    p_ref: np.ndarray,
) -> float:
    p_interp = resample_profile(y, p, y_ref)
    num = abs(np.vdot(p_interp, p_ref))
    den = max(float(np.linalg.norm(p_interp) * np.linalg.norm(p_ref)), 1e-12)
    return float(num / den)


def optimize_stage2_for_candidate(solver: Mstab17SupersonicSolver, cr: float, ci: float) -> tuple[float, float]:
    opt = minimize_scalar(
        lambda ln_p_right: solver.stage2_objective(ln_p_right, cr, ci),
        bounds=(-15.0, 5.0),
        method="bounded",
    )
    return float(opt.x), float(opt.fun)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shooting_boxes = parse_shooting_boxes(args.shooting_box)
    alpha = float(args.alpha)
    mach_values = [float(m) for m in args.mach_values]

    reference_solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=float(args.reference_mach),
        n_points=int(args.n_points),
        mapping_kind=args.mapping_kind,
        mapping_scale=args.mapping_scale,
        cubic_delta=args.cubic_delta,
        xi_max=args.xi_max,
    )
    ref_shooting_solver, ref_shooting_result = solve_shooting(
        alpha=alpha,
        mach=float(args.reference_mach),
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * float(args.reference_mach))),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=int(args.shooting_max_iter),
        grid_size=int(args.shooting_grid_size),
    )
    del ref_shooting_solver
    ref_guess = (float(ref_shooting_result.cr), float(ref_shooting_result.ci))
    ref_high_modes = [mode for mode in reference_solver.finite_modes() if mode["cr"] >= args.high_cr_threshold]
    if not ref_high_modes:
        raise RuntimeError("Aucune famille haute trouvee au Mach de reference.")
    reference_high_mode = min(
        ref_high_modes,
        key=lambda mode: reference_solver.spectral_distance(mode, ref_guess, ci_weight=args.ci_weight),
    )
    y_ref, p_ref = normalize_pressure_profile(
        reference_solver.y,
        reference_high_mode["vector"],
        reference_solver.n_points,
    )
    ref_stats = profile_stats(y_ref, p_ref)

    family_rows: list[dict] = []
    shooting_rows: list[dict] = []
    summary_rows: list[dict] = []

    for mach in mach_values:
        local_shooting_solver, local_shooting_result = solve_shooting(
            alpha=alpha,
            mach=mach,
            cr_min=0.03,
            cr_max=min(0.7, max(0.35, 0.5 * mach)),
            ci_min=0.001,
            ci_max=0.12,
            max_iter=int(args.shooting_max_iter),
            grid_size=int(args.shooting_grid_size),
        )
        local_guess = (float(local_shooting_result.cr), float(local_shooting_result.ci))

        gep_solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        modes = [mode for mode in gep_solver.finite_modes() if mode["cr"] >= -1e-10]
        if not modes:
            summary_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "n_points": int(args.n_points),
                    "has_low_family": False,
                    "has_high_family": False,
                    "default_shooting_cr": float(local_shooting_result.cr),
                    "default_shooting_ci": float(local_shooting_result.ci),
                    "default_stage1_mismatch": float(local_shooting_result.stage1_mismatch),
                    "default_stage2_mismatch": float(local_shooting_result.stage2_mismatch),
                }
            )
            continue

        low_mode = min(
            modes,
            key=lambda mode: gep_solver.spectral_distance(mode, local_guess, ci_weight=args.ci_weight),
        )
        high_candidates = [mode for mode in modes if mode["cr"] >= args.high_cr_threshold]
        high_mode = None if not high_candidates else min(
            high_candidates,
            key=lambda mode: gep_solver.spectral_distance(mode, local_guess, ci_weight=args.ci_weight),
        )

        for family_name, mode in [("low", low_mode), ("high", high_mode)]:
            if mode is None:
                continue
            y, p = normalize_pressure_profile(gep_solver.y, mode["vector"], gep_solver.n_points)
            stats = profile_stats(y, p)
            overlap_ref = overlap_between_profiles(y, p, y_ref, p_ref)
            stage1_mismatch = float(local_shooting_solver.stage1_mismatch(float(mode["cr"]), float(mode["ci"])))
            stage2_ln_p_right, stage2_mismatch = optimize_stage2_for_candidate(
                local_shooting_solver,
                float(mode["cr"]),
                float(mode["ci"]),
            )
            family_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "n_points": int(args.n_points),
                    "family": family_name,
                    "cand_cr": float(mode["cr"]),
                    "cand_ci": float(mode["ci"]),
                    "cand_omega_i": float(mode["omega_i"]),
                    "distance_to_default_shooting": float(
                        gep_solver.spectral_distance(mode, local_guess, ci_weight=args.ci_weight)
                    ),
                    "abs_delta_cr_to_default_shooting": abs(float(mode["cr"]) - float(local_shooting_result.cr)),
                    "abs_delta_ci_to_default_shooting": abs(float(mode["ci"]) - float(local_shooting_result.ci)),
                    "shooting_stage1_mismatch_at_candidate": stage1_mismatch,
                    "shooting_stage2_mismatch_at_candidate": stage2_mismatch,
                    "shooting_stage2_ln_p_start_right_at_candidate": stage2_ln_p_right,
                    "overlap_to_reference_high": overlap_ref,
                    "reference_high_cr": float(reference_high_mode["cr"]),
                    "reference_high_ci": float(reference_high_mode["ci"]),
                    "reference_peak_y": ref_stats["peak_y"],
                    "reference_centroid_y": ref_stats["centroid_y"],
                    "reference_spread_y": ref_stats["spread_y"],
                    "reference_phase_span": ref_stats["phase_span"],
                    "default_shooting_cr": float(local_shooting_result.cr),
                    "default_shooting_ci": float(local_shooting_result.ci),
                    "default_shooting_omega_i": float(local_shooting_result.omega_i),
                    "default_shooting_stage1_mismatch": float(local_shooting_result.stage1_mismatch),
                    "default_shooting_stage2_mismatch": float(local_shooting_result.stage2_mismatch),
                    "default_shooting_success": bool(local_shooting_result.success),
                    "default_shooting_spectral_success": bool(local_shooting_result.spectral_success),
                    **stats,
                }
            )

        for box_label, cr_min, cr_max, ci_min, ci_max in shooting_boxes:
            _, shooting_result = solve_shooting(
                alpha=alpha,
                mach=mach,
                cr_min=cr_min,
                cr_max=cr_max,
                ci_min=ci_min,
                ci_max=ci_max,
                max_iter=int(args.shooting_max_iter),
                grid_size=int(args.shooting_grid_size),
            )
            shooting_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "box_label": box_label,
                    "cr_min": cr_min,
                    "cr_max": cr_max,
                    "ci_min": ci_min,
                    "ci_max": ci_max,
                    "shooting_cr": float(shooting_result.cr),
                    "shooting_ci": float(shooting_result.ci),
                    "shooting_omega_i": float(shooting_result.omega_i),
                    "stage1_mismatch": float(shooting_result.stage1_mismatch),
                    "stage2_mismatch": float(shooting_result.stage2_mismatch),
                    "spectral_success": bool(shooting_result.spectral_success),
                    "mode_success": bool(shooting_result.mode_success),
                    "success": bool(shooting_result.success),
                }
            )

        summary_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "n_points": int(args.n_points),
                "has_low_family": True,
                "has_high_family": high_mode is not None,
                "low_cr": float(low_mode["cr"]),
                "low_ci": float(low_mode["ci"]),
                "low_stage1_mismatch": float(local_shooting_solver.stage1_mismatch(float(low_mode["cr"]), float(low_mode["ci"]))),
                "low_distance_to_default_shooting": float(
                    gep_solver.spectral_distance(low_mode, local_guess, ci_weight=args.ci_weight)
                ),
                "high_cr": np.nan if high_mode is None else float(high_mode["cr"]),
                "high_ci": np.nan if high_mode is None else float(high_mode["ci"]),
                "high_stage1_mismatch": np.nan
                if high_mode is None
                else float(local_shooting_solver.stage1_mismatch(float(high_mode["cr"]), float(high_mode["ci"]))),
                "high_distance_to_default_shooting": np.nan
                if high_mode is None
                else float(gep_solver.spectral_distance(high_mode, local_guess, ci_weight=args.ci_weight)),
                "default_shooting_cr": float(local_shooting_result.cr),
                "default_shooting_ci": float(local_shooting_result.ci),
                "default_stage1_mismatch": float(local_shooting_result.stage1_mismatch),
                "default_stage2_mismatch": float(local_shooting_result.stage2_mismatch),
                "default_shooting_success": bool(local_shooting_result.success),
            }
        )

    family_df = pd.DataFrame(family_rows).sort_values(["Mach", "family"]).reset_index(drop=True)
    shooting_df = pd.DataFrame(shooting_rows).sort_values(["Mach", "box_label"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach"]).reset_index(drop=True)

    families_path = OUTPUT_DIR / f"{args.output_stem}_families.csv"
    sensitivity_path = OUTPUT_DIR / f"{args.output_stem}_shooting_sensitivity.csv"
    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"

    family_df.to_csv(families_path, index=False)
    shooting_df.to_csv(sensitivity_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nFamilies CSV: {families_path}")
    print(f"Shooting sensitivity CSV: {sensitivity_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
