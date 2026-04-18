from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic local des familles modales supersoniques.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--n-points-values", type=int, nargs="+", default=[561, 801])
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--high-cr-threshold", type=float, default=0.6)
    parser.add_argument("--reference-mach", type=float, default=1.30)
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


def resample_profile(y: np.ndarray, p: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    pr = np.interp(y_ref, y, np.real(p))
    pi = np.interp(y_ref, y, np.imag(p))
    return pr + 1j * pi


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    profile_payload: dict[str, np.ndarray] = {}

    reference_profiles: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}

    # Build reference high-family profile at the reference Mach for each resolution.
    for n_points in args.n_points_values:
        solver = NotebookStyleDenseGEPSolver(
            alpha=args.alpha,
            Mach=float(args.reference_mach),
            n_points=int(n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        shooting_cr, shooting_ci, *_ = solve_shooting(args.alpha, float(args.reference_mach))
        shooting_guess = (shooting_cr, shooting_ci)
        modes = [mode for mode in solver.finite_modes() if mode["cr"] >= args.high_cr_threshold]
        if not modes:
            continue
        ref_mode = min(modes, key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight))
        y_ref, p_ref = normalize_pressure_profile(solver.y, ref_mode["vector"], solver.n_points)
        reference_profiles[int(n_points)] = (y_ref, p_ref, ref_mode)

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
            if not modes:
                continue

            low_mode = min(modes, key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight))
            high_candidates = [mode for mode in modes if mode["cr"] >= args.high_cr_threshold]
            high_mode = None if not high_candidates else min(
                high_candidates,
                key=lambda mode: solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight),
            )

            families = [("low", low_mode)]
            if high_mode is not None:
                families.append(("high", high_mode))

            local_profiles: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
            for family_name, mode in families:
                y, p = normalize_pressure_profile(solver.y, mode["vector"], solver.n_points)
                stats = profile_stats(y, p)
                dist = float(solver.spectral_distance(mode, shooting_guess, ci_weight=args.ci_weight))
                profile_key = f"M{mach:.3f}_N{int(n_points)}_{family_name}"
                profile_payload[f"{profile_key}_y"] = y
                profile_payload[f"{profile_key}_pr"] = np.real(p)
                profile_payload[f"{profile_key}_pi"] = np.imag(p)
                local_profiles[family_name] = (y, p, stats)

                overlap_ref = np.nan
                if int(n_points) in reference_profiles:
                    y_ref, p_ref, ref_mode = reference_profiles[int(n_points)]
                    p_interp = resample_profile(y, p, y_ref)
                    num = abs(np.vdot(p_interp, p_ref))
                    den = max(float(np.linalg.norm(p_interp) * np.linalg.norm(p_ref)), 1e-12)
                    overlap_ref = float(num / den)
                    ref_cr = float(ref_mode["cr"])
                    ref_ci = float(ref_mode["ci"])
                else:
                    ref_cr = np.nan
                    ref_ci = np.nan

                summary_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "family": family_name,
                        "cand_cr": float(mode["cr"]),
                        "cand_ci": float(mode["ci"]),
                        "cand_omega_i": float(mode["omega_i"]),
                        "distance_to_shooting": dist,
                        "shooting_cr": float(shooting_cr),
                        "shooting_ci": float(shooting_ci),
                        "shooting_omega_i": float(shooting_omega_i),
                        "shooting_spectral_success": bool(shooting_ok),
                        "reference_high_cr": ref_cr,
                        "reference_high_ci": ref_ci,
                        "overlap_to_reference_high": overlap_ref,
                        **stats,
                    }
                )

            if "low" in local_profiles and "high" in local_profiles:
                y_low, p_low, stats_low = local_profiles["low"]
                y_high, p_high, stats_high = local_profiles["high"]
                y_common = np.linspace(max(y_low[0], y_high[0]), min(y_low[-1], y_high[-1]), 512)
                p_low_i = resample_profile(y_low, p_low, y_common)
                p_high_i = resample_profile(y_high, p_high, y_common)
                num = abs(np.vdot(p_low_i, p_high_i))
                den = max(float(np.linalg.norm(p_low_i) * np.linalg.norm(p_high_i)), 1e-12)
                overlap = float(num / den)
                comparison_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(n_points),
                        "low_cr": float(low_mode["cr"]),
                        "low_ci": float(low_mode["ci"]),
                        "high_cr": float(high_mode["cr"]),
                        "high_ci": float(high_mode["ci"]),
                        "low_distance_to_shooting": float(
                            solver.spectral_distance(low_mode, shooting_guess, ci_weight=args.ci_weight)
                        ),
                        "high_distance_to_shooting": float(
                            solver.spectral_distance(high_mode, shooting_guess, ci_weight=args.ci_weight)
                        ),
                        "low_high_overlap": overlap,
                        "delta_peak_y": float(stats_high["peak_y"] - stats_low["peak_y"]),
                        "delta_centroid_y": float(stats_high["centroid_y"] - stats_low["centroid_y"]),
                        "delta_phase_span": float(stats_high["phase_span"] - stats_low["phase_span"]),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "n_points", "family"]).reset_index(drop=True)
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["Mach", "n_points"]).reset_index(drop=True)

    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    comparison_path = OUTPUT_DIR / f"{args.output_stem}_comparison.csv"
    npz_path = OUTPUT_DIR / f"{args.output_stem}_profiles.npz"

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    np.savez(npz_path, **profile_payload)

    print(f"Wrote {summary_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
