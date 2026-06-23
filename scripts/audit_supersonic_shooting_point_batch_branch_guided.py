from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    build_blumen_targets,
    load_digitized_long,
)
from scripts.audit_supersonic_shooting_ci_map import ci_primary_score  # noqa: E402
from scripts.audit_supersonic_shooting_point_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    default_box_robustness_metrics,
    dedup_seeds,
    generic_seed_list,
    plot_diagnostics,
    plot_modes_pdf,
    plot_status_map,
    select_best_candidate_with_box_audit,
    success_label,
)
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


DEFAULT_REFERENCE_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral.csv"
DEFAULT_MODAL_REFERENCE_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pointwise supersonique M intermediaire guide par branche : "
            "le score cible utilise un c_r interpole depuis les lignes voisines "
            "et un c_i cible de Blumen."
        )
    )
    parser.add_argument("--mach", type=float, default=1.40)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.125000,0.137500,0.150000,0.162500,0.168750,0.175000,0.181250,0.187500,0.193750,0.200000",
        help="Liste CSV d'alpha a evaluer.",
    )
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--modal-reference-csv", type=Path, default=DEFAULT_MODAL_REFERENCE_CSV)
    parser.add_argument("--alpha-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=500.0)
    parser.add_argument("--y-limit-factor", type=float, default=6.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.015, 0.03, 0.06, 0.10])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.008, 0.015, 0.03, 0.06])
    parser.add_argument("--retry-growth", type=float, default=1.75)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ci-weight", type=float, default=4.0)
    parser.add_argument("--cr-weight", type=float, default=0.35)
    parser.add_argument("--continuity-weight", type=float, default=0.20)
    parser.add_argument("--include-generic-seeds", action="store_true")
    parser.add_argument("--visible-threshold-ratio", type=float, default=0.02)
    parser.add_argument("--visible-min-half-width", type=float, default=8.0)
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument("--box-robustness-factors", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--box-robustness-max-rel-l2", type=float, default=0.15)
    parser.add_argument("--box-robustness-max-peak-shift", type=float, default=0.75)
    parser.add_argument("--box-robustness-max-center8-delta", type=float, default=0.10)
    parser.add_argument("--box-robustness-max-edge-growth", type=float, default=1.25)
    parser.add_argument(
        "--box-selection-max-candidates",
        type=int,
        default=12,
        help="Nombre maximum de candidats bruts validated a tester avec la boite. <=0 teste tous les candidats.",
    )
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting_point_batch_M140_branch_guided")
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    return parser


def parse_alpha_csv(raw_value: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        alpha = round(float(token), 8)
        if alpha in seen:
            continue
        seen.add(alpha)
        values.append(float(token))
    if not values:
        raise ValueError("La liste d'alpha est vide.")
    return values


def find_reference_row(
    df: pd.DataFrame,
    *,
    mach: float,
    alpha: float,
    alpha_tolerance: float,
) -> pd.Series | None:
    same_mach = df[np.isclose(df["Mach"].to_numpy(dtype=float), float(mach), atol=1.0e-10, rtol=0.0)].copy()
    if same_mach.empty:
        return None
    alpha_delta = np.abs(same_mach["alpha"].to_numpy(dtype=float) - float(alpha))
    same_mach = same_mach.assign(_alpha_delta=alpha_delta)
    exact = same_mach[same_mach["_alpha_delta"] <= float(alpha_tolerance)].sort_values(["_alpha_delta", "alpha"])
    if not exact.empty:
        return exact.iloc[0]
    nearest = same_mach.sort_values(["_alpha_delta", "alpha"]).iloc[0]
    if float(nearest["_alpha_delta"]) <= float(alpha_tolerance):
        return nearest
    return None


def interpolate_reference_bundle_for_mach(
    df: pd.DataFrame,
    *,
    mach: float,
    alpha: float,
    alpha_tolerance: float,
    source_type: str,
) -> dict[str, object] | None:
    exact_row = find_reference_row(df, mach=float(mach), alpha=float(alpha), alpha_tolerance=float(alpha_tolerance))
    if exact_row is not None:
        return {
            "Mach": float(exact_row["Mach"]),
            "alpha": float(exact_row["alpha"]),
            "reference_cr": float(exact_row["reference_cr"]),
            "reference_ci": float(exact_row["reference_ci"]),
            "source_type": str(source_type),
            "source_label": str(exact_row.get("source_label", "")),
        }

    same_mach = df[np.isclose(df["Mach"].to_numpy(dtype=float), float(mach), atol=1.0e-10, rtol=0.0)].copy()
    if same_mach.empty:
        return None
    same_mach = same_mach.sort_values(["alpha", "reference_ci", "reference_cr"]).reset_index(drop=True)
    alpha_values = same_mach["alpha"].to_numpy(dtype=float)
    lower_candidates = np.where(alpha_values < float(alpha) - float(alpha_tolerance))[0]
    upper_candidates = np.where(alpha_values > float(alpha) + float(alpha_tolerance))[0]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        return None
    lower_row = same_mach.iloc[int(lower_candidates[-1])]
    upper_row = same_mach.iloc[int(upper_candidates[0])]
    lower_alpha = float(lower_row["alpha"])
    upper_alpha = float(upper_row["alpha"])
    if not np.isfinite(lower_alpha) or not np.isfinite(upper_alpha) or upper_alpha <= lower_alpha:
        return None

    weight = (float(alpha) - lower_alpha) / (upper_alpha - lower_alpha)
    weight = float(np.clip(weight, 0.0, 1.0))
    lower_label = str(lower_row.get("source_label", ""))
    upper_label = str(upper_row.get("source_label", ""))
    if lower_label == upper_label:
        source_label = lower_label
    else:
        source_label = f"{lower_label}|{upper_label}".strip("|")

    return {
        "Mach": float(mach),
        "alpha": float(alpha),
        "reference_cr": (1.0 - weight) * float(lower_row["reference_cr"]) + weight * float(upper_row["reference_cr"]),
        "reference_ci": (1.0 - weight) * float(lower_row["reference_ci"]) + weight * float(upper_row["reference_ci"]),
        "source_type": f"{source_type}_interp",
        "source_label": source_label,
    }


def build_reference_bundle(
    *,
    spectral_df: pd.DataFrame,
    modal_df: pd.DataFrame,
    mach: float,
    alpha: float,
    alpha_tolerance: float,
) -> dict[str, object] | None:
    modal_bundle = interpolate_reference_bundle_for_mach(
        modal_df,
        mach=float(mach),
        alpha=float(alpha),
        alpha_tolerance=float(alpha_tolerance),
        source_type="modal",
    )
    if modal_bundle is not None:
        return modal_bundle
    spectral_bundle = interpolate_reference_bundle_for_mach(
        spectral_df,
        mach=float(mach),
        alpha=float(alpha),
        alpha_tolerance=float(alpha_tolerance),
        source_type="spectral",
    )
    if spectral_bundle is None:
        return None
    return spectral_bundle


def pick_bracketing_guides(
    *,
    spectral_df: pd.DataFrame,
    modal_df: pd.DataFrame,
    target_mach: float,
    target_alpha: float,
    alpha_tolerance: float,
) -> tuple[dict[str, object], dict[str, object]]:
    available_machs = sorted(float(value) for value in spectral_df["Mach"].to_numpy(dtype=float))
    lower_machs = [mach for mach in available_machs if mach < float(target_mach)]
    upper_machs = [mach for mach in available_machs if mach > float(target_mach)]
    if not lower_machs or not upper_machs:
        raise RuntimeError(f"Impossible de bracketter M={target_mach:.3f} avec les references disponibles.")
    lower_mach = lower_machs[-1]
    upper_mach = upper_machs[0]
    lower_bundle = build_reference_bundle(
        spectral_df=spectral_df,
        modal_df=modal_df,
        mach=float(lower_mach),
        alpha=float(target_alpha),
        alpha_tolerance=float(alpha_tolerance),
    )
    upper_bundle = build_reference_bundle(
        spectral_df=spectral_df,
        modal_df=modal_df,
        mach=float(upper_mach),
        alpha=float(target_alpha),
        alpha_tolerance=float(alpha_tolerance),
    )
    if lower_bundle is None or upper_bundle is None:
        raise RuntimeError(
            f"References manquantes pour bracketter alpha={target_alpha:.6f}, M={target_mach:.3f} "
            f"(lower={lower_mach:.3f}, upper={upper_mach:.3f})."
        )
    return lower_bundle, upper_bundle


def build_guided_target(
    *,
    target_mach: float,
    target_alpha: float,
    lower_bundle: dict[str, object],
    upper_bundle: dict[str, object],
    blumen_cr: float,
    blumen_ci: float,
) -> dict[str, object]:
    lower_mach = float(lower_bundle["Mach"])
    upper_mach = float(upper_bundle["Mach"])
    weight = (float(target_mach) - lower_mach) / (upper_mach - lower_mach)
    interp_cr = (1.0 - weight) * float(lower_bundle["reference_cr"]) + weight * float(upper_bundle["reference_cr"])
    interp_ci = (1.0 - weight) * float(lower_bundle["reference_ci"]) + weight * float(upper_bundle["reference_ci"])
    target_ci = float(blumen_ci) if np.isfinite(blumen_ci) else float(interp_ci)
    target_ci_source = "blumen_ci" if np.isfinite(blumen_ci) else "interpolated_branch_ci"
    return {
        "alpha": float(target_alpha),
        "Mach": float(target_mach),
        "lower_mach": float(lower_mach),
        "upper_mach": float(upper_mach),
        "interp_weight": float(weight),
        "lower_alpha": float(lower_bundle["alpha"]),
        "upper_alpha": float(upper_bundle["alpha"]),
        "lower_cr": float(lower_bundle["reference_cr"]),
        "lower_ci": float(lower_bundle["reference_ci"]),
        "upper_cr": float(upper_bundle["reference_cr"]),
        "upper_ci": float(upper_bundle["reference_ci"]),
        "lower_source_type": str(lower_bundle["source_type"]),
        "upper_source_type": str(upper_bundle["source_type"]),
        "lower_source_label": str(lower_bundle["source_label"]),
        "upper_source_label": str(upper_bundle["source_label"]),
        "interp_cr": float(interp_cr),
        "interp_ci": float(interp_ci),
        "guide_target_cr": float(interp_cr),
        "guide_target_ci": float(target_ci),
        "guide_target_ci_source": str(target_ci_source),
        "blumen_cr": float(blumen_cr) if np.isfinite(blumen_cr) else np.nan,
        "blumen_ci": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
    }


def guided_seed_list(
    *,
    guide: dict[str, object],
    include_generic_seeds: bool,
) -> list[tuple[str, float, float]]:
    lower_cr = float(guide["lower_cr"])
    lower_ci = float(guide["lower_ci"])
    upper_cr = float(guide["upper_cr"])
    upper_ci = float(guide["upper_ci"])
    interp_cr = float(guide["interp_cr"])
    interp_ci = float(guide["interp_ci"])
    target_ci = float(guide["guide_target_ci"])
    seeds = [
        ("lower_branch", lower_cr, lower_ci),
        ("upper_branch", upper_cr, upper_ci),
        ("interp_branch", interp_cr, interp_ci),
        ("interp_cr_target_ci", interp_cr, target_ci),
        ("lower_cr_target_ci", lower_cr, target_ci),
        ("upper_cr_target_ci", upper_cr, target_ci),
        ("interp_branch_ci_blend", interp_cr, 0.5 * (interp_ci + target_ci)),
        ("lower_branch_ci_blend", lower_cr, 0.5 * (lower_ci + target_ci)),
        ("upper_branch_ci_blend", upper_cr, 0.5 * (upper_ci + target_ci)),
    ]
    blumen_cr = float(guide["blumen_cr"])
    if np.isfinite(blumen_cr) and np.isfinite(target_ci):
        seeds.append(("blumen_ci_guided", blumen_cr, target_ci))
    if include_generic_seeds:
        seeds.extend(generic_seed_list())
    return dedup_seeds(seeds)


def guided_selection_metric(
    *,
    shooting_cr: float,
    shooting_ci: float,
    guide: dict[str, object],
    stage1_mismatch: float,
    stage2_mismatch: float,
    cfg: dict[str, object],
) -> tuple[float, str]:
    score = ci_primary_score(
        shooting_cr=float(shooting_cr),
        shooting_ci=float(shooting_ci),
        blumen_cr=float(guide["guide_target_cr"]),
        blumen_ci=float(guide["guide_target_ci"]),
        previous_mach=(float(guide["lower_cr"]), float(guide["lower_ci"])),
        previous_alpha=(float(guide["upper_cr"]), float(guide["upper_ci"])),
        ci_weight=float(cfg["ci_weight"]),
        cr_weight=float(cfg["cr_weight"]),
        continuity_weight=float(cfg["continuity_weight"]),
    )
    score += 0.05 * float(stage1_mismatch) + 0.001 * float(stage2_mismatch)
    return float(score), "distance_to_guided_branch_target"


def evaluate_point(
    point: tuple[float, float],
    cfg: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    alpha, mach = point
    cr_points = load_digitized_long(Path(str(cfg["cr_points"])))
    ci_points = load_digitized_long(Path(str(cfg["ci_points"])))
    target_df = build_blumen_targets([float(mach)], float(alpha), cr_points, ci_points)
    target = target_df.iloc[0]
    blumen_cr = float(target["blumen_cr"])
    blumen_ci = float(target["blumen_ci"])
    ci_available = bool(np.isfinite(blumen_ci))
    cr_available = bool(np.isfinite(blumen_cr))

    spectral_df = pd.read_csv(Path(str(cfg["reference_csv"])))
    modal_df = pd.read_csv(Path(str(cfg["modal_reference_csv"])))
    lower_bundle, upper_bundle = pick_bracketing_guides(
        spectral_df=spectral_df,
        modal_df=modal_df,
        target_mach=float(mach),
        target_alpha=float(alpha),
        alpha_tolerance=float(cfg["alpha_tolerance"]),
    )
    guide = build_guided_target(
        target_mach=float(mach),
        target_alpha=float(alpha),
        lower_bundle=lower_bundle,
        upper_bundle=upper_bundle,
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
    )
    seeds = guided_seed_list(guide=guide, include_generic_seeds=bool(cfg["include_generic_seeds"]))

    candidate_rows: list[dict[str, object]] = []
    try:
        for seed_name, cr_center, ci_center in seeds:
            for cr_half in cfg["cr_half_windows"]:
                for ci_half in cfg["ci_half_windows"]:
                    solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                        alpha=float(alpha),
                        mach=float(mach),
                        match_y=float(cfg["match_y"]),
                        use_mapping=bool(cfg["use_mapping"]),
                        mapping_scale=float(cfg["mapping_scale"]),
                        min_y_limit=float(cfg["min_y_limit"]),
                        max_y_limit=float(cfg["max_y_limit"]),
                        y_limit_factor=float(cfg["y_limit_factor"]),
                        amp_lower_bound=float(cfg["amp_lower_bound"]),
                        amp_upper_bound=float(cfg["amp_upper_bound"]),
                        cr_center=float(cr_center),
                        ci_center=float(ci_center),
                        cr_half_window=float(cr_half),
                        ci_half_window=float(ci_half),
                        retry_growth=float(cfg["retry_growth"]),
                        max_retries=int(cfg["max_retries"]),
                        max_iter=int(cfg["max_iter"]),
                        grid_size=int(cfg["grid_size"]),
                    )
                    metric_value, metric_name = guided_selection_metric(
                        shooting_cr=float(result.cr),
                        shooting_ci=float(result.ci),
                        guide=guide,
                        stage1_mismatch=float(result.stage1_mismatch),
                        stage2_mismatch=float(result.stage2_mismatch),
                        cfg=cfg,
                    )
                    candidate_rows.append(
                        {
                            "alpha": float(alpha),
                            "Mach": float(mach),
                            "blumen_cr": float(blumen_cr),
                            "blumen_ci": float(blumen_ci),
                            "blumen_cr_available": bool(cr_available),
                            "blumen_ci_available": bool(ci_available),
                            "guide_target_cr": float(guide["guide_target_cr"]),
                            "guide_target_ci": float(guide["guide_target_ci"]),
                            "guide_target_ci_source": str(guide["guide_target_ci_source"]),
                            "guide_lower_cr": float(guide["lower_cr"]),
                            "guide_lower_ci": float(guide["lower_ci"]),
                            "guide_upper_cr": float(guide["upper_cr"]),
                            "guide_upper_ci": float(guide["upper_ci"]),
                            "guide_lower_mach": float(guide["lower_mach"]),
                            "guide_upper_mach": float(guide["upper_mach"]),
                            "guide_lower_source_type": str(guide["lower_source_type"]),
                            "guide_upper_source_type": str(guide["upper_source_type"]),
                            "seed_name": str(seed_name),
                            "seed_cr_center": float(cr_center),
                            "seed_ci_center": float(ci_center),
                            "requested_cr_half_window": float(cr_half),
                            "requested_ci_half_window": float(ci_half),
                            "used_cr_half_window": float(used_cr_half),
                            "used_ci_half_window": float(used_ci_half),
                            "retry_index": int(retry_idx),
                            "shooting_cr": float(result.cr),
                            "shooting_ci": float(result.ci),
                            "shooting_omega_i": float(result.omega_i),
                            "err_cr_abs": abs(float(result.cr) - blumen_cr) if cr_available else np.nan,
                            "err_ci_abs": abs(float(result.ci) - blumen_ci) if ci_available else np.nan,
                            "err_ci_rel": (
                                abs(float(result.ci) - blumen_ci) / max(abs(blumen_ci), 1e-12) if ci_available else np.nan
                            ),
                            "guide_err_cr_abs": abs(float(result.cr) - float(guide["guide_target_cr"])),
                            "guide_err_ci_abs": abs(float(result.ci) - float(guide["guide_target_ci"])),
                            "stage1_mismatch": float(result.stage1_mismatch),
                            "stage2_mismatch": float(result.stage2_mismatch),
                            "ln_p_start_right": float(result.ln_p_start_right),
                            "spectral_success": bool(result.spectral_success),
                            "mode_success": bool(result.mode_success),
                            "success": bool(result.success),
                            "status": success_label(bool(result.spectral_success), bool(result.mode_success)),
                            "selection_metric": float(metric_value),
                            "selection_metric_name": str(metric_name),
                            "y_limit": float(result.y_limit),
                        }
                    )

        best, fields, diag, final_status, box_rejection_applied, box_selection_diag = select_best_candidate_with_box_audit(
            alpha=float(alpha),
            mach=float(mach),
            candidate_rows=candidate_rows,
            cfg=cfg,
        )

        summary_row = {
            "alpha": float(alpha),
            "Mach": float(mach),
            "blumen_cr": float(blumen_cr),
            "blumen_ci": float(blumen_ci),
            "blumen_cr_available": bool(cr_available),
            "blumen_ci_available": bool(ci_available),
            "guide_target_cr": float(guide["guide_target_cr"]),
            "guide_target_ci": float(guide["guide_target_ci"]),
            "guide_target_ci_source": str(guide["guide_target_ci_source"]),
            "guide_lower_mach": float(guide["lower_mach"]),
            "guide_upper_mach": float(guide["upper_mach"]),
            "guide_lower_alpha": float(guide["lower_alpha"]),
            "guide_upper_alpha": float(guide["upper_alpha"]),
            "guide_lower_cr": float(guide["lower_cr"]),
            "guide_lower_ci": float(guide["lower_ci"]),
            "guide_upper_cr": float(guide["upper_cr"]),
            "guide_upper_ci": float(guide["upper_ci"]),
            "guide_lower_source_type": str(guide["lower_source_type"]),
            "guide_upper_source_type": str(guide["upper_source_type"]),
            "guide_lower_source_label": str(guide["lower_source_label"]),
            "guide_upper_source_label": str(guide["upper_source_label"]),
            "guide_interp_weight": float(guide["interp_weight"]),
            "n_seeds": int(len(seeds)),
            "n_candidates": int(len(candidate_rows)),
            "n_success_candidates": int(sum(bool(row["success"]) for row in candidate_rows)),
            "n_spectral_success_candidates": int(sum(bool(row["spectral_success"]) for row in candidate_rows)),
            "n_mode_success_candidates": int(sum(bool(row["mode_success"]) for row in candidate_rows)),
            "best_seed_name": str(best["seed_name"]),
            "best_shooting_cr": float(best["shooting_cr"]),
            "best_shooting_ci": float(best["shooting_ci"]),
            "best_shooting_omega_i": float(best["shooting_omega_i"]),
            "best_err_cr_abs": float(best["err_cr_abs"]) if cr_available else np.nan,
            "best_err_ci_abs": float(best["err_ci_abs"]) if ci_available else np.nan,
            "best_err_ci_rel": float(best["err_ci_rel"]) if ci_available else np.nan,
            "best_guide_err_cr_abs": float(best["guide_err_cr_abs"]),
            "best_guide_err_ci_abs": float(best["guide_err_ci_abs"]),
            "best_stage1_mismatch": float(best["stage1_mismatch"]),
            "best_stage2_mismatch": float(best["stage2_mismatch"]),
            "best_ln_p_start_right": float(best["ln_p_start_right"]),
            "best_spectral_success": bool(best["spectral_success"]),
            "best_mode_success": bool(best["mode_success"]),
            "best_success": bool(best["success"]) and not box_rejection_applied,
            "best_raw_status": str(best["status"]),
            "best_status": final_status,
            "box_rejection_applied": bool(box_rejection_applied),
            "best_selection_metric": float(best["selection_metric"]),
            "best_selection_metric_name": str(best["selection_metric_name"]),
            "best_retry_index": int(best["retry_index"]),
            "best_used_cr_half_window": float(best["used_cr_half_window"]),
            "best_used_ci_half_window": float(best["used_ci_half_window"]),
            "best_y_limit": float(best["y_limit"]),
            **box_selection_diag,
            **diag,
            "exception": "",
        }

        field_rows: list[dict[str, object]] = []
        for y_value, rho_value, u_value, v_value, p_value in zip(
            fields["y"], fields["rho"], fields["u"], fields["v"], fields["p"]
        ):
            field_rows.append(
                {
                    "alpha": float(alpha),
                    "Mach": float(mach),
                    "best_status": final_status,
                    "best_raw_status": str(best["status"]),
                    "y": float(y_value),
                    "rho_real": float(np.real(rho_value)),
                    "rho_imag": float(np.imag(rho_value)),
                    "u_real": float(np.real(u_value)),
                    "u_imag": float(np.imag(u_value)),
                    "v_real": float(np.real(v_value)),
                    "v_imag": float(np.imag(v_value)),
                    "p_real": float(np.real(p_value)),
                    "p_imag": float(np.imag(p_value)),
                }
            )
        return summary_row, candidate_rows, field_rows
    except Exception as exc:  # noqa: BLE001
        summary_row = {
            "alpha": float(alpha),
            "Mach": float(mach),
            "blumen_cr": float(blumen_cr),
            "blumen_ci": float(blumen_ci),
            "blumen_cr_available": bool(cr_available),
            "blumen_ci_available": bool(ci_available),
            "guide_target_cr": float(guide["guide_target_cr"]),
            "guide_target_ci": float(guide["guide_target_ci"]),
            "guide_target_ci_source": str(guide["guide_target_ci_source"]),
            "guide_lower_mach": float(guide["lower_mach"]),
            "guide_upper_mach": float(guide["upper_mach"]),
            "guide_lower_alpha": float(guide["lower_alpha"]),
            "guide_upper_alpha": float(guide["upper_alpha"]),
            "guide_lower_cr": float(guide["lower_cr"]),
            "guide_lower_ci": float(guide["lower_ci"]),
            "guide_upper_cr": float(guide["upper_cr"]),
            "guide_upper_ci": float(guide["upper_ci"]),
            "guide_lower_source_type": str(guide["lower_source_type"]),
            "guide_upper_source_type": str(guide["upper_source_type"]),
            "guide_lower_source_label": str(guide["lower_source_label"]),
            "guide_upper_source_label": str(guide["upper_source_label"]),
            "guide_interp_weight": float(guide["interp_weight"]),
            "n_seeds": int(len(seeds)),
            "n_candidates": int(len(candidate_rows)),
            "n_success_candidates": int(sum(bool(row["success"]) for row in candidate_rows)),
            "n_spectral_success_candidates": int(sum(bool(row["spectral_success"]) for row in candidate_rows)),
            "n_mode_success_candidates": int(sum(bool(row["mode_success"]) for row in candidate_rows)),
            "best_seed_name": "",
            "best_shooting_cr": np.nan,
            "best_shooting_ci": np.nan,
            "best_shooting_omega_i": np.nan,
            "best_err_cr_abs": np.nan,
            "best_err_ci_abs": np.nan,
            "best_err_ci_rel": np.nan,
            "best_guide_err_cr_abs": np.nan,
            "best_guide_err_ci_abs": np.nan,
            "best_stage1_mismatch": np.nan,
            "best_stage2_mismatch": np.nan,
            "best_ln_p_start_right": np.nan,
            "best_spectral_success": False,
            "best_mode_success": False,
            "best_success": False,
            "best_raw_status": "exception",
            "best_status": "exception",
            "box_rejection_applied": False,
            "best_selection_metric": np.nan,
            "best_selection_metric_name": "",
            "best_retry_index": np.nan,
            "best_used_cr_half_window": np.nan,
            "best_used_ci_half_window": np.nan,
            "best_y_limit": np.nan,
            "centroid_abs_y": np.nan,
            "spread_abs_y": np.nan,
            "peak_y": np.nan,
            "centroid_abs_y_center8": np.nan,
            "spread_abs_y_center8": np.nan,
            "peak_y_center8": np.nan,
            "center8_mass_fraction": np.nan,
            "left_mass_fraction": np.nan,
            "right_mass_fraction": np.nan,
            "left_boundary_amp_fraction": np.nan,
            "right_boundary_amp_fraction": np.nan,
            "edge_amp_fraction_max": np.nan,
            "p_left_boundary_amp_fraction": np.nan,
            "p_right_boundary_amp_fraction": np.nan,
            "p_edge_amp_fraction_max": np.nan,
            "rho_left_boundary_amp_fraction": np.nan,
            "rho_right_boundary_amp_fraction": np.nan,
            "rho_edge_amp_fraction_max": np.nan,
            "u_left_boundary_amp_fraction": np.nan,
            "u_right_boundary_amp_fraction": np.nan,
            "u_edge_amp_fraction_max": np.nan,
            "v_left_boundary_amp_fraction": np.nan,
            "v_right_boundary_amp_fraction": np.nan,
            "v_edge_amp_fraction_max": np.nan,
            "max_field_edge_amp_fraction": np.nan,
            "box_truncation_suspect_p": False,
            "box_truncation_suspect_any_field": False,
            "left_relative_mach": np.nan,
            "left_relative_regime": "",
            "right_relative_mach": np.nan,
            "right_relative_regime": "",
            "box_selection_tested_candidates": 0,
            "box_selection_rejected_candidates": 0,
            "box_selected_rank": np.nan,
            "box_selection_max_candidates": int(cfg.get("box_selection_max_candidates", 12)),
            "box_selection_note": "exception",
            **default_box_robustness_metrics(
                sorted(
                    {
                        float(factor)
                        for factor in cfg.get("box_robustness_factors", [])
                        if np.isfinite(float(factor)) and float(factor) > 1.0 + 1.0e-9
                    }
                )
            ),
            "exception": repr(exc),
        }
        return summary_row, candidate_rows, []


def build_guide_report(
    *,
    mach: float,
    alphas: list[float],
    reference_csv: Path,
    modal_reference_csv: Path,
    alpha_tolerance: float,
    cr_points: Path,
    ci_points: Path,
) -> pd.DataFrame:
    spectral_df = pd.read_csv(reference_csv)
    modal_df = pd.read_csv(modal_reference_csv)
    cr_points_df = load_digitized_long(cr_points)
    ci_points_df = load_digitized_long(ci_points)
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        target_df = build_blumen_targets([float(mach)], float(alpha), cr_points_df, ci_points_df)
        target = target_df.iloc[0]
        lower_bundle, upper_bundle = pick_bracketing_guides(
            spectral_df=spectral_df,
            modal_df=modal_df,
            target_mach=float(mach),
            target_alpha=float(alpha),
            alpha_tolerance=float(alpha_tolerance),
        )
        rows.append(
            build_guided_target(
                target_mach=float(mach),
                target_alpha=float(alpha),
                lower_bundle=lower_bundle,
                upper_bundle=upper_bundle,
                blumen_cr=float(target["blumen_cr"]),
                blumen_ci=float(target["blumen_ci"]),
            )
        )
    return pd.DataFrame(rows).sort_values(["alpha", "Mach"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_alpha_csv(str(args.alphas))
    points = [(float(alpha), float(args.mach)) for alpha in alphas]
    guide_df = build_guide_report(
        mach=float(args.mach),
        alphas=alphas,
        reference_csv=Path(args.reference_csv),
        modal_reference_csv=Path(args.modal_reference_csv),
        alpha_tolerance=float(args.alpha_tolerance),
        cr_points=Path(args.cr_points),
        ci_points=Path(args.ci_points),
    )
    guide_path = output_dir / f"{args.output_stem}_guide_targets.csv"
    guide_df.to_csv(guide_path, index=False)

    print("Supersonic shooting point audit (branch-guided)")
    print(f"points: {' '.join(f'{alpha:.3f}:{args.mach:.3f}' for alpha in alphas)}")
    print(f"workers={int(args.workers)}")
    print(
        f"box: min={float(args.min_y_limit):.1f} max={float(args.max_y_limit):.1f} "
        f"factor={float(args.y_limit_factor):.2f} amp=[{float(args.amp_lower_bound):.1f},{float(args.amp_upper_bound):.1f}]"
    )
    print(
        "box robustness: "
        f"factors={','.join(f'{float(value):.2f}' for value in args.box_robustness_factors)} "
        f"relL2<={float(args.box_robustness_max_rel_l2):.3f} "
        f"peak<={float(args.box_robustness_max_peak_shift):.3f} "
        f"center8<={float(args.box_robustness_max_center8_delta):.3f} "
        f"edge_growth<={float(args.box_robustness_max_edge_growth):.3f} "
        f"selection_max={int(args.box_selection_max_candidates)}"
    )
    print(f"guide-targets={guide_path}")
    if args.dry_run:
        print("\nGuide targets:")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(guide_df.to_string(index=False))
        print(f"Wrote {guide_path}")
        return

    cfg = {
        "reference_csv": str(args.reference_csv),
        "modal_reference_csv": str(args.modal_reference_csv),
        "alpha_tolerance": float(args.alpha_tolerance),
        "match_y": float(args.match_y),
        "use_mapping": bool(args.use_mapping),
        "mapping_scale": float(args.mapping_scale),
        "min_y_limit": float(args.min_y_limit),
        "max_y_limit": float(args.max_y_limit),
        "y_limit_factor": float(args.y_limit_factor),
        "amp_lower_bound": float(args.amp_lower_bound),
        "amp_upper_bound": float(args.amp_upper_bound),
        "cr_half_windows": [float(value) for value in args.cr_half_windows],
        "ci_half_windows": [float(value) for value in args.ci_half_windows],
        "retry_growth": float(args.retry_growth),
        "max_retries": int(args.max_retries),
        "max_iter": int(args.max_iter),
        "grid_size": int(args.grid_size),
        "ci_weight": float(args.ci_weight),
        "cr_weight": float(args.cr_weight),
        "continuity_weight": float(args.continuity_weight),
        "include_generic_seeds": bool(args.include_generic_seeds),
        "edge_amp_threshold": float(args.edge_amp_threshold),
        "box_robustness_factors": [float(value) for value in args.box_robustness_factors],
        "box_robustness_max_rel_l2": float(args.box_robustness_max_rel_l2),
        "box_robustness_max_peak_shift": float(args.box_robustness_max_peak_shift),
        "box_robustness_max_center8_delta": float(args.box_robustness_max_center8_delta),
        "box_robustness_max_edge_growth": float(args.box_robustness_max_edge_growth),
        "box_selection_max_candidates": int(args.box_selection_max_candidates),
        "cr_points": str(args.cr_points),
        "ci_points": str(args.ci_points),
    }

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = {executor.submit(evaluate_point, point, cfg): point for point in points}
        for future in as_completed(futures):
            alpha, mach = futures[future]
            summary_row, point_candidates, point_fields = future.result()
            summary_rows.append(summary_row)
            candidate_rows.extend(point_candidates)
            field_rows.extend(point_fields)
            print(
                f"[point] alpha={alpha:.3f} Mach={mach:.3f} "
                f"status={summary_row['best_status']} "
                f"ci={float(summary_row['best_shooting_ci']) if np.isfinite(summary_row['best_shooting_ci']) else np.nan:.5f} "
                f"cr={float(summary_row['best_shooting_cr']) if np.isfinite(summary_row['best_shooting_cr']) else np.nan:.5f} "
                f"guide_err_cr={float(summary_row['best_guide_err_cr_abs']) if np.isfinite(summary_row['best_guide_err_cr_abs']) else np.nan:.3e} "
                f"guide_err_ci={float(summary_row['best_guide_err_ci_abs']) if np.isfinite(summary_row['best_guide_err_ci_abs']) else np.nan:.3e} "
                f"box_any={bool(summary_row['box_truncation_suspect_any_field'])} "
                f"box_robust={bool(summary_row.get('box_robustness_pass', False))} "
                f"stage1={float(summary_row['best_stage1_mismatch']) if np.isfinite(summary_row['best_stage1_mismatch']) else np.nan:.3e} "
                f"stage2={float(summary_row['best_stage2_mismatch']) if np.isfinite(summary_row['best_stage2_mismatch']) else np.nan:.3e}"
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["alpha", "Mach"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["alpha", "Mach", "success", "selection_metric"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
    fields_df = pd.DataFrame(field_rows)
    if not fields_df.empty:
        fields_df = fields_df.sort_values(["alpha", "Mach", "y"]).reset_index(drop=True)

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    fields_path = output_dir / f"{args.output_stem}_fields.csv"
    points_path = output_dir / f"{args.output_stem}_points.txt"
    status_map_path = output_dir / f"{args.output_stem}_status_map.png"
    diagnostics_path = output_dir / f"{args.output_stem}_diagnostics.png"
    modes_pdf_path = output_dir / f"{args.output_stem}_modes.pdf"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    points_path.write_text("\n".join(f"{alpha:.6f}:{args.mach:.6f}" for alpha in alphas) + "\n", encoding="utf-8")
    plot_status_map(summary_df, status_map_path)
    plot_diagnostics(summary_df, diagnostics_path)
    if not fields_df.empty:
        plot_modes_pdf(
            summary_df,
            fields_df,
            threshold_ratio=float(args.visible_threshold_ratio),
            min_half_width=float(args.visible_min_half_width),
            output_path=modes_pdf_path,
        )

    print("\nGuide targets:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(guide_df.to_string(index=False))

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 280):
        print(summary_df.to_string(index=False))

    print(f"Wrote {guide_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {fields_path}")
    print(f"Wrote {points_path}")
    print(f"Wrote {status_map_path}")
    print(f"Wrote {diagnostics_path}")
    if not fields_df.empty:
        print(f"Wrote {modes_pdf_path}")


if __name__ == "__main__":
    main()
