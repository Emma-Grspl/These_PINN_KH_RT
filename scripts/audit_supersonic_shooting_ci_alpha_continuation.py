from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from scripts.audit_supersonic_shooting_ci_map import (  # noqa: E402
    build_seed_list,
    ci_primary_score,
    extended_profile_diagnostics,
)
from scripts.audit_supersonic_shooting_point_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    boundary_amplitude_metrics,
    dedup_seeds,
    generic_seed_list,
    infer_regimes,
    plot_modes_pdf,
    success_label,
)
from scripts.audit_supersonic_shooting_visual_validation import reconstruct_shooting_fields  # noqa: E402
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


@dataclass(frozen=True)
class LineSpec:
    mach: float
    anchor_alpha: float
    alphas: tuple[float, ...]

    @property
    def line_id(self) -> str:
        return f"M{self.mach:.2f}_anchor{self.anchor_alpha:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuation locale stricte du shooting supersonique pour des lignes c_i(alpha) a Mach fixe, "
            "avec ancre fiable puis propagation sequentielle a gauche/droite."
        )
    )
    parser.add_argument(
        "--line-specs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Spec de ligne sous la forme mach:anchor_alpha:alpha1,alpha2,... "
            "Ex: 1.20:0.20:0.15,0.175,0.20,0.225,0.25"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=500.0)
    parser.add_argument("--y-limit-factor", type=float, default=6.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.015, 0.03])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.008, 0.015])
    parser.add_argument("--retry-growth", type=float, default=1.60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ci-weight", type=float, default=4.0)
    parser.add_argument("--cr-weight", type=float, default=0.35)
    parser.add_argument("--continuity-weight", type=float, default=1.0)
    parser.add_argument("--acceptance-mode", choices=["modal", "spectral"], default="modal")
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument("--max-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-err-ci-abs", type=float, default=1.0e-2)
    parser.add_argument("--max-delta-ci", type=float, default=2.5e-2)
    parser.add_argument("--max-delta-cr", type=float, default=8.0e-2)
    parser.add_argument("--no-anchor-generic-seeds", action="store_false", dest="anchor_include_generic_seeds")
    parser.add_argument("--continuation-generic-seeds", action="store_true", dest="continuation_include_generic_seeds")
    parser.set_defaults(anchor_include_generic_seeds=True, continuation_include_generic_seeds=False)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def parse_line_spec(raw: str) -> LineSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Line spec invalide: {raw!r}")
    mach = float(parts[0])
    anchor_alpha = float(parts[1])
    alpha_values = tuple(sorted({float(item) for item in parts[2].split(",") if item.strip()}))
    if not alpha_values:
        raise ValueError(f"Aucun alpha dans line spec {raw!r}")
    if not any(np.isclose(alpha, anchor_alpha) for alpha in alpha_values):
        raise ValueError(f"anchor_alpha={anchor_alpha:.6f} absent de la liste {raw!r}")
    return LineSpec(mach=mach, anchor_alpha=anchor_alpha, alphas=alpha_values)


def build_cfg(args: argparse.Namespace) -> dict[str, object]:
    return {
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
        "acceptance_mode": str(args.acceptance_mode),
        "edge_amp_threshold": float(args.edge_amp_threshold),
        "max_stage1": float(args.max_stage1),
        "max_err_ci_abs": float(args.max_err_ci_abs),
        "max_delta_ci": float(args.max_delta_ci),
        "max_delta_cr": float(args.max_delta_cr),
        "anchor_include_generic_seeds": bool(args.anchor_include_generic_seeds),
        "continuation_include_generic_seeds": bool(args.continuation_include_generic_seeds),
        "cr_points": str(args.cr_points),
        "ci_points": str(args.ci_points),
    }


def continuity_penalty(
    *,
    shooting_cr: float,
    shooting_ci: float,
    previous_cr: float | None,
    previous_ci: float | None,
    ci_weight: float,
    cr_weight: float,
    continuity_weight: float,
) -> float:
    if previous_cr is None or previous_ci is None:
        return 0.0
    return float(
        continuity_weight
        * np.hypot(
            0.5 * cr_weight * (shooting_cr - previous_cr),
            ci_weight * (shooting_ci - previous_ci),
        )
    )


def continuation_selection_metric(
    *,
    target_available: bool,
    ci_available: bool,
    shooting_cr: float,
    shooting_ci: float,
    blumen_cr: float,
    blumen_ci: float,
    previous_cr: float | None,
    previous_ci: float | None,
    spectral_success: bool,
    mode_success: bool,
    stage1_mismatch: float,
    stage2_mismatch: float,
    ci_weight: float,
    cr_weight: float,
    continuity_weight: float,
) -> tuple[float, str]:
    penalty = 0.05 * float(stage1_mismatch) + 0.001 * float(stage2_mismatch)
    bonus = 0.02 if bool(spectral_success) else 0.0
    bonus += 0.05 if bool(mode_success) else 0.0
    continuity = continuity_penalty(
        shooting_cr=float(shooting_cr),
        shooting_ci=float(shooting_ci),
        previous_cr=previous_cr,
        previous_ci=previous_ci,
        ci_weight=float(ci_weight),
        cr_weight=float(cr_weight),
        continuity_weight=float(continuity_weight),
    )
    if target_available:
        score = ci_primary_score(
            shooting_cr=float(shooting_cr),
            shooting_ci=float(shooting_ci),
            blumen_cr=float(blumen_cr),
            blumen_ci=float(blumen_ci),
            previous_mach=None,
            previous_alpha=None if previous_cr is None or previous_ci is None else (float(previous_cr), float(previous_ci)),
            ci_weight=float(ci_weight),
            cr_weight=float(cr_weight),
            continuity_weight=float(continuity_weight),
        )
        return float(score + penalty - bonus), "distance_to_blumen_plus_previous"
    if ci_available:
        score = float(ci_weight) * abs(float(shooting_ci) - float(blumen_ci))
        return float(score + continuity + penalty - bonus), "distance_to_blumen_ci_plus_previous"
    if previous_cr is not None and previous_ci is not None:
        return float(continuity + penalty - bonus - 0.25 * float(shooting_ci)), "distance_to_previous_fallback"
    return float(-shooting_ci + penalty - bonus), "max_ci_anchor_fallback"


def preferred_success_flag(row: dict[str, object], *, acceptance_mode: str) -> bool:
    if str(acceptance_mode) == "spectral":
        return bool(row["spectral_success"])
    return bool(row["success"])


def build_seeds_for_step(
    *,
    blumen_cr: float,
    blumen_ci: float,
    target_available: bool,
    ci_available: bool,
    previous_cr: float | None,
    previous_ci: float | None,
    include_generic_seeds: bool,
) -> list[tuple[str, float, float]]:
    seeds: list[tuple[str, float, float]] = []
    if previous_cr is None or previous_ci is None:
        if target_available:
            seeds.extend(
                build_seed_list(
                    blumen_cr=float(blumen_cr),
                    blumen_ci=float(blumen_ci),
                    previous_mach=None,
                    previous_alpha=None,
                )
            )
        if include_generic_seeds:
            seeds.extend(generic_seed_list())
        return dedup_seeds(seeds)

    seeds.append(("previous", float(previous_cr), float(previous_ci)))
    if target_available:
        seeds.append(("blumen", float(blumen_cr), float(blumen_ci)))
        seeds.append(
            (
                "blend_previous_blumen",
                0.5 * (float(previous_cr) + float(blumen_cr)),
                0.5 * (float(previous_ci) + float(blumen_ci)),
            )
        )
    elif ci_available:
        seeds.append(("previous_cr_target_ci", float(previous_cr), float(blumen_ci)))
        seeds.append(("previous_cr_blend_ci", float(previous_cr), 0.5 * (float(previous_ci) + float(blumen_ci))))
    if include_generic_seeds:
        seeds.extend(generic_seed_list())
    return dedup_seeds(seeds)


def candidate_row_base(
    *,
    line: LineSpec,
    alpha: float,
    target: pd.Series,
    direction: str,
    step_index: int,
    previous_alpha: float | None,
    previous_cr: float | None,
    previous_ci: float | None,
) -> dict[str, object]:
    blumen_cr = float(target["blumen_cr"])
    blumen_ci = float(target["blumen_ci"])
    ci_available = bool(np.isfinite(blumen_ci))
    cr_available = bool(np.isfinite(blumen_cr))
    target_available = bool(ci_available and cr_available)
    return {
        "line_id": line.line_id,
        "anchor_alpha": float(line.anchor_alpha),
        "alpha": float(alpha),
        "Mach": float(line.mach),
        "continuation_direction": str(direction),
        "continuation_step_index": int(step_index),
        "continuation_prev_alpha": np.nan if previous_alpha is None else float(previous_alpha),
        "continuation_prev_cr": np.nan if previous_cr is None else float(previous_cr),
        "continuation_prev_ci": np.nan if previous_ci is None else float(previous_ci),
        "blumen_cr": blumen_cr,
        "blumen_ci": blumen_ci,
        "blumen_cr_available": bool(cr_available),
        "blumen_ci_available": bool(ci_available),
        "blumen_target_available": bool(target_available),
    }


def evaluate_step(
    *,
    line: LineSpec,
    alpha: float,
    target: pd.Series,
    cfg: dict[str, object],
    direction: str,
    step_index: int,
    previous_alpha: float | None,
    previous_cr: float | None,
    previous_ci: float | None,
    include_generic_seeds: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    base = candidate_row_base(
        line=line,
        alpha=float(alpha),
        target=target,
        direction=direction,
        step_index=step_index,
        previous_alpha=previous_alpha,
        previous_cr=previous_cr,
        previous_ci=previous_ci,
    )
    blumen_cr = float(base["blumen_cr"])
    blumen_ci = float(base["blumen_ci"])
    ci_available = bool(base["blumen_ci_available"])
    cr_available = bool(base["blumen_cr_available"])
    target_available = bool(base["blumen_target_available"])

    seeds = build_seeds_for_step(
        blumen_cr=blumen_cr,
        blumen_ci=blumen_ci,
        target_available=target_available,
        ci_available=ci_available,
        previous_cr=previous_cr,
        previous_ci=previous_ci,
        include_generic_seeds=include_generic_seeds,
    )
    if not seeds:
        raise RuntimeError(f"Aucune seed disponible pour line={line.line_id}, alpha={alpha:.3f}")

    candidate_rows: list[dict[str, object]] = []
    for seed_name, cr_center, ci_center in seeds:
        for cr_half in cfg["cr_half_windows"]:
            for ci_half in cfg["ci_half_windows"]:
                solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                    alpha=float(alpha),
                    mach=float(line.mach),
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
                metric_value, metric_name = continuation_selection_metric(
                    target_available=bool(target_available),
                    ci_available=bool(ci_available),
                    shooting_cr=float(result.cr),
                    shooting_ci=float(result.ci),
                    blumen_cr=blumen_cr,
                    blumen_ci=blumen_ci,
                    previous_cr=previous_cr,
                    previous_ci=previous_ci,
                    spectral_success=bool(result.spectral_success),
                    mode_success=bool(result.mode_success),
                    stage1_mismatch=float(result.stage1_mismatch),
                    stage2_mismatch=float(result.stage2_mismatch),
                    ci_weight=float(cfg["ci_weight"]),
                    cr_weight=float(cfg["cr_weight"]),
                    continuity_weight=float(cfg["continuity_weight"]),
                )
                candidate_rows.append(
                    {
                        **base,
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
                        "delta_cr_prev": (
                            abs(float(result.cr) - float(previous_cr)) if previous_cr is not None else np.nan
                        ),
                        "delta_ci_prev": (
                            abs(float(result.ci) - float(previous_ci)) if previous_ci is not None else np.nan
                        ),
                        "stage1_mismatch": float(result.stage1_mismatch),
                        "stage2_mismatch": float(result.stage2_mismatch),
                        "ln_p_start_right": float(result.ln_p_start_right),
                        "spectral_success": bool(result.spectral_success),
                        "mode_success": bool(result.mode_success),
                        "success": bool(result.success),
                        "status": success_label(bool(result.spectral_success), bool(result.mode_success)),
                        "selection_metric": float(metric_value),
                        "selection_metric_name": str(metric_name),
                        "success_priority": (
                            0
                            if preferred_success_flag(
                                {
                                    "spectral_success": bool(result.spectral_success),
                                    "success": bool(result.success),
                                },
                                acceptance_mode=str(cfg["acceptance_mode"]),
                            )
                            else 1
                        ),
                        "y_limit": float(result.y_limit),
                    }
                )

    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            0 if preferred_success_flag(row, acceptance_mode=str(cfg["acceptance_mode"])) else 1,
            float(row["selection_metric"]),
            float(row["stage1_mismatch"] + row["stage2_mismatch"]),
        ),
    )
    best = ranked[0]
    fields = reconstruct_shooting_fields(
        alpha=float(alpha),
        mach=float(line.mach),
        cr=float(best["shooting_cr"]),
        ci=float(best["shooting_ci"]),
        ln_p_start_right=float(best["ln_p_start_right"]),
        match_y=float(cfg["match_y"]),
        use_mapping=bool(cfg["use_mapping"]),
        mapping_scale=float(cfg["mapping_scale"]),
        min_y_limit=float(cfg["min_y_limit"]),
        max_y_limit=float(cfg["max_y_limit"]),
        y_limit_factor=float(cfg["y_limit_factor"]),
    )
    y_fields = np.asarray(fields["y"], dtype=float)
    p_fields = np.asarray(fields["p"], dtype=np.complex128)
    rho_fields = np.asarray(fields["rho"], dtype=np.complex128)
    u_fields = np.asarray(fields["u"], dtype=np.complex128)
    v_fields = np.asarray(fields["v"], dtype=np.complex128)
    diag = extended_profile_diagnostics(y_fields, p_fields)
    p_boundary = boundary_amplitude_metrics(p_fields, prefix="p")
    rho_boundary = boundary_amplitude_metrics(rho_fields, prefix="rho")
    u_boundary = boundary_amplitude_metrics(u_fields, prefix="u")
    v_boundary = boundary_amplitude_metrics(v_fields, prefix="v")
    diag.update(p_boundary)
    diag.update(rho_boundary)
    diag.update(u_boundary)
    diag.update(v_boundary)
    diag.update(infer_regimes(mach=float(line.mach), cr=float(best["shooting_cr"]), ci=float(best["shooting_ci"])))
    diag["edge_amp_fraction_max"] = float(p_boundary["p_edge_amp_fraction_max"])
    diag["left_boundary_amp_fraction"] = float(p_boundary["p_left_boundary_amp_fraction"])
    diag["right_boundary_amp_fraction"] = float(p_boundary["p_right_boundary_amp_fraction"])
    diag["max_field_edge_amp_fraction"] = float(
        max(
            p_boundary["p_edge_amp_fraction_max"],
            rho_boundary["rho_edge_amp_fraction_max"],
            u_boundary["u_edge_amp_fraction_max"],
            v_boundary["v_edge_amp_fraction_max"],
        )
    )
    diag["box_truncation_suspect_p"] = bool(float(p_boundary["p_edge_amp_fraction_max"]) > float(cfg["edge_amp_threshold"]))
    diag["box_truncation_suspect_any_field"] = bool(
        float(diag["max_field_edge_amp_fraction"]) > float(cfg["edge_amp_threshold"])
    )

    summary_row = {
        **base,
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
        "best_delta_cr_prev": float(best["delta_cr_prev"]) if previous_cr is not None else np.nan,
        "best_delta_ci_prev": float(best["delta_ci_prev"]) if previous_ci is not None else np.nan,
        "best_stage1_mismatch": float(best["stage1_mismatch"]),
        "best_stage2_mismatch": float(best["stage2_mismatch"]),
        "best_ln_p_start_right": float(best["ln_p_start_right"]),
        "best_spectral_success": bool(best["spectral_success"]),
        "best_mode_success": bool(best["mode_success"]),
        "best_success": bool(best["success"]),
        "best_status": str(best["status"]),
        "best_selection_metric": float(best["selection_metric"]),
        "best_selection_metric_name": str(best["selection_metric_name"]),
        "best_retry_index": int(best["retry_index"]),
        "best_used_cr_half_window": float(best["used_cr_half_window"]),
        "best_used_ci_half_window": float(best["used_ci_half_window"]),
        "best_y_limit": float(best["y_limit"]),
        **diag,
        "continuation_anchor": bool(np.isclose(alpha, line.anchor_alpha)),
        "continuation_accepted": False,
        "continuation_state": "",
        "continuation_stop_reason": "",
        "acceptance_mode": str(cfg["acceptance_mode"]),
        "exception": "",
    }

    field_rows: list[dict[str, object]] = []
    for y_value, rho_value, u_value, v_value, p_value in zip(
        fields["y"], fields["rho"], fields["u"], fields["v"], fields["p"]
    ):
        field_rows.append(
            {
                "line_id": line.line_id,
                "alpha": float(alpha),
                "Mach": float(line.mach),
                "best_status": str(best["status"]),
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


def continuation_acceptance(
    summary_row: dict[str, object],
    *,
    cfg: dict[str, object],
) -> tuple[bool, str]:
    acceptance_mode = str(cfg["acceptance_mode"])
    if acceptance_mode == "modal":
        if str(summary_row["best_status"]) != "validated":
            return False, "status_not_validated"
    elif acceptance_mode == "spectral":
        if not bool(summary_row["best_spectral_success"]):
            return False, "spectral_not_validated"
    else:
        raise ValueError(f"acceptance_mode inconnu: {acceptance_mode!r}")
    if not np.isfinite(summary_row["best_stage1_mismatch"]) or float(summary_row["best_stage1_mismatch"]) > float(cfg["max_stage1"]):
        return False, "stage1_too_large"
    if bool(summary_row["box_truncation_suspect_any_field"]):
        return False, "box_truncation_suspect"
    if bool(summary_row["blumen_ci_available"]) and np.isfinite(summary_row["best_err_ci_abs"]):
        if float(summary_row["best_err_ci_abs"]) > float(cfg["max_err_ci_abs"]):
            return False, "ci_error_too_large"
    if np.isfinite(summary_row["best_delta_ci_prev"]) and float(summary_row["best_delta_ci_prev"]) > float(cfg["max_delta_ci"]):
        return False, "delta_ci_too_large"
    if np.isfinite(summary_row["best_delta_cr_prev"]) and float(summary_row["best_delta_cr_prev"]) > float(cfg["max_delta_cr"]):
        return False, "delta_cr_too_large"
    return True, ""


def not_run_row(
    *,
    line: LineSpec,
    alpha: float,
    direction: str,
    step_index: int,
    previous_alpha: float | None,
    previous_cr: float | None,
    previous_ci: float | None,
    target: pd.Series,
    stop_reason: str,
    cfg: dict[str, object],
) -> dict[str, object]:
    base = candidate_row_base(
        line=line,
        alpha=float(alpha),
        target=target,
        direction=direction,
        step_index=step_index,
        previous_alpha=previous_alpha,
        previous_cr=previous_cr,
        previous_ci=previous_ci,
    )
    return {
        **base,
        "n_seeds": 0,
        "n_candidates": 0,
        "n_success_candidates": 0,
        "n_spectral_success_candidates": 0,
        "n_mode_success_candidates": 0,
        "best_seed_name": "",
        "best_shooting_cr": np.nan,
        "best_shooting_ci": np.nan,
        "best_shooting_omega_i": np.nan,
        "best_err_cr_abs": np.nan,
        "best_err_ci_abs": np.nan,
        "best_err_ci_rel": np.nan,
        "best_delta_cr_prev": np.nan,
        "best_delta_ci_prev": np.nan,
        "best_stage1_mismatch": np.nan,
        "best_stage2_mismatch": np.nan,
        "best_ln_p_start_right": np.nan,
        "best_spectral_success": False,
        "best_mode_success": False,
        "best_success": False,
        "best_status": "not_run",
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
        "continuation_anchor": bool(np.isclose(alpha, line.anchor_alpha)),
        "continuation_accepted": False,
        "continuation_state": "not_run_after_reject",
        "continuation_stop_reason": str(stop_reason),
        "acceptance_mode": str(cfg["acceptance_mode"]),
        "exception": "",
    }


def target_lookup_for_line(line: LineSpec, cr_points: pd.DataFrame, ci_points: pd.DataFrame) -> dict[float, pd.Series]:
    targets: dict[float, pd.Series] = {}
    for alpha in line.alphas:
        target_df = build_blumen_targets([float(line.mach)], float(alpha), cr_points, ci_points)
        targets[float(alpha)] = target_df.iloc[0]
    return targets


def evaluate_line(line: LineSpec, cfg: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cr_points = load_digitized_long(Path(str(cfg["cr_points"])))
    ci_points = load_digitized_long(Path(str(cfg["ci_points"])))
    targets = target_lookup_for_line(line, cr_points, ci_points)

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []

    alpha_values = list(line.alphas)
    anchor_index = next(idx for idx, alpha in enumerate(alpha_values) if np.isclose(alpha, line.anchor_alpha))

    anchor_row, anchor_candidates, anchor_fields = evaluate_step(
        line=line,
        alpha=float(line.anchor_alpha),
        target=targets[float(line.anchor_alpha)],
        cfg=cfg,
        direction="anchor",
        step_index=0,
        previous_alpha=None,
        previous_cr=None,
        previous_ci=None,
        include_generic_seeds=bool(cfg["anchor_include_generic_seeds"]),
    )
    anchor_ok, anchor_reason = continuation_acceptance(anchor_row, cfg=cfg)
    anchor_row["continuation_accepted"] = bool(anchor_ok)
    anchor_row["continuation_state"] = "anchor_accepted" if anchor_ok else "anchor_rejected"
    anchor_row["continuation_stop_reason"] = "" if anchor_ok else str(anchor_reason)
    summary_rows.append(anchor_row)
    candidate_rows.extend(anchor_candidates)
    field_rows.extend(anchor_fields)

    if not anchor_ok:
        for direction, indices in (
            ("left", range(anchor_index - 1, -1, -1)),
            ("right", range(anchor_index + 1, len(alpha_values))),
        ):
            for offset, idx in enumerate(indices, start=1):
                alpha = float(alpha_values[idx])
                summary_rows.append(
                    not_run_row(
                        line=line,
                        alpha=alpha,
                        direction=direction,
                        step_index=offset,
                        previous_alpha=float(line.anchor_alpha),
                        previous_cr=float(anchor_row["best_shooting_cr"]) if np.isfinite(anchor_row["best_shooting_cr"]) else None,
                        previous_ci=float(anchor_row["best_shooting_ci"]) if np.isfinite(anchor_row["best_shooting_ci"]) else None,
                        target=targets[alpha],
                        stop_reason=f"anchor_rejected:{anchor_reason}",
                        cfg=cfg,
                    )
                )
        return summary_rows, candidate_rows, field_rows

    for direction, indices in (
        ("left", range(anchor_index - 1, -1, -1)),
        ("right", range(anchor_index + 1, len(alpha_values))),
    ):
        previous_alpha = float(line.anchor_alpha)
        previous_cr = float(anchor_row["best_shooting_cr"])
        previous_ci = float(anchor_row["best_shooting_ci"])
        stopped = False
        stop_reason = ""

        for offset, idx in enumerate(indices, start=1):
            alpha = float(alpha_values[idx])
            if stopped:
                summary_rows.append(
                    not_run_row(
                        line=line,
                        alpha=alpha,
                        direction=direction,
                        step_index=offset,
                        previous_alpha=previous_alpha,
                        previous_cr=previous_cr,
                        previous_ci=previous_ci,
                        target=targets[alpha],
                        stop_reason=stop_reason,
                        cfg=cfg,
                    )
                )
                continue

            step_row, step_candidates, step_fields = evaluate_step(
                line=line,
                alpha=alpha,
                target=targets[alpha],
                cfg=cfg,
                direction=direction,
                step_index=offset,
                previous_alpha=previous_alpha,
                previous_cr=previous_cr,
                previous_ci=previous_ci,
                include_generic_seeds=bool(cfg["continuation_include_generic_seeds"]),
            )
            accepted, reason = continuation_acceptance(step_row, cfg=cfg)
            step_row["continuation_accepted"] = bool(accepted)
            step_row["continuation_state"] = "continued_accepted" if accepted else "continued_rejected"
            step_row["continuation_stop_reason"] = "" if accepted else str(reason)
            summary_rows.append(step_row)
            candidate_rows.extend(step_candidates)
            field_rows.extend(step_fields)

            if accepted:
                previous_alpha = alpha
                previous_cr = float(step_row["best_shooting_cr"])
                previous_ci = float(step_row["best_shooting_ci"])
            else:
                stopped = True
                stop_reason = f"{direction}_stopped_at_alpha={alpha:.3f}:{reason}"

    summary_rows = sorted(summary_rows, key=lambda row: (float(row["Mach"]), float(row["alpha"])))
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            str(row["line_id"]),
            float(row["alpha"]),
            0 if bool(row["success"]) else 1,
            float(row["selection_metric"]),
        ),
    )
    field_rows = sorted(field_rows, key=lambda row: (str(row["line_id"]), float(row["alpha"]), float(row["y"])))
    return summary_rows, candidate_rows, field_rows


def plot_continuation_lines(summary_df: pd.DataFrame, output_path: Path) -> None:
    machs = sorted(float(value) for value in summary_df["Mach"].dropna().unique())
    ncols = 2 if len(machs) > 1 else 1
    nrows = int(np.ceil(len(machs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.2 * nrows), squeeze=False, sharex=True)
    state_colors = {
        "anchor_accepted": "#15803D",
        "continued_accepted": "#2563EB",
        "continued_rejected": "#DC2626",
        "anchor_rejected": "#7F1D1D",
        "not_run_after_reject": "#9CA3AF",
    }
    for ax, mach in zip(axes.ravel(), machs):
        sub = summary_df[np.isclose(summary_df["Mach"].to_numpy(dtype=float), mach)].sort_values("alpha")
        if sub.empty:
            ax.set_visible(False)
            continue
        alpha = sub["alpha"].to_numpy(dtype=float)
        ci_blumen = sub["blumen_ci"].to_numpy(dtype=float)
        ci_shoot = sub["best_shooting_ci"].to_numpy(dtype=float)
        ax.plot(alpha, ci_shoot, color="#111827", linewidth=1.5, label="shooting")
        finite_ci = np.isfinite(ci_blumen)
        if np.any(finite_ci):
            ax.plot(alpha[finite_ci], ci_blumen[finite_ci], color="#2563EB", linestyle="--", linewidth=1.3, label="Blumen $c_i$")
        for state, sub_state in sub.groupby("continuation_state", sort=False):
            finite = np.isfinite(sub_state["best_shooting_ci"].to_numpy(dtype=float))
            ax.scatter(
                sub_state["alpha"].to_numpy(dtype=float)[finite],
                sub_state["best_shooting_ci"].to_numpy(dtype=float)[finite],
                s=46,
                color=state_colors.get(str(state), "#4B5563"),
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
                label=str(state) if str(state) not in ax.get_legend_handles_labels()[1] else None,
            )
            not_run = sub_state[~np.isfinite(sub_state["best_shooting_ci"].to_numpy(dtype=float))]
            if not not_run.empty:
                ax.scatter(
                    not_run["alpha"],
                    np.zeros(len(not_run)),
                    marker="x",
                    s=40,
                    color=state_colors.get(str(state), "#4B5563"),
                    linewidths=1.1,
                    zorder=4,
                    label=str(state) if str(state) not in ax.get_legend_handles_labels()[1] else None,
                )
        ax.set_title(f"Mach = {mach:.2f}")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$c_i$")
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes.ravel()[len(machs) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=True)
    acceptance_mode = str(summary_df["acceptance_mode"].dropna().iloc[0]) if "acceptance_mode" in summary_df.columns and not summary_df.empty else "modal"
    title = "strict local continuation" if acceptance_mode == "modal" else "spectral-only local continuation"
    fig.suptitle(rf"Supersonic shooting: {title} for $c_i(\alpha)$", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_continuation_errors(summary_df: pd.DataFrame, output_path: Path) -> None:
    machs = sorted(float(value) for value in summary_df["Mach"].dropna().unique())
    ncols = 2 if len(machs) > 1 else 1
    nrows = int(np.ceil(len(machs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.2 * nrows), squeeze=False, sharex=True)
    for ax, mach in zip(axes.ravel(), machs):
        sub = summary_df[np.isclose(summary_df["Mach"].to_numpy(dtype=float), mach)].sort_values("alpha")
        if sub.empty:
            ax.set_visible(False)
            continue
        alpha = sub["alpha"].to_numpy(dtype=float)
        err_abs = sub["best_err_ci_abs"].to_numpy(dtype=float)
        finite = np.isfinite(err_abs)
        if np.any(finite):
            ax.plot(alpha[finite], err_abs[finite], color="#B45309", marker="o", linewidth=1.5, markersize=4.5)
        else:
            ax.text(0.5, 0.5, "No Blumen $c_i$ on this line", ha="center", va="center", transform=ax.transAxes)
        ax.axhline(1.0e-2, color="#DC2626", linestyle="--", linewidth=1.0, alpha=0.8, label="guard")
        ax.set_title(f"Mach = {mach:.2f}")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$|c_i^{shoot} - c_i^{Blumen}|$")
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes.ravel()[len(machs) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=True)
    acceptance_mode = str(summary_df["acceptance_mode"].dropna().iloc[0]) if "acceptance_mode" in summary_df.columns and not summary_df.empty else "modal"
    title = "strict continuation" if acceptance_mode == "modal" else "spectral-only continuation"
    fig.suptitle(rf"Supersonic shooting: {title} $c_i$ error", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    line_specs = [parse_line_spec(raw) for raw in args.line_specs]
    cfg = build_cfg(args)

    mode_label = "strict local continuation" if str(args.acceptance_mode) == "modal" else "spectral-only local continuation"
    print(f"Supersonic shooting {mode_label}")
    for line in line_specs:
        print(
            f"line={line.line_id} mach={line.mach:.3f} anchor={line.anchor_alpha:.3f} "
            f"alphas={' '.join(f'{alpha:.3f}' for alpha in line.alphas)}"
        )
    print(f"workers={int(args.workers)}")
    print(
        f"box: min={float(args.min_y_limit):.1f} max={float(args.max_y_limit):.1f} "
        f"factor={float(args.y_limit_factor):.2f} amp=[{float(args.amp_lower_bound):.1f},{float(args.amp_upper_bound):.1f}]"
    )
    print(
        f"guards: stage1<={float(args.max_stage1):.3e} "
        f"err_ci<={float(args.max_err_ci_abs):.3e} "
        f"delta_ci<={float(args.max_delta_ci):.3e} "
        f"delta_cr<={float(args.max_delta_cr):.3e}"
    )
    print(f"acceptance-mode={args.acceptance_mode}")

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = {executor.submit(evaluate_line, line, cfg): line for line in line_specs}
        for future in as_completed(futures):
            line = futures[future]
            line_summary, line_candidates, line_fields = future.result()
            summary_rows.extend(line_summary)
            candidate_rows.extend(line_candidates)
            field_rows.extend(line_fields)
            accepted = sum(bool(row["continuation_accepted"]) for row in line_summary if row["continuation_state"] != "not_run_after_reject")
            print(f"[line] {line.line_id} completed | accepted={accepted}/{len(line_summary)}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    fields_df = pd.DataFrame(field_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["Mach", "alpha", "continuation_direction", "continuation_step_index", "success_priority", "selection_metric"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)
    if not fields_df.empty:
        fields_df = fields_df.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    fields_path = output_dir / f"{args.output_stem}_fields.csv"
    lines_path = output_dir / f"{args.output_stem}_ci_alpha_lines.png"
    errors_path = output_dir / f"{args.output_stem}_ci_alpha_errors.png"
    modes_path = output_dir / f"{args.output_stem}_modes.pdf"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    plot_continuation_lines(summary_df, lines_path)
    plot_continuation_errors(summary_df, errors_path)
    if not fields_df.empty:
        plot_modes_pdf(
            summary_df,
            fields_df,
            threshold_ratio=0.02,
            min_half_width=8.0,
            output_path=modes_path,
        )

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 320):
        print(summary_df.to_string(index=False))

    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {fields_path}")
    print(f"Wrote {lines_path}")
    print(f"Wrote {errors_path}")
    if not fields_df.empty:
        print(f"Wrote {modes_path}")


if __name__ == "__main__":
    main()
