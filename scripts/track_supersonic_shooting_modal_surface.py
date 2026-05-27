from __future__ import annotations

import argparse
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
)
from scripts.audit_supersonic_shooting_ci_map import extended_profile_diagnostics  # noqa: E402
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


DEFAULT_POINTS_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral.csv"
DEFAULT_ANCHORS_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal.csv"
DEFAULT_ANCHOR_FIELDS_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_fields.csv"


@dataclass(frozen=True)
class PointKey:
    mach: float
    alpha: float


@dataclass
class AcceptedState:
    summary_row: dict[str, object]
    field_bundle: dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuation modale 2D du shooting supersonique a partir d'ancres modales fiables "
            "et d'une reference spectrale sur une grille (alpha, Mach)."
        )
    )
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    parser.add_argument("--anchor-fields-csv", type=Path, default=DEFAULT_ANCHOR_FIELDS_CSV)
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
    parser.add_argument("--mode-distance-weight", type=float, default=0.60)
    parser.add_argument("--mode-mask-ratio", type=float, default=0.15)
    parser.add_argument("--mode-common-points", type=int, default=801)
    parser.add_argument("--max-mode-distance", type=float, default=0.35)
    parser.add_argument("--max-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-stage2", type=float, default=1.0e-1)
    parser.add_argument("--max-err-ci-abs", type=float, default=1.0e-2)
    parser.add_argument("--max-delta-ci", type=float, default=2.5e-2)
    parser.add_argument("--max-delta-cr", type=float, default=8.0e-2)
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument("--include-generic-seeds", action="store_true")
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting_modal_surface_core")
    return parser


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
        "mode_distance_weight": float(args.mode_distance_weight),
        "mode_mask_ratio": float(args.mode_mask_ratio),
        "mode_common_points": int(args.mode_common_points),
        "max_mode_distance": float(args.max_mode_distance),
        "max_stage1": float(args.max_stage1),
        "max_stage2": float(args.max_stage2),
        "max_err_ci_abs": float(args.max_err_ci_abs),
        "max_delta_ci": float(args.max_delta_ci),
        "max_delta_cr": float(args.max_delta_cr),
        "edge_amp_threshold": float(args.edge_amp_threshold),
        "include_generic_seeds": bool(args.include_generic_seeds),
        "cr_points": str(args.cr_points),
        "ci_points": str(args.ci_points),
    }


def point_key(row: pd.Series) -> PointKey:
    return PointKey(mach=float(row["Mach"]), alpha=float(row["alpha"]))


def continuity_penalty(
    *,
    shooting_cr: float,
    shooting_ci: float,
    previous_cr: float,
    previous_ci: float,
    ci_weight: float,
    cr_weight: float,
    continuity_weight: float,
) -> float:
    return float(
        continuity_weight
        * np.hypot(
            0.5 * cr_weight * (shooting_cr - previous_cr),
            ci_weight * (shooting_ci - previous_ci),
        )
    )


def build_neighbors(points_df: pd.DataFrame) -> dict[PointKey, set[PointKey]]:
    neighbors: dict[PointKey, set[PointKey]] = {}
    work = points_df[["Mach", "alpha"]].drop_duplicates().copy()
    for _, row in work.iterrows():
        neighbors[point_key(row)] = set()

    for mach, sub in work.groupby("Mach", sort=True):
        ordered = sub.sort_values("alpha").reset_index(drop=True)
        keys = [point_key(row) for _, row in ordered.iterrows()]
        for left, right in zip(keys[:-1], keys[1:]):
            neighbors[left].add(right)
            neighbors[right].add(left)

    for alpha, sub in work.groupby("alpha", sort=True):
        ordered = sub.sort_values("Mach").reset_index(drop=True)
        keys = [point_key(row) for _, row in ordered.iterrows()]
        for lower, upper in zip(keys[:-1], keys[1:]):
            neighbors[lower].add(upper)
            neighbors[upper].add(lower)

    return neighbors


def complex_interp(x_new: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    real = np.interp(x_new, x, np.real(z))
    imag = np.interp(x_new, x, np.imag(z))
    return real + 1j * imag


def normalize_mode(p: np.ndarray) -> np.ndarray:
    amp = np.abs(p)
    peak = int(np.argmax(amp))
    peak_amp = float(amp[peak])
    if peak_amp <= 0.0:
        return p.copy()
    phase = np.angle(p[peak])
    return p * np.exp(-1j * phase) / peak_amp


def mode_distance(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    mask_ratio: float,
    common_points: int,
) -> dict[str, float]:
    y_ref = np.asarray(reference["y"], dtype=float)
    p_ref = np.asarray(reference["p"], dtype=np.complex128)
    y_cand = np.asarray(candidate["y"], dtype=float)
    p_cand = np.asarray(candidate["p"], dtype=np.complex128)

    overlap_min = max(float(np.min(y_ref)), float(np.min(y_cand)))
    overlap_max = min(float(np.max(y_ref)), float(np.max(y_cand)))
    if not np.isfinite(overlap_min) or not np.isfinite(overlap_max) or overlap_max <= overlap_min:
        return {"mode_distance": 1.0e9, "env_distance": 1.0e9, "phase_distance": 1.0e9, "shape_distance": 1.0e9}

    y_common = np.linspace(overlap_min, overlap_max, int(common_points))
    p_ref_common = normalize_mode(complex_interp(y_common, y_ref, p_ref))
    p_cand_common = normalize_mode(complex_interp(y_common, y_cand, p_cand))

    amp_ref = np.abs(p_ref_common)
    amp_cand = np.abs(p_cand_common)
    amp_peak = float(np.max(amp_ref))
    mask = amp_ref >= float(mask_ratio) * max(amp_peak, 1.0e-12)
    if not np.any(mask):
        mask = amp_ref >= 0.05 * max(amp_peak, 1.0e-12)
    if not np.any(mask):
        return {"mode_distance": 1.0e9, "env_distance": 1.0e9, "phase_distance": 1.0e9, "shape_distance": 1.0e9}

    env_distance = float(np.sqrt(np.mean((amp_cand[mask] - amp_ref[mask]) ** 2)))
    phase_diff = np.angle(p_cand_common[mask] * np.conj(p_ref_common[mask]))
    phase_distance = float(np.sqrt(np.mean(phase_diff**2)))

    gamma_ref = np.gradient(p_ref_common, y_common) / np.where(np.abs(p_ref_common) > 1.0e-4, p_ref_common, np.nan + 0j)
    gamma_cand = np.gradient(p_cand_common, y_common) / np.where(np.abs(p_cand_common) > 1.0e-4, p_cand_common, np.nan + 0j)
    gamma_diff = np.abs(gamma_cand - gamma_ref)
    gamma_distance = float(np.sqrt(np.nanmean(gamma_diff[mask] ** 2))) if np.any(np.isfinite(gamma_diff[mask])) else 5.0
    gamma_distance = min(gamma_distance, 5.0)

    diag_ref = extended_profile_diagnostics(y_common, p_ref_common)
    diag_cand = extended_profile_diagnostics(y_common, p_cand_common)
    spread_scale = max(float(diag_ref["spread_abs_y"]), 1.0)
    shape_terms = [
        abs(float(diag_cand["centroid_abs_y"]) - float(diag_ref["centroid_abs_y"])) / spread_scale,
        abs(float(diag_cand["spread_abs_y"]) - float(diag_ref["spread_abs_y"])) / spread_scale,
        abs(float(diag_cand["left_mass_fraction"]) - float(diag_ref["left_mass_fraction"])),
        abs(float(diag_cand["right_mass_fraction"]) - float(diag_ref["right_mass_fraction"])),
        abs(float(diag_cand["center8_mass_fraction"]) - float(diag_ref["center8_mass_fraction"])),
    ]
    shape_distance = float(np.mean(shape_terms))
    total = float(2.0 * env_distance + 0.35 * phase_distance + 0.15 * gamma_distance + 0.5 * shape_distance)
    return {
        "mode_distance": total,
        "env_distance": env_distance,
        "phase_distance": phase_distance,
        "shape_distance": shape_distance,
    }


def bundle_from_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "y": np.asarray(fields["y"], dtype=float),
        "rho": np.asarray(fields["rho"], dtype=np.complex128),
        "u": np.asarray(fields["u"], dtype=np.complex128),
        "v": np.asarray(fields["v"], dtype=np.complex128),
        "p": np.asarray(fields["p"], dtype=np.complex128),
    }


def bundle_from_dataframe(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y": df["y"].to_numpy(dtype=float),
        "rho": df["rho_real"].to_numpy(dtype=float) + 1j * df["rho_imag"].to_numpy(dtype=float),
        "u": df["u_real"].to_numpy(dtype=float) + 1j * df["u_imag"].to_numpy(dtype=float),
        "v": df["v_real"].to_numpy(dtype=float) + 1j * df["v_imag"].to_numpy(dtype=float),
        "p": df["p_real"].to_numpy(dtype=float) + 1j * df["p_imag"].to_numpy(dtype=float),
    }


def build_seeds(
    *,
    target_cr: float,
    target_ci: float,
    previous_cr: float,
    previous_ci: float,
    include_generic_seeds: bool,
) -> list[tuple[str, float, float]]:
    seeds = [
        ("previous", float(previous_cr), float(previous_ci)),
        ("target", float(target_cr), float(target_ci)),
        ("blend_previous_target", 0.5 * (float(previous_cr) + float(target_cr)), 0.5 * (float(previous_ci) + float(target_ci))),
        ("previous_cr_target_ci", float(previous_cr), float(target_ci)),
    ]
    if include_generic_seeds:
        seeds.extend(generic_seed_list())
    return dedup_seeds(seeds)


def target_metric(
    *,
    shooting_cr: float,
    shooting_ci: float,
    target_cr: float,
    target_ci: float,
    previous_cr: float,
    previous_ci: float,
    stage1_mismatch: float,
    stage2_mismatch: float,
    mode_distance_value: float,
    cfg: dict[str, object],
) -> float:
    spectral_term = np.hypot(
        float(cfg["ci_weight"]) * (float(shooting_ci) - float(target_ci)),
        0.5 * float(cfg["cr_weight"]) * (float(shooting_cr) - float(target_cr)),
    )
    continuity = continuity_penalty(
        shooting_cr=float(shooting_cr),
        shooting_ci=float(shooting_ci),
        previous_cr=float(previous_cr),
        previous_ci=float(previous_ci),
        ci_weight=float(cfg["ci_weight"]),
        cr_weight=float(cfg["cr_weight"]),
        continuity_weight=float(cfg["continuity_weight"]),
    )
    return float(
        spectral_term
        + continuity
        + float(cfg["mode_distance_weight"]) * float(mode_distance_value)
        + 0.05 * float(stage1_mismatch)
        + 0.005 * float(stage2_mismatch)
    )


def attempt_point(
    *,
    point_row: pd.Series,
    parent_state: AcceptedState,
    cfg: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray] | None]:
    key = point_key(point_row)
    parent_row = parent_state.summary_row
    parent_bundle = parent_state.field_bundle
    target_cr = float(point_row["reference_cr"])
    target_ci = float(point_row["reference_ci"])
    blumen_cr = float(point_row["blumen_cr"]) if np.isfinite(point_row["blumen_cr"]) else np.nan
    blumen_ci = float(point_row["blumen_ci"]) if np.isfinite(point_row["blumen_ci"]) else np.nan

    seeds = build_seeds(
        target_cr=target_cr,
        target_ci=target_ci,
        previous_cr=float(parent_row["best_shooting_cr"]),
        previous_ci=float(parent_row["best_shooting_ci"]),
        include_generic_seeds=bool(cfg["include_generic_seeds"]),
    )

    candidate_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_bundle: dict[str, np.ndarray] | None = None

    for seed_name, cr_center, ci_center in seeds:
        for cr_half in cfg["cr_half_windows"]:
            for ci_half in cfg["ci_half_windows"]:
                solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                    alpha=float(key.alpha),
                    mach=float(key.mach),
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
                fields = reconstruct_shooting_fields(
                    alpha=float(key.alpha),
                    mach=float(key.mach),
                    cr=float(result.cr),
                    ci=float(result.ci),
                    ln_p_start_right=float(result.ln_p_start_right),
                    match_y=float(cfg["match_y"]),
                    use_mapping=bool(cfg["use_mapping"]),
                    mapping_scale=float(cfg["mapping_scale"]),
                    min_y_limit=float(cfg["min_y_limit"]),
                    max_y_limit=float(cfg["max_y_limit"]),
                    y_limit_factor=float(cfg["y_limit_factor"]),
                )
                bundle = bundle_from_fields(fields)
                mode_metrics = mode_distance(
                    parent_bundle,
                    bundle,
                    mask_ratio=float(cfg["mode_mask_ratio"]),
                    common_points=int(cfg["mode_common_points"]),
                )
                p_boundary = boundary_amplitude_metrics(bundle["p"], prefix="p")
                rho_boundary = boundary_amplitude_metrics(bundle["rho"], prefix="rho")
                u_boundary = boundary_amplitude_metrics(bundle["u"], prefix="u")
                v_boundary = boundary_amplitude_metrics(bundle["v"], prefix="v")
                diag = extended_profile_diagnostics(bundle["y"], bundle["p"])
                diag.update(p_boundary)
                diag.update(rho_boundary)
                diag.update(u_boundary)
                diag.update(v_boundary)
                diag.update(infer_regimes(mach=float(key.mach), cr=float(result.cr), ci=float(result.ci)))
                max_field_edge = float(
                    max(
                        p_boundary["p_edge_amp_fraction_max"],
                        rho_boundary["rho_edge_amp_fraction_max"],
                        u_boundary["u_edge_amp_fraction_max"],
                        v_boundary["v_edge_amp_fraction_max"],
                    )
                )
                selection_metric = target_metric(
                    shooting_cr=float(result.cr),
                    shooting_ci=float(result.ci),
                    target_cr=target_cr,
                    target_ci=target_ci,
                    previous_cr=float(parent_row["best_shooting_cr"]),
                    previous_ci=float(parent_row["best_shooting_ci"]),
                    stage1_mismatch=float(result.stage1_mismatch),
                    stage2_mismatch=float(result.stage2_mismatch),
                    mode_distance_value=float(mode_metrics["mode_distance"]),
                    cfg=cfg,
                )
                row = {
                    "Mach": float(key.mach),
                    "alpha": float(key.alpha),
                    "parent_mach": float(parent_row["Mach"]),
                    "parent_alpha": float(parent_row["alpha"]),
                    "blumen_cr": blumen_cr,
                    "blumen_ci": blumen_ci,
                    "reference_cr": target_cr,
                    "reference_ci": target_ci,
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
                    "delta_cr_target": abs(float(result.cr) - target_cr),
                    "delta_ci_target": abs(float(result.ci) - target_ci),
                    "delta_cr_prev": abs(float(result.cr) - float(parent_row["best_shooting_cr"])),
                    "delta_ci_prev": abs(float(result.ci) - float(parent_row["best_shooting_ci"])),
                    "err_ci_abs": abs(float(result.ci) - blumen_ci) if np.isfinite(blumen_ci) else np.nan,
                    "stage1_mismatch": float(result.stage1_mismatch),
                    "stage2_mismatch": float(result.stage2_mismatch),
                    "ln_p_start_right": float(result.ln_p_start_right),
                    "spectral_success": bool(result.spectral_success),
                    "mode_success": bool(result.mode_success),
                    "success": bool(result.success),
                    "status": success_label(bool(result.spectral_success), bool(result.mode_success)),
                    "mode_distance_prev": float(mode_metrics["mode_distance"]),
                    "mode_env_distance_prev": float(mode_metrics["env_distance"]),
                    "mode_phase_distance_prev": float(mode_metrics["phase_distance"]),
                    "mode_shape_distance_prev": float(mode_metrics["shape_distance"]),
                    "selection_metric": float(selection_metric),
                    "y_limit": float(result.y_limit),
                    "box_truncation_suspect_any_field": bool(max_field_edge > float(cfg["edge_amp_threshold"])),
                    "max_field_edge_amp_fraction": max_field_edge,
                    **diag,
                }
                row["accepted_modal"] = bool(
                    row["spectral_success"]
                    and row["mode_success"]
                    and row["status"] == "validated"
                    and np.isfinite(row["stage1_mismatch"])
                    and float(row["stage1_mismatch"]) <= float(cfg["max_stage1"])
                    and np.isfinite(row["stage2_mismatch"])
                    and float(row["stage2_mismatch"]) <= float(cfg["max_stage2"])
                    and not bool(row["box_truncation_suspect_any_field"])
                    and float(row["delta_ci_prev"]) <= float(cfg["max_delta_ci"])
                    and float(row["delta_cr_prev"]) <= float(cfg["max_delta_cr"])
                    and float(row["mode_distance_prev"]) <= float(cfg["max_mode_distance"])
                    and (not np.isfinite(row["err_ci_abs"]) or float(row["err_ci_abs"]) <= float(cfg["max_err_ci_abs"]))
                )
                candidate_rows.append(row)
                if best_summary is None or (
                    (0 if bool(row["accepted_modal"]) else 1, float(row["selection_metric"]))
                    < (0 if bool(best_summary["accepted_modal"]) else 1, float(best_summary["selection_metric"]))
                ):
                    best_summary = row
                    best_bundle = bundle

    if best_summary is None:
        raise RuntimeError(f"Aucun candidat evalue pour M={key.mach:.3f}, alpha={key.alpha:.3f}")
    return best_summary, candidate_rows, best_bundle


def fields_rows_from_bundle(summary_row: dict[str, object], bundle: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    parent_mach = summary_row.get("parent_mach", np.nan)
    parent_alpha = summary_row.get("parent_alpha", np.nan)
    for y_value, rho_value, u_value, v_value, p_value in zip(
        bundle["y"], bundle["rho"], bundle["u"], bundle["v"], bundle["p"]
    ):
        rows.append(
            {
                "Mach": float(summary_row["Mach"]),
                "alpha": float(summary_row["alpha"]),
                "best_status": str(summary_row["status"]),
                "parent_mach": float(parent_mach) if np.isfinite(parent_mach) else np.nan,
                "parent_alpha": float(parent_alpha) if np.isfinite(parent_alpha) else np.nan,
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
    return rows


def load_anchor_states(anchor_df: pd.DataFrame, anchor_fields_df: pd.DataFrame, cfg: dict[str, object]) -> dict[PointKey, AcceptedState]:
    accepted: dict[PointKey, AcceptedState] = {}
    for _, row in anchor_df.iterrows():
        key = point_key(row)
        sub = anchor_fields_df[
            np.isclose(anchor_fields_df["Mach"].to_numpy(dtype=float), key.mach)
            & np.isclose(anchor_fields_df["alpha"].to_numpy(dtype=float), key.alpha)
        ].copy()
        if sub.empty:
            fields = reconstruct_shooting_fields(
                alpha=float(key.alpha),
                mach=float(key.mach),
                cr=float(row["reference_cr"]),
                ci=float(row["reference_ci"]),
                ln_p_start_right=float(row.get("best_ln_p_start_right", -20.0)),
                match_y=float(cfg["match_y"]),
                use_mapping=bool(cfg["use_mapping"]),
                mapping_scale=float(cfg["mapping_scale"]),
                min_y_limit=float(cfg["min_y_limit"]),
                max_y_limit=float(cfg["max_y_limit"]),
                y_limit_factor=float(cfg["y_limit_factor"]),
            )
            bundle = bundle_from_fields(fields)
        else:
            bundle = bundle_from_dataframe(sub.sort_values("y"))
        summary_row = {
            **row.to_dict(),
            "parent_mach": np.nan,
            "parent_alpha": np.nan,
            "best_shooting_cr": float(row["reference_cr"]),
            "best_shooting_ci": float(row["reference_ci"]),
            "status": "anchor",
            "accepted_modal": True,
        }
        accepted[key] = AcceptedState(summary_row=summary_row, field_bundle=bundle)
    return accepted


def build_unresolved_rows(points_df: pd.DataFrame, accepted: dict[PointKey, AcceptedState]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    accepted_keys = set(accepted.keys())
    for _, row in points_df.iterrows():
        key = point_key(row)
        if key in accepted_keys:
            continue
        rows.append(
            {
                "Mach": float(key.mach),
                "alpha": float(key.alpha),
                "parent_mach": np.nan,
                "parent_alpha": np.nan,
                "blumen_cr": float(row["blumen_cr"]) if np.isfinite(row["blumen_cr"]) else np.nan,
                "blumen_ci": float(row["blumen_ci"]) if np.isfinite(row["blumen_ci"]) else np.nan,
                "reference_cr": float(row["reference_cr"]),
                "reference_ci": float(row["reference_ci"]),
                "status": "unresolved",
                "accepted_modal": False,
            }
        )
    return rows


def plot_status_map(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    accepted = summary_df[summary_df["accepted_modal"].fillna(False)]
    unresolved = summary_df[~summary_df["accepted_modal"].fillna(False)]
    if not unresolved.empty:
        ax.scatter(
            unresolved["Mach"],
            unresolved["alpha"],
            s=70,
            marker="x",
            color="#9CA3AF",
            linewidths=1.4,
            label="Unresolved modal",
            zorder=2,
        )
    if not accepted.empty:
        ax.scatter(
            accepted["Mach"],
            accepted["alpha"],
            s=85,
            marker="o",
            facecolor="#0F766E",
            edgecolor="black",
            linewidth=0.6,
            label="Accepted modal",
            zorder=3,
        )
    anchor_mask = summary_df["status"].astype(str).eq("anchor")
    anchors = summary_df[anchor_mask]
    if not anchors.empty:
        ax.scatter(
            anchors["Mach"],
            anchors["alpha"],
            s=130,
            marker="*",
            facecolor="#D97706",
            edgecolor="black",
            linewidth=0.7,
            label="Modal anchors",
            zorder=4,
        )
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Supersonic shooting modal surface tracking")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_cfg(args)

    points_df = pd.read_csv(args.points_csv).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    anchor_df = pd.read_csv(args.anchor_csv).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    anchor_fields_df = pd.read_csv(args.anchor_fields_csv).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    neighbors = build_neighbors(points_df)
    accepted = load_anchor_states(anchor_df, anchor_fields_df, cfg)
    anchor_keys = {point_key(row) for _, row in anchor_df.iterrows()}

    print("Supersonic shooting modal surface tracking")
    print(f"points_csv={args.points_csv}")
    print(f"anchor_csv={args.anchor_csv}")
    print(f"n_points={len(points_df)} n_anchors={len(anchor_df)}")
    print(
        f"guards: stage1<={float(cfg['max_stage1']):.3e} stage2<={float(cfg['max_stage2']):.3e} "
        f"mode_distance<={float(cfg['max_mode_distance']):.3f}"
    )

    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    for state in accepted.values():
        field_rows.extend(fields_rows_from_bundle(state.summary_row, state.field_bundle))
    iteration = 0
    progress = True
    while progress:
        iteration += 1
        progress = False
        pending_keys = [point_key(row) for _, row in points_df.iterrows() if point_key(row) not in accepted]
        print(f"[iter {iteration}] pending={len(pending_keys)} accepted={len(accepted)}")
        for key in pending_keys:
            parent_states = [accepted[parent] for parent in sorted(neighbors[key], key=lambda item: (item.mach, item.alpha)) if parent in accepted]
            if not parent_states:
                continue
            point_row = points_df[
                np.isclose(points_df["Mach"].to_numpy(dtype=float), key.mach)
                & np.isclose(points_df["alpha"].to_numpy(dtype=float), key.alpha)
            ].iloc[0]
            best_summary: dict[str, object] | None = None
            best_bundle: dict[str, np.ndarray] | None = None
            best_rows: list[dict[str, object]] = []
            for parent_state in parent_states:
                summary_row, attempt_rows, bundle = attempt_point(point_row=point_row, parent_state=parent_state, cfg=cfg)
                candidate_rows.extend(attempt_rows)
                if best_summary is None or (
                    (0 if bool(summary_row["accepted_modal"]) else 1, float(summary_row["selection_metric"]))
                    < (0 if bool(best_summary["accepted_modal"]) else 1, float(best_summary["selection_metric"]))
                ):
                    best_summary = summary_row
                    best_bundle = bundle
                    best_rows = attempt_rows
            if best_summary is not None and bool(best_summary["accepted_modal"]) and best_bundle is not None:
                best_summary["status"] = "anchor" if key in anchor_keys else "validated"
                accepted[key] = AcceptedState(summary_row=best_summary, field_bundle=best_bundle)
                field_rows.extend(fields_rows_from_bundle(best_summary, best_bundle))
                progress = True
                print(
                    f"[accept] M={key.mach:.3f} alpha={key.alpha:.3f} "
                    f"parent=({best_summary['parent_mach']:.3f},{best_summary['parent_alpha']:.3f}) "
                    f"mode_dist={float(best_summary['mode_distance_prev']):.3f} "
                    f"stage1={float(best_summary['stage1_mismatch']):.3e} stage2={float(best_summary['stage2_mismatch']):.3e}"
                )

    accepted_rows = [state.summary_row for state in accepted.values()]
    unresolved_rows = build_unresolved_rows(points_df, accepted)
    summary_df = pd.DataFrame(accepted_rows + unresolved_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["Mach", "alpha", "accepted_modal", "selection_metric"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
    fields_df = pd.DataFrame(field_rows)
    if not fields_df.empty:
        fields_df = fields_df.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    fields_path = output_dir / f"{args.output_stem}_fields.csv"
    status_map_path = output_dir / f"{args.output_stem}_status_map.png"
    modes_path = output_dir / f"{args.output_stem}_modes.pdf"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    plot_status_map(summary_df, status_map_path)
    if not fields_df.empty:
        plot_modes_pdf(summary_df, fields_df, threshold_ratio=0.02, min_half_width=8.0, output_path=modes_path)

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {fields_path}")
    print(f"Wrote {status_map_path}")
    if not fields_df.empty:
        print(f"Wrote {modes_path}")


if __name__ == "__main__":
    main()
