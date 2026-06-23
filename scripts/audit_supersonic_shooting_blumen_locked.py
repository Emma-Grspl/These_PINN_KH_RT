from __future__ import annotations

import argparse
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
    apply_box_rejection,
    boundary_amplitude_metrics,
    default_box_robustness_metrics,
    infer_regimes,
    plot_diagnostics,
    plot_modes_pdf,
    plot_status_map,
    run_box_robustness_audit,
)
from scripts.audit_supersonic_shooting_visual_validation import (  # noqa: E402
    reconstruct_shooting_fields,
)
from scripts.audit_supersonic_shooting_ci_map import extended_profile_diagnostics  # noqa: E402
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


DEFAULT_ANCHOR_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "validated_modal_points"
    / "supersonic_validated_modal_points.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Shooting supersonique verrouille sur les branches Blumen fiables. "
            "Contrairement a l'audit exploratoire, ce script n'utilise pas de seeds generiques "
            "et ne propage que les points acceptes par les criteres stricts."
        )
    )
    parser.add_argument("--points", type=str, nargs="*", default=None, help="Liste de couples alpha:Mach.")
    parser.add_argument("--mach", type=float, default=None, help="Mach fixe si --alphas est fourni.")
    parser.add_argument("--alphas", type=str, default=None, help="Liste CSV d'alpha pour un Mach fixe.")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha fixe si --mach-values est fourni.")
    parser.add_argument("--mach-values", type=str, default=None, help="Liste CSV de Mach pour un alpha fixe.")
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHOR_CSV)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, required=True)

    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=2000.0)
    parser.add_argument("--y-limit-factor", type=float, default=18.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.004, 0.008, 0.015, 0.03])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.002, 0.004, 0.008, 0.015])
    parser.add_argument("--retry-growth", type=float, default=1.35)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--robust-top-k", type=int, default=8)

    parser.add_argument("--max-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-stage2", type=float, default=1.0e-4)
    parser.add_argument("--max-ci-abs", type=float, default=8.0e-3)
    parser.add_argument("--max-ci-rel", type=float, default=0.10)
    parser.add_argument("--max-cr-abs", type=float, default=3.5e-2)
    parser.add_argument("--max-edge-amp", type=float, default=5.0e-2)
    parser.add_argument("--box-robustness-factors", type=float, nargs="+", default=[1.25, 1.50])
    parser.add_argument("--box-robustness-max-rel-l2", type=float, default=0.25)
    parser.add_argument("--box-robustness-max-peak-shift", type=float, default=1.0)
    parser.add_argument("--box-robustness-max-center8-delta", type=float, default=0.10)
    parser.add_argument("--box-robustness-max-edge-growth", type=float, default=1.25)
    parser.add_argument("--require-box-robustness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--visible-threshold-ratio", type=float, default=0.02)
    parser.add_argument("--visible-min-half-width", type=float, default=8.0)
    return parser


def parse_csv_floats(raw_value: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        key = round(value, 10)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("Liste vide.")
    return values


def parse_points(raw_points: list[str]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for item in raw_points:
        alpha_raw, mach_raw = item.split(":")
        key = (round(float(alpha_raw), 10), round(float(mach_raw), 10))
        if key in seen:
            continue
        seen.add(key)
        points.append((float(alpha_raw), float(mach_raw)))
    return points


def resolve_points(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.points:
        return parse_points(list(args.points))
    if args.mach is not None and args.alphas is not None:
        return [(float(alpha), float(args.mach)) for alpha in parse_csv_floats(args.alphas)]
    if args.alpha is not None and args.mach_values is not None:
        return [(float(args.alpha), float(mach)) for mach in parse_csv_floats(args.mach_values)]
    raise ValueError("Fournir --points, ou --mach avec --alphas, ou --alpha avec --mach-values.")


def load_anchor_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"alpha", "Mach", "reference_cr", "reference_ci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("validated")]
    for column in ("alpha", "Mach", "reference_cr", "reference_ci"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["alpha", "Mach", "reference_cr", "reference_ci"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Aucune ancre valide dans {path}.")
    return out


def dedup_seeds(seeds: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[str, float, float]] = []
    for name, cr, ci in seeds:
        if not np.isfinite(cr) or not np.isfinite(ci) or ci <= 0.0:
            continue
        key = (round(float(cr), 7), round(float(ci), 7))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, float(cr), float(ci)))
    return out


def interpolated_anchor_seed(
    anchors: pd.DataFrame,
    *,
    alpha: float,
    mach: float,
    fixed_column: str,
    varying_column: str,
    seed_name: str,
) -> tuple[str, float, float] | None:
    fixed_value = float(mach if fixed_column == "Mach" else alpha)
    varying_value = float(alpha if varying_column == "alpha" else mach)
    sub = anchors[np.isclose(anchors[fixed_column].to_numpy(dtype=float), fixed_value, atol=1.0e-9)].copy()
    if len(sub) < 2:
        return None
    sub = sub.sort_values(varying_column)
    x = sub[varying_column].to_numpy(dtype=float)
    if varying_value < float(np.min(x)) or varying_value > float(np.max(x)):
        return None
    cr = float(np.interp(varying_value, x, sub["reference_cr"].to_numpy(dtype=float)))
    ci = float(np.interp(varying_value, x, sub["reference_ci"].to_numpy(dtype=float)))
    return seed_name, cr, ci


def anchor_seeds(anchors: pd.DataFrame, *, alpha: float, mach: float, k_nearest: int = 4) -> list[tuple[str, float, float]]:
    seeds: list[tuple[str, float, float]] = []
    exact = anchors[
        np.isclose(anchors["alpha"].to_numpy(dtype=float), float(alpha), atol=1.0e-9)
        & np.isclose(anchors["Mach"].to_numpy(dtype=float), float(mach), atol=1.0e-9)
    ]
    for _, row in exact.iterrows():
        seeds.append(("anchor_exact", float(row["reference_cr"]), float(row["reference_ci"])))

    interp_alpha = interpolated_anchor_seed(
        anchors,
        alpha=float(alpha),
        mach=float(mach),
        fixed_column="Mach",
        varying_column="alpha",
        seed_name="anchor_interp_alpha",
    )
    if interp_alpha is not None:
        seeds.append(interp_alpha)
    interp_mach = interpolated_anchor_seed(
        anchors,
        alpha=float(alpha),
        mach=float(mach),
        fixed_column="alpha",
        varying_column="Mach",
        seed_name="anchor_interp_mach",
    )
    if interp_mach is not None:
        seeds.append(interp_mach)

    scaled_distance = np.hypot(
        (anchors["alpha"].to_numpy(dtype=float) - float(alpha)) / 0.05,
        (anchors["Mach"].to_numpy(dtype=float) - float(mach)) / 0.10,
    )
    nearest_idx = np.argsort(scaled_distance)[: max(1, int(k_nearest))]
    nearest = anchors.iloc[nearest_idx].copy()
    weights = 1.0 / np.maximum(scaled_distance[nearest_idx], 1.0e-6)
    weights = weights / np.sum(weights)
    cr = float(np.sum(weights * nearest["reference_cr"].to_numpy(dtype=float)))
    ci = float(np.sum(weights * nearest["reference_ci"].to_numpy(dtype=float)))
    seeds.append(("anchor_knn", cr, ci))
    for rank, (_, row) in enumerate(nearest.iterrows(), start=1):
        seeds.append((f"anchor_nearest_{rank}", float(row["reference_cr"]), float(row["reference_ci"])))
    return dedup_seeds(seeds)


def candidate_selection_score(
    row: dict[str, object],
    *,
    blumen_cr: float,
    blumen_ci: float,
    target_available: bool,
    previous: tuple[float, float] | None,
) -> float:
    score = 0.0 if bool(row["success"]) else 50.0
    score += 10.0 * float(row["stage1_mismatch"])
    score += 0.01 * float(row["stage2_mismatch"])
    if target_available:
        score += ci_primary_score(
            shooting_cr=float(row["shooting_cr"]),
            shooting_ci=float(row["shooting_ci"]),
            blumen_cr=float(blumen_cr),
            blumen_ci=float(blumen_ci),
            previous_mach=None,
            previous_alpha=None,
            ci_weight=5.0,
            cr_weight=1.0,
            continuity_weight=0.0,
        )
    if previous is not None:
        score += 0.35 * np.hypot(float(row["shooting_cr"]) - previous[0], 5.0 * (float(row["shooting_ci"]) - previous[1]))
    return float(score)


def evaluate_mode_diagnostics(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    cfg: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fields = reconstruct_shooting_fields(
        alpha=float(alpha),
        mach=float(mach),
        cr=float(cr),
        ci=float(ci),
        ln_p_start_right=float(ln_p_start_right),
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

    diag: dict[str, object] = {}
    diag.update(extended_profile_diagnostics(y_fields, p_fields))
    p_boundary = boundary_amplitude_metrics(p_fields, prefix="p")
    rho_boundary = boundary_amplitude_metrics(rho_fields, prefix="rho")
    u_boundary = boundary_amplitude_metrics(u_fields, prefix="u")
    v_boundary = boundary_amplitude_metrics(v_fields, prefix="v")
    diag.update(p_boundary)
    diag.update(rho_boundary)
    diag.update(u_boundary)
    diag.update(v_boundary)
    diag.update(infer_regimes(mach=float(mach), cr=float(cr), ci=float(ci)))
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
    diag["box_truncation_suspect_p"] = bool(float(p_boundary["p_edge_amp_fraction_max"]) > float(cfg["max_edge_amp"]))
    diag["box_truncation_suspect_any_field"] = bool(float(diag["max_field_edge_amp_fraction"]) > float(cfg["max_edge_amp"]))
    diag.update(
        run_box_robustness_audit(
            alpha=float(alpha),
            mach=float(mach),
            cr=float(cr),
            ci=float(ci),
            ln_p_start_right=float(ln_p_start_right),
            cfg=cfg,
            base_fields=fields,
        )
    )

    field_rows: list[dict[str, object]] = []
    for y_value, rho_value, u_value, v_value, p_value in zip(
        fields["y"], fields["rho"], fields["u"], fields["v"], fields["p"]
    ):
        field_rows.append(
            {
                "alpha": float(alpha),
                "Mach": float(mach),
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
    return diag, field_rows


def strict_acceptance(row: dict[str, object], diag: dict[str, object], args: argparse.Namespace) -> tuple[bool, str]:
    reasons: list[str] = []
    if not bool(row["spectral_success"]):
        reasons.append("spectral")
    if not bool(row["mode_success"]):
        reasons.append("mode")
    if float(row["stage1_mismatch"]) > float(args.max_stage1):
        reasons.append("stage1")
    if float(row["stage2_mismatch"]) > float(args.max_stage2):
        reasons.append("stage2")
    if bool(row["target_available"]):
        if float(row["err_ci_abs"]) > float(args.max_ci_abs):
            reasons.append("ci_abs")
        if float(row["err_ci_rel"]) > float(args.max_ci_rel):
            reasons.append("ci_rel")
        if bool(row["blumen_cr_available"]) and float(row["err_cr_abs"]) > float(args.max_cr_abs):
            reasons.append("cr_abs")
    if bool(diag.get("box_truncation_suspect_any_field", False)):
        reasons.append("edge")
    if bool(args.require_box_robustness) and not bool(diag.get("box_robustness_pass", False)):
        reasons.append("box")
    if reasons:
        return False, ";".join(reasons)
    return True, "accepted"


def evaluate_point(
    *,
    alpha: float,
    mach: float,
    targets_df: pd.DataFrame,
    anchors: pd.DataFrame,
    previous: tuple[float, float] | None,
    args: argparse.Namespace,
    cfg: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], tuple[float, float] | None]:
    target = targets_df[np.isclose(targets_df["Mach"].to_numpy(dtype=float), float(mach))].iloc[0]
    blumen_cr = float(target["blumen_cr"])
    blumen_ci = float(target["blumen_ci"])
    cr_available = bool(np.isfinite(blumen_cr))
    ci_available = bool(np.isfinite(blumen_ci))
    target_available = bool(cr_available and ci_available)

    seeds: list[tuple[str, float, float]] = []
    if target_available:
        seeds.append(("blumen", blumen_cr, blumen_ci))
    seeds.extend(anchor_seeds(anchors, alpha=float(alpha), mach=float(mach)))
    if previous is not None:
        seeds.append(("previous_accepted", previous[0], previous[1]))
        if target_available:
            seeds.append(("blend_previous_blumen", 0.5 * (previous[0] + blumen_cr), 0.5 * (previous[1] + blumen_ci)))
    seeds = dedup_seeds(seeds)
    if not seeds:
        raise RuntimeError(f"Aucune seed fiable disponible pour alpha={alpha:.3f}, Mach={mach:.3f}.")

    candidate_rows: list[dict[str, object]] = []
    for seed_name, cr_center, ci_center in seeds:
        for cr_half in args.cr_half_windows:
            for ci_half in args.ci_half_windows:
                solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                    alpha=float(alpha),
                    mach=float(mach),
                    match_y=float(args.match_y),
                    use_mapping=bool(args.use_mapping),
                    mapping_scale=float(args.mapping_scale),
                    min_y_limit=float(args.min_y_limit),
                    max_y_limit=float(args.max_y_limit),
                    y_limit_factor=float(args.y_limit_factor),
                    amp_lower_bound=float(args.amp_lower_bound),
                    amp_upper_bound=float(args.amp_upper_bound),
                    cr_center=float(cr_center),
                    ci_center=float(ci_center),
                    cr_half_window=float(cr_half),
                    ci_half_window=float(ci_half),
                    retry_growth=float(args.retry_growth),
                    max_retries=int(args.max_retries),
                    max_iter=int(args.max_iter),
                    grid_size=int(args.grid_size),
                )
                row = {
                    "alpha": float(alpha),
                    "Mach": float(mach),
                    "blumen_cr": float(blumen_cr),
                    "blumen_ci": float(blumen_ci),
                    "blumen_cr_available": bool(cr_available),
                    "blumen_ci_available": bool(ci_available),
                    "target_available": bool(target_available),
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
                    "err_ci_rel": abs(float(result.ci) - blumen_ci) / max(abs(blumen_ci), 1.0e-12) if ci_available else np.nan,
                    "stage1_mismatch": float(result.stage1_mismatch),
                    "stage2_mismatch": float(result.stage2_mismatch),
                    "ln_p_start_right": float(result.ln_p_start_right),
                    "y_limit": float(result.y_limit),
                    "spectral_success": bool(result.spectral_success),
                    "mode_success": bool(result.mode_success),
                    "success": bool(result.success),
                }
                row["pre_robust_score"] = candidate_selection_score(
                    row,
                    blumen_cr=blumen_cr,
                    blumen_ci=blumen_ci,
                    target_available=target_available,
                    previous=previous,
                )
                candidate_rows.append(row)

    ranked = sorted(candidate_rows, key=lambda row: float(row["pre_robust_score"]))
    robust_top = ranked[: max(1, int(args.robust_top_k))]
    best_summary_candidate: dict[str, object] | None = None
    best_diag: dict[str, object] = {}
    best_fields: list[dict[str, object]] = []
    best_accept = False
    best_reason = "not_evaluated"

    for rank, row in enumerate(robust_top, start=1):
        diag, fields = evaluate_mode_diagnostics(
            alpha=float(alpha),
            mach=float(mach),
            cr=float(row["shooting_cr"]),
            ci=float(row["shooting_ci"]),
            ln_p_start_right=float(row["ln_p_start_right"]),
            cfg=cfg,
        )
        accepted, reason = strict_acceptance(row, diag, args)
        row["robust_rank"] = int(rank)
        row["strict_accept"] = bool(accepted)
        row["strict_rejection_reason"] = str(reason)
        row.update({f"diag_{key}": value for key, value in diag.items()})
        if best_summary_candidate is None:
            best_summary_candidate = row
            best_diag = diag
            best_fields = fields
            best_accept = accepted
            best_reason = reason
        if accepted:
            best_summary_candidate = row
            best_diag = diag
            best_fields = fields
            best_accept = True
            best_reason = reason
            break

    for row in ranked:
        row.setdefault("robust_rank", np.nan)
        row.setdefault("strict_accept", False)
        row.setdefault("strict_rejection_reason", "not_robust_evaluated")

    if best_summary_candidate is None:
        raise RuntimeError(f"Aucun candidat evalue pour alpha={alpha:.3f}, Mach={mach:.3f}.")

    raw_status = "validated" if bool(best_summary_candidate["success"]) else "failed"
    if bool(args.require_box_robustness):
        final_status, box_rejection_applied = apply_box_rejection(raw_status, best_diag)
    else:
        final_status = raw_status
        box_rejection_applied = False
    if not best_accept:
        final_status = "strict_rejected"

    summary_row = {
        "alpha": float(alpha),
        "Mach": float(mach),
        "blumen_cr": float(blumen_cr),
        "blumen_ci": float(blumen_ci),
        "blumen_cr_available": bool(cr_available),
        "blumen_ci_available": bool(ci_available),
        "blumen_target_available": bool(target_available),
        "n_seeds": int(len(seeds)),
        "n_candidates": int(len(candidate_rows)),
        "n_robust_evaluated": int(len(robust_top)),
        "n_success_candidates": int(sum(bool(row["success"]) for row in candidate_rows)),
        "best_seed_name": str(best_summary_candidate["seed_name"]),
        "best_shooting_cr": float(best_summary_candidate["shooting_cr"]),
        "best_shooting_ci": float(best_summary_candidate["shooting_ci"]),
        "best_shooting_omega_i": float(best_summary_candidate["shooting_omega_i"]),
        "best_err_cr_abs": float(best_summary_candidate["err_cr_abs"]) if cr_available else np.nan,
        "best_err_ci_abs": float(best_summary_candidate["err_ci_abs"]) if ci_available else np.nan,
        "best_err_ci_rel": float(best_summary_candidate["err_ci_rel"]) if ci_available else np.nan,
        "best_stage1_mismatch": float(best_summary_candidate["stage1_mismatch"]),
        "best_stage2_mismatch": float(best_summary_candidate["stage2_mismatch"]),
        "best_ln_p_start_right": float(best_summary_candidate["ln_p_start_right"]),
        "best_spectral_success": bool(best_summary_candidate["spectral_success"]),
        "best_mode_success": bool(best_summary_candidate["mode_success"]),
        "best_success": bool(best_accept),
        "best_raw_status": str(raw_status),
        "best_status": str(final_status),
        "box_rejection_applied": bool(box_rejection_applied),
        "strict_accept": bool(best_accept),
        "strict_rejection_reason": str(best_reason),
        "best_selection_metric": float(best_summary_candidate["pre_robust_score"]),
        "best_selection_metric_name": "blumen_locked_strict",
        "best_retry_index": int(best_summary_candidate["retry_index"]),
        "best_used_cr_half_window": float(best_summary_candidate["used_cr_half_window"]),
        "best_used_ci_half_window": float(best_summary_candidate["used_ci_half_window"]),
        "best_y_limit": float(best_summary_candidate["y_limit"]),
        **best_diag,
        "exception": "",
    }
    for field_row in best_fields:
        field_row["best_status"] = str(final_status)
        field_row["best_raw_status"] = str(raw_status)
    next_previous = (
        (float(best_summary_candidate["shooting_cr"]), float(best_summary_candidate["shooting_ci"]))
        if best_accept
        else previous
    )
    return summary_row, ranked, best_fields, next_previous


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = resolve_points(args)
    anchors = load_anchor_points(args.anchor_csv)
    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)

    cfg = {
        "match_y": float(args.match_y),
        "use_mapping": bool(args.use_mapping),
        "mapping_scale": float(args.mapping_scale),
        "min_y_limit": float(args.min_y_limit),
        "max_y_limit": float(args.max_y_limit),
        "y_limit_factor": float(args.y_limit_factor),
        "amp_lower_bound": float(args.amp_lower_bound),
        "amp_upper_bound": float(args.amp_upper_bound),
        "max_edge_amp": float(args.max_edge_amp),
        "box_robustness_factors": [float(value) for value in args.box_robustness_factors],
        "box_robustness_max_rel_l2": float(args.box_robustness_max_rel_l2),
        "box_robustness_max_peak_shift": float(args.box_robustness_max_peak_shift),
        "box_robustness_max_center8_delta": float(args.box_robustness_max_center8_delta),
        "box_robustness_max_edge_growth": float(args.box_robustness_max_edge_growth),
    }

    print("Supersonic shooting Blumen-locked continuation")
    print(f"points: {' '.join(f'{alpha:.5f}:{mach:.5f}' for alpha, mach in points)}")
    print(f"anchor_csv={args.anchor_csv}")
    print(
        f"strict: stage1<={args.max_stage1:g} stage2<={args.max_stage2:g} "
        f"ci_abs<={args.max_ci_abs:g} ci_rel<={args.max_ci_rel:g} cr_abs<={args.max_cr_abs:g} "
        f"box_required={bool(args.require_box_robustness)}"
    )

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    previous: tuple[float, float] | None = None

    for alpha, mach in points:
        targets_df = build_blumen_targets([float(mach)], float(alpha), cr_points, ci_points)
        try:
            summary_row, point_candidates, point_fields, previous = evaluate_point(
                alpha=float(alpha),
                mach=float(mach),
                targets_df=targets_df,
                anchors=anchors,
                previous=previous,
                args=args,
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            target = targets_df.iloc[0]
            summary_row = {
                "alpha": float(alpha),
                "Mach": float(mach),
                "blumen_cr": float(target["blumen_cr"]),
                "blumen_ci": float(target["blumen_ci"]),
                "blumen_cr_available": bool(np.isfinite(float(target["blumen_cr"]))),
                "blumen_ci_available": bool(np.isfinite(float(target["blumen_ci"]))),
                "blumen_target_available": bool(
                    np.isfinite(float(target["blumen_cr"])) and np.isfinite(float(target["blumen_ci"]))
                ),
                "n_seeds": 0,
                "n_candidates": 0,
                "n_robust_evaluated": 0,
                "n_success_candidates": 0,
                "best_seed_name": "",
                "best_shooting_cr": np.nan,
                "best_shooting_ci": np.nan,
                "best_shooting_omega_i": np.nan,
                "best_err_cr_abs": np.nan,
                "best_err_ci_abs": np.nan,
                "best_err_ci_rel": np.nan,
                "best_stage1_mismatch": np.nan,
                "best_stage2_mismatch": np.nan,
                "best_ln_p_start_right": np.nan,
                "best_spectral_success": False,
                "best_mode_success": False,
                "best_success": False,
                "best_raw_status": "exception",
                "best_status": "exception",
                "box_rejection_applied": False,
                "strict_accept": False,
                "strict_rejection_reason": "exception",
                "best_selection_metric": np.nan,
                "best_selection_metric_name": "",
                "best_retry_index": np.nan,
                "best_used_cr_half_window": np.nan,
                "best_used_ci_half_window": np.nan,
                "best_y_limit": np.nan,
                **default_box_robustness_metrics([float(value) for value in args.box_robustness_factors]),
                "exception": repr(exc),
            }
            point_candidates = []
            point_fields = []

        summary_rows.append(summary_row)
        candidate_rows.extend(point_candidates)
        field_rows.extend(point_fields)
        print(
            f"[point] alpha={alpha:.5f} Mach={mach:.5f} status={summary_row['best_status']} "
            f"accept={bool(summary_row['strict_accept'])} "
            f"cr={float(summary_row['best_shooting_cr']) if np.isfinite(summary_row['best_shooting_cr']) else np.nan:.6f} "
            f"ci={float(summary_row['best_shooting_ci']) if np.isfinite(summary_row['best_shooting_ci']) else np.nan:.6f} "
            f"err_ci={float(summary_row['best_err_ci_abs']) if np.isfinite(summary_row['best_err_ci_abs']) else np.nan:.3e} "
            f"reason={summary_row['strict_rejection_reason']}"
        )
        if bool(args.stop_on_failure) and not bool(summary_row["strict_accept"]):
            break

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    fields_df = pd.DataFrame(field_rows)
    accepted_df = summary_df[summary_df["strict_accept"].astype(bool)].copy()

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    fields_path = output_dir / f"{args.output_stem}_fields.csv"
    accepted_path = output_dir / f"{args.output_stem}_accepted_reference.csv"
    points_path = output_dir / f"{args.output_stem}_points.txt"
    status_map_path = output_dir / f"{args.output_stem}_status_map.png"
    diagnostics_path = output_dir / f"{args.output_stem}_diagnostics.png"
    modes_pdf_path = output_dir / f"{args.output_stem}_modes.pdf"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    accepted_df.to_csv(accepted_path, index=False)
    points_path.write_text("\n".join(f"{alpha:.6f}:{mach:.6f}" for alpha, mach in points) + "\n", encoding="utf-8")

    if not summary_df.empty:
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

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))
    print(f"\nACCEPTED {len(accepted_df)}/{len(summary_df)}")
    for path in (summary_path, candidates_path, fields_path, accepted_path, points_path, status_map_path, diagnostics_path):
        print(f"Wrote {path}")
    if not fields_df.empty:
        print(f"Wrote {modes_pdf_path}")


if __name__ == "__main__":
    main()
