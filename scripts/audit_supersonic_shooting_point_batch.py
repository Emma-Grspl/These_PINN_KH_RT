from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    classify_relative_regime,
    extended_profile_diagnostics,
)
from scripts.audit_supersonic_shooting_visual_validation import (  # noqa: E402
    compute_visible_xlim,
    reconstruct_shooting_fields,
)
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "shooting"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pointwise supersonique au shooting sur plusieurs couples alpha:Mach, "
            "avec parallelisme CPU et metriques de succes."
        )
    )
    parser.add_argument("--points", type=str, nargs="*", default=None, help="Liste de couples alpha:Mach.")
    parser.add_argument("--mach", type=float, default=None, help="Mach fixe si on fournit seulement une liste d'alpha.")
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Liste CSV d'alpha a evaluer pour un Mach fixe. Ex: 0.125,0.1375,0.15",
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
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.015, 0.03, 0.06, 0.10])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.008, 0.015, 0.03])
    parser.add_argument("--retry-growth", type=float, default=1.75)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ci-weight", type=float, default=4.0)
    parser.add_argument("--cr-weight", type=float, default=0.35)
    parser.add_argument("--continuity-weight", type=float, default=0.20)
    parser.add_argument("--no-generic-seeds", action="store_false", dest="include_generic_seeds")
    parser.set_defaults(include_generic_seeds=True)
    parser.add_argument("--visible-threshold-ratio", type=float, default=0.02)
    parser.add_argument("--visible-min-half-width", type=float, default=8.0)
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument(
        "--box-robustness-factors",
        type=float,
        nargs="+",
        default=[1.5, 2.0],
        help="Facteurs multiplicatifs utilises pour agrandir la boite et tester la robustesse modale.",
    )
    parser.add_argument("--box-robustness-max-rel-l2", type=float, default=0.15)
    parser.add_argument("--box-robustness-max-peak-shift", type=float, default=0.75)
    parser.add_argument("--box-robustness-max-center8-delta", type=float, default=0.10)
    parser.add_argument("--box-robustness-max-edge-growth", type=float, default=1.25)
    parser.add_argument("--output-stem", type=str, required=True)
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


def parse_points(values: list[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for item in values:
        alpha_raw, mach_raw = item.split(":")
        key = (round(float(alpha_raw), 8), round(float(mach_raw), 8))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(alpha_raw), float(mach_raw)))
    return out


def resolve_points(*, raw_points: list[str] | None, mach: float | None, alphas_csv: str | None) -> list[tuple[float, float]]:
    if raw_points:
        return parse_points(list(raw_points))
    if mach is None or alphas_csv is None:
        raise ValueError("Fournir soit --points, soit --mach avec --alphas.")
    return [(float(alpha), float(mach)) for alpha in parse_alpha_csv(str(alphas_csv))]


def generic_seed_list() -> list[tuple[str, float, float]]:
    return [
        ("generic_00", 0.00, 0.015),
        ("generic_01", 0.05, 0.025),
        ("generic_02", 0.10, 0.040),
        ("generic_03", 0.18, 0.055),
        ("generic_04", 0.26, 0.070),
        ("generic_05", 0.34, 0.085),
    ]


def dedup_seeds(seeds: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[str, float, float]] = []
    for seed_name, cr_center, ci_center in seeds:
        key = (round(float(cr_center), 6), round(float(ci_center), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((seed_name, float(cr_center), float(ci_center)))
    return out


def success_label(spectral_success: bool, mode_success: bool) -> str:
    if spectral_success and mode_success:
        return "validated"
    if spectral_success and not mode_success:
        return "spectral_only"
    if mode_success and not spectral_success:
        return "mode_only"
    return "failed"


def apply_box_rejection(status: str, diag: dict[str, object]) -> tuple[str, bool]:
    """
    Downgrade a mathematically matched candidate when the reconstructed mode is not
    stable under the box-size audit. The raw spectral/mode flags stay available in
    the CSV, but best_status/best_success must not advertise such points as valid.
    """
    if str(status) != "validated":
        return str(status), False

    truncation_suspect = bool(diag.get("box_truncation_suspect_any_field", False))
    robustness_enabled = bool(diag.get("box_robustness_enabled", False))
    robustness_failed = robustness_enabled and not bool(diag.get("box_robustness_pass", False))
    if truncation_suspect or robustness_failed:
        return "box_rejected", True
    return str(status), False


def selection_metric(
    *,
    target_available: bool,
    ci_available: bool,
    shooting_cr: float,
    shooting_ci: float,
    blumen_cr: float,
    blumen_ci: float,
    spectral_success: bool,
    mode_success: bool,
    stage1_mismatch: float,
    stage2_mismatch: float,
    ci_weight: float,
    cr_weight: float,
    continuity_weight: float,
) -> tuple[float, str]:
    if target_available:
        score = ci_primary_score(
            shooting_cr=float(shooting_cr),
            shooting_ci=float(shooting_ci),
            blumen_cr=float(blumen_cr),
            blumen_ci=float(blumen_ci),
            previous_mach=None,
            previous_alpha=None,
            ci_weight=float(ci_weight),
            cr_weight=float(cr_weight),
            continuity_weight=float(continuity_weight),
        )
        return float(score), "distance_to_blumen"
    if ci_available:
        penalty = 0.05 * float(stage1_mismatch) + 0.001 * float(stage2_mismatch)
        bonus = 0.02 if bool(spectral_success) else 0.0
        bonus += 0.05 if bool(mode_success) else 0.0
        score = abs(float(shooting_ci) - float(blumen_ci)) - bonus + penalty
        return float(score), "distance_to_blumen_ci_only"
    penalty = 0.05 * float(stage1_mismatch) + 0.001 * float(stage2_mismatch)
    bonus = 0.02 if bool(spectral_success) else 0.0
    bonus += 0.05 if bool(mode_success) else 0.0
    return float(-shooting_ci - bonus + penalty), "max_ci_fallback"


def boundary_amplitude_metrics(field: np.ndarray, *, prefix: str) -> dict[str, float]:
    field_abs = np.abs(field)
    peak = max(float(np.max(field_abs)), 1e-12)
    return {
        f"{prefix}_left_boundary_amp_fraction": float(field_abs[0] / peak),
        f"{prefix}_right_boundary_amp_fraction": float(field_abs[-1] / peak),
        f"{prefix}_edge_amp_fraction_max": float(max(field_abs[0], field_abs[-1]) / peak),
    }


def infer_regimes(*, mach: float, cr: float, ci: float) -> dict[str, object]:
    c = complex(float(cr), float(ci))
    left_rel_mach, left_regime = classify_relative_regime(float(mach), -1.0, c)
    right_rel_mach, right_regime = classify_relative_regime(float(mach), 1.0, c)
    return {
        "left_relative_mach": float(left_rel_mach),
        "left_relative_regime": str(left_regime),
        "right_relative_mach": float(right_rel_mach),
        "right_relative_regime": str(right_regime),
    }


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def box_factor_label(factor: float) -> str:
    return f"x{float(factor):.2f}".replace(".", "p")


def interpolate_complex_field(y_source: np.ndarray, field_source: np.ndarray, y_target: np.ndarray) -> np.ndarray:
    real_part = np.interp(y_target, y_source, np.real(field_source))
    imag_part = np.interp(y_target, y_source, np.imag(field_source))
    return real_part + 1j * imag_part


def normalized_field_l2(field: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm = np.sqrt(max(trapezoid_compat(np.abs(field) ** 2, y), 1e-12))
    return field / norm


def compare_mode_shapes(
    *,
    base_fields: dict[str, np.ndarray],
    variant_fields: dict[str, np.ndarray],
) -> dict[str, float]:
    y_base = np.asarray(base_fields["y"], dtype=float)
    y_variant = np.asarray(variant_fields["y"], dtype=float)
    overlap_min = max(float(np.min(y_base)), float(np.min(y_variant)))
    overlap_max = min(float(np.max(y_base)), float(np.max(y_variant)))
    if not np.isfinite(overlap_min) or not np.isfinite(overlap_max) or overlap_max <= overlap_min:
        raise RuntimeError("Recouvrement vide entre le mode de base et le mode reconstruit sur grande boite.")

    n_points = int(min(max(len(y_base), len(y_variant), 401), 2001))
    y_common = np.linspace(overlap_min, overlap_max, n_points)
    base_p = interpolate_complex_field(y_base, np.asarray(base_fields["p"], dtype=np.complex128), y_common)
    variant_p = interpolate_complex_field(y_variant, np.asarray(variant_fields["p"], dtype=np.complex128), y_common)

    inner = np.vdot(variant_p, base_p)
    if np.abs(inner) > 0.0:
        phase = np.exp(-1j * np.angle(inner))
    else:
        phase = 1.0 + 0.0j

    rel_l2_by_field: dict[str, float] = {}
    for name in ("p", "rho", "u", "v"):
        base_interp = interpolate_complex_field(y_base, np.asarray(base_fields[name], dtype=np.complex128), y_common)
        variant_interp = interpolate_complex_field(y_variant, np.asarray(variant_fields[name], dtype=np.complex128), y_common) * phase
        base_norm = normalized_field_l2(base_interp, y_common)
        variant_norm = normalized_field_l2(variant_interp, y_common)
        rel_l2_by_field[name] = float(
            np.sqrt(trapezoid_compat(np.abs(base_norm - variant_norm) ** 2, y_common) / max(trapezoid_compat(np.abs(base_norm) ** 2, y_common), 1e-12))
        )

    base_diag = extended_profile_diagnostics(y_common, base_p)
    variant_diag = extended_profile_diagnostics(y_common, variant_p * phase)
    base_peak = max(float(np.max(np.abs(base_p))), 1e-12)
    variant_peak = max(float(np.max(np.abs(variant_p))), 1e-12)
    base_edge = max(float(np.abs(base_p[0])), float(np.abs(base_p[-1]))) / base_peak
    variant_edge = max(float(np.abs(variant_p[0])), float(np.abs(variant_p[-1]))) / variant_peak

    return {
        "p_rel_l2": float(rel_l2_by_field["p"]),
        "rho_rel_l2": float(rel_l2_by_field["rho"]),
        "u_rel_l2": float(rel_l2_by_field["u"]),
        "v_rel_l2": float(rel_l2_by_field["v"]),
        "max_rel_l2": float(max(rel_l2_by_field.values())),
        "peak_shift_abs": abs(float(variant_diag["peak_y"]) - float(base_diag["peak_y"])),
        "center8_mass_fraction_delta": abs(
            float(variant_diag["center8_mass_fraction"]) - float(base_diag["center8_mass_fraction"])
        ),
        "edge_growth_ratio": float(variant_edge / max(base_edge, 1e-12)),
        "base_half_span": float(max(abs(y_base[0]), abs(y_base[-1]))),
        "variant_half_span": float(max(abs(y_variant[0]), abs(y_variant[-1]))),
    }


def default_box_robustness_metrics(factors: list[float]) -> dict[str, object]:
    metrics: dict[str, object] = {
        "box_robustness_enabled": bool(factors),
        "box_robustness_pass": True,
        "box_robustness_note": "not_run" if factors else "disabled",
        "box_robustness_n_variants": int(len(factors)),
        "box_robustness_max_rel_l2": np.nan,
        "box_robustness_max_peak_shift": np.nan,
        "box_robustness_max_center8_delta": np.nan,
        "box_robustness_max_edge_growth": np.nan,
        "box_robustness_min_half_span_gain": np.nan,
    }
    for factor in factors:
        label = box_factor_label(float(factor))
        metrics[f"box_robustness_p_rel_l2_{label}"] = np.nan
        metrics[f"box_robustness_rho_rel_l2_{label}"] = np.nan
        metrics[f"box_robustness_u_rel_l2_{label}"] = np.nan
        metrics[f"box_robustness_v_rel_l2_{label}"] = np.nan
        metrics[f"box_robustness_max_rel_l2_{label}"] = np.nan
        metrics[f"box_robustness_peak_shift_{label}"] = np.nan
        metrics[f"box_robustness_center8_delta_{label}"] = np.nan
        metrics[f"box_robustness_edge_growth_{label}"] = np.nan
        metrics[f"box_robustness_half_span_gain_{label}"] = np.nan
    return metrics


def run_box_robustness_audit(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    cfg: dict[str, object],
    base_fields: dict[str, np.ndarray],
) -> dict[str, object]:
    factors = sorted(
        {
            float(factor)
            for factor in cfg["box_robustness_factors"]
            if np.isfinite(float(factor)) and float(factor) > 1.0 + 1.0e-9
        }
    )
    metrics = default_box_robustness_metrics(factors)
    if not factors:
        return metrics

    failures: list[str] = []
    max_rel_l2 = 0.0
    max_peak_shift = 0.0
    max_center8_delta = 0.0
    max_edge_growth = 0.0
    min_half_span_gain = np.inf
    base_half_span = float(max(abs(base_fields["y"][0]), abs(base_fields["y"][-1])))

    for factor in factors:
        label = box_factor_label(float(factor))
        try:
            variant_fields = reconstruct_shooting_fields(
                alpha=float(alpha),
                mach=float(mach),
                cr=float(cr),
                ci=float(ci),
                ln_p_start_right=float(ln_p_start_right),
                match_y=float(cfg["match_y"]),
                use_mapping=bool(cfg["use_mapping"]),
                mapping_scale=float(cfg["mapping_scale"]),
                min_y_limit=float(cfg["min_y_limit"]),
                max_y_limit=float(cfg["max_y_limit"]) * float(factor),
                y_limit_factor=float(cfg["y_limit_factor"]) * float(factor),
            )
            compare = compare_mode_shapes(base_fields=base_fields, variant_fields=variant_fields)
        except Exception as exc:  # noqa: BLE001
            metrics["box_robustness_pass"] = False
            metrics["box_robustness_note"] = f"reconstruction_failed_{label}"
            failures.append(f"{label}:{exc!r}")
            continue

        metrics[f"box_robustness_p_rel_l2_{label}"] = float(compare["p_rel_l2"])
        metrics[f"box_robustness_rho_rel_l2_{label}"] = float(compare["rho_rel_l2"])
        metrics[f"box_robustness_u_rel_l2_{label}"] = float(compare["u_rel_l2"])
        metrics[f"box_robustness_v_rel_l2_{label}"] = float(compare["v_rel_l2"])
        metrics[f"box_robustness_max_rel_l2_{label}"] = float(compare["max_rel_l2"])
        metrics[f"box_robustness_peak_shift_{label}"] = float(compare["peak_shift_abs"])
        metrics[f"box_robustness_center8_delta_{label}"] = float(compare["center8_mass_fraction_delta"])
        metrics[f"box_robustness_edge_growth_{label}"] = float(compare["edge_growth_ratio"])
        metrics[f"box_robustness_half_span_gain_{label}"] = float(compare["variant_half_span"] / max(base_half_span, 1e-12))

        max_rel_l2 = max(max_rel_l2, float(compare["max_rel_l2"]))
        max_peak_shift = max(max_peak_shift, float(compare["peak_shift_abs"]))
        max_center8_delta = max(max_center8_delta, float(compare["center8_mass_fraction_delta"]))
        max_edge_growth = max(max_edge_growth, float(compare["edge_growth_ratio"]))
        min_half_span_gain = min(min_half_span_gain, float(compare["variant_half_span"] / max(base_half_span, 1e-12)))

        if float(compare["max_rel_l2"]) > float(cfg["box_robustness_max_rel_l2"]):
            failures.append(f"{label}:rel_l2")
        if float(compare["peak_shift_abs"]) > float(cfg["box_robustness_max_peak_shift"]):
            failures.append(f"{label}:peak_shift")
        if float(compare["center8_mass_fraction_delta"]) > float(cfg["box_robustness_max_center8_delta"]):
            failures.append(f"{label}:center8")
        if float(compare["edge_growth_ratio"]) > float(cfg["box_robustness_max_edge_growth"]):
            failures.append(f"{label}:edge_growth")

    metrics["box_robustness_max_rel_l2"] = float(max_rel_l2) if factors else np.nan
    metrics["box_robustness_max_peak_shift"] = float(max_peak_shift) if factors else np.nan
    metrics["box_robustness_max_center8_delta"] = float(max_center8_delta) if factors else np.nan
    metrics["box_robustness_max_edge_growth"] = float(max_edge_growth) if factors else np.nan
    metrics["box_robustness_min_half_span_gain"] = float(min_half_span_gain) if np.isfinite(min_half_span_gain) else np.nan
    if failures:
        metrics["box_robustness_pass"] = False
        metrics["box_robustness_note"] = ";".join(failures)
    else:
        metrics["box_robustness_note"] = "stable_across_box_growth"
    return metrics


def evaluate_point(point: tuple[float, float], cfg: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    alpha, mach = point
    cr_points = load_digitized_long(Path(str(cfg["cr_points"])))
    ci_points = load_digitized_long(Path(str(cfg["ci_points"])))
    target_df = build_blumen_targets([float(mach)], float(alpha), cr_points, ci_points)
    target = target_df.iloc[0]
    blumen_cr = float(target["blumen_cr"])
    blumen_ci = float(target["blumen_ci"])
    ci_available = bool(np.isfinite(blumen_ci))
    cr_available = bool(np.isfinite(blumen_cr))
    target_available = bool(np.isfinite(blumen_cr) and np.isfinite(blumen_ci))

    seeds: list[tuple[str, float, float]] = []
    if target_available:
        seeds.extend(
            build_seed_list(
                blumen_cr=float(blumen_cr),
                blumen_ci=float(blumen_ci),
                previous_mach=None,
                previous_alpha=None,
            )
        )
    if bool(cfg["include_generic_seeds"]):
        seeds.extend(generic_seed_list())
    seeds = dedup_seeds(seeds)
    if not seeds:
        raise RuntimeError(f"Aucune seed disponible pour alpha={alpha:.3f}, Mach={mach:.3f}.")

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
                    metric_value, metric_name = selection_metric(
                        target_available=bool(target_available),
                        ci_available=bool(ci_available),
                        shooting_cr=float(result.cr),
                        shooting_ci=float(result.ci),
                        blumen_cr=float(blumen_cr),
                        blumen_ci=float(blumen_ci),
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
                            "alpha": float(alpha),
                            "Mach": float(mach),
                            "blumen_cr": float(blumen_cr),
                            "blumen_ci": float(blumen_ci),
                            "blumen_cr_available": bool(cr_available),
                            "blumen_ci_available": bool(ci_available),
                            "blumen_target_available": bool(target_available),
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

        ranked = sorted(
            candidate_rows,
            key=lambda row: (
                0 if bool(row["success"]) else 1,
                float(row["selection_metric"]),
                float(row["stage1_mismatch"] + row["stage2_mismatch"]),
            ),
        )
        best = ranked[0]
        fields = reconstruct_shooting_fields(
            alpha=float(alpha),
            mach=float(mach),
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
        diag.update(infer_regimes(mach=float(mach), cr=float(best["shooting_cr"]), ci=float(best["shooting_ci"])))
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
        diag["box_truncation_suspect_any_field"] = bool(float(diag["max_field_edge_amp_fraction"]) > float(cfg["edge_amp_threshold"]))
        diag.update(
            run_box_robustness_audit(
                alpha=float(alpha),
                mach=float(mach),
                cr=float(best["shooting_cr"]),
                ci=float(best["shooting_ci"]),
                ln_p_start_right=float(best["ln_p_start_right"]),
                cfg=cfg,
                base_fields=fields,
            )
        )
        final_status, box_rejection_applied = apply_box_rejection(str(best["status"]), diag)

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
            "blumen_target_available": bool(target_available),
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


def plot_status_map(summary_df: pd.DataFrame, output_path: Path) -> None:
    color_map = {
        "validated": "#15803D",
        "spectral_only": "#D97706",
        "mode_only": "#7C3AED",
        "box_rejected": "#64748B",
        "failed": "#DC2626",
        "exception": "#111827",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for status, sub in summary_df.groupby("best_status", sort=False):
        ax.scatter(
            sub["Mach"],
            sub["alpha"],
            s=70,
            label=status,
            color=color_map.get(str(status), "#4B5563"),
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Supersonic shooting point audit: status map")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    status_colors = {
        "validated": "#15803D",
        "spectral_only": "#D97706",
        "mode_only": "#7C3AED",
        "box_rejected": "#64748B",
        "failed": "#DC2626",
        "exception": "#111827",
    }
    colors = [status_colors.get(str(status), "#4B5563") for status in summary_df["best_status"]]
    axes[0, 0].scatter(summary_df["Mach"], summary_df["best_stage1_mismatch"], c=colors, s=65, edgecolors="black", linewidths=0.4)
    axes[0, 0].set_title("stage1 mismatch")
    axes[0, 1].scatter(summary_df["Mach"], summary_df["best_stage2_mismatch"], c=colors, s=65, edgecolors="black", linewidths=0.4)
    axes[0, 1].set_title("stage2 mismatch")
    axes[1, 0].scatter(summary_df["Mach"], summary_df["max_field_edge_amp_fraction"], c=colors, s=65, edgecolors="black", linewidths=0.4)
    axes[1, 0].set_title("max boundary amplitude fraction across fields")
    axes[1, 1].scatter(summary_df["Mach"], summary_df["center8_mass_fraction"], c=colors, s=65, edgecolors="black", linewidths=0.4)
    axes[1, 1].set_title("center8 mass fraction")
    for ax in axes.ravel():
        ax.set_xlabel(r"Mach $M$")
        ax.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_modes_pdf(
    summary_df: pd.DataFrame,
    fields_df: pd.DataFrame,
    *,
    threshold_ratio: float,
    min_half_width: float,
    output_path: Path,
) -> None:
    field_names = ["rho", "u", "v", "p"]
    field_titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]
    with PdfPages(output_path) as pdf:
        for _, row in summary_df.sort_values(["alpha", "Mach"]).iterrows():
            alpha = float(row["alpha"])
            mach = float(row["Mach"])
            prof = fields_df[
                np.isclose(fields_df["alpha"].to_numpy(dtype=float), alpha)
                & np.isclose(fields_df["Mach"].to_numpy(dtype=float), mach)
            ].copy()
            if prof.empty:
                continue
            y = prof["y"].to_numpy(dtype=float)
            complex_fields = [
                prof["rho_real"].to_numpy(dtype=float) + 1j * prof["rho_imag"].to_numpy(dtype=float),
                prof["u_real"].to_numpy(dtype=float) + 1j * prof["u_imag"].to_numpy(dtype=float),
                prof["v_real"].to_numpy(dtype=float) + 1j * prof["v_imag"].to_numpy(dtype=float),
                prof["p_real"].to_numpy(dtype=float) + 1j * prof["p_imag"].to_numpy(dtype=float),
            ]
            x_limits = compute_visible_xlim(
                y,
                complex_fields,
                threshold_ratio=float(threshold_ratio),
                min_half_width=float(min_half_width),
            )

            fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
            for ax, field_name, title in zip(axes.ravel(), field_names, field_titles):
                ax.plot(y, prof[f"{field_name}_real"], color="black", linewidth=1.8, label="Real")
                ax.plot(y, prof[f"{field_name}_imag"], color="#D97706", linestyle="--", linewidth=1.3, label="Imag")
                ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                ax.set_title(title)
                ax.set_xlim(*x_limits)
                ax.grid(True, alpha=0.25)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(
                f"alpha={alpha:.3f}, M={mach:.3f} | status={row['best_status']}\n"
                f"c_r={float(row['best_shooting_cr']):.5f}, c_i={float(row['best_shooting_ci']):.5f}, "
                f"stage1={float(row['best_stage1_mismatch']):.3e}, stage2={float(row['best_stage2_mismatch']):.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    points = resolve_points(raw_points=args.points, mach=args.mach, alphas_csv=args.alphas)
    cfg = {
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
        "cr_points": str(args.cr_points),
        "ci_points": str(args.ci_points),
    }

    print("Supersonic shooting point audit")
    print(f"points: {' '.join(f'{alpha:.3f}:{mach:.3f}' for alpha, mach in points)}")
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
        f"edge_growth<={float(args.box_robustness_max_edge_growth):.3f}"
    )

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
                f"err_ci={float(summary_row['best_err_ci_abs']) if np.isfinite(summary_row['best_err_ci_abs']) else np.nan:.3e} "
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
    points_path.write_text("\n".join(f"{alpha:.6f}:{mach:.6f}" for alpha, mach in points) + "\n", encoding="utf-8")
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
