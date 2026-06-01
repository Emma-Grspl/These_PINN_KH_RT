from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

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
from scripts.audit_supersonic_shooting_ci_alpha_continuation import (  # noqa: E402
    LineSpec,
    build_cfg,
    evaluate_line,
    line_anchor_key,
    parse_line_spec,
    plot_continuation_errors,
    plot_continuation_lines,
)
from scripts.audit_supersonic_shooting_point_batch import DEFAULT_OUTPUT_DIR, plot_modes_pdf  # noqa: E402


DEFAULT_MODAL_REFERENCE_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal.csv"
DEFAULT_MODAL_FIELDS_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_fields.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Densification spectrale supersonique sur des Mach intermediaires, "
            "avec ancre modale interpolee entre deux lignes de reference voisines."
        )
    )
    parser.add_argument(
        "--line-specs",
        type=str,
        nargs="+",
        default=["1.40:0.150000:0.150000,0.162500,0.168750,0.175000,0.181250,0.187500,0.193750,0.200000"],
        help=(
            "Spec de ligne sous la forme mach:anchor_alpha:alpha1,alpha2,... "
            "Ex: 1.40:0.150000:0.150000,0.162500,0.168750,0.175000,0.181250,0.187500,0.193750,0.200000"
        ),
    )
    parser.add_argument("--modal-reference-csv", type=Path, default=DEFAULT_MODAL_REFERENCE_CSV)
    parser.add_argument("--modal-fields-csv", type=Path, default=DEFAULT_MODAL_FIELDS_CSV)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--anchor-alpha-tolerance", type=float, default=5.0e-4)
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
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument("--max-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-err-ci-abs", type=float, default=1.0e-2)
    parser.add_argument("--max-delta-ci", type=float, default=2.5e-2)
    parser.add_argument("--max-delta-cr", type=float, default=8.0e-2)
    parser.add_argument("--continuation-generic-seeds", action="store_true", dest="continuation_include_generic_seeds")
    parser.set_defaults(continuation_include_generic_seeds=False, anchor_include_generic_seeds=False)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting_intermediate_mach_spectral")
    return parser


def serialize_line_specs(line_specs: list[LineSpec]) -> list[str]:
    return [
        f"{line.mach:.2f}:{line.anchor_alpha:.6f}:{','.join(f'{alpha:.6f}' for alpha in line.alphas)}"
        for line in line_specs
    ]


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def interpolate_scalar(lower_value: object, upper_value: object, weight: float) -> float:
    lower = float(lower_value)
    upper = float(upper_value)
    return float((1.0 - weight) * lower + weight * upper)


def reference_fields_to_complex(df: pd.DataFrame, field: str) -> np.ndarray:
    return df[f"{field}_real"].to_numpy(dtype=float) + 1j * df[f"{field}_imag"].to_numpy(dtype=float)


def interpolate_reference_fields(
    *,
    lower_fields: pd.DataFrame,
    upper_fields: pd.DataFrame,
    target_mach: float,
    target_alpha: float,
    weight: float,
) -> pd.DataFrame:
    lower_sorted = lower_fields.sort_values("y").reset_index(drop=True)
    upper_sorted = upper_fields.sort_values("y").reset_index(drop=True)
    y_min = max(float(lower_sorted["y"].min()), float(upper_sorted["y"].min()))
    y_max = min(float(lower_sorted["y"].max()), float(upper_sorted["y"].max()))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        raise RuntimeError(
            f"Pas de domaine commun pour interpoler les champs a M={target_mach:.3f}, alpha={target_alpha:.6f}"
        )
    n_common = max(len(lower_sorted), len(upper_sorted))
    y_common = np.linspace(y_min, y_max, int(n_common), dtype=float)

    rows: dict[str, np.ndarray] = {
        "Mach": np.full(n_common, float(target_mach)),
        "alpha": np.full(n_common, float(target_alpha)),
        "y": y_common,
    }
    for field in ("rho", "u", "v", "p"):
        lower_complex = reference_fields_to_complex(lower_sorted, field)
        upper_complex = reference_fields_to_complex(upper_sorted, field)
        lower_interp = interp_complex(lower_sorted["y"].to_numpy(dtype=float), lower_complex, y_common)
        upper_interp = interp_complex(upper_sorted["y"].to_numpy(dtype=float), upper_complex, y_common)
        target_complex = (1.0 - float(weight)) * lower_interp + float(weight) * upper_interp
        rows[f"{field}_real"] = np.real(target_complex)
        rows[f"{field}_imag"] = np.imag(target_complex)
    return pd.DataFrame(rows)


def build_interpolated_anchor_summary(
    *,
    line: LineSpec,
    target: pd.Series,
    lower_row: pd.Series,
    upper_row: pd.Series,
    weight: float,
    lower_mach: float,
    upper_mach: float,
) -> dict[str, object]:
    best_cr = interpolate_scalar(lower_row["reference_cr"], upper_row["reference_cr"], weight)
    best_ci = interpolate_scalar(lower_row["reference_ci"], upper_row["reference_ci"], weight)
    exact_modal_reference = bool(np.isclose(lower_mach, upper_mach))

    blumen_cr = float(target["blumen_cr"])
    blumen_ci = float(target["blumen_ci"])
    cr_available = bool(np.isfinite(blumen_cr))
    ci_available = bool(np.isfinite(blumen_ci))
    err_cr = abs(best_cr - blumen_cr) if cr_available else np.nan
    err_ci = abs(best_ci - blumen_ci) if ci_available else np.nan
    err_ci_rel = err_ci / max(abs(blumen_ci), 1.0e-12) if ci_available else np.nan

    return {
        "Mach": float(line.mach),
        "alpha": float(line.anchor_alpha),
        "reference_cr": float(best_cr),
        "reference_ci": float(best_ci),
        "reference_omega_i": float(line.anchor_alpha) * float(best_ci),
        "best_seed_name": "interpolated_modal_reference",
        "best_shooting_cr": float(best_cr),
        "best_shooting_ci": float(best_ci),
        "best_shooting_omega_i": float(line.anchor_alpha) * float(best_ci),
        "best_err_cr_abs": float(err_cr) if exact_modal_reference and np.isfinite(err_cr) else np.nan,
        "best_err_ci_abs": float(err_ci) if exact_modal_reference and np.isfinite(err_ci) else np.nan,
        "best_err_ci_rel": float(err_ci_rel) if exact_modal_reference and np.isfinite(err_ci_rel) else np.nan,
        "best_stage1_mismatch": 0.0,
        "best_stage2_mismatch": 0.0,
        "best_ln_p_start_right": interpolate_scalar(
            lower_row.get("best_ln_p_start_right", -20.0),
            upper_row.get("best_ln_p_start_right", -20.0),
            weight,
        ),
        "best_spectral_success": True,
        "best_mode_success": True,
        "best_success": True,
        "best_status": "validated" if exact_modal_reference else "interpolated_anchor",
        "best_selection_metric": 0.0,
        "best_selection_metric_name": "interpolated_locked_reference",
        "best_retry_index": 0,
        "best_used_cr_half_window": np.nan,
        "best_used_ci_half_window": np.nan,
        "best_y_limit": np.nan,
        "blumen_cr": blumen_cr if cr_available else np.nan,
        "blumen_ci": blumen_ci if ci_available else np.nan,
        "blumen_cr_available": bool(cr_available) if exact_modal_reference else False,
        "blumen_ci_available": bool(ci_available) if exact_modal_reference else False,
        "source_label": "interpolated_modal_reference_mach",
        "source_csv": "",
        "anchor_lower_mach": float(lower_mach),
        "anchor_upper_mach": float(upper_mach),
        "anchor_interp_weight": float(weight),
    }


def pick_bracketing_modal_rows(
    modal_reference_df: pd.DataFrame,
    *,
    target_mach: float,
    target_alpha: float,
    alpha_tolerance: float,
) -> tuple[pd.Series, pd.Series, float, float]:
    same_alpha = modal_reference_df[
        np.isclose(
            modal_reference_df["alpha"].to_numpy(dtype=float),
            float(target_alpha),
            atol=float(alpha_tolerance),
            rtol=0.0,
        )
    ].copy()
    if same_alpha.empty:
        raise RuntimeError(
            f"Aucune ancre modale proche de alpha={target_alpha:.6f} pour construire M={target_mach:.3f}"
        )
    same_alpha = same_alpha.sort_values("Mach").reset_index(drop=True)
    exact = same_alpha[np.isclose(same_alpha["Mach"].to_numpy(dtype=float), float(target_mach), atol=1.0e-10, rtol=0.0)]
    if not exact.empty:
        row = exact.iloc[0]
        mach = float(row["Mach"])
        return row, row, mach, mach

    lower = same_alpha[same_alpha["Mach"].to_numpy(dtype=float) < float(target_mach)]
    upper = same_alpha[same_alpha["Mach"].to_numpy(dtype=float) > float(target_mach)]
    if lower.empty or upper.empty:
        available = ", ".join(f"{float(value):.3f}" for value in same_alpha["Mach"].to_numpy(dtype=float))
        raise RuntimeError(
            f"Impossible de bracketter M={target_mach:.3f} a alpha={target_alpha:.6f}. Machs disponibles: {available}"
        )
    lower_row = lower.iloc[-1]
    upper_row = upper.iloc[0]
    return lower_row, upper_row, float(lower_row["Mach"]), float(upper_row["Mach"])


def build_intermediate_anchor_overrides(
    line_specs: list[LineSpec],
    *,
    modal_reference_csv: Path,
    modal_fields_csv: Path,
    alpha_tolerance: float,
    cr_points: pd.DataFrame,
    ci_points: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    if not modal_reference_csv.exists():
        raise FileNotFoundError(f"Modal reference csv introuvable: {modal_reference_csv}")
    if not modal_fields_csv.exists():
        raise FileNotFoundError(f"Modal fields csv introuvable: {modal_fields_csv}")

    modal_reference_df = pd.read_csv(modal_reference_csv).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    modal_fields_df = pd.read_csv(modal_fields_csv).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

    overrides: dict[str, dict[str, object]] = {}
    report_rows: list[dict[str, object]] = []

    for line in line_specs:
        target_df = build_blumen_targets([float(line.mach)], float(line.anchor_alpha), cr_points, ci_points)
        target = target_df.iloc[0]
        lower_row, upper_row, lower_mach, upper_mach = pick_bracketing_modal_rows(
            modal_reference_df,
            target_mach=float(line.mach),
            target_alpha=float(line.anchor_alpha),
            alpha_tolerance=float(alpha_tolerance),
        )

        if np.isclose(lower_mach, upper_mach):
            weight = 0.0
        else:
            weight = float((float(line.mach) - lower_mach) / (upper_mach - lower_mach))

        lower_fields = modal_fields_df[
            np.isclose(modal_fields_df["Mach"].to_numpy(dtype=float), lower_mach, atol=1.0e-10, rtol=0.0)
            & np.isclose(
                modal_fields_df["alpha"].to_numpy(dtype=float),
                float(lower_row["alpha"]),
                atol=float(alpha_tolerance),
                rtol=0.0,
            )
        ].copy()
        upper_fields = modal_fields_df[
            np.isclose(modal_fields_df["Mach"].to_numpy(dtype=float), upper_mach, atol=1.0e-10, rtol=0.0)
            & np.isclose(
                modal_fields_df["alpha"].to_numpy(dtype=float),
                float(upper_row["alpha"]),
                atol=float(alpha_tolerance),
                rtol=0.0,
            )
        ].copy()
        reference_fields = None
        if not lower_fields.empty and not upper_fields.empty:
            if np.isclose(lower_mach, upper_mach):
                reference_fields = lower_fields.copy()
                reference_fields.loc[:, "Mach"] = float(line.mach)
                reference_fields.loc[:, "alpha"] = float(line.anchor_alpha)
            else:
                reference_fields = interpolate_reference_fields(
                    lower_fields=lower_fields,
                    upper_fields=upper_fields,
                    target_mach=float(line.mach),
                    target_alpha=float(line.anchor_alpha),
                    weight=float(weight),
                )

        summary_row = build_interpolated_anchor_summary(
            line=line,
            target=target,
            lower_row=lower_row,
            upper_row=upper_row,
            weight=float(weight),
            lower_mach=float(lower_mach),
            upper_mach=float(upper_mach),
        )
        overrides[line_anchor_key(line.mach, line.anchor_alpha)] = {
            "summary_row": summary_row,
            "reference_fields": reference_fields,
        }
        report_rows.append(
            {
                "line_id": line.line_id,
                "target_mach": float(line.mach),
                "anchor_alpha": float(line.anchor_alpha),
                "lower_mach": float(lower_mach),
                "upper_mach": float(upper_mach),
                "interp_weight": float(weight),
                "interpolated_cr": float(summary_row["best_shooting_cr"]),
                "interpolated_ci": float(summary_row["best_shooting_ci"]),
                "blumen_ci": float(summary_row["blumen_ci"]) if np.isfinite(summary_row["blumen_ci"]) else np.nan,
                "anchor_source_mode": "exact_modal_reference" if np.isclose(lower_mach, upper_mach) else "interpolated_modal_reference",
            }
        )
    report_df = pd.DataFrame(report_rows).sort_values(["target_mach", "anchor_alpha"]).reset_index(drop=True)
    return overrides, report_df


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    line_specs = [parse_line_spec(raw) for raw in args.line_specs]
    line_specs_text = serialize_line_specs(line_specs)
    line_specs_path = output_dir / f"{args.output_stem}_line_specs.txt"
    line_specs_path.write_text("\n".join(line_specs_text) + "\n", encoding="utf-8")

    cr_points = load_digitized_long(Path(str(args.cr_points)))
    ci_points = load_digitized_long(Path(str(args.ci_points)))
    anchor_overrides, anchor_report_df = build_intermediate_anchor_overrides(
        line_specs,
        modal_reference_csv=args.modal_reference_csv,
        modal_fields_csv=args.modal_fields_csv,
        alpha_tolerance=float(args.anchor_alpha_tolerance),
        cr_points=cr_points,
        ci_points=ci_points,
    )
    anchor_report_path = output_dir / f"{args.output_stem}_anchor_report.csv"
    anchor_report_df.to_csv(anchor_report_path, index=False)

    print("Supersonic intermediate-Mach spectral densification")
    print(f"modal_reference_csv={args.modal_reference_csv}")
    print(f"modal_fields_csv={args.modal_fields_csv}")
    print(f"workers={int(args.workers)}")
    print(f"anchor_alpha_tolerance={float(args.anchor_alpha_tolerance):.3e}")
    for raw in line_specs_text:
        print(f"line-spec={raw}")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nInterpolated anchors:")
        print(anchor_report_df.to_string(index=False))
    print(f"Wrote {line_specs_path}")
    print(f"Wrote {anchor_report_path}")

    if args.dry_run:
        return

    cfg = build_cfg(args)
    cfg["acceptance_mode"] = "spectral"
    cfg["anchor_overrides"] = anchor_overrides

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
        plot_modes_pdf(summary_df, fields_df, threshold_ratio=0.02, min_half_width=8.0, output_path=modes_path)

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
