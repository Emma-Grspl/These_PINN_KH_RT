from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "shooting"
DEFAULT_POINTS_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral.csv"
DEFAULT_ANCHOR_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal.csv"
DEFAULT_ANCHOR_FIELDS_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_fields.csv"
DEFAULT_BRANCH_GUIDED_SUMMARY = DEFAULT_OUTPUT_DIR / "supersonic_shooting_point_batch_M140_branch_guided_summary.csv"
DEFAULT_BRANCH_GUIDED_FIELDS = DEFAULT_OUTPUT_DIR / "supersonic_shooting_point_batch_M140_branch_guided_fields.csv"
DEFAULT_OUTPUT_POINTS = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral_M140_branch_guided.csv"
DEFAULT_OUTPUT_ANCHORS = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_M140_branch_guided.csv"
DEFAULT_OUTPUT_ANCHOR_FIELDS = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_fields_M140_branch_guided.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_M140_branch_guided_report.csv"


REFERENCE_COLS = [
    "alpha",
    "Mach",
    "reference_cr",
    "reference_ci",
    "reference_omega_i",
    "blumen_cr",
    "blumen_ci",
    "blumen_cr_available",
    "blumen_ci_available",
    "best_err_cr_abs",
    "best_err_ci_abs",
    "best_err_ci_rel",
    "best_stage1_mismatch",
    "best_stage2_mismatch",
    "best_spectral_success",
    "best_mode_success",
    "best_status",
    "line_id",
    "anchor_alpha",
    "continuation_direction",
    "continuation_step_index",
    "continuation_anchor",
    "continuation_accepted",
    "continuation_stop_reason",
    "acceptance_mode",
    "trusted_spectral",
    "trusted_modal",
    "source_csv",
    "source_label",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare experimental spectral/modal reference CSV copies for the supersonic "
            "modal-surface tracker by injecting a branch-guided M=1.4 line."
        )
    )
    parser.add_argument("--branch-guided-summary", type=Path, default=DEFAULT_BRANCH_GUIDED_SUMMARY)
    parser.add_argument("--branch-guided-fields", type=Path, default=DEFAULT_BRANCH_GUIDED_FIELDS)
    parser.add_argument("--points-ref", type=Path, default=DEFAULT_POINTS_REF)
    parser.add_argument("--anchor-ref", type=Path, default=DEFAULT_ANCHOR_REF)
    parser.add_argument("--anchor-fields-ref", type=Path, default=DEFAULT_ANCHOR_FIELDS_REF)
    parser.add_argument("--output-points", type=Path, default=DEFAULT_OUTPUT_POINTS)
    parser.add_argument("--output-anchors", type=Path, default=DEFAULT_OUTPUT_ANCHORS)
    parser.add_argument("--output-anchor-fields", type=Path, default=DEFAULT_OUTPUT_ANCHOR_FIELDS)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--points-statuses",
        type=str,
        nargs="+",
        default=["validated", "spectral_only"],
        help="Accepted statuses to inject into the experimental spectral points line.",
    )
    parser.add_argument(
        "--anchor-statuses",
        type=str,
        nargs="+",
        default=["validated"],
        help="Accepted statuses to promote as experimental modal anchors.",
    )
    parser.add_argument("--max-anchor-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-anchor-stage2", type=float, default=1.0e-1)
    parser.add_argument("--max-point-stage1", type=float, default=5.0e-2)
    parser.add_argument("--source-label-points", type=str, default="branch_guided_M140_points")
    parser.add_argument("--source-label-anchors", type=str, default="branch_guided_M140_modal_candidates")
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def row_key(mach: float, alpha: float) -> tuple[float, float]:
    return (round(float(mach), 8), round(float(alpha), 8))


def ensure_reference_cols(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in REFERENCE_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    return work[REFERENCE_COLS]


def ensure_anchor_fields_cols(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "line_id",
        "alpha",
        "Mach",
        "best_status",
        "y",
        "rho_real",
        "rho_imag",
        "u_real",
        "u_imag",
        "v_real",
        "v_imag",
        "p_real",
        "p_imag",
        "source_csv",
        "source_label",
    ]
    work = df.copy()
    for col in required:
        if col not in work.columns:
            work[col] = pd.NA
    return work[required]


def merge_reference(existing_df: pd.DataFrame, new_df: pd.DataFrame, *, replace_existing: bool) -> tuple[pd.DataFrame, int]:
    if existing_df.empty:
        merged = new_df.sort_values(["Mach", "alpha"]).reset_index(drop=True)
        return merged, len(new_df)
    if new_df.empty:
        return existing_df.copy(), 0

    existing = existing_df.copy()
    new = new_df.copy()
    new["__key__"] = [row_key(m, a) for m, a in zip(new["Mach"], new["alpha"])]
    existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}

    if replace_existing:
        replace_keys = set(new["__key__"].tolist())
        existing = existing[[row_key(m, a) not in replace_keys for m, a in zip(existing["Mach"], existing["alpha"])]].copy()
        added = len(new)
    else:
        new = new[~new["__key__"].isin(existing_keys)].copy()
        added = len(new)

    new = new.drop(columns="__key__", errors="ignore")
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    return merged, added


def merge_fields(existing_df: pd.DataFrame, new_df: pd.DataFrame, *, replace_existing: bool) -> tuple[pd.DataFrame, int]:
    if existing_df.empty:
        merged = new_df.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
        added_keys = len({row_key(m, a) for m, a in zip(new_df["Mach"], new_df["alpha"])})
        return merged, added_keys
    if new_df.empty:
        return existing_df.copy(), 0

    existing = existing_df.copy()
    new = new_df.copy()
    new_keys = {row_key(m, a) for m, a in zip(new["Mach"], new["alpha"])}

    if replace_existing:
        existing = existing[[row_key(m, a) not in new_keys for m, a in zip(existing["Mach"], existing["alpha"])]].copy()
        added = len(new_keys)
    else:
        existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}
        keep_keys = new_keys - existing_keys
        new = new[[row_key(m, a) in keep_keys for m, a in zip(new["Mach"], new["alpha"])]].copy()
        added = len(keep_keys)

    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
    return merged, added


def build_line_id(mach: float, alpha: float) -> str:
    return f"M{mach:.2f}_branch_guided_{alpha:.6f}".replace(".", "p")


def summary_rows_to_reference(
    summary_df: pd.DataFrame,
    *,
    source_csv: Path,
    source_label: str,
    trusted_modal: bool,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=REFERENCE_COLS)

    work = summary_df.copy()
    work["reference_cr"] = work["best_shooting_cr"]
    work["reference_ci"] = work["best_shooting_ci"]
    work["reference_omega_i"] = work["best_shooting_omega_i"]
    work["line_id"] = [
        build_line_id(float(mach), float(alpha))
        for mach, alpha in zip(work["Mach"], work["alpha"])
    ]
    work["anchor_alpha"] = work["alpha"].astype(float)
    work["continuation_direction"] = "branch_guided"
    work["continuation_step_index"] = 0
    work["continuation_anchor"] = trusted_modal
    work["continuation_accepted"] = True
    work["continuation_stop_reason"] = pd.NA
    work["acceptance_mode"] = "spectral"
    work["trusted_spectral"] = True
    work["trusted_modal"] = bool(trusted_modal)
    work["source_csv"] = str(source_csv)
    work["source_label"] = str(source_label)
    return ensure_reference_cols(work)


def decorate_anchor_fields(
    fields_df: pd.DataFrame,
    *,
    anchor_ref_df: pd.DataFrame,
    source_csv: Path,
    source_label: str,
) -> pd.DataFrame:
    if fields_df.empty:
        return ensure_anchor_fields_cols(fields_df)

    line_id_map = {
        row_key(mach, alpha): line_id
        for mach, alpha, line_id in zip(anchor_ref_df["Mach"], anchor_ref_df["alpha"], anchor_ref_df["line_id"])
    }
    work = fields_df.copy()
    work["line_id"] = [
        line_id_map.get(row_key(mach, alpha), build_line_id(float(mach), float(alpha)))
        for mach, alpha in zip(work["Mach"], work["alpha"])
    ]
    work["source_csv"] = str(source_csv)
    work["source_label"] = str(source_label)
    return ensure_anchor_fields_cols(work)


def main() -> None:
    args = build_parser().parse_args()

    summary_df = pd.read_csv(args.branch_guided_summary)
    fields_df = pd.read_csv(args.branch_guided_fields)
    points_ref_df = pd.read_csv(args.points_ref)
    anchor_ref_df = pd.read_csv(args.anchor_ref)
    anchor_fields_ref_df = pd.read_csv(args.anchor_fields_ref)

    point_statuses = {str(value) for value in args.points_statuses}
    anchor_statuses = {str(value) for value in args.anchor_statuses}

    points_mask = (
        summary_df["best_status"].astype(str).isin(point_statuses)
        & summary_df["best_spectral_success"].fillna(False).astype(bool)
        & np.isfinite(summary_df["best_shooting_cr"])
        & np.isfinite(summary_df["best_shooting_ci"])
        & np.isfinite(summary_df["best_stage1_mismatch"])
        & (summary_df["best_stage1_mismatch"].astype(float) <= float(args.max_point_stage1))
        & ~summary_df["box_truncation_suspect_any_field"].fillna(False).astype(bool)
    )
    anchor_mask = (
        summary_df["best_status"].astype(str).isin(anchor_statuses)
        & summary_df["best_spectral_success"].fillna(False).astype(bool)
        & summary_df["best_mode_success"].fillna(False).astype(bool)
        & np.isfinite(summary_df["best_shooting_cr"])
        & np.isfinite(summary_df["best_shooting_ci"])
        & np.isfinite(summary_df["best_stage1_mismatch"])
        & (summary_df["best_stage1_mismatch"].astype(float) <= float(args.max_anchor_stage1))
        & np.isfinite(summary_df["best_stage2_mismatch"])
        & (summary_df["best_stage2_mismatch"].astype(float) <= float(args.max_anchor_stage2))
        & ~summary_df["box_truncation_suspect_any_field"].fillna(False).astype(bool)
    )

    point_summary = summary_df[points_mask].copy().sort_values(["Mach", "alpha"]).reset_index(drop=True)
    anchor_summary = summary_df[anchor_mask].copy().sort_values(["Mach", "alpha"]).reset_index(drop=True)

    point_ref_add = summary_rows_to_reference(
        point_summary,
        source_csv=args.branch_guided_summary,
        source_label=args.source_label_points,
        trusted_modal=False,
    )
    anchor_ref_add = summary_rows_to_reference(
        anchor_summary,
        source_csv=args.branch_guided_summary,
        source_label=args.source_label_anchors,
        trusted_modal=True,
    )

    anchor_keys = {row_key(mach, alpha) for mach, alpha in zip(anchor_summary["Mach"], anchor_summary["alpha"])}
    anchor_fields_add = fields_df[
        [row_key(mach, alpha) in anchor_keys for mach, alpha in zip(fields_df["Mach"], fields_df["alpha"])]
    ].copy()
    anchor_fields_add = decorate_anchor_fields(
        anchor_fields_add,
        anchor_ref_df=anchor_ref_add,
        source_csv=args.branch_guided_fields,
        source_label=args.source_label_anchors,
    )

    merged_points_ref, n_points_added = merge_reference(
        ensure_reference_cols(points_ref_df),
        point_ref_add,
        replace_existing=bool(args.replace_existing),
    )
    merged_anchor_ref, n_anchor_added = merge_reference(
        ensure_reference_cols(anchor_ref_df),
        anchor_ref_add,
        replace_existing=bool(args.replace_existing),
    )
    merged_anchor_fields, n_anchor_field_keys_added = merge_fields(
        ensure_anchor_fields_cols(anchor_fields_ref_df),
        anchor_fields_add,
        replace_existing=bool(args.replace_existing),
    )

    args.output_points.parent.mkdir(parents=True, exist_ok=True)
    args.output_anchors.parent.mkdir(parents=True, exist_ok=True)
    args.output_anchor_fields.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    merged_points_ref.to_csv(args.output_points, index=False)
    merged_anchor_ref.to_csv(args.output_anchors, index=False)
    merged_anchor_fields.to_csv(args.output_anchor_fields, index=False)

    report_rows: list[dict[str, object]] = []
    for _, row in point_summary.iterrows():
        key = row_key(row["Mach"], row["alpha"])
        report_rows.append(
            {
                "kind": "point",
                "Mach": float(row["Mach"]),
                "alpha": float(row["alpha"]),
                "best_status": str(row["best_status"]),
                "best_shooting_cr": float(row["best_shooting_cr"]),
                "best_shooting_ci": float(row["best_shooting_ci"]),
                "best_stage1_mismatch": float(row["best_stage1_mismatch"]),
                "best_stage2_mismatch": float(row["best_stage2_mismatch"]),
                "best_mode_success": bool(row["best_mode_success"]),
                "trusted_modal": bool(key in anchor_keys),
            }
        )
    report_df = pd.DataFrame(report_rows).sort_values(["kind", "Mach", "alpha"]).reset_index(drop=True)
    report_df.to_csv(args.report_path, index=False)

    print("Prepared experimental supersonic modal-surface inputs")
    print(f"branch_guided_summary={args.branch_guided_summary}")
    print(f"branch_guided_fields={args.branch_guided_fields}")
    print(f"points_added={n_points_added} total_points={len(merged_points_ref)}")
    print(f"anchors_added={n_anchor_added} total_anchors={len(merged_anchor_ref)}")
    print(f"anchor_field_keys_added={n_anchor_field_keys_added} total_anchor_field_rows={len(merged_anchor_fields)}")
    print(f"Wrote {args.output_points}")
    print(f"Wrote {args.output_anchors}")
    print(f"Wrote {args.output_anchor_fields}")
    print(f"Wrote {args.report_path}")

    if not point_summary.empty:
        print("\nInjected M=1.4 spectral line:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                point_summary[
                    [
                        "Mach",
                        "alpha",
                        "best_status",
                        "best_shooting_cr",
                        "best_shooting_ci",
                        "best_stage1_mismatch",
                        "best_stage2_mismatch",
                        "best_mode_success",
                    ]
                ].to_string(index=False)
            )

    if not anchor_summary.empty:
        print("\nPromoted M=1.4 modal anchors:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                anchor_summary[
                    [
                        "Mach",
                        "alpha",
                        "best_status",
                        "best_shooting_cr",
                        "best_shooting_ci",
                        "best_stage1_mismatch",
                        "best_stage2_mismatch",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
