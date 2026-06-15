from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "shooting"
DEFAULT_SPECTRAL_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral.csv"
DEFAULT_MODAL_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal.csv"
DEFAULT_MODAL_FIELDS_REF = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_modal_fields.csv"

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
            "Merge des sorties pointwise/branch-guided supersoniques dans les CSV de reference, "
            "avec classification stricte gold/silver/ambigu/reject."
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--source-label", type=str, required=True)
    parser.add_argument("--continuation-direction", type=str, default="branch_guided")
    parser.add_argument("--spectral-ref", type=Path, default=DEFAULT_SPECTRAL_REF)
    parser.add_argument("--modal-ref", type=Path, default=DEFAULT_MODAL_REF)
    parser.add_argument("--modal-fields-ref", type=Path, default=DEFAULT_MODAL_FIELDS_REF)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--include-silver-spectral", action="store_true")
    parser.add_argument("--include-internal-mode-only", action="store_true")
    parser.add_argument("--max-gold-ci-rel", type=float, default=0.10)
    parser.add_argument("--max-silver-ci-rel", type=float, default=0.20)
    parser.add_argument("--max-gold-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-silver-stage1", type=float, default=7.5e-2)
    parser.add_argument("--max-stage2", type=float, default=1.0e-2)
    parser.add_argument("--max-gold-edge", type=float, default=2.0e-2)
    parser.add_argument("--max-silver-edge", type=float, default=5.0e-2)
    parser.add_argument("--forbid-selection-metric", type=str, nargs="*", default=["max_ci_fallback"])
    parser.add_argument("--report", type=Path, default=None)
    return parser


def row_key(mach: float, alpha: float) -> tuple[float, float]:
    return (round(float(mach), 8), round(float(alpha), 8))


def finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def classify_row(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    if not truthy(row.get("best_spectral_success", False)):
        return "reject", "spectral_success_false"
    if not truthy(row.get("best_mode_success", False)):
        return "spectral_only", "mode_success_false"
    if str(row.get("best_status", "")) != "validated":
        return "reject", "status_not_validated"

    metric_name = str(row.get("best_selection_metric_name", ""))
    if metric_name in set(args.forbid_selection_metric):
        return "reject", f"forbidden_metric:{metric_name}"

    stage1 = finite_float(row.get("best_stage1_mismatch"))
    stage2 = finite_float(row.get("best_stage2_mismatch"))
    edge = finite_float(row.get("max_field_edge_amp_fraction", row.get("edge_amp_fraction_max")))
    box_any = truthy(row.get("box_truncation_suspect_any_field", False))
    ci_available = truthy(row.get("blumen_ci_available", False))
    ci_rel = finite_float(row.get("best_err_ci_rel"))

    if stage2 > float(args.max_stage2):
        return "reject", "stage2_too_large"
    if box_any:
        return "reject", "box_truncation_suspect"

    ci_ok_gold = (not ci_available) or (np.isfinite(ci_rel) and ci_rel <= float(args.max_gold_ci_rel))
    ci_ok_silver = (not ci_available) or (np.isfinite(ci_rel) and ci_rel <= float(args.max_silver_ci_rel))

    if ci_available and ci_ok_gold and stage1 <= float(args.max_gold_stage1) and edge <= float(args.max_gold_edge):
        return "gold", "strict_gold"
    if ci_available and ci_ok_silver and stage1 <= float(args.max_silver_stage1) and edge <= float(args.max_silver_edge):
        return "silver_spectral", "loose_spectral_or_modal"
    if (not ci_available) and stage1 <= float(args.max_gold_stage1) and edge <= float(args.max_gold_edge):
        return "internal_mode_only", "no_blumen_ci_but_clean_mode"
    if ci_available:
        return "reject", "ci_or_quality_threshold_failed"
    return "reject", "no_blumen_ci_and_not_clean_enough"


def ensure_reference_cols(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in REFERENCE_COLS:
        if col not in work.columns:
            work[col] = pd.NA
    return work[REFERENCE_COLS]


def build_reference_rows(summary_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = summary_df.copy()
    classes: list[str] = []
    reasons: list[str] = []
    for _, row in work.iterrows():
        cls, reason = classify_row(row, args)
        classes.append(cls)
        reasons.append(reason)
    work["strict_class"] = classes
    work["strict_reason"] = reasons

    accept_spectral = work["strict_class"].isin(["gold"])
    if args.include_silver_spectral:
        accept_spectral |= work["strict_class"].isin(["silver_spectral"])
    if args.include_internal_mode_only:
        accept_spectral |= work["strict_class"].isin(["internal_mode_only"])
    accepted = work[accept_spectral].copy()
    if accepted.empty:
        return pd.DataFrame(columns=REFERENCE_COLS), work

    accepted["reference_cr"] = accepted["best_shooting_cr"]
    accepted["reference_ci"] = accepted["best_shooting_ci"]
    accepted["reference_omega_i"] = accepted["best_shooting_omega_i"]
    accepted["line_id"] = accepted.apply(lambda r: f"M{float(r.Mach):.2f}_{r.strict_class}_a{float(r.alpha):.6f}", axis=1)
    accepted["anchor_alpha"] = accepted["alpha"]
    accepted["continuation_direction"] = str(args.continuation_direction)
    accepted["continuation_step_index"] = 0
    accepted["continuation_anchor"] = True
    accepted["continuation_accepted"] = True
    accepted["continuation_stop_reason"] = accepted["strict_reason"]
    accepted["acceptance_mode"] = accepted["strict_class"].map(
        {
            "gold": "branch_guided_gold",
            "silver_spectral": "branch_guided_silver_spectral",
            "internal_mode_only": "branch_guided_internal_mode_only",
        }
    )
    accepted["trusted_spectral"] = accepted["strict_class"].isin(["gold", "silver_spectral"])
    accepted["trusted_modal"] = accepted["strict_class"].isin(["gold"])
    if args.include_internal_mode_only:
        accepted.loc[accepted["strict_class"].eq("internal_mode_only"), "trusted_modal"] = True
    accepted["source_csv"] = str(args.summary)
    accepted["source_label"] = str(args.source_label)
    return ensure_reference_cols(accepted), work


def merge_reference(existing_df: pd.DataFrame, new_df: pd.DataFrame, *, replace_existing: bool) -> pd.DataFrame:
    if new_df.empty:
        return existing_df.copy()
    if existing_df.empty:
        return new_df.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    existing = existing_df.copy()
    new = new_df.copy()
    new["__key__"] = [row_key(m, a) for m, a in zip(new["Mach"], new["alpha"])]
    if replace_existing:
        replace_keys = set(new["__key__"])
        existing = existing[[row_key(m, a) not in replace_keys for m, a in zip(existing["Mach"], existing["alpha"])]].copy()
    else:
        existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}
        new = new[~new["__key__"].isin(existing_keys)].copy()
    new = new.drop(columns="__key__", errors="ignore")
    merged = pd.concat([existing, new], ignore_index=True)
    return merged.sort_values(["Mach", "alpha"]).reset_index(drop=True)


def merge_fields(existing_df: pd.DataFrame, fields_df: pd.DataFrame, modal_ref_df: pd.DataFrame, *, replace_existing: bool, source_label: str, source_csv: Path) -> pd.DataFrame:
    modal_keys = {row_key(m, a) for m, a in zip(modal_ref_df["Mach"], modal_ref_df["alpha"])}
    if fields_df.empty or not modal_keys:
        return existing_df.copy()

    new = fields_df[[row_key(m, a) in modal_keys for m, a in zip(fields_df["Mach"], fields_df["alpha"])]].copy()
    if new.empty:
        return existing_df.copy()
    line_map = {
        row_key(row.Mach, row.alpha): str(row.line_id)
        for row in modal_ref_df.itertuples(index=False)
    }
    new["line_id"] = [line_map[row_key(m, a)] for m, a in zip(new["Mach"], new["alpha"])]
    new["source_csv"] = str(source_csv)
    new["source_label"] = str(source_label)

    if existing_df.empty:
        return new.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

    existing = existing_df.copy()
    if replace_existing:
        existing = existing[[row_key(m, a) not in modal_keys for m, a in zip(existing["Mach"], existing["alpha"])]].copy()
    else:
        existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}
        keep_keys = modal_keys - existing_keys
        new = new[[row_key(m, a) in keep_keys for m, a in zip(new["Mach"], new["alpha"])]].copy()
    merged = pd.concat([existing, new], ignore_index=True)
    return merged.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)


def backup_file(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def main() -> None:
    args = build_parser().parse_args()
    summary_df = pd.read_csv(args.summary)
    fields_df = pd.read_csv(args.fields)

    new_ref_df, report_df = build_reference_rows(summary_df, args)
    new_modal_df = new_ref_df[new_ref_df["trusted_modal"].fillna(False).astype(bool)].copy()

    if args.report is None:
        report_path = args.summary.with_name(args.summary.stem + "_strict_report.csv")
    else:
        report_path = args.report
    report_df.to_csv(report_path, index=False)

    existing_spectral_df = pd.read_csv(args.spectral_ref) if args.spectral_ref.exists() else pd.DataFrame(columns=REFERENCE_COLS)
    existing_modal_df = pd.read_csv(args.modal_ref) if args.modal_ref.exists() else pd.DataFrame(columns=REFERENCE_COLS)
    existing_fields_df = pd.read_csv(args.modal_fields_ref) if args.modal_fields_ref.exists() else pd.DataFrame()

    if args.backup_dir is not None:
        backup_file(args.spectral_ref, args.backup_dir)
        backup_file(args.modal_ref, args.backup_dir)
        backup_file(args.modal_fields_ref, args.backup_dir)

    merged_spectral_df = merge_reference(existing_spectral_df, new_ref_df, replace_existing=bool(args.replace_existing))
    merged_modal_df = merge_reference(existing_modal_df, new_modal_df, replace_existing=bool(args.replace_existing))
    merged_fields_df = merge_fields(
        existing_fields_df,
        fields_df,
        new_modal_df,
        replace_existing=bool(args.replace_existing),
        source_label=str(args.source_label),
        source_csv=args.fields,
    )

    merged_spectral_df.to_csv(args.spectral_ref, index=False)
    merged_modal_df.to_csv(args.modal_ref, index=False)
    merged_fields_df.to_csv(args.modal_fields_ref, index=False)

    counts = report_df["strict_class"].value_counts(dropna=False).to_dict()
    print("Strict classification:", counts)
    print(f"accepted_spectral={len(new_ref_df)} accepted_modal={len(new_modal_df)}")
    print(f"Wrote {args.spectral_ref}")
    print(f"Wrote {args.modal_ref}")
    print(f"Wrote {args.modal_fields_ref}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
