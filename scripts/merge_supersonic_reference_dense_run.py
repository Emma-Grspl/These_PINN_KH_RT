from __future__ import annotations

import argparse
from pathlib import Path
import shutil

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
        description="Merge an incremental supersonic dense shooting run into the local spectral/modal reference CSVs."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--spectral-ref", type=Path, default=DEFAULT_SPECTRAL_REF)
    parser.add_argument("--modal-ref", type=Path, default=DEFAULT_MODAL_REF)
    parser.add_argument("--modal-fields-ref", type=Path, default=DEFAULT_MODAL_FIELDS_REF)
    parser.add_argument("--source-label", type=str, default="modal_front_spectral_dense_M15")
    parser.add_argument("--backup-dir", type=Path, default=None)
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


def summary_to_reference(summary_df: pd.DataFrame, *, source_csv: Path, source_label: str, modal_only: bool) -> pd.DataFrame:
    accepted = summary_df[summary_df["continuation_accepted"].fillna(False)].copy()
    accepted = accepted[accepted["acceptance_mode"].astype(str) == "spectral"].copy()
    if modal_only:
        accepted = accepted[accepted["best_mode_success"].fillna(False)].copy()
        accepted = accepted[accepted["best_status"].astype(str) == "validated"].copy()
    if accepted.empty:
        return pd.DataFrame(columns=REFERENCE_COLS)

    accepted["reference_cr"] = accepted["best_shooting_cr"]
    accepted["reference_ci"] = accepted["best_shooting_ci"]
    accepted["reference_omega_i"] = accepted["best_shooting_omega_i"]
    accepted["trusted_spectral"] = True
    accepted["trusted_modal"] = modal_only
    accepted["source_csv"] = str(source_csv)
    accepted["source_label"] = str(source_label)
    return ensure_reference_cols(accepted)


def merge_reference(existing_df: pd.DataFrame, new_df: pd.DataFrame, *, replace_existing: bool) -> tuple[pd.DataFrame, int]:
    if existing_df.empty:
        merged = new_df.sort_values(["Mach", "alpha"]).reset_index(drop=True)
        return merged, len(new_df)
    if new_df.empty:
        return existing_df.copy(), 0

    existing = existing_df.copy()
    new = new_df.copy()
    existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}
    new["__key__"] = [row_key(m, a) for m, a in zip(new["Mach"], new["alpha"])]

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


def merge_modal_fields(
    existing_fields_df: pd.DataFrame,
    new_fields_df: pd.DataFrame,
    modal_keys: set[tuple[float, float]],
    *,
    replace_existing: bool,
) -> tuple[pd.DataFrame, int]:
    if new_fields_df.empty or not modal_keys:
        return existing_fields_df.copy(), 0
    new = new_fields_df[
        [row_key(m, a) in modal_keys for m, a in zip(new_fields_df["Mach"], new_fields_df["alpha"])]
    ].copy()
    if new.empty:
        return existing_fields_df.copy(), 0

    if existing_fields_df.empty:
        return new.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True), len(modal_keys)

    existing = existing_fields_df.copy()
    if replace_existing:
        existing = existing[
            [row_key(m, a) not in modal_keys for m, a in zip(existing["Mach"], existing["alpha"])]
        ].copy()
        added = len(modal_keys)
    else:
        existing_keys = {row_key(m, a) for m, a in zip(existing["Mach"], existing["alpha"])}
        keep_keys = modal_keys - existing_keys
        new = new[[row_key(m, a) in keep_keys for m, a in zip(new["Mach"], new["alpha"])]].copy()
        added = len(keep_keys)

    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
    return merged, added


def backup_file(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


def main() -> None:
    args = build_parser().parse_args()

    summary_df = pd.read_csv(args.summary)
    fields_df = pd.read_csv(args.fields)

    existing_spectral_df = pd.read_csv(args.spectral_ref) if args.spectral_ref.exists() else pd.DataFrame(columns=REFERENCE_COLS)
    existing_modal_df = pd.read_csv(args.modal_ref) if args.modal_ref.exists() else pd.DataFrame(columns=REFERENCE_COLS)
    existing_modal_fields_df = pd.read_csv(args.modal_fields_ref) if args.modal_fields_ref.exists() else pd.DataFrame()

    new_spectral_df = summary_to_reference(
        summary_df,
        source_csv=args.summary,
        source_label=args.source_label,
        modal_only=False,
    )
    new_modal_df = summary_to_reference(
        summary_df,
        source_csv=args.summary,
        source_label=args.source_label,
        modal_only=True,
    )

    merged_spectral_df, n_spectral_added = merge_reference(
        existing_spectral_df,
        new_spectral_df,
        replace_existing=bool(args.replace_existing),
    )
    merged_modal_df, n_modal_added = merge_reference(
        existing_modal_df,
        new_modal_df,
        replace_existing=bool(args.replace_existing),
    )
    modal_keys = {row_key(m, a) for m, a in zip(new_modal_df["Mach"], new_modal_df["alpha"])}
    merged_modal_fields_df, n_modal_field_keys_added = merge_modal_fields(
        existing_modal_fields_df,
        fields_df,
        modal_keys,
        replace_existing=bool(args.replace_existing),
    )

    if args.backup_dir is not None:
        backup_file(args.spectral_ref, args.backup_dir)
        backup_file(args.modal_ref, args.backup_dir)
        backup_file(args.modal_fields_ref, args.backup_dir)

    merged_spectral_df.to_csv(args.spectral_ref, index=False)
    merged_modal_df.to_csv(args.modal_ref, index=False)
    merged_modal_fields_df.to_csv(args.modal_fields_ref, index=False)

    print("Supersonic reference CSVs updated")
    print(f"spectral_added={n_spectral_added} total={len(merged_spectral_df)}")
    print(f"modal_added={n_modal_added} total={len(merged_modal_df)}")
    print(f"modal_field_key_added={n_modal_field_keys_added} total_rows={len(merged_modal_fields_df)}")
    print(f"Wrote {args.spectral_ref}")
    print(f"Wrote {args.modal_ref}")
    print(f"Wrote {args.modal_fields_ref}")


if __name__ == "__main__":
    main()
