from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_supersonic_shooting_point_batch import DEFAULT_OUTPUT_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construit des tables de reference supersoniques a partir des continuations shooting "
            "spectrales et modales."
        )
    )
    parser.add_argument("--spectral-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--spectral-fields", type=Path, nargs="*", default=[])
    parser.add_argument("--modal-summaries", type=Path, nargs="*", default=[])
    parser.add_argument("--modal-fields", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default="supersonic_reference")
    return parser


def load_csvs(paths: list[Path], *, label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        df["source_label"] = str(label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_err_key(series: pd.Series) -> float:
    value = series.get("best_err_ci_abs", np.nan)
    return float(value) if np.isfinite(value) else 1.0e9


def deduplicate_reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["dedup_err_ci_key"] = work["best_err_ci_abs"].where(np.isfinite(work["best_err_ci_abs"]), 1.0e9)
    work["dedup_stage1_key"] = work["best_stage1_mismatch"].where(np.isfinite(work["best_stage1_mismatch"]), 1.0e9)
    work["dedup_select_key"] = work["best_selection_metric"].where(np.isfinite(work["best_selection_metric"]), 1.0e9)
    work = work.sort_values(
        ["Mach", "alpha", "dedup_err_ci_key", "dedup_stage1_key", "dedup_select_key"],
        ascending=[True, True, True, True, True],
    )
    work = work.drop_duplicates(subset=["Mach", "alpha"], keep="first").reset_index(drop=True)
    return work.drop(columns=["dedup_err_ci_key", "dedup_stage1_key", "dedup_select_key"])


def build_spectral_reference(spectral_df: pd.DataFrame, modal_reference_df: pd.DataFrame) -> pd.DataFrame:
    spectral = spectral_df.copy()
    if spectral.empty:
        return spectral
    accepted = spectral[spectral["continuation_accepted"].fillna(False)].copy()
    accepted = accepted[accepted["acceptance_mode"].astype(str) == "spectral"].copy()
    accepted = deduplicate_reference_rows(accepted)

    modal_keys: set[tuple[float, float]] = set()
    if not modal_reference_df.empty:
        for _, row in modal_reference_df.iterrows():
            modal_keys.add((round(float(row["Mach"]), 8), round(float(row["alpha"]), 8)))

    accepted["reference_cr"] = accepted["best_shooting_cr"]
    accepted["reference_ci"] = accepted["best_shooting_ci"]
    accepted["reference_omega_i"] = accepted["best_shooting_omega_i"]
    accepted["trusted_spectral"] = True
    accepted["trusted_modal"] = [
        (round(float(row["Mach"]), 8), round(float(row["alpha"]), 8)) in modal_keys for _, row in accepted.iterrows()
    ]

    cols = [
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
    return accepted[cols].sort_values(["Mach", "alpha"]).reset_index(drop=True)


def build_modal_reference(modal_df: pd.DataFrame, spectral_df: pd.DataFrame) -> pd.DataFrame:
    modal = modal_df.copy()
    if not modal.empty:
        accepted = modal[modal["continuation_accepted"].fillna(False)].copy()
        accepted = accepted[accepted["acceptance_mode"].astype(str) == "modal"].copy()
        accepted = accepted[accepted["best_mode_success"].fillna(False)].copy()
    else:
        spectral = spectral_df.copy()
        if spectral.empty:
            return pd.DataFrame()
        accepted = spectral[spectral["continuation_accepted"].fillna(False)].copy()
        accepted = accepted[accepted["best_mode_success"].fillna(False)].copy()
        accepted = accepted[accepted["best_status"].astype(str) == "validated"].copy()
        if accepted.empty:
            return accepted
        accepted["source_label"] = accepted["source_label"].astype(str) + "_modal_fallback"
    accepted = deduplicate_reference_rows(accepted)

    accepted["reference_cr"] = accepted["best_shooting_cr"]
    accepted["reference_ci"] = accepted["best_shooting_ci"]
    accepted["reference_omega_i"] = accepted["best_shooting_omega_i"]
    accepted["trusted_spectral"] = accepted["best_spectral_success"].fillna(False)
    accepted["trusted_modal"] = True

    cols = [
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
    return accepted[cols].sort_values(["Mach", "alpha"]).reset_index(drop=True)


def build_modal_fields_reference(fields_df: pd.DataFrame, modal_reference_df: pd.DataFrame) -> pd.DataFrame:
    if fields_df.empty or modal_reference_df.empty:
        return pd.DataFrame()
    keys = modal_reference_df[["Mach", "alpha"]].drop_duplicates().copy()
    fields = fields_df.merge(keys, on=["Mach", "alpha"], how="inner")
    return fields.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectral_df = load_csvs(list(args.spectral_summaries), label="spectral_summary")
    spectral_fields_df = load_csvs(list(args.spectral_fields), label="spectral_fields")
    modal_df = load_csvs(list(args.modal_summaries), label="modal_summary")
    modal_fields_df = load_csvs(list(args.modal_fields), label="modal_fields")

    modal_ref = build_modal_reference(modal_df, spectral_df)
    spectral_ref = build_spectral_reference(spectral_df, modal_ref)
    fields_source_df = modal_fields_df if not modal_fields_df.empty else spectral_fields_df
    modal_fields_ref = build_modal_fields_reference(fields_source_df, modal_ref)

    spectral_path = args.output_dir / f"{args.output_stem}_spectral.csv"
    modal_path = args.output_dir / f"{args.output_stem}_modal.csv"
    modal_fields_path = args.output_dir / f"{args.output_stem}_modal_fields.csv"

    spectral_ref.to_csv(spectral_path, index=False)
    modal_ref.to_csv(modal_path, index=False)
    if not modal_fields_ref.empty:
        modal_fields_ref.to_csv(modal_fields_path, index=False)

    print("Supersonic reference tables built")
    print(f"spectral_rows={len(spectral_ref)}")
    print(f"modal_rows={len(modal_ref)}")
    print(f"modal_field_rows={len(modal_fields_ref)}")
    print(f"Wrote {spectral_path}")
    print(f"Wrote {modal_path}")
    if not modal_fields_ref.empty:
        print(f"Wrote {modal_fields_path}")


if __name__ == "__main__":
    main()
