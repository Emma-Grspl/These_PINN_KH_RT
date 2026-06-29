from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def read_launch_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing launch log: {path}")

    df = pd.read_csv(path, sep="\t")
    if "job_id" not in df.columns:
        raise SystemExit(f"Bad launch log format: {path}")

    df = df[df["job_id"].astype(str).str.fullmatch(r"\d+").fillna(False)].copy()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--launch-log", required=True)
    ap.add_argument("--ci-min", type=float, default=1e-4)
    ap.add_argument("--ci-max", type=float, default=0.080)
    ap.add_argument("--stage2-max", type=float, default=1e-8)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    launch_log = Path(args.launch_log)
    launch = read_launch_log(launch_log)

    audit_root = Path("assets/classic_supersonic/multicandidate_audits")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"assets/classic_supersonic/shooting/_incoming_{args.campaign}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    status_rows = []
    candidate_rows = []

    for _, r in launch.iterrows():
        job_id = str(r["job_id"])
        stem = str(r["stem"])
        audit_dir = audit_root / stem
        all_csv = audit_dir / "all_candidates.csv"
        point_csv = audit_dir / "point_summary.csv"
        out_log = Path("slurm/log") / f"KH_shoot_multi_{job_id}.out"
        err_log = Path("slurm/log") / f"KH_shoot_multi_{job_id}.err"

        state = "missing"
        if all_csv.exists():
            state = "done"
            try:
                df = pd.read_csv(all_csv)
                df["job_id"] = job_id
                df["launch_alpha"] = r["alpha"]
                df["launch_Mach"] = r["Mach"]
                df["launch_seed"] = r["seed"]
                df["stem"] = stem
                df["audit_dir"] = str(audit_dir)
                candidate_rows.append(df)
            except Exception as e:
                state = f"csv_read_error:{e}"
        elif out_log.exists() or err_log.exists():
            state = "started_no_csv"

        status_rows.append({
            "job_id": job_id,
            "alpha": r["alpha"],
            "Mach": r["Mach"],
            "seed": r["seed"],
            "stem": stem,
            "campaign": r["campaign"],
            "state": state,
            "audit_dir": str(audit_dir),
            "out_log": str(out_log),
            "err_log": str(err_log),
        })

    status = pd.DataFrame(status_rows)
    status_path = out_dir / f"{args.campaign}_job_status.csv"
    status.to_csv(status_path, index=False)

    print("\n=== JOB STATUS ===")
    print(status["state"].value_counts(dropna=False).to_string())
    print(f"\nWrote {status_path}")

    if not candidate_rows:
        print("\nNo all_candidates.csv found yet.")
        return

    raw = pd.concat(candidate_rows, ignore_index=True)

    for c in [
        "alpha", "Mach", "cr_init", "ci_init",
        "cr_final", "ci_final",
        "stage1_mismatch", "stage2_mismatch",
    ]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw_path = out_dir / f"{args.campaign}_raw_candidates.csv"
    raw.to_csv(raw_path, index=False)

    valid = raw.copy()

    if "candidate_source" in valid.columns:
        valid = valid[valid["candidate_source"].astype(str).eq("manual_grid")].copy()

    valid = valid[
        valid["alpha"].notna()
        & valid["Mach"].notna()
        & valid["cr_final"].notna()
        & valid["ci_final"].notna()
        & valid["ci_final"].between(args.ci_min, args.ci_max)
    ].copy()

    if "stage2_mismatch" in valid.columns:
        valid = valid[
            valid["stage2_mismatch"].notna()
            & (valid["stage2_mismatch"] <= args.stage2_max)
        ].copy()

    valid_path = out_dir / f"{args.campaign}_valid_candidates.csv"
    valid.to_csv(valid_path, index=False)

    if valid.empty:
        print(f"\nWrote raw:   {raw_path}")
        print(f"Wrote valid: {valid_path}")
        print("No valid candidate after filters.")
        return

    sort_cols = ["alpha", "Mach"]
    if "stage1_mismatch" in valid.columns:
        sort_cols.append("stage1_mismatch")
    if "stage2_mismatch" in valid.columns:
        sort_cols.append("stage2_mismatch")

    valid = valid.sort_values(sort_cols, ascending=True)

    best = (
        valid.groupby(["alpha", "Mach"], as_index=False, group_keys=False)
        .head(1)
        .copy()
    )

    best["reference_cr"] = best["cr_final"]
    best["reference_ci"] = best["ci_final"]
    best["reference_omega_i"] = best["alpha"] * best["ci_final"]
    best["reference_kind"] = "supersonic_classical_shooting"
    best["validation_level"] = "shooting_root_only_no_box"
    best["best_spectral_success"] = True
    best["best_mode_success"] = False
    best["trusted_spectral"] = True
    best["trusted_modal"] = False
    best["source_label"] = args.campaign

    best_cols = [
        "alpha", "Mach",
        "reference_cr", "reference_ci", "reference_omega_i",
        "cr_init", "ci_init", "cr_final", "ci_final",
        "stage1_mismatch", "stage2_mismatch",
        "candidate_source", "status", "accept", "reason", "reject_reasons",
        "job_id", "launch_seed", "stem", "audit_dir",
        "reference_kind", "validation_level",
        "best_spectral_success", "best_mode_success",
        "trusted_spectral", "trusted_modal", "source_label",
    ]
    best_cols = [c for c in best_cols if c in best.columns]

    best_path = out_dir / f"{args.campaign}_selected_reference_points.csv"
    best[best_cols].to_csv(best_path, index=False)

    print(f"\nWrote raw:      {raw_path}")
    print(f"Wrote valid:    {valid_path}")
    print(f"Wrote selected: {best_path}")

    print("\n=== SELECTED BEST POINTS ===")
    print(best[best_cols].to_string(index=False))

    print("\n=== COVERAGE BY MACH ===")
    cov = best.groupby("Mach")["alpha"].nunique().reset_index(name="n_alpha_selected")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
