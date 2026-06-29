from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-process supersonic multicandidate audit using an exploratory ci interval instead of strict Blumen reference."
    )
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--ci-min", type=float, default=0.01)
    p.add_argument("--ci-max", type=float, default=0.03)
    p.add_argument("--strict-stage1", type=float, default=5e-2)
    p.add_argument("--strict-stage2", type=float, default=1e-4)
    p.add_argument("--strict-box-max-rel-l2", type=float, default=0.15)
    p.add_argument("--strict-peak-shift", type=float, default=0.75)
    p.add_argument("--require-box", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-peak", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = args.audit_dir
    input_csv = audit_dir / "all_candidates.csv"
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    required = ["alpha", "Mach", "candidate_id", "ci_final", "stage1_mismatch", "stage2_mismatch"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {input_csv}: {missing}")

    df["ci_range_min"] = float(args.ci_min)
    df["ci_range_max"] = float(args.ci_max)
    df["ci_range_ok"] = df["ci_final"].between(float(args.ci_min), float(args.ci_max), inclusive="both")

    df["stage1_ok_ci_range"] = df["stage1_mismatch"].astype(float) <= float(args.strict_stage1)
    df["stage2_ok_ci_range"] = df["stage2_mismatch"].astype(float) <= float(args.strict_stage2)

    if "box_robustness_pass" in df.columns:
        box_pass = _as_bool_series(df["box_robustness_pass"])
    elif "box_robustness_max_rel_l2" in df.columns:
        box_pass = df["box_robustness_max_rel_l2"].astype(float) <= float(args.strict_box_max_rel_l2)
    else:
        box_pass = pd.Series(False, index=df.index)

    if args.require_box:
        df["box_ok_ci_range"] = box_pass
    else:
        df["box_ok_ci_range"] = True

    if "peak_shift" in df.columns:
        peak_pass = df["peak_shift"].astype(float) <= float(args.strict_peak_shift)
    else:
        peak_pass = pd.Series(False, index=df.index)

    if args.require_peak:
        df["peak_ok_ci_range"] = peak_pass
    else:
        df["peak_ok_ci_range"] = True

    df["accept_ci_range"] = (
        df["ci_range_ok"]
        & df["stage1_ok_ci_range"]
        & df["stage2_ok_ci_range"]
        & df["box_ok_ci_range"]
        & df["peak_ok_ci_range"]
    )

    reasons = []
    for _, row in df.iterrows():
        r = []
        if not bool(row["ci_range_ok"]):
            r.append("ci_range")
        if not bool(row["stage1_ok_ci_range"]):
            r.append("stage1")
        if not bool(row["stage2_ok_ci_range"]):
            r.append("stage2")
        if not bool(row["box_ok_ci_range"]):
            r.append("box")
        if not bool(row["peak_ok_ci_range"]):
            r.append("peak_shift")
        reasons.append(";".join(r) if r else "accepted")

    df["reject_reasons_ci_range"] = reasons
    df["status_ci_range"] = np.where(df["accept_ci_range"], "ci_range_accepted", "ci_range_rejected")

    # Ranking: accepted first, then ci-range-compatible, then more robust box, then smaller peak shift, then smaller stage1.
    work = df.copy()
    work["_rank_accept"] = (~work["accept_ci_range"]).astype(int)
    work["_rank_ci"] = (~work["ci_range_ok"]).astype(int)
    work["_rank_box"] = work.get("box_robustness_max_rel_l2", pd.Series(np.inf, index=work.index)).astype(float).fillna(np.inf)
    work["_rank_peak"] = work.get("peak_shift", pd.Series(np.inf, index=work.index)).astype(float).fillna(np.inf)
    work["_rank_stage1"] = work["stage1_mismatch"].astype(float).fillna(np.inf)

    best_rows = []
    for (alpha, mach), sub in work.groupby(["alpha", "Mach"], sort=True):
        sub = sub.sort_values(["_rank_accept", "_rank_ci", "_rank_box", "_rank_peak", "_rank_stage1"])
        best = sub.iloc[0].copy()
        best_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "n_candidates": int(len(sub)),
                "n_accepted_ci_range": int(sub["accept_ci_range"].sum()),
                "best_candidate_id": best["candidate_id"],
                "best_candidate_source": best.get("candidate_source", ""),
                "best_ci_final": best["ci_final"],
                "best_ci_range_ok": bool(best["ci_range_ok"]),
                "best_accept_ci_range": bool(best["accept_ci_range"]),
                "best_reject_reasons_ci_range": best["reject_reasons_ci_range"],
                "best_stage1_mismatch": best["stage1_mismatch"],
                "best_stage2_mismatch": best["stage2_mismatch"],
                "best_box_robustness_max_rel_l2": best.get("box_robustness_max_rel_l2", np.nan),
                "best_box_robustness_pass": best.get("box_robustness_pass", np.nan),
                "best_peak_shift": best.get("peak_shift", np.nan),
            }
        )

    summary = pd.DataFrame(best_rows)

    df.drop(columns=[c for c in df.columns if c.startswith("_rank_")], errors="ignore").to_csv(
        audit_dir / "all_candidates_ci_range.csv", index=False
    )
    df[df["accept_ci_range"]].to_csv(audit_dir / "accepted_points_ci_range.csv", index=False)
    df[~df["accept_ci_range"]].to_csv(audit_dir / "rejected_candidates_ci_range.csv", index=False)
    summary.to_csv(audit_dir / "point_summary_ci_range.csv", index=False)

    print(f"Wrote {audit_dir / 'all_candidates_ci_range.csv'}")
    print(f"Wrote {audit_dir / 'accepted_points_ci_range.csv'}")
    print(f"Wrote {audit_dir / 'rejected_candidates_ci_range.csv'}")
    print(f"Wrote {audit_dir / 'point_summary_ci_range.csv'}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
