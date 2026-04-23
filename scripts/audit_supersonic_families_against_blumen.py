from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def numeric_level(label: str) -> float | None:
    try:
        return float(str(label).strip())
    except ValueError:
        return None


def load_digitized_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"level", "Mach", "alpha"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} doit contenir les colonnes {sorted(required)}")
    df = df.copy()
    df["level_value"] = df["level"].map(numeric_level)
    return df.dropna(subset=["level_value", "Mach", "alpha"]).reset_index(drop=True)


def estimate_level_from_isolines(points: pd.DataFrame, mach: float, alpha: float) -> float:
    """Interpole un niveau c_r/c_i depuis des isolignes digitalisees.

    Pour chaque isoligne, on estime alpha(M). Puis on interpole le niveau dont
    alpha(M, niveau)=alpha. Cela suppose une branche locale mono-valuee dans la
    zone auditee, ce qui est le cas pour les points alpha=0.2, M=1.2..1.3 vises.
    """
    anchors: list[tuple[float, float]] = []
    for level, sub in points.groupby("level_value"):
        sub = sub[["Mach", "alpha"]].sort_values("Mach").drop_duplicates("Mach")
        if len(sub) < 2:
            continue
        mach_values = sub["Mach"].to_numpy(dtype=float)
        alpha_values = sub["alpha"].to_numpy(dtype=float)
        if mach < float(np.min(mach_values)) or mach > float(np.max(mach_values)):
            continue
        alpha_on_curve = float(np.interp(mach, mach_values, alpha_values))
        anchors.append((alpha_on_curve, float(level)))

    if len(anchors) < 2:
        return float("nan")

    alpha_grid = np.array([item[0] for item in anchors], dtype=float)
    level_grid = np.array([item[1] for item in anchors], dtype=float)
    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    level_grid = level_grid[order]
    if alpha < float(np.min(alpha_grid)) or alpha > float(np.max(alpha_grid)):
        return float("nan")
    return float(np.interp(alpha, alpha_grid, level_grid))


def build_blumen_targets(
    mach_values: Iterable[float],
    alpha: float,
    cr_points: pd.DataFrame,
    ci_points: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mach in mach_values:
        rows.append(
            {
                "alpha": float(alpha),
                "Mach": float(mach),
                "blumen_cr": estimate_level_from_isolines(cr_points, float(mach), float(alpha)),
                "blumen_ci": estimate_level_from_isolines(ci_points, float(mach), float(alpha)),
            }
        )
    return pd.DataFrame(rows)


def _append_candidate(
    rows: list[dict],
    *,
    source_file: Path,
    source_kind: str,
    label: str,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    omega_i: float | None = None,
    extra: dict | None = None,
) -> None:
    row = {
        "source_file": source_file.name,
        "source_kind": source_kind,
        "label": label,
        "alpha": float(alpha),
        "Mach": float(mach),
        "candidate_cr": float(cr),
        "candidate_ci": float(ci),
        "candidate_omega_i": float(alpha * ci if omega_i is None else omega_i),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def load_candidates_from_csv(path: Path, alpha_filter: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[dict] = []

    def keep_alpha(value: float) -> bool:
        return alpha_filter is None or abs(float(value) - float(alpha_filter)) < 1e-9

    if {"alpha", "Mach", "cand_cr", "cand_ci"}.issubset(df.columns):
        for _, row in df.iterrows():
            if not keep_alpha(row["alpha"]):
                continue
            extra_cols = [
                "n_points",
                "family",
                "distance_to_shooting",
                "distance_to_default_shooting",
                "shooting_stage1_mismatch_at_candidate",
                "shooting_stage2_mismatch_at_candidate",
                "stage1_mismatch_at_candidate",
                "stage2_mismatch_at_candidate",
                "overlap_to_reference_high",
                "overlap_to_previous",
            ]
            extra = {col: row[col] for col in extra_cols if col in df.columns}
            label_parts = []
            if "family" in df.columns:
                label_parts.append(str(row["family"]))
            if "n_points" in df.columns:
                label_parts.append(f"N={int(row['n_points'])}")
            _append_candidate(
                rows,
                source_file=path,
                source_kind="candidate_family",
                label=" ".join(label_parts) or "candidate",
                alpha=row["alpha"],
                mach=row["Mach"],
                cr=row["cand_cr"],
                ci=row["cand_ci"],
                omega_i=row["cand_omega_i"] if "cand_omega_i" in df.columns else None,
                extra=extra,
            )

    if {"alpha", "Mach", "gep_cr", "gep_ci"}.issubset(df.columns):
        for _, row in df.iterrows():
            if not keep_alpha(row["alpha"]):
                continue
            extra = {col: row[col] for col in ["N", "score", "selection_source", "accepted"] if col in df.columns}
            label = str(row["selection_source"]) if "selection_source" in df.columns else "gep_surface"
            _append_candidate(
                rows,
                source_file=path,
                source_kind="gep_surface",
                label=label,
                alpha=row["alpha"],
                mach=row["Mach"],
                cr=row["gep_cr"],
                ci=row["gep_ci"],
                omega_i=row["gep_omega_i"] if "gep_omega_i" in df.columns else None,
                extra=extra,
            )

    if {"Mach", "rep_cr", "rep_ci"}.issubset(df.columns):
        for _, row in df.iterrows():
            if alpha_filter is None:
                alpha = np.nan
            else:
                alpha = float(alpha_filter)
            extra = {col: row[col] for col in ["cluster_id", "n_modes", "distance_to_shooting"] if col in df.columns}
            label = f"cluster={int(row['cluster_id'])}" if "cluster_id" in df.columns else "cluster"
            _append_candidate(
                rows,
                source_file=path,
                source_kind="cluster_rep",
                label=label,
                alpha=alpha,
                mach=row["Mach"],
                cr=row["rep_cr"],
                ci=row["rep_ci"],
                omega_i=row["rep_omega_i"] if "rep_omega_i" in df.columns else None,
                extra=extra,
            )

    if {"alpha", "Mach", "default_shooting_cr", "default_shooting_ci"}.issubset(df.columns):
        for _, row in df.iterrows():
            if not keep_alpha(row["alpha"]):
                continue
            extra = {
                col: row[col]
                for col in [
                    "default_shooting_success",
                    "default_shooting_spectral_success",
                    "default_stage1_mismatch",
                    "default_stage2_mismatch",
                    "default_shooting_stage1_mismatch",
                    "default_shooting_stage2_mismatch",
                ]
                if col in df.columns
            }
            _append_candidate(
                rows,
                source_file=path,
                source_kind="default_shooting",
                label="shooting_default",
                alpha=row["alpha"],
                mach=row["Mach"],
                cr=row["default_shooting_cr"],
                ci=row["default_shooting_ci"],
                omega_i=row["default_shooting_omega_i"] if "default_shooting_omega_i" in df.columns else None,
                extra=extra,
            )

    return pd.DataFrame(rows)


def load_all_candidates(paths: list[Path], alpha: float) -> pd.DataFrame:
    parts = []
    for path in paths:
        if path.exists():
            part = load_candidates_from_csv(path, alpha_filter=alpha)
            if not part.empty:
                parts.append(part)
    if not parts:
        raise RuntimeError("Aucun candidat GEP/shooting lisible dans les CSV fournis.")
    candidates = pd.concat(parts, ignore_index=True)
    dedup_cols = ["source_file", "source_kind", "label", "alpha", "Mach", "candidate_cr", "candidate_ci"]
    return candidates.drop_duplicates(subset=[col for col in dedup_cols if col in candidates.columns]).reset_index(drop=True)


def score_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(targets, on=["alpha", "Mach"], how="left")
    merged["err_cr"] = (merged["candidate_cr"] - merged["blumen_cr"]).abs()
    merged["err_ci"] = (merged["candidate_ci"] - merged["blumen_ci"]).abs()
    cr_scale = max(float(np.nanmedian(merged["err_cr"])), 0.05)
    ci_scale = max(float(np.nanmedian(merged["err_ci"])), 0.01)
    merged["score_blumen"] = merged["err_cr"] / cr_scale + merged["err_ci"] / ci_scale
    return merged.sort_values(["Mach", "score_blumen", "source_kind", "label"]).reset_index(drop=True)


def summarize_by_label(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source_kind", "label", "source_file"]
    rows = []
    for key, sub in scored.dropna(subset=["score_blumen"]).groupby(group_cols, dropna=False):
        rows.append(
            {
                "source_kind": key[0],
                "label": key[1],
                "source_file": key[2],
                "n_points": int(len(sub)),
                "mean_err_cr": float(sub["err_cr"].mean()),
                "max_err_cr": float(sub["err_cr"].max()),
                "mean_err_ci": float(sub["err_ci"].mean()),
                "max_err_ci": float(sub["err_ci"].max()),
                "mean_score_blumen": float(sub["score_blumen"].mean()),
                "max_score_blumen": float(sub["score_blumen"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_score_blumen", "max_score_blumen"]).reset_index(drop=True)


def plot_comparison(scored: pd.DataFrame, targets: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    axes[0].plot(targets["Mach"], targets["blumen_cr"], "k*-", linewidth=2.5, markersize=9, label="Blumen")
    axes[1].plot(targets["Mach"], targets["blumen_ci"], "k*-", linewidth=2.5, markersize=9, label="Blumen")

    preferred = ["default_shooting", "candidate_family", "cluster_rep", "gep_surface"]
    for source_kind in preferred:
        sub_kind = scored[scored["source_kind"] == source_kind]
        for label, sub in sub_kind.groupby("label", sort=False):
            if len(sub) < 1:
                continue
            sub = sub.sort_values("Mach")
            text = f"{source_kind}:{label}"
            axes[0].plot(sub["Mach"], sub["candidate_cr"], "o-", linewidth=1.2, markersize=4, alpha=0.75, label=text)
            axes[1].plot(sub["Mach"], sub["candidate_ci"], "o-", linewidth=1.2, markersize=4, alpha=0.75, label=text)

    axes[0].set_ylabel(r"$c_r$")
    axes[1].set_ylabel(r"$c_i$")
    for ax in axes:
        ax.set_xlabel(r"Mach $M$")
        ax.grid(True, linestyle=":", alpha=0.35)
    axes[0].set_title(r"Real phase speed")
    axes[1].set_title(r"Imaginary phase speed")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit des familles supersoniques contre Blumen c_r/c_i.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--mach-values", type=float, nargs="+", default=[1.20, 1.25, 1.275, 1.30])
    parser.add_argument("--cr-points", type=Path, default=ROOT_DIR / "assets" / "blumen" / "supersonic_cr_digitized_points.csv")
    parser.add_argument("--ci-points", type=Path, default=ROOT_DIR / "assets" / "blumen" / "supersonic_ci_digitized_points.csv")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        nargs="*",
        default=[
            DEFAULT_OUTPUT_DIR / "supersonic_shooting_vs_gep_families_a020_m120_130_families.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_shooting_vs_gep_families_a020_m120_130_summary.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_local_mode_families_a020_m120_130_summary.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_cluster_family_audit_a020_m120_130_families.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_family_clustering_a020_m120_130_clusters.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_branch_beam_search_a020_m120_130_surface.csv",
            DEFAULT_OUTPUT_DIR / "supersonic_branch_continuation_blumen_a020_m120_130_surface.csv",
        ],
    )
    parser.add_argument("--output-stem", type=str, default="supersonic_blumen_cr_ci_family_audit_a020_m120_130")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    targets = build_blumen_targets(args.mach_values, args.alpha, cr_points, ci_points)
    candidates = load_all_candidates(list(args.candidate_csv), alpha=args.alpha)
    scored = score_candidates(candidates, targets)
    summary = summarize_by_label(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = args.output_dir / f"{args.output_stem}_targets.csv"
    scored_path = args.output_dir / f"{args.output_stem}_candidates.csv"
    summary_path = args.output_dir / f"{args.output_stem}_summary.csv"
    figure_path = args.output_dir / f"{args.output_stem}_comparison.png"
    targets.to_csv(targets_path, index=False)
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_comparison(scored, targets, figure_path)

    print("Blumen targets:")
    print(targets.to_string(index=False))
    print("\nBest labels:")
    print(summary.head(12).to_string(index=False))
    print(f"\nTargets CSV: {targets_path}")
    print(f"Candidates CSV: {scored_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
