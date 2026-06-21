from __future__ import annotations

import argparse
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

from classical_solver.supersonic.blumen_reference import estimate_blumen_ci, load_digitized_curves


DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "validated_modal_points"
BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
REFERENCE_MODAL_CSV = ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_modal.csv"
REFERENCE_FIELDS_CSV = ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_modal_fields.csv"
M120_SUMMARY_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_pointwise_core_M12_2026-06-17"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_core_band_M12_summary.csv"
)
M120_FIELDS_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_pointwise_core_M12_2026-06-17"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_core_band_M12_fields.csv"
)
M140_SUMMARY_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_point_batch_M140_branch_guided_reconfirm_2026-06-09"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_point_batch_M140_branch_guided_reconfirm_summary.csv"
)
M140_FIELDS_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_point_batch_M140_branch_guided_reconfirm_2026-06-09"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_point_batch_M140_branch_guided_reconfirm_fields.csv"
)
M170_SUMMARY_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_pointwise_core_M17_2026-06-16"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_core_band_M17_manual_summary_corrected.csv"
)
M170_FIELDS_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_pointwise_core_M17_2026-06-16"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_core_band_M17_manual_fields.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clean package of validated supersonic modal points."
    )
    parser.add_argument("--mach-values", type=float, nargs="+", default=[1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    parser.add_argument("--output-stem", type=str, default="supersonic_validated_modal_points")
    return parser


def box_robust_mask(df: pd.DataFrame) -> pd.Series:
    if "box_robustness_enabled" in df.columns and "box_robustness_pass" in df.columns:
        enabled = df["box_robustness_enabled"].fillna(False).astype(bool)
        passed = df["box_robustness_pass"].fillna(False).astype(bool)
        return (~enabled) | passed
    return pd.Series(True, index=df.index)


def normalize_reference_modal_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[
        work["best_status"].astype(str).eq("validated")
        & work["best_spectral_success"].astype(bool)
        & work["best_mode_success"].astype(bool)
        & work["trusted_modal"].astype(bool)
        & box_robust_mask(work)
    ].copy()
    work["reference_cr"] = work["reference_cr"].astype(float)
    work["reference_ci"] = work["reference_ci"].astype(float)
    work["stage1_mismatch"] = work["best_stage1_mismatch"].astype(float)
    work["stage2_mismatch"] = work["best_stage2_mismatch"].astype(float)
    work["status"] = work["best_status"].astype(str)
    work["source_group"] = "reference_core_local_modal"
    keep = [
        "alpha",
        "Mach",
        "reference_cr",
        "reference_ci",
        "stage1_mismatch",
        "stage2_mismatch",
        "status",
        "source_group",
        "source_csv",
        "source_label",
        "line_id",
    ]
    return work[keep].copy()


def normalize_m140_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[
        work["best_status"].astype(str).eq("validated")
        & work["best_spectral_success"].astype(bool)
        & work["best_mode_success"].astype(bool)
        & box_robust_mask(work)
    ].copy()
    work["reference_cr"] = work["best_shooting_cr"].astype(float)
    work["reference_ci"] = work["best_shooting_ci"].astype(float)
    work["stage1_mismatch"] = work["best_stage1_mismatch"].astype(float)
    work["stage2_mismatch"] = work["best_stage2_mismatch"].astype(float)
    work["status"] = work["best_status"].astype(str)
    work["source_group"] = "M140_branch_guided_reconfirm"
    work["source_csv"] = str(M140_SUMMARY_CSV.relative_to(ROOT_DIR))
    work["source_label"] = "M140_branch_guided_reconfirm"
    work["line_id"] = [f"M1.40_a{alpha:.5f}" for alpha in work["alpha"].astype(float)]
    keep = [
        "alpha",
        "Mach",
        "reference_cr",
        "reference_ci",
        "stage1_mismatch",
        "stage2_mismatch",
        "status",
        "source_group",
        "source_csv",
        "source_label",
        "line_id",
    ]
    return work[keep].copy()


def normalize_m120_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[
        work["alpha"].astype(float).isin([0.15, 0.175, 0.20])
        & work["best_status"].astype(str).eq("validated")
        & work["best_spectral_success"].astype(bool)
        & work["best_mode_success"].astype(bool)
        & (~work["box_truncation_suspect_any_field"].astype(bool))
        & box_robust_mask(work)
        & work["blumen_ci_available"].astype(bool)
        & (work["best_err_ci_rel"].astype(float) <= 0.10)
    ].copy()
    work["reference_cr"] = work["best_shooting_cr"].astype(float)
    work["reference_ci"] = work["best_shooting_ci"].astype(float)
    work["stage1_mismatch"] = work["best_stage1_mismatch"].astype(float)
    work["stage2_mismatch"] = work["best_stage2_mismatch"].astype(float)
    work["status"] = work["best_status"].astype(str)
    work["source_group"] = "M120_pointwise_core"
    work["source_csv"] = str(M120_SUMMARY_CSV.relative_to(ROOT_DIR))
    work["source_label"] = "M120_pointwise_core"
    work["line_id"] = [f"M1.20_a{alpha:.5f}" for alpha in work["alpha"].astype(float)]
    keep = [
        "alpha",
        "Mach",
        "reference_cr",
        "reference_ci",
        "stage1_mismatch",
        "stage2_mismatch",
        "status",
        "source_group",
        "source_csv",
        "source_label",
        "line_id",
    ]
    return work[keep].copy()


def normalize_m170_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[
        work["best_status"].astype(str).eq("validated")
        & work["best_spectral_success"].astype(bool)
        & work["best_mode_success"].astype(bool)
        & (~work["box_truncation_suspect_any_field"].astype(bool))
        & box_robust_mask(work)
        & work["blumen_ci_available"].astype(bool)
        & (work["best_err_ci_rel"].astype(float) <= 0.10)
    ].copy()
    work["reference_cr"] = work["best_shooting_cr"].astype(float)
    work["reference_ci"] = work["best_shooting_ci"].astype(float)
    work["stage1_mismatch"] = work["best_stage1_mismatch"].astype(float)
    work["stage2_mismatch"] = work["best_stage2_mismatch"].astype(float)
    work["status"] = work["best_status"].astype(str)
    work["source_group"] = "M170_pointwise_corrected"
    work["source_csv"] = str(M170_SUMMARY_CSV.relative_to(ROOT_DIR))
    work["source_label"] = "M170_pointwise_corrected"
    work["line_id"] = [f"M1.70_a{alpha:.5f}" for alpha in work["alpha"].astype(float)]
    keep = [
        "alpha",
        "Mach",
        "reference_cr",
        "reference_ci",
        "stage1_mismatch",
        "stage2_mismatch",
        "status",
        "source_group",
        "source_csv",
        "source_label",
        "line_id",
    ]
    return work[keep].copy()


def load_validated_points(mach_values: list[float]) -> pd.DataFrame:
    modal_ref = normalize_reference_modal_rows(pd.read_csv(REFERENCE_MODAL_CSV))
    modal_ref = modal_ref[
        modal_ref["Mach"].astype(float).isin(
            [mach for mach in mach_values if not np.isclose(mach, 1.2) and not np.isclose(mach, 1.4) and not np.isclose(mach, 1.7)]
        )
    ]

    frames = [modal_ref]
    if any(np.isclose(mach, 1.2) for mach in mach_values) and M120_SUMMARY_CSV.exists():
        m120 = normalize_m120_rows(pd.read_csv(M120_SUMMARY_CSV))
        frames.append(m120)
    if any(np.isclose(mach, 1.4) for mach in mach_values):
        m140 = normalize_m140_rows(pd.read_csv(M140_SUMMARY_CSV))
        frames.append(m140)
    if any(np.isclose(mach, 1.7) for mach in mach_values) and M170_SUMMARY_CSV.exists():
        m170 = normalize_m170_rows(pd.read_csv(M170_SUMMARY_CSV))
        frames.append(m170)

    merged = pd.concat(frames, ignore_index=True)
    merged["alpha"] = merged["alpha"].astype(float)
    merged["Mach"] = merged["Mach"].astype(float)
    merged["reference_cr"] = merged["reference_cr"].astype(float)
    merged["reference_ci"] = merged["reference_ci"].astype(float)
    merged["point_id"] = [f"M{mach:.2f}_a{alpha:.5f}" for mach, alpha in zip(merged["Mach"], merged["alpha"])]
    merged = merged.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    return merged


def attach_corrected_blumen_errors(df: pd.DataFrame) -> pd.DataFrame:
    curves = load_digitized_curves(BLUMEN_DIR)
    work = df.copy()
    work["blumen_ci_corrected"] = [
        estimate_blumen_ci(float(alpha), float(mach), curves)
        for alpha, mach in zip(work["alpha"], work["Mach"])
    ]
    work["err_ci_abs_corrected"] = (work["reference_ci"] - work["blumen_ci_corrected"]).abs()
    work["err_ci_rel_corrected"] = work["err_ci_abs_corrected"] / work["blumen_ci_corrected"].abs()
    return work


def keep_strict_gold_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    mask = (
        np.isfinite(work["blumen_ci_corrected"].to_numpy(dtype=float))
        & np.isfinite(work["err_ci_abs_corrected"].to_numpy(dtype=float))
        & np.isfinite(work["err_ci_rel_corrected"].to_numpy(dtype=float))
    )
    return work.loc[mask].reset_index(drop=True)


def blumen_points_df() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for curve in load_digitized_curves(BLUMEN_DIR):
        if curve.get("family") != "ci_level" or curve.get("level") is None:
            continue
        level = float(curve["level"])
        for _, row in curve["data"].iterrows():
            rows.append(
                {
                    "Mach": float(row["Mach"]),
                    "alpha": float(row["alpha"]),
                    "ci_level": level,
                }
            )
    return pd.DataFrame(rows)


def plot_overlay(points_df: pd.DataFrame, output_path: Path) -> None:
    blumen_df = blumen_points_df()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    ax_full, ax_zoom = axes

    colors = {1.2: "#0891b2", 1.3: "#2563eb", 1.4: "#dc2626", 1.5: "#059669", 1.6: "#7c3aed", 1.7: "#d97706"}

    for ax in axes:
        ax.scatter(
            blumen_df["Mach"],
            blumen_df["alpha"],
            s=18,
            c="0.75",
            alpha=0.75,
            label="Points Blumen ($c_i$)",
        )
        for mach in sorted(points_df["Mach"].unique()):
            sub = points_df[np.isclose(points_df["Mach"], mach)]
            ax.scatter(
                sub["Mach"],
                sub["alpha"],
                s=95,
                marker="D",
                color=colors.get(round(float(mach), 1), "black"),
                edgecolors="black",
                linewidths=0.7,
                label=f"Points valides M={mach:.1f}",
                zorder=4,
            )
        ax.set_xlabel(r"Mach $M$")
        ax.set_ylabel(r"$\alpha$")
        ax.grid(True, linestyle=":", alpha=0.3)

    ax_full.set_title("Blumen digitise vs points robustes (spectral + modal)")
    ax_full.set_xlim(0.95, 2.05)
    ax_full.set_ylim(0.0, 0.5)

    ax_zoom.set_title("Zoom sur la zone utile M = 1.2 a 1.7")
    ax_zoom.set_xlim(1.15, 1.78)
    ax_zoom.set_ylim(0.08, 0.22)

    counts = points_df.groupby("Mach").size().to_dict()
    count_text = "\n".join([f"M={mach:.1f}: {count} points" for mach, count in sorted(counts.items())])
    ax_zoom.text(
        0.98,
        0.04,
        count_text,
        transform=ax_zoom.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    handles, labels = ax_full.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax_full.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_fields_df() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_FIELDS_CSV).copy()
    ref["source_group"] = "reference_core_local_modal"
    frames = [ref]
    if M120_FIELDS_CSV.exists():
        m120 = pd.read_csv(M120_FIELDS_CSV).copy()
        m120["source_group"] = "M120_pointwise_core"
        frames.append(m120)
    m140 = pd.read_csv(M140_FIELDS_CSV).copy()
    m140["source_group"] = "M140_branch_guided_reconfirm"
    frames.append(m140)
    if M170_FIELDS_CSV.exists():
        m170 = pd.read_csv(M170_FIELDS_CSV).copy()
        m170["source_group"] = "M170_pointwise_corrected"
        frames.append(m170)
    return pd.concat(frames, ignore_index=True)


def point_fields(fields_df: pd.DataFrame, alpha: float, mach: float, source_group: str) -> pd.DataFrame:
    sub = fields_df[
        np.isclose(fields_df["alpha"].astype(float), float(alpha), atol=1e-8)
        & np.isclose(fields_df["Mach"].astype(float), float(mach), atol=1e-8)
        & fields_df["source_group"].astype(str).eq(source_group)
    ].copy()
    if len(sub) == 0:
        return sub
    return sub.sort_values("y").reset_index(drop=True)


def build_modes_pdf(points_df: pd.DataFrame, output_path: Path) -> None:
    fields_df = load_fields_df()
    points = points_df.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    with PdfPages(output_path) as pdf:
        ncols, nrows = 2, 3
        per_page = ncols * nrows
        for start in range(0, len(points), per_page):
            chunk = points.iloc[start : start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 12.0))
            axes = axes.ravel()

            for ax in axes[len(chunk) :]:
                ax.axis("off")

            for ax, (_, row) in zip(axes, chunk.iterrows()):
                sub = point_fields(fields_df, float(row["alpha"]), float(row["Mach"]), str(row["source_group"]))
                if len(sub) == 0:
                    ax.text(0.5, 0.5, "fields missing", ha="center", va="center")
                    ax.axis("off")
                    continue

                y = sub["y"].to_numpy(dtype=float)
                p_real = sub["p_real"].to_numpy(dtype=float)
                p_imag = sub["p_imag"].to_numpy(dtype=float)
                p_abs = np.sqrt(p_real**2 + p_imag**2)
                norm = max(float(np.max(p_abs)), 1e-12)

                ax.plot(y, p_real / norm, color="#2563eb", linewidth=1.4, label=r"$\Re(p)$ / max|p|")
                ax.plot(y, p_abs / norm, color="#dc2626", linewidth=1.2, linestyle="--", label=r"$|p|$ / max|p|")
                ax.axhline(0.0, color="0.75", linewidth=0.7)
                ax.grid(True, linestyle=":", alpha=0.25)
                ax.set_title(
                    f"M={float(row['Mach']):.2f}, alpha={float(row['alpha']):.5f}\n"
                    f"ci={float(row['reference_ci']):.5f}, |Δci|={float(row['err_ci_abs_corrected']):.3e}"
                )
                ax.set_xlabel("y")
                ax.set_ylabel("normalized mode")

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
            fig.suptitle("Modes associes aux points valides (spectral + modal)", y=0.995, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.975])
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_validated_points([float(value) for value in args.mach_values])
    points_df = attach_corrected_blumen_errors(points_df)
    points_df = keep_strict_gold_rows(points_df)

    csv_path = output_dir / f"{args.output_stem}.csv"
    png_path = output_dir / f"{args.output_stem}_overlay.png"
    pdf_path = output_dir / f"{args.output_stem}_modes.pdf"

    points_df.to_csv(csv_path, index=False)
    plot_overlay(points_df, png_path)
    build_modes_pdf(points_df, pdf_path)

    print(csv_path)
    print(png_path)
    print(pdf_path)
    print(points_df[["alpha", "Mach", "reference_ci", "blumen_ci_corrected", "err_ci_abs_corrected", "err_ci_rel_corrected", "stage1_mismatch", "stage2_mismatch", "source_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
