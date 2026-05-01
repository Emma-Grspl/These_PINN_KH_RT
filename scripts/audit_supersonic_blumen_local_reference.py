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

from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    estimate_level_from_isolines,
    load_digitized_long,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local de la reference supersonique digitalisee de Blumen."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--quantity", choices=["cr", "ci", "both"], default="cr")
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--local-mach-half-window", type=float, default=0.08)
    parser.add_argument("--local-alpha-half-window", type=float, default=0.08)
    parser.add_argument("--neighbor-levels", type=int, default=2)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def numeric_level(level: object) -> float | None:
    try:
        return float(str(level).strip())
    except ValueError:
        return None


def normalize_points(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    out["level_value"] = out["level"].map(numeric_level)
    return out.dropna(subset=["level_value", "Mach", "alpha"]).reset_index(drop=True)


def build_anchors(points: pd.DataFrame, mach: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for level, sub in points.groupby("level_value"):
        sub = sub[["Mach", "alpha"]].sort_values("Mach").drop_duplicates("Mach")
        mach_values = sub["Mach"].to_numpy(dtype=float)
        alpha_values = sub["alpha"].to_numpy(dtype=float)
        if len(sub) < 2 or mach < float(np.min(mach_values)) or mach > float(np.max(mach_values)):
            continue

        idx_right = int(np.searchsorted(mach_values, mach, side="left"))
        idx_left = max(idx_right - 1, 0)
        idx_right = min(idx_right, len(mach_values) - 1)

        left_mach = float(mach_values[idx_left])
        right_mach = float(mach_values[idx_right])
        left_alpha = float(alpha_values[idx_left])
        right_alpha = float(alpha_values[idx_right])
        rows.append(
            {
                "level_value": float(level),
                "alpha_on_curve": float(np.interp(mach, mach_values, alpha_values)),
                "left_mach": left_mach,
                "right_mach": right_mach,
                "left_alpha": left_alpha,
                "right_alpha": right_alpha,
                "mach_bracket_width": float(abs(right_mach - left_mach)),
                "n_points_on_curve": int(len(sub)),
                "mach_min_on_curve": float(np.min(mach_values)),
                "mach_max_on_curve": float(np.max(mach_values)),
            }
        )
    return pd.DataFrame(rows).sort_values("alpha_on_curve").reset_index(drop=True)


def summarize_target_for_mach(
    *,
    points: pd.DataFrame,
    mach: float,
    alpha_target: float,
    quantity: str,
    local_mach_half_window: float,
    local_alpha_half_window: float,
    neighbor_levels: int,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    anchors = build_anchors(points, mach)
    if anchors.empty:
        raise RuntimeError(f"Aucun support local trouvable pour Mach={mach:.6f} ({quantity}).")

    target_level = float(estimate_level_from_isolines(points, mach, alpha_target))
    if not np.isfinite(target_level):
        raise RuntimeError(f"Interpolation du niveau Blumen impossible pour Mach={mach:.6f} ({quantity}).")

    alpha_values = anchors["alpha_on_curve"].to_numpy(dtype=float)
    level_values = anchors["level_value"].to_numpy(dtype=float)
    insert_idx = int(np.searchsorted(alpha_values, alpha_target, side="left"))
    lower_idx = max(insert_idx - 1, 0)
    upper_idx = min(insert_idx, len(anchors) - 1)

    lower = anchors.iloc[lower_idx]
    upper = anchors.iloc[upper_idx]

    if len(anchors) == 1:
        lower = upper = anchors.iloc[0]

    alpha_gap_lower = float(alpha_target - lower["alpha_on_curve"])
    alpha_gap_upper = float(upper["alpha_on_curve"] - alpha_target)
    bracket_alpha_span = float(upper["alpha_on_curve"] - lower["alpha_on_curve"])
    bracket_level_span = float(upper["level_value"] - lower["level_value"])

    if upper_idx == lower_idx or np.isclose(bracket_alpha_span, 0.0):
        local_dlevel_dalpha = np.nan
        local_dalpha_dlevel = np.nan
        alpha_position_in_bracket = np.nan
    else:
        local_dlevel_dalpha = float(bracket_level_span / bracket_alpha_span)
        local_dalpha_dlevel = float(bracket_alpha_span / bracket_level_span) if not np.isclose(bracket_level_span, 0.0) else np.nan
        alpha_position_in_bracket = float((alpha_target - lower["alpha_on_curve"]) / bracket_alpha_span)

    anchors["level_distance_to_target"] = np.abs(anchors["level_value"] - target_level)
    support_levels = anchors.nsmallest(max(2 * neighbor_levels + 1, 2), "level_distance_to_target")["level_value"].tolist()
    support_levels = sorted(set(float(level) for level in support_levels))

    local_points = points[
        points["level_value"].isin(support_levels)
        & (points["Mach"] >= mach - local_mach_half_window)
        & (points["Mach"] <= mach + local_mach_half_window)
        & (points["alpha"] >= alpha_target - local_alpha_half_window)
        & (points["alpha"] <= alpha_target + local_alpha_half_window)
    ].copy()
    local_points["distance_to_target_raw"] = np.hypot(local_points["Mach"] - mach, local_points["alpha"] - alpha_target)

    nearest_raw_distance = float(local_points["distance_to_target_raw"].min()) if not local_points.empty else np.nan
    support_local_counts = local_points.groupby("level_value").size().to_dict()
    support_global_counts = points[points["level_value"].isin(support_levels)].groupby("level_value").size().to_dict()

    support_anchor_rows = anchors[anchors["level_value"].isin(support_levels)].copy()
    support_anchor_rows.insert(0, "target_mach", float(mach))
    support_anchor_rows.insert(0, "Mach", float(mach))
    support_anchor_rows.insert(0, "alpha_target", float(alpha_target))
    support_anchor_rows.insert(0, "quantity", quantity)
    support_anchor_rows.insert(0, "target_level", float(target_level))

    summary = {
        "quantity": quantity,
        "alpha_target": float(alpha_target),
        "Mach": float(mach),
        "target_level": float(target_level),
        "n_anchor_levels": int(len(anchors)),
        "lower_level": float(lower["level_value"]),
        "upper_level": float(upper["level_value"]),
        "lower_alpha_on_curve": float(lower["alpha_on_curve"]),
        "upper_alpha_on_curve": float(upper["alpha_on_curve"]),
        "alpha_gap_lower": alpha_gap_lower,
        "alpha_gap_upper": alpha_gap_upper,
        "bracket_alpha_span": bracket_alpha_span,
        "bracket_level_span": bracket_level_span,
        "local_dlevel_dalpha": local_dlevel_dalpha,
        "local_dalpha_dlevel": local_dalpha_dlevel,
        "alpha_position_in_bracket": alpha_position_in_bracket,
        "support_levels": ",".join(f"{level:.3f}" for level in support_levels),
        "n_support_levels": int(len(support_levels)),
        "n_local_support_points": int(len(local_points)),
        "nearest_raw_distance": nearest_raw_distance,
    }
    for idx, level in enumerate(support_levels, start=1):
        summary[f"support_level_{idx}"] = float(level)
        summary[f"support_level_{idx}_global_points"] = int(support_global_counts.get(level, 0))
        summary[f"support_level_{idx}_local_points"] = int(support_local_counts.get(level, 0))

    local_points.insert(0, "target_level", float(target_level))
    local_points.insert(0, "target_mach", float(mach))
    local_points.insert(0, "alpha_target", float(alpha_target))
    local_points.insert(0, "quantity", quantity)

    return summary, support_anchor_rows, local_points


def plot_local_pdf(
    *,
    quantity: str,
    summary_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    local_points_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for mach in summary_df["Mach"].tolist():
            row = summary_df[summary_df["Mach"] == mach].iloc[0]
            sub_anchors = anchors_df[anchors_df["Mach"] == mach].copy()
            if "target_mach" in local_points_df.columns:
                sub_local = local_points_df[np.isclose(local_points_df["target_mach"], mach)].copy()
            else:
                sub_local = local_points_df.iloc[0:0]

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

            for level, sub in sub_local.groupby("level_value", sort=True):
                axes[0].scatter(sub["Mach"], sub["alpha"], s=30, label=f"{quantity}={level:.3f}")
            axes[0].axvline(float(mach), color="black", linestyle=":", linewidth=1.2)
            axes[0].axhline(float(row["alpha_target"]), color="tab:red", linestyle=":", linewidth=1.2)
            axes[0].scatter([float(mach)], [float(row["alpha_target"])], marker="*", s=180, color="gold", edgecolor="black", zorder=5)
            axes[0].set_title(f"Voisinage brut des points digitalises | M={mach:.3f}")
            axes[0].set_xlabel("Mach")
            axes[0].set_ylabel(r"$\alpha$")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(fontsize=8, loc="best")

            axes[1].plot(sub_anchors["level_value"], sub_anchors["alpha_on_curve"], "o-", color="tab:blue")
            axes[1].axhline(float(row["alpha_target"]), color="tab:red", linestyle=":", linewidth=1.2, label=r"$\alpha$ cible")
            axes[1].axvline(float(row["target_level"]), color="tab:green", linestyle="--", linewidth=1.2, label="niveau interpole")
            axes[1].scatter([float(row["lower_level"])], [float(row["lower_alpha_on_curve"])], color="black", s=60, zorder=5, label="borne basse")
            axes[1].scatter([float(row["upper_level"])], [float(row["upper_alpha_on_curve"])], color="tab:orange", s=60, zorder=5, label="borne haute")
            axes[1].set_title(f"Interpolation locale {quantity}(M) -> alpha | M={mach:.3f}")
            axes[1].set_xlabel(rf"${quantity}$")
            axes[1].set_ylabel(r"$\alpha(M, niveau)$")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(fontsize=8, loc="best")

            fig.suptitle(
                f"Audit local Blumen {quantity} | alpha={float(row['alpha_target']):.3f}, M={mach:.3f}\n"
                f"target={float(row['target_level']):.6f} | span_alpha={float(row['bracket_alpha_span']):.3e} | "
                f"d{quantity}/dalpha={float(row['local_dlevel_dalpha']):.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)


def run_quantity_audit(
    *,
    quantity: str,
    points_path: Path,
    alpha_target: float,
    mach_values: list[float],
    local_mach_half_window: float,
    local_alpha_half_window: float,
    neighbor_levels: int,
    output_stem: str,
) -> None:
    points = normalize_points(load_digitized_long(points_path))
    summaries: list[dict[str, float | str]] = []
    anchor_rows: list[pd.DataFrame] = []
    local_rows: list[pd.DataFrame] = []

    for mach in mach_values:
        summary, support_anchor_rows, local_points = summarize_target_for_mach(
            points=points,
            mach=float(mach),
            alpha_target=float(alpha_target),
            quantity=quantity,
            local_mach_half_window=float(local_mach_half_window),
            local_alpha_half_window=float(local_alpha_half_window),
            neighbor_levels=int(neighbor_levels),
        )
        summaries.append(summary)
        anchor_rows.append(support_anchor_rows)
        local_rows.append(local_points)

    summary_df = pd.DataFrame(summaries).sort_values("Mach").reset_index(drop=True)
    anchors_df = pd.concat(anchor_rows, ignore_index=True) if anchor_rows else pd.DataFrame()
    local_points_df = pd.concat(local_rows, ignore_index=True) if local_rows else pd.DataFrame()

    summary_path = DEFAULT_OUTPUT_DIR / f"{output_stem}_{quantity}_summary.csv"
    anchors_path = DEFAULT_OUTPUT_DIR / f"{output_stem}_{quantity}_anchors.csv"
    local_points_path = DEFAULT_OUTPUT_DIR / f"{output_stem}_{quantity}_local_points.csv"
    pdf_path = DEFAULT_OUTPUT_DIR / f"{output_stem}_{quantity}_local_audit.pdf"

    summary_df.to_csv(summary_path, index=False)
    anchors_df.to_csv(anchors_path, index=False)
    local_points_df.to_csv(local_points_path, index=False)
    plot_local_pdf(
        quantity=quantity,
        summary_df=summary_df,
        anchors_df=anchors_df,
        local_points_df=local_points_df,
        output_path=pdf_path,
    )

    print(f"\nLocal Blumen reference audit for {quantity}:")
    print(summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {anchors_path}")
    print(f"Wrote {local_points_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    quantities = ["cr", "ci"] if args.quantity == "both" else [args.quantity]
    point_map = {
        "cr": args.cr_points,
        "ci": args.ci_points,
    }

    for quantity in quantities:
        run_quantity_audit(
            quantity=quantity,
            points_path=point_map[quantity],
            alpha_target=float(args.alpha),
            mach_values=[float(m) for m in args.mach_values],
            local_mach_half_window=float(args.local_mach_half_window),
            local_alpha_half_window=float(args.local_alpha_half_window),
            neighbor_levels=int(args.neighbor_levels),
            output_stem=str(args.output_stem),
        )


if __name__ == "__main__":
    main()
