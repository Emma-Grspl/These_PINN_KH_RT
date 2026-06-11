from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.supersonic.blumen_reference import estimate_blumen_ci, load_digitized_curves


BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "plots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic visuel des courbes supersoniques de Blumen a partir des points digitalises."
    )
    parser.add_argument("--mach-focus", type=float, default=1.4)
    parser.add_argument("--output-stem", type=str, default="supersonic_blumen_digitized_diagnostic")
    return parser


def ci_level_curves() -> list[dict]:
    curves = load_digitized_curves(BLUMEN_DIR)
    return sorted(
        [curve for curve in curves if curve["family"] == "ci_level" and curve["level"] is not None],
        key=lambda item: float(item["level"]),
    )


def curve_alpha_at_mach(curve: dict, mach: float) -> float:
    df = curve["data"][["Mach", "alpha"]].dropna().sort_values("Mach").reset_index(drop=True)
    mach_values = df["Mach"].to_numpy(dtype=float)
    alpha_values = df["alpha"].to_numpy(dtype=float)
    if mach < float(np.min(mach_values)) or mach > float(np.max(mach_values)):
        return float("nan")
    return float(np.interp(mach, mach_values, alpha_values))


def curve_intersections_at_mach(curve: dict, mach: float) -> list[float]:
    df = curve["data"].reset_index(drop=True)
    mach_values = df["Mach"].to_numpy(dtype=float)
    alpha_values = df["alpha"].to_numpy(dtype=float)
    intersections: list[float] = []
    for i in range(len(df) - 1):
        x0 = float(mach_values[i])
        x1 = float(mach_values[i + 1])
        y0 = float(alpha_values[i])
        y1 = float(alpha_values[i + 1])
        if (mach - x0) * (mach - x1) > 0:
            continue
        if np.isclose(x0, x1):
            if np.isclose(mach, x0):
                intersections.extend([y0, y1])
            continue
        t = (mach - x0) / (x1 - x0)
        if 0.0 <= t <= 1.0:
            intersections.append(y0 + t * (y1 - y0))
    intersections = sorted(float(y) for y in intersections)
    dedup: list[float] = []
    for value in intersections:
        if not dedup or not np.isclose(value, dedup[-1], atol=1e-6):
            dedup.append(value)
    return dedup


def build_summary(curves: list[dict], mach_focus: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for curve in curves:
        level = float(curve["level"])
        alpha_at_focus = curve_alpha_at_mach(curve, mach_focus)
        intersections = curve_intersections_at_mach(curve, mach_focus)
        rows.append(
            {
                "ci_level": level,
                "alpha_at_focus_mach": alpha_at_focus,
                "n_intersections_at_focus_mach": int(len(intersections)),
                "intersections_at_focus_mach": " ".join(f"{value:.6f}" for value in intersections),
                "mach_min": float(curve["data"]["Mach"].min()),
                "mach_max": float(curve["data"]["Mach"].max()),
                "n_points": int(len(curve["data"])),
            }
        )
    return pd.DataFrame(rows)


def plot_diagnostic(curves: list[dict], summary_df: pd.DataFrame, mach_focus: float, output_path: Path) -> None:
    colors = {
        0.01: "#1d4ed8",
        0.03: "#059669",
        0.05: "#d97706",
        0.07: "#dc2626",
        0.10: "#7c3aed",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    ax_full, ax_zoom = axes

    for curve in curves:
        level = float(curve["level"])
        df_raw = curve["data"].copy().reset_index(drop=True)
        df = df_raw.sort_values("Mach")
        color = colors.get(round(level, 2), "black")
        label = f"c_i = {level:.2f}"
        for ax in axes:
            # Main line: raw digitization order, i.e. the actual traced polyline.
            ax.plot(df_raw["Mach"], df_raw["alpha"], color=color, linewidth=2.0, alpha=0.95, label=label)
            # Raw digitized points are kept visible to spot broken or multi-valued datasets immediately.
            ax.scatter(
                df_raw["Mach"],
                df_raw["alpha"],
                color=color,
                s=18,
                alpha=0.55,
                edgecolors="none",
            )
            # Dashed line: current estimator assumption, kept only as a diagnostic.
            ax.plot(df["Mach"], df["alpha"], color=color, linewidth=1.0, alpha=0.35, linestyle="--")

        intersections = curve_intersections_at_mach(curve, mach_focus)
        for alpha_focus in intersections:
            ax_full.scatter([mach_focus], [alpha_focus], color=color, s=44, marker="x", linewidths=1.5)
            ax_zoom.scatter([mach_focus], [alpha_focus], color=color, s=44, marker="x", linewidths=1.5)
            ax_zoom.text(
                mach_focus + 0.01,
                alpha_focus + 0.003,
                f"{level:.2f}",
                color=color,
                fontsize=9,
            )

    for ax in axes:
        ax.axvline(mach_focus, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_xlabel(r"Mach $M$")
        ax.set_ylabel(r"$\alpha$")

    ax_full.set_title("Isolignes supersoniques de Blumen\ntrait plein = ordre brut, tirets = tri par Mach")
    ax_full.set_xlim(0.95, 2.05)
    ax_full.set_ylim(0.0, 0.5)

    ax_zoom.set_title(f"Zoom autour de M = {mach_focus:.2f}")
    ax_zoom.set_xlim(max(0.95, mach_focus - 0.25), min(2.05, mach_focus + 0.25))
    ax_zoom.set_ylim(0.0, 0.18)

    handles, labels = ax_full.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax_full.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

    ci_from_interp = [(alpha, estimate_blumen_ci(alpha, mach_focus, curves)) for alpha in [0.125, 0.1375, 0.15, 0.1625, 0.16875]]
    intersection_rows = []
    for curve in curves:
        vals = curve_intersections_at_mach(curve, mach_focus)
        if vals:
            intersection_rows.append(f"{curve['level']:.2f}: " + ", ".join(f"{v:.3f}" for v in vals))
    summary_text = "\n".join(
        ["intersections at M=1.4"]
        + intersection_rows
        + [""]
        + [f"estimator alpha={alpha:.4f} -> {ci:.4f}" for alpha, ci in ci_from_interp]
    )
    ax_zoom.text(
        0.98,
        0.98,
        summary_text,
        transform=ax_zoom.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    curves = ci_level_curves()
    summary_df = build_summary(curves, mach_focus=float(args.mach_focus))

    output_dir = DEFAULT_OUTPUT_DIR
    png_path = output_dir / f"{args.output_stem}.png"
    csv_path = output_dir / f"{args.output_stem}_levels.csv"
    plot_diagnostic(curves, summary_df, mach_focus=float(args.mach_focus), output_path=png_path)
    summary_df.to_csv(csv_path, index=False)

    print(png_path)
    print(csv_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
