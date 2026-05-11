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

from classical_solver.supersonic.blumen_reference import load_digitized_curves
from classical_solver.supersonic.shooting_supersonic import sample_supersonic_growth_map


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruction visuelle propre des isolignes supersoniques de Blumen en c_i."
    )
    parser.add_argument("--mach-min", type=float, default=1.0)
    parser.add_argument("--mach-max", type=float, default=2.0)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.50)
    parser.add_argument("--num-mach", type=int, default=31)
    parser.add_argument("--num-alpha", type=int, default=31)
    parser.add_argument("--ci-max", type=float, default=0.12)
    parser.add_argument("--cr-max", type=float, default=0.35)
    parser.add_argument("--tracking-weight", type=float, default=2e-2)
    parser.add_argument("--output-stem", type=str, default="supersonic_blumen_ci_visual")
    return parser


def load_ci_curves() -> list[dict]:
    curves = load_digitized_curves(DATA_DIR)
    return [curve for curve in curves if curve["family"] in {"ci_level", "ci_special"} and curve["level"] is not None]


def build_anchor_points(curves: list[dict]) -> list[dict]:
    anchors: list[dict] = []
    for curve in curves:
        level = float(curve["level"])
        for _, row in curve["data"].iterrows():
            mach = float(row["Mach"])
            alpha = float(row["alpha"])
            if mach < 1.0 or alpha < 0.0:
                continue
            anchors.append(
                {
                    "Mach": mach,
                    "alpha": alpha,
                    "cr_seed": 0.03,
                    "ci_seed": level,
                }
            )
    return anchors


def contour_levels(curves: list[dict]) -> list[float]:
    return sorted({float(curve["level"]) for curve in curves})


def style_for_curve(curve: dict) -> dict:
    if curve["family"] == "ci_special":
        return {"color": "black", "linewidth": 1.0, "linestyle": (0, (5, 3)), "alpha": 0.9}
    return {"color": "black", "linewidth": 1.4, "linestyle": "-", "alpha": 0.95}


def label_position(df: pd.DataFrame, mach_target: float) -> tuple[float, float]:
    idx = (df["Mach"] - mach_target).abs().idxmin()
    row = df.loc[idx]
    return float(row["Mach"]), float(row["alpha"])


def plot_blumen_reference(curves: list[dict], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for curve in curves:
        style = style_for_curve(curve)
        ax.plot(curve["data"]["Mach"], curve["data"]["alpha"], **style)

    main_levels = [curve for curve in curves if curve["family"] == "ci_level"]
    targets = [
        (0.10, 1.15, "0·1"),
        (0.07, 1.32, "0·07"),
        (0.05, 1.48, "0·05"),
        (0.03, 1.63, "0·03"),
        (0.01, 1.88, "0·01"),
    ]
    for level, mach_target, label in targets:
        curve = next((item for item in main_levels if abs(float(item["level"]) - level) < 1e-12), None)
        if curve is None:
            continue
        x, y = label_position(curve["data"], mach_target)
        ax.text(x + 0.02, y - 0.01, label, fontsize=9)

    for level, label, mach_target, alpha_shift in [
        (0.01, r"$c_r = 0,\; c_i = 0.01$", 1.02, 0.018),
        (0.02, r"$c_r = 0,\; c_i = 0.02$", 1.08, 0.010),
    ]:
        curve = next(
            (item for item in curves if item["family"] == "ci_special" and abs(float(item["level"]) - level) < 1e-12),
            None,
        )
        if curve is None:
            continue
        x, y = label_position(curve["data"], mach_target)
        ax.text(x - 0.01, y + alpha_shift, label, fontsize=8.5)

    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(0.0, 0.50)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=10)
    ax.set_title(r"Blumen 1975: isolignes supersoniques de $c_i$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.grid(True, linestyle=":", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shooting_overlay(df: pd.DataFrame, curves: list[dict], output_path: Path) -> None:
    pivot = df.pivot(index="alpha", columns="Mach", values="ci").sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    ci_values = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    levels = contour_levels(curves)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    contour = ax.contour(
        mach_grid,
        alpha_grid,
        ci_values,
        levels=levels,
        cmap="viridis",
        linewidths=2.0,
    )
    ax.clabel(contour, fmt="%0.02f", fontsize=8)

    for curve in curves:
        style = style_for_curve(curve)
        ax.plot(curve["data"]["Mach"], curve["data"]["alpha"], **style)

    ax.set_xlim(float(np.min(machs)), float(np.max(machs)))
    ax.set_ylim(float(np.min(alphas)), float(np.max(alphas)))
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"Nombre d'onde $\alpha$")
    ax.set_title(r"Reconstruction visuelle: isolignes de $c_i$ du shooting vs Blumen")
    ax.grid(True, linestyle=":", alpha=0.25)

    legend_lines = [
        plt.Line2D([], [], color="black", linewidth=1.4, label=r"Blumen $c_i$ (principal)"),
        plt.Line2D([], [], color="black", linewidth=1.0, linestyle=(0, (5, 3)), label=r"Blumen $c_r=0$"),
        plt.Line2D([], [], color=plt.get_cmap("viridis")(0.7), linewidth=2.0, label=r"Shooting $c_i$"),
    ]
    ax.legend(handles=legend_lines, loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_pointwise_curve_levels(curves: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for curve in curves:
        df = curve["data"]
        for _, row in df.iterrows():
            rows.append(
                {
                    "family": str(curve["family"]),
                    "label": str(curve["label"]),
                    "level": float(curve["level"]),
                    "Mach": float(row["Mach"]),
                    "alpha": float(row["alpha"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_ci_curves()
    anchors = build_anchor_points(curves)
    machs = np.linspace(float(args.mach_min), float(args.mach_max), int(args.num_mach))
    alphas = np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.num_alpha))

    print("Reconstruction visuelle Blumen supersonique en c_i")
    print(f"Mach range: {float(args.mach_min):.3f} -> {float(args.mach_max):.3f} ({int(args.num_mach)} points)")
    print(f"alpha range: {float(args.alpha_min):.3f} -> {float(args.alpha_max):.3f} ({int(args.num_alpha)} points)")
    print(f"c_i levels: {' '.join(f'{level:.2f}' for level in contour_levels(curves))}")
    print(f"output stem: {args.output_stem}")

    growth_df = sample_supersonic_growth_map(
        alphas,
        machs,
        ci_max=float(args.ci_max),
        cr_max=float(args.cr_max),
        anchor_points=anchors,
        tracking_weight=float(args.tracking_weight),
    )
    points_df = summarize_pointwise_curve_levels(curves)

    growth_csv = output_dir / f"{args.output_stem}_growth_map.csv"
    points_csv = output_dir / f"{args.output_stem}_blumen_points.csv"
    ref_png = output_dir / f"{args.output_stem}_blumen_reference.png"
    overlay_png = output_dir / f"{args.output_stem}_shooting_overlay.png"

    growth_df.to_csv(growth_csv, index=False)
    points_df.to_csv(points_csv, index=False)
    plot_blumen_reference(curves, ref_png)
    plot_shooting_overlay(growth_df, curves, overlay_png)

    print(f"Wrote {growth_csv}")
    print(f"Wrote {points_csv}")
    print(f"Wrote {ref_png}")
    print(f"Wrote {overlay_png}")


if __name__ == "__main__":
    main()
