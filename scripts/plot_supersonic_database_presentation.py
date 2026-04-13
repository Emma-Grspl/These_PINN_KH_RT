from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT_DIR = Path(__file__).resolve().parents[1]
BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def parse_reference_level(csv_path: Path) -> tuple[float | None, str, str]:
    stem = csv_path.stem.strip().replace("_", ".").replace(",", ".")
    lower = stem.lower()
    if lower.startswith("ci"):
        value = float(lower[2:] or "0") / 100.0
        return value, fr"$c_r = 0,\; c_i = {value:.2f}$", "ci_special"
    if lower.startswith("cr"):
        value = float(lower[2:] or "0")
        return value, fr"$c_i = 0,\; c_r = {value:.2f}$", "cr_special"
    numeric = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    if not numeric:
        return None, stem, "unknown"
    value = float(numeric)
    return value, fr"$c_i = {value:.2f}$", "ci_level"


def load_digitized_curves() -> list[dict]:
    curves = []
    for csv_file in sorted(BLUMEN_DIR.glob("*.csv")):
        level, label, family = parse_reference_level(csv_file)
        df = (
            pd.read_csv(
                csv_file,
                header=None,
                names=["Mach", "alpha"],
                sep=";",
                decimal=",",
                engine="python",
            )
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        curves.append({"level": level, "label": label, "family": family, "data": df})
    return curves


def reshape_field(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mach_values = np.sort(df["Mach"].unique())
    alpha_values = np.sort(df["alpha"].unique())
    pivot = df.pivot(index="Mach", columns="alpha", values=value_col).sort_index().sort_index(axis=1)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    return aa, mm, pivot.to_numpy(dtype=float)


def filter_curves_to_domain(curves: list[dict], df: pd.DataFrame) -> list[dict]:
    alpha_min = float(df["alpha"].min())
    alpha_max = float(df["alpha"].max())
    mach_min = float(df["Mach"].min())
    mach_max = float(df["Mach"].max())
    local_curves = []
    for curve in curves:
        sub = curve["data"]
        mask = (
            (sub["alpha"] >= alpha_min - 1e-9)
            & (sub["alpha"] <= alpha_max + 1e-9)
            & (sub["Mach"] >= mach_min - 1e-9)
            & (sub["Mach"] <= mach_max + 1e-9)
        )
        clipped = sub.loc[mask].copy()
        if len(clipped) >= 2:
            local_curves.append({**curve, "data": clipped})
    return local_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plots de presentation pour la base supersonique.")
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.surface_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = filter_curves_to_domain(load_digitized_curves(), df)

    aa, mm, accepted = reshape_field(df, "accepted")
    _, _, gep_ci = reshape_field(df, "gep_ci")
    _, _, gep_cr = reshape_field(df, "gep_cr")

    accepted_mask = accepted >= 0.5
    gep_ci_masked = np.where(accepted_mask, gep_ci, np.nan)
    gep_cr_masked = np.where(accepted_mask, gep_cr, np.nan)

    # 1. Binary acceptance map.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmap = ListedColormap(["#d9d9d9", "#1b9e77"])
    pcm = ax.pcolormesh(aa, mm, accepted, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Base supersonique : points retenus")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    cbar = fig.colorbar(pcm, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["rejetes", "acceptes"])
    ax.grid(True, linestyle=":", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "05_supersonic_acceptance_map_clean.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 2. Accepted-only ci map with Blumen ci curves.
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    pcm = ax.pcolormesh(aa, mm, gep_ci_masked, shading="auto", cmap="YlOrRd")
    valid_levels = [level for level in [0.006, 0.007, 0.008, 0.009] if np.nanmin(gep_ci_masked) <= level <= np.nanmax(gep_ci_masked)]
    if valid_levels:
        contours = ax.contour(aa, mm, gep_ci_masked, levels=valid_levels, colors="#7f0000", linewidths=1.2)
        ax.clabel(contours, fmt="%.3f", fontsize=8)
    for curve in curves:
        if curve["family"] not in {"ci_level", "ci_special"}:
            continue
        level = curve["level"]
        if level is not None and level > 0.03:
            continue
        style = "-" if curve["family"] == "ci_level" else "--"
        ax.plot(
            curve["data"]["alpha"],
            curve["data"]["Mach"],
            color="black",
            linewidth=1.6,
            linestyle=style,
            alpha=0.9,
        )
        tail = curve["data"].iloc[-1]
        ax.text(float(tail["alpha"]) + 0.0004, float(tail["Mach"]), curve["label"], fontsize=8, color="black")
    ax.set_title(r"Carte de $c_i$ acceptee avec isolignes Blumen")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, linestyle=":", alpha=0.25)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$c_i$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "06_supersonic_ci_vs_blumen_clean.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 3. Accepted-only branch map on c_r.
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    pcm = ax.pcolormesh(aa, mm, gep_cr_masked, shading="auto", cmap="viridis")
    ax.contour(aa, mm, accepted, levels=[0.5], colors="white", linewidths=1.0)
    ax.set_title(r"Branche retenue sur les points acceptes : $c_r$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, linestyle=":", alpha=0.25)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$c_r$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "07_supersonic_cr_branch_map_clean.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(args.output_dir / "05_supersonic_acceptance_map_clean.png")
    print(args.output_dir / "06_supersonic_ci_vs_blumen_clean.png")
    print(args.output_dir / "07_supersonic_cr_branch_map_clean.png")


if __name__ == "__main__":
    main()
