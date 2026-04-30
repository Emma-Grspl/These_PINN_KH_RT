from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse visuelle de la base de modes GEP supersoniques.")
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def reshape_field(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mach_values = np.sort(df["Mach"].unique())
    alpha_values = np.sort(df["alpha"].unique())
    pivot = df.pivot(index="Mach", columns="alpha", values=value_col).sort_index().sort_index(axis=1)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    return aa, mm, pivot.to_numpy(dtype=float)


def local_jump_metric(field: np.ndarray) -> np.ndarray:
    jump = np.zeros_like(field, dtype=float)
    if field.shape[1] > 1:
        jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(np.diff(field, axis=1)))
        jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(np.diff(field, axis=1)))
    if field.shape[0] > 1:
        jump[1:, :] = np.maximum(jump[1:, :], np.abs(np.diff(field, axis=0)))
        jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(np.diff(field, axis=0)))
    return jump


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.surface_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aa, mm, accepted = reshape_field(df, "accepted")
    _, _, gep_cr = reshape_field(df, "gep_cr")
    _, _, gep_ci = reshape_field(df, "gep_ci")
    _, _, overlap = reshape_field(df.fillna({"overlap_to_previous": 0.0}), "overlap_to_previous")

    jump_cr = local_jump_metric(gep_cr)
    jump_ci = local_jump_metric(gep_ci)

    # 1. Validity / accepted map
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    pcm = ax.pcolormesh(aa, mm, accepted, shading="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Solveur supersonique : points acceptes")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("accepted")
    fig.tight_layout()
    fig.savefig(args.output_dir / "05_supersonic_acceptance_map.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 2. Selected branch by c_r
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    pcm = ax.pcolormesh(aa, mm, gep_cr, shading="auto", cmap="viridis")
    ax.contour(aa, mm, accepted, levels=[0.5], colors="white", linewidths=1.2)
    ax.set_title(r"Branche selectionnee : carte de $c_r$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$c_r$")
    fig.tight_layout()
    fig.savefig(args.output_dir / "06_supersonic_selected_cr_map.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # 3. Local continuity diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)
    panels = [
        (jump_cr, r"Saut local max sur $c_r$", "magma"),
        (jump_ci, r"Saut local max sur $c_i$", "magma"),
        (1.0 - overlap, r"$1-$overlap\_to\_previous$", "cividis"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        pcm = ax.pcolormesh(aa, mm, field, shading="auto", cmap=cmap)
        ax.contour(aa, mm, accepted, levels=[0.5], colors="white", linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(pcm, ax=ax)
    fig.savefig(args.output_dir / "07_supersonic_branch_continuity_diagnostics.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Save summary CSV for worst discontinuities
    summary = pd.DataFrame(
        {
            "alpha": aa.reshape(-1),
            "Mach": mm.reshape(-1),
            "accepted": accepted.reshape(-1),
            "gep_cr": gep_cr.reshape(-1),
            "gep_ci": gep_ci.reshape(-1),
            "jump_cr": jump_cr.reshape(-1),
            "jump_ci": jump_ci.reshape(-1),
            "one_minus_overlap": (1.0 - overlap).reshape(-1),
        }
    )
    summary.sort_values(["jump_cr", "jump_ci", "one_minus_overlap"], ascending=False).to_csv(
        args.output_dir / "supersonic_branch_diagnostics_top.csv",
        index=False,
    )

    print(f"Acceptance map: {args.output_dir / '05_supersonic_acceptance_map.png'}")
    print(f"Selected c_r map: {args.output_dir / '06_supersonic_selected_cr_map.png'}")
    print(f"Branch continuity diagnostics: {args.output_dir / '07_supersonic_branch_continuity_diagnostics.png'}")
    print(f"Diagnostics CSV: {args.output_dir / 'supersonic_branch_diagnostics_top.csv'}")


if __name__ == "__main__":
    main()
