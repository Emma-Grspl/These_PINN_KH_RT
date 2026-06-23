from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.kh_subsonic_sampling_2d import build_reference_surface, resolve_reference_cache_path
from src.training.kh_subsonic_trainer_2d_hybrid4ci import load_stage0_checkpoint


def _parse_float_list(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        import json

        return [float(item) for item in json.loads(text)]
    return [float(item) for item in text.replace(",", " ").split()]


def _model_ci_surface(
    model,
    *,
    alpha_values: np.ndarray,
    mach_values: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    aa, mm = np.meshgrid(alpha_values, mach_values)
    alpha_t = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_t = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_t, mach_t).detach().cpu().numpy().reshape(-1)
    return pd.DataFrame(
        {
            "alpha": aa.reshape(-1),
            "Mach": mm.reshape(-1),
            "ci_pinn": ci_pred,
        }
    )


def plot_anchor_fit(anchor_df: pd.DataFrame, output_path: Path) -> dict[str, float]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].scatter(anchor_df["ci_reference"], anchor_df["ci_pred"], s=55, color="tab:blue")
    lo = float(min(anchor_df["ci_reference"].min(), anchor_df["ci_pred"].min()))
    hi = float(max(anchor_df["ci_reference"].max(), anchor_df["ci_pred"].max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.2)
    axes[0].set_title("Anchors Stage 0: classique vs PINN")
    axes[0].set_xlabel(r"$c_i^{classic}$")
    axes[0].set_ylabel(r"$c_i^{PINN}$")
    axes[0].grid(True, alpha=0.25)

    scatter = axes[1].scatter(
        anchor_df["alpha"],
        anchor_df["Mach"],
        c=anchor_df["ci_rel_err"],
        cmap="magma",
        s=75,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[1].set_title("Erreur relative sur les anchors")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$M$")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label=r"$|c_i^{PINN}-c_i^{classic}|/|c_i^{classic}|$")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "ci_anchor_mae_abs": float(anchor_df["ci_abs_err"].mean()),
        "ci_anchor_max_abs": float(anchor_df["ci_abs_err"].max()),
        "ci_anchor_mae_rel": float(anchor_df["ci_rel_err"].mean()),
        "ci_anchor_max_rel": float(anchor_df["ci_rel_err"].max()),
    }


def plot_surface_comparison(grid_df: pd.DataFrame, output_path: Path) -> dict[str, float]:
    pivot_ref = grid_df.pivot(index="Mach", columns="alpha", values="ci_reference").sort_index().sort_index(axis=1)
    pivot_pred = grid_df.pivot(index="Mach", columns="alpha", values="ci_pinn").sort_index().sort_index(axis=1)
    pivot_rel = grid_df.pivot(index="Mach", columns="alpha", values="ci_rel_err").sort_index().sort_index(axis=1)
    alpha_values = pivot_ref.columns.to_numpy(dtype=float)
    mach_values = pivot_ref.index.to_numpy(dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    fields = [
        (pivot_ref.to_numpy(dtype=float), r"Classique $c_i(\alpha, M)$", "viridis"),
        (pivot_pred.to_numpy(dtype=float), r"PINN $c_i(\alpha, M)$", "viridis"),
        (pivot_rel.to_numpy(dtype=float), r"Erreur relative sur $c_i$", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, fields):
        pcm = ax.pcolormesh(aa, mm, field, shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$M$")
        ax.grid(True, alpha=0.20)
        fig.colorbar(pcm, ax=ax)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "ci_grid_mae_abs": float(grid_df["ci_abs_err"].mean()),
        "ci_grid_max_abs": float(grid_df["ci_abs_err"].max()),
        "ci_grid_mae_rel": float(grid_df["ci_rel_err"].mean()),
        "ci_grid_max_rel": float(grid_df["ci_rel_err"].max()),
    }


def plot_relative_error_map(grid_df: pd.DataFrame, output_path: Path) -> None:
    pivot_rel = grid_df.pivot(index="Mach", columns="alpha", values="ci_rel_err").sort_index().sort_index(axis=1)
    alpha_values = pivot_rel.columns.to_numpy(dtype=float)
    mach_values = pivot_rel.index.to_numpy(dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    pcm = ax.pcolormesh(aa, mm, pivot_rel.to_numpy(dtype=float), shading="auto", cmap="magma")
    ax.set_title(r"Carte d'erreur relative sur $c_i$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, alpha=0.20)
    fig.colorbar(pcm, ax=ax, label=r"$|c_i^{PINN}-c_i^{classic}|/|c_i^{classic}|$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render light diagnostics for the subsonic 2D hybrid4ci Stage 0 spectral lock."
    )
    parser.add_argument("--stage0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/pinn_subsonic/2d_hybrid4ci_diagnostics"))
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--grid-alpha", type=int, default=61)
    parser.add_argument("--grid-mach", type=int, default=31)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
    model, config, checkpoint = load_stage0_checkpoint(args.stage0_dir, device=device)

    anchors_df = pd.read_csv(args.stage0_dir / "anchors_used.csv")
    alpha_anchor = torch.tensor(anchors_df["alpha"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    mach_anchor = torch.tensor(anchors_df["Mach"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred_anchor = model.get_ci(alpha_anchor, mach_anchor).detach().cpu().numpy().reshape(-1)
    anchors_df["ci_pred"] = ci_pred_anchor
    anchors_df["ci_abs_err"] = np.abs(anchors_df["ci_pred"] - anchors_df["ci_reference"])
    anchors_df["ci_rel_err"] = anchors_df["ci_abs_err"] / np.maximum(
        np.abs(anchors_df["ci_reference"].to_numpy(dtype=float)),
        1e-12,
    )
    anchors_df.to_csv(output_dir / "anchors_diagnostic_table.csv", index=False)

    mach_values = _parse_float_list(config.get("mach_values_json") or config.get("mach_values"))
    if not mach_values:
        mach_values = sorted(anchors_df["Mach"].drop_duplicates().to_list())
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.grid_alpha), dtype=float)
    mach_grid = np.linspace(min(mach_values), max(mach_values), int(args.grid_mach), dtype=float)

    reference_cache = args.reference_cache
    if reference_cache is None:
        reference_cache = resolve_reference_cache_path()

    reference_df = build_reference_surface(
        alpha_values=alpha_values,
        mach_values=mach_grid,
        reference_cache=reference_cache,
    )
    pred_df = _model_ci_surface(model, alpha_values=alpha_values, mach_values=mach_grid, device=device)
    grid_df = reference_df.merge(pred_df, on=["alpha", "Mach"], how="inner")
    grid_df["ci_abs_err"] = np.abs(grid_df["ci_pinn"] - grid_df["ci_reference"])
    grid_df["ci_rel_err"] = grid_df["ci_abs_err"] / np.maximum(np.abs(grid_df["ci_reference"]), 1e-12)
    grid_df.to_csv(output_dir / "ci_surface_diagnostic_table.csv", index=False)

    summary = {}
    summary.update(plot_anchor_fit(anchors_df, output_dir / "01_stage0_ci_anchor_fit.png"))
    summary.update(plot_surface_comparison(grid_df, output_dir / "02_ci_surface_classic_vs_pinn.png"))
    plot_relative_error_map(grid_df, output_dir / "03_ci_relative_error_map.png")

    if isinstance(checkpoint, dict):
        best_metrics = checkpoint.get("best_metrics")
        if isinstance(best_metrics, dict):
            for key, value in best_metrics.items():
                summary[f"stage0_best_{key}"] = float(value)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "diagnostics_summary.csv", index=False)
    print(f"Diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
