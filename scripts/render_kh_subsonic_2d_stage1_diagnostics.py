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
from src.training.kh_subsonic_trainer_2d_stage1 import load_stage0_model_from_checkpoint


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


def _load_stage1_model(stage1_dir: Path, *, device: torch.device):
    checkpoint = torch.load(stage1_dir / "model_best.pt", map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported Stage 1 checkpoint format in {stage1_dir / 'model_best.pt'}.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        model_config = checkpoint.get("config")
    if not isinstance(model_config, dict):
        config_df = pd.read_csv(stage1_dir / "config.csv")
        model_config = config_df.iloc[0].to_dict()
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config_df = pd.read_csv(stage1_dir / "config.csv")
        config = config_df.iloc[0].to_dict()
    from src.models.kh_subsonic_pinn_2d import build_kh_subsonic_pinn_2d_from_config

    model = build_kh_subsonic_pinn_2d_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def _build_prediction_surface(model, alpha_values: np.ndarray, mach_values: np.ndarray, *, device: torch.device) -> pd.DataFrame:
    aa, mm = np.meshgrid(alpha_values, mach_values)
    alpha_t = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_t = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_t, mach_t).detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({"alpha": aa.reshape(-1), "Mach": mm.reshape(-1), "ci_pinn": ci_pred})


def plot_ci_curves_by_mach(
    grid_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    *,
    mach_values: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(mach_values), 1, figsize=(8.0, 3.0 * len(mach_values)), sharex=True)
    if len(mach_values) == 1:
        axes = [axes]
    for ax, mach in zip(axes, mach_values):
        sub = grid_df[np.isclose(grid_df["Mach"], float(mach), atol=1e-12)].sort_values("alpha")
        anc = anchors_df[np.isclose(anchors_df["Mach"], float(mach), atol=1e-12)].sort_values("alpha")
        ax.plot(sub["alpha"], sub["ci_reference"], color="black", linewidth=1.6, label="Classique")
        ax.plot(sub["alpha"], sub["ci_pinn"], color="tab:orange", linewidth=1.6, linestyle="--", label="PINN Stage 1")
        ax.scatter(anc["alpha"], anc["ci_reference"], color="black", s=28, zorder=3, label="Anchors")
        ax.scatter(anc["alpha"], anc["ci_pred_stage1"], color="tab:orange", s=28, zorder=3)
        ax.set_ylabel(r"$c_i$")
        ax.set_title(f"Mach = {mach:.2f}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$\alpha$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ci_error_heatmap(grid_df: pd.DataFrame, output_path: Path) -> None:
    work = grid_df.copy()
    work["ci_error_display"] = np.where(
        np.abs(work["ci_reference"].to_numpy(dtype=float)) > 1e-4,
        work["ci_abs_err"].to_numpy(dtype=float) / np.abs(work["ci_reference"].to_numpy(dtype=float)),
        work["ci_abs_err"].to_numpy(dtype=float),
    )
    pivot = work.pivot(index="Mach", columns="alpha", values="ci_error_display").sort_index().sort_index(axis=1)
    alpha_values = pivot.columns.to_numpy(dtype=float)
    mach_values = pivot.index.to_numpy(dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pcm = ax.pcolormesh(aa, mm, pivot.to_numpy(dtype=float), shading="auto", cmap="magma")
    ax.set_title(r"Erreur $c_i$: relative instable / absolue neutre")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, alpha=0.20)
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    left_keys = ["loss_total", "loss_pde", "loss_bc", "loss_norm", "loss_phase"]
    for key in left_keys:
        if key in history.columns:
            axes[0].plot(history["epoch"], history[key], label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss history")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    right_keys = ["ci_anchor_max_abs", "ci_anchor_max_rel_unstable", "ci_neutral_max_abs"]
    for key in right_keys:
        if key in history.columns:
            axes[1].plot(history["epoch"], history[key], label=key)
    axes[1].set_yscale("log")
    axes[1].set_title("Spectral lock audit")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_panel(
    classic_fields: dict[str, np.ndarray] | None,
    pinn_fields: dict[str, np.ndarray],
    *,
    alpha: float,
    mach: float,
    output_path: Path,
) -> None:
    titles = [("rho", r"$\hat{\rho}$"), ("u", r"$\hat{u}$"), ("v", r"$\hat{v}$"), ("p", r"$\hat{p}$")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for ax, (field_name, title) in zip(axes.flat, titles):
        if classic_fields is not None:
            ax.plot(classic_fields["y"], np.real(classic_fields[field_name]), color="black", linewidth=1.4, label="Classic Re")
            ax.plot(classic_fields["y"], np.imag(classic_fields[field_name]), color="black", linewidth=1.0, linestyle=":", label="Classic Im")
        ax.plot(pinn_fields["y"], np.real(pinn_fields[field_name]), color="tab:orange", linewidth=1.4, label="PINN Re")
        ax.plot(pinn_fields["y"], np.imag(pinn_fields[field_name]), color="tab:orange", linewidth=1.0, linestyle=":", label="PINN Im")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 1 mode diagnostic | alpha={alpha:.2f}, Mach={mach:.2f}")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def maybe_render_mode_panels(stage1_dir: Path, model, *, device: torch.device, output_dir: Path) -> None:
    try:
        from scripts.evaluate_kh_subsonic_pinn_2d_ci_modes import load_classic_full_mode, load_pinn_full_mode
    except Exception as exc:  # pragma: no cover - optional path
        print(f"Skipping modal comparison import: {exc}")
        return

    mode_specs = [
        (0.30, 0.50, "04_stage1_modes_M05_alpha030_2x2.png"),
        (0.50, 0.50, "05_stage1_modes_M05_alpha050_2x2.png"),
        (0.70, 0.50, "06_stage1_modes_M05_alpha070_2x2.png"),
    ]
    for alpha, mach, filename in mode_specs:
        try:
            classic_fields, _ = load_classic_full_mode(alpha, mach)
        except Exception as exc:  # pragma: no cover - optional path
            print(f"Classic modal diagnostic unavailable at alpha={alpha:.2f}, Mach={mach:.2f}: {exc}")
            classic_fields = None
        try:
            pinn_fields, _ = load_pinn_full_mode(model, alpha=alpha, mach=mach, n_y=1001, device=device)
            _plot_mode_panel(classic_fields, pinn_fields, alpha=alpha, mach=mach, output_path=output_dir / filename)
        except Exception as exc:  # pragma: no cover - optional path
            print(f"Skipping modal diagnostic at alpha={alpha:.2f}, Mach={mach:.2f}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render diagnostics for the 2D hybrid4ci Stage 1 run.")
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/pinn_subsonic/baseline_2D_Stage1"))
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--grid-alpha", type=int, default=61)
    parser.add_argument("--grid-mach", type=int, default=31)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")

    model, config, checkpoint = _load_stage1_model(args.stage1_dir, device=device)
    history = pd.read_csv(args.stage1_dir / "history.csv")
    anchors_df = pd.read_csv(args.stage1_dir / "anchor_predictions_stage1.csv")

    mach_values = _parse_float_list(config.get("mach_values_json") or config.get("mach_values"))
    if not mach_values:
        mach_values = sorted(anchors_df["Mach"].drop_duplicates().to_list())

    alpha_grid = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.grid_alpha), dtype=float)
    mach_grid = np.linspace(min(mach_values), max(mach_values), int(args.grid_mach), dtype=float)

    reference_cache = args.reference_cache if args.reference_cache is not None else resolve_reference_cache_path()
    reference_df = build_reference_surface(alpha_values=alpha_grid, mach_values=mach_grid, reference_cache=reference_cache)
    pred_df = _build_prediction_surface(model, alpha_grid, mach_grid, device=device)
    grid_df = reference_df.merge(pred_df, on=["alpha", "Mach"], how="inner")
    grid_df["ci_abs_err"] = np.abs(grid_df["ci_pinn"] - grid_df["ci_reference"])
    grid_df.to_csv(args.output_dir / "stage1_ci_surface_table.csv", index=False)

    plot_ci_curves_by_mach(
        grid_df,
        anchors_df,
        mach_values=mach_values,
        output_path=args.output_dir / "01_stage1_ci_vs_alpha_by_mach.png",
    )
    plot_ci_error_heatmap(grid_df, args.output_dir / "02_stage1_ci_error_heatmap.png")
    plot_loss_history(history, args.output_dir / "03_stage1_loss_history.png")
    maybe_render_mode_panels(args.stage1_dir, model, device=device, output_dir=args.output_dir)

    summary = {
        "stage0_checkpoint": checkpoint.get("stage0_checkpoint", ""),
        "best_epoch": checkpoint.get("best_epoch", np.nan),
        "ci_anchor_max_abs": float(anchors_df["ci_abs_err"].max()) if not anchors_df.empty else np.nan,
        "ci_anchor_mae_abs": float(anchors_df["ci_abs_err"].mean()) if not anchors_df.empty else np.nan,
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "diagnostics_summary.csv", index=False)
    print(f"Stage 1 diagnostics written to {args.output_dir}")


if __name__ == "__main__":
    main()
