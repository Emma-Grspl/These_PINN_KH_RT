from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.kh_subsonic_sampling import SubsonicReferenceCache2D
from src.models.kh_subsonic_pinn import KHSubsonicMultiMachPINN


def build_model_from_config(config: pd.Series) -> KHSubsonicMultiMachPINN:
    mode_hidden_dim = None if "mode_hidden_dim" not in config.index or pd.isna(config["mode_hidden_dim"]) else int(config["mode_hidden_dim"])
    ci_hidden_dim = None if "ci_hidden_dim" not in config.index or pd.isna(config["ci_hidden_dim"]) else int(config["ci_hidden_dim"])
    return KHSubsonicMultiMachPINN(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        mach_min=float(config["mach_min"]),
        mach_max=float(config["mach_max"]),
        hidden_dim=int(config["hidden_dim"]),
        mode_hidden_dim=mode_hidden_dim,
        ci_hidden_dim=ci_hidden_dim,
        mode_depth=int(config["mode_depth"]),
        ci_depth=int(config["ci_depth"]),
        activation=str(config["activation"]),
        fourier_features=int(config["fourier_features"]),
        fourier_scale=float(config["fourier_scale"]),
        initial_ci=float(config["initial_ci"]),
        mapping_scale=float(config["mapping_scale"]),
        trainable_mapping_scale=bool(config["trainable_mapping_scale"]),
        mode_representation=str(config["mode_representation"]) if "mode_representation" in config.index else "cartesian",
    )


def load_model(
    run_dir: Path,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[KHSubsonicMultiMachPINN, pd.Series, pd.DataFrame]:
    config_df = pd.read_csv(run_dir / "config.csv")
    history = pd.read_csv(run_dir / "history.csv")
    config = config_df.iloc[0]
    model = build_model_from_config(config)
    state_dict = torch.load(checkpoint_path or (run_dir / "model_best.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, history


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["loss"], label="loss totale")
    for key in ("loss_pde", "loss_bc", "loss_norm", "loss_phase", "loss_ci_supervision"):
        axes[0].plot(history["epoch"], history[key], alpha=0.8, label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Historique des losses")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    audited = history.dropna(subset=["audit_ci_mae"])
    axes[1].plot(audited["epoch"], audited["audit_ci_mae"], label="audit ci MAE")
    axes[1].plot(audited["epoch"], audited["audit_ci_max_abs"], label="audit ci max abs")
    axes[1].set_yscale("log")
    axes[1].set_title("Audit spectral 2D")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_reference_surface(config: pd.Series, *, num_alpha: int, num_mach: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = SubsonicReferenceCache2D.build(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        mach_min=float(config["mach_min"]),
        mach_max=float(config["mach_max"]),
        num_alpha=int(config["n_reference_alpha"]),
        num_mach=int(config["n_reference_mach"]),
    )
    return cache.audit_grid(num_alpha=num_alpha, num_mach=num_mach)


def build_prediction_surface(
    model: KHSubsonicMultiMachPINN,
    config: pd.Series,
    *,
    num_alpha: int,
    num_mach: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), num_alpha, dtype=float)
    mach_values = np.linspace(float(config["mach_min"]), float(config["mach_max"]), num_mach, dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    alpha_tensor = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_tensor = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor, mach_tensor).cpu().numpy().reshape(aa.shape)
    return aa, mm, ci_pred


def plot_surfaces(
    aa: np.ndarray,
    mm: np.ndarray,
    ci_ref: np.ndarray,
    ci_pred: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    ci_abs_err = np.abs(ci_pred - ci_ref)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    panels = [
        (ci_ref, r"Reference $c_i(\alpha, M)$", "viridis"),
        (ci_pred, r"PINN $c_i(\alpha, M)$", "viridis"),
        (ci_abs_err, r"$|c_i^{PINN}-c_i^{ref}|$", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        pcm = ax.pcolormesh(aa, mm, field, shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(pcm, ax=ax)

    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        {
            "alpha": aa.reshape(-1),
            "mach": mm.reshape(-1),
            "ci_reference": ci_ref.reshape(-1),
            "ci_pinn": ci_pred.reshape(-1),
            "ci_abs_err": ci_abs_err.reshape(-1),
        }
    )


def plot_error_map(
    aa: np.ndarray,
    mm: np.ndarray,
    ci_abs_err: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    pcm = ax.pcolormesh(aa, mm, ci_abs_err, shading="auto", cmap="magma")
    ax.set_title(r"Carte d'erreur absolue sur $c_i$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation d'un run PINN KH subsonique 2D.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-alpha", type=int, default=61)
    parser.add_argument("--num-mach", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = args.run_dir
    model, config, history = load_model(run_dir, device, checkpoint_path=args.checkpoint)

    history_png = run_dir / "history_diagnostics_2d.png"
    surface_png = run_dir / "ci_surface_vs_reference.png"
    error_png = run_dir / "ci_error_map.png"
    surface_csv = run_dir / "ci_surface_vs_reference.csv"
    top_csv = run_dir / "ci_surface_top_errors.csv"

    plot_history(history, history_png)
    aa_ref, mm_ref, ci_ref = build_reference_surface(config, num_alpha=args.num_alpha, num_mach=args.num_mach)
    aa_pred, mm_pred, ci_pred = build_prediction_surface(
        model,
        config,
        num_alpha=args.num_alpha,
        num_mach=args.num_mach,
        device=device,
    )
    if aa_ref.shape != aa_pred.shape or mm_ref.shape != mm_pred.shape:
        raise RuntimeError("Reference and prediction grids do not match.")

    df = plot_surfaces(aa_ref, mm_ref, ci_ref, ci_pred, surface_png)
    df.to_csv(surface_csv, index=False)
    plot_error_map(aa_ref, mm_ref, np.abs(ci_pred - ci_ref), error_png)

    top_df = df.sort_values("ci_abs_err", ascending=False).head(args.top_k)
    top_df.to_csv(top_csv, index=False)

    print(f"History plot: {history_png}")
    print(f"CI surface plot: {surface_png}")
    print(f"CI error map: {error_png}")
    print(f"CI surface CSV: {surface_csv}")
    print(f"Top-error CSV: {top_csv}")


if __name__ == "__main__":
    main()
