from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN
from src.physics.kh_subsonic_residual import xi_to_y


def build_model_from_config(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return KHSubsonicFixedMachPINN(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        hidden_dim=int(config["hidden_dim"]),
        mode_depth=int(config["mode_depth"]),
        ci_depth=int(config["ci_depth"]),
        activation=str(config["activation"]),
        fourier_features=int(config["fourier_features"]),
        fourier_scale=float(config["fourier_scale"]),
        initial_ci=float(config["initial_ci"]),
        mapping_scale=float(config["mapping_scale"]),
        trainable_mapping_scale=bool(config["trainable_mapping_scale"]),
    )


def solve_reference_curve(mach: float, alpha_min: float, alpha_max: float, num_alpha: int) -> pd.DataFrame:
    rows = []
    for alpha in np.linspace(alpha_min, alpha_max, num_alpha):
        result = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve()
        rows.append(
            {
                "alpha": float(alpha),
                "ci_reference": float(result.ci),
                "omega_i_reference": float(result.omega_i),
            }
        )
    return pd.DataFrame(rows)


def load_model(run_dir: Path, device: torch.device) -> tuple[KHSubsonicFixedMachPINN, pd.Series, pd.DataFrame]:
    config_df = pd.read_csv(run_dir / "config.csv")
    history = pd.read_csv(run_dir / "history.csv")
    config = config_df.iloc[0]
    model = build_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
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
    axes[1].set_title("Audit spectral")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ci_curve(
    model: KHSubsonicFixedMachPINN,
    config: pd.Series,
    output_path: Path,
    *,
    num_alpha: int,
    device: torch.device,
) -> pd.DataFrame:
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), num_alpha)
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)

    reference_df = solve_reference_curve(
        mach=float(config["mach"]),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        num_alpha=num_alpha,
    )
    plot_df = reference_df.copy()
    plot_df["ci_pinn"] = ci_pred
    plot_df["ci_abs_err"] = np.abs(plot_df["ci_pinn"] - plot_df["ci_reference"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(plot_df["alpha"], plot_df["ci_reference"], label="classique", linewidth=2.0)
    axes[0].plot(plot_df["alpha"], plot_df["ci_pinn"], "--", label="PINN", linewidth=2.0)
    axes[0].set_title(fr"$c_i(\alpha)$ a M={float(config['mach']):.3f}")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(plot_df["alpha"], plot_df["ci_abs_err"], color="tab:red")
    axes[1].set_title(r"Erreur absolue sur $c_i$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$|c_i^{PINN} - c_i^{ref}|$")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def plot_modes_pdf(
    model: KHSubsonicFixedMachPINN,
    config: pd.Series,
    output_path: Path,
    *,
    alpha_values: list[float],
    n_y: int,
    device: torch.device,
) -> None:
    xi = torch.linspace(-0.98, 0.98, n_y, device=device).view(-1, 1)
    mapping_scale = model.get_mapping_scale().detach()
    y = xi_to_y(xi, mapping_scale).detach().cpu().numpy().reshape(-1)

    with PdfPages(output_path) as pdf:
        for alpha in alpha_values:
            alpha_tensor = torch.full_like(xi, float(alpha))
            with torch.no_grad():
                pred = model(xi, alpha_tensor)
                ci_pred = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())

            p_r = pred[:, 0].detach().cpu().numpy()
            p_i = pred[:, 1].detach().cpu().numpy()
            norm = max(np.max(np.abs(p_r)), np.max(np.abs(p_i)), 1e-12)
            p_r = p_r / norm
            p_i = p_i / norm

            ci_ref = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(config["mach"])).solve().ci

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].plot(y, p_r, label="Re(p)")
            axes[0].plot(y, p_i, "--", label="Im(p)")
            axes[0].set_title(fr"Mode PINN pour $\alpha={alpha:.3f}$")
            axes[0].set_xlabel("y")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(y, np.tanh(y), label=r"$U(y)=\tanh(y)$")
            axes[1].set_title(
                fr"$c_i^{{PINN}}={ci_pred:.5f}$, $c_i^{{ref}}={ci_ref:.5f}$"
            )
            axes[1].set_xlabel("y")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.suptitle(fr"Prototype PINN subsonique | M={float(config['mach']):.3f}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation d'un run PINN KH subsonique a Mach fixe.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-alpha", type=int, default=81)
    parser.add_argument("--mode-alpha", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--n-y", type=int, default=801)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    run_dir = args.run_dir
    model, config, history = load_model(run_dir, device)

    history_png = run_dir / "history_diagnostics.png"
    ci_png = run_dir / "ci_curve_vs_reference.png"
    ci_csv = run_dir / "ci_curve_vs_reference.csv"
    modes_pdf = run_dir / "pinn_modes.pdf"

    plot_history(history, history_png)
    ci_df = plot_ci_curve(model, config, ci_png, num_alpha=args.num_alpha, device=device)
    ci_df.to_csv(ci_csv, index=False)
    plot_modes_pdf(
        model,
        config,
        modes_pdf,
        alpha_values=[float(alpha) for alpha in args.mode_alpha],
        n_y=args.n_y,
        device=device,
    )

    print(f"History plot: {history_png}")
    print(f"CI curve plot: {ci_png}")
    print(f"CI curve CSV: {ci_csv}")
    print(f"Modes PDF: {modes_pdf}")


if __name__ == "__main__":
    main()
