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

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.data.kh_subsonic_sampling_2d import build_reference_surface
from src.physics.kh_subsonic_pressure_first_2d import (
    build_pressure_first_model_from_config,
    pressure_first_value_and_derivatives,
)
from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative


def _parse_float_list(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "nan":
        return []
    if text.startswith("["):
        import json

        return [float(item) for item in json.loads(text)]
    return [float(item) for item in text.replace(",", " ").split()]


def _load_stage1quater_model(stage1quater_dir: Path, *, device: torch.device):
    checkpoint_path = stage1quater_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported Stage 1quater checkpoint format in {checkpoint_path}.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError(f"Missing model_config in {checkpoint_path}.")
    model = build_pressure_first_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    config_df = pd.read_csv(stage1quater_dir / "config.csv")
    config = config_df.iloc[0].to_dict()
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
        ax.plot(sub["alpha"], sub["ci_pinn"], color="tab:orange", linewidth=1.6, linestyle="--", label="PINN Stage 1quater")
        ax.scatter(anc["alpha"], anc["ci_reference"], color="black", s=28, zorder=3, label="Anchors")
        if "ci_pred_stage1" in anc.columns:
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
    ci_ref = work["ci_reference"].to_numpy(dtype=float)
    ci_abs_err = work["ci_abs_err"].to_numpy(dtype=float)
    work["ci_error_display"] = np.where(
        np.abs(ci_ref) > 1e-4,
        ci_abs_err / np.maximum(np.abs(ci_ref), 1.0e-12),
        ci_abs_err,
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
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), constrained_layout=True)
    left_keys = [
        "loss_total",
        "loss_pde_pressure",
        "loss_bc_robin",
        "loss_gauge",
        "loss_center_pde",
        "loss_ci_anchor",
    ]
    for key in left_keys:
        if key in history.columns:
            axes[0].plot(history["epoch"], history[key], label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss history")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    right_keys = [
        "ci_anchor_max_abs",
        "ci_anchor_max_rel_unstable",
        "ci_neutral_max_abs",
    ]
    for key in right_keys:
        if key in history.columns:
            axes[1].plot(history["epoch"], history[key], label=key)
    axes[1].set_yscale("log")
    axes[1].set_title("Spectral lock")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def _normalize_extended_mode(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
    gamma: np.ndarray,
    p_y: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho, p_y = u * phase, v * phase, p * phase, rho * phase, p_y * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho, p_y = -u, -v, -p, -rho, -p_y

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return {
        "y": np.asarray(y, dtype=float),
        "u": np.asarray(u / scale, dtype=np.complex128),
        "v": np.asarray(v / scale, dtype=np.complex128),
        "p": np.asarray(p / scale, dtype=np.complex128),
        "rho": np.asarray(rho / scale, dtype=np.complex128),
        "p_y": np.asarray(p_y / scale, dtype=np.complex128),
        "gamma": np.asarray(gamma, dtype=np.complex128),
    }


def load_classic_full_mode_extended(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
    solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    k_left = np.asarray(sol_left.y[0])
    q_left = np.asarray(sol_left.y[1])
    ln_p_left = np.asarray(sol_left.y[2])
    phi_left = np.asarray(sol_left.y[3])
    k_right = np.asarray(sol_right.y[0])
    q_right = np.asarray(sol_right.y[1])
    ln_p_right = np.asarray(sol_right.y[2])
    phi_right = np.asarray(sol_right.y[3])

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    mask_left = y_left < 0.0
    y = np.concatenate([y_left[mask_left], y_right[::-1]])
    p = np.concatenate([p_left[mask_left], p_right[::-1]])
    gamma = np.concatenate([gamma_left[mask_left], gamma_right[::-1]])

    p_y = gamma * p
    c = 1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)

    return _normalize_extended_mode(y, u, v, p, rho, gamma, p_y), float(result.ci)


def load_pinn_full_mode_extended(model, alpha: float, mach: float, *, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    y = torch.linspace(-float(model.ymax), float(model.ymax), int(n_y), dtype=torch.float32, device=device).view(-1, 1)
    y.requires_grad_(True)
    alpha_t = torch.full_like(y, float(alpha))
    mach_t = torch.full_like(y, float(mach))

    with torch.enable_grad():
        fields = pressure_first_value_and_derivatives(model, y, alpha_t, mach_t)

    p = fields["p"]
    p_y = fields["p_y"]
    ci = float(fields["ci"][0, 0].detach().cpu().item())
    c = torch.complex(torch.zeros_like(fields["ci"]), fields["ci"])
    u_bar = torch.complex(base_velocity(y), torch.zeros_like(y))
    du_bar = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    i_alpha = torch.complex(torch.zeros_like(alpha_t), alpha_t)
    denom = i_alpha * (u_bar - c)
    denom = torch.where(torch.abs(denom) > 1e-8, denom, denom + torch.complex(torch.full_like(denom.real, 1e-8), torch.zeros_like(denom.real)))
    gamma = p_y / torch.where(
        torch.abs(p) > 1e-8,
        p,
        p + torch.complex(torch.full_like(p.real, 1e-8), torch.zeros_like(p.real)),
    )
    v = -p_y / denom
    u = -(du_bar * v + i_alpha * p) / denom
    rho = p * (float(mach) ** 2)

    return (
        _normalize_extended_mode(
            y.detach().cpu().numpy().reshape(-1),
            u.detach().cpu().numpy().reshape(-1),
            v.detach().cpu().numpy().reshape(-1),
            p.detach().cpu().numpy().reshape(-1),
            rho.detach().cpu().numpy().reshape(-1),
            gamma.detach().cpu().numpy().reshape(-1),
            p_y.detach().cpu().numpy().reshape(-1),
        ),
        ci,
    )


def compute_extended_mode_metrics(
    classic_mode: dict[str, np.ndarray],
    pinn_mode: dict[str, np.ndarray],
) -> dict[str, float]:
    y_common = np.asarray(classic_mode["y"], dtype=float)
    metrics: dict[str, float] = {}
    for field in ("p", "rho", "u", "v", "gamma", "p_y"):
        ref = np.asarray(classic_mode[field], dtype=np.complex128)
        pred = _interp_complex(np.asarray(pinn_mode["y"], dtype=float), np.asarray(pinn_mode[field], dtype=np.complex128), y_common)
        denom = max(np.linalg.norm(ref), 1e-12)
        metrics[f"{field}_rel"] = float(np.linalg.norm(pred - ref) / denom)

    amp_ref = np.abs(np.asarray(classic_mode["p"], dtype=np.complex128))
    amp_pred = np.abs(_interp_complex(np.asarray(pinn_mode["y"], dtype=float), np.asarray(pinn_mode["p"], dtype=np.complex128), y_common))
    amp_denom = max(np.linalg.norm(amp_ref), 1e-12)
    metrics["amp_rel"] = float(np.linalg.norm(amp_pred - amp_ref) / amp_denom)

    phase_ref = np.unwrap(np.angle(np.asarray(classic_mode["p"], dtype=np.complex128)))
    phase_pred = np.unwrap(np.angle(_interp_complex(np.asarray(pinn_mode["y"], dtype=float), np.asarray(pinn_mode["p"], dtype=np.complex128), y_common)))
    metrics["phase_rmse"] = float(np.sqrt(np.mean((phase_pred - phase_ref) ** 2)))
    return metrics


def _plot_mode_fields(
    classic_mode: dict[str, np.ndarray],
    pinn_mode: dict[str, np.ndarray],
    *,
    title: str,
    output_path: Path,
) -> None:
    fields = ("rho", "u", "v", "p")
    y_ref = np.asarray(classic_mode["y"], dtype=float)
    y_pred = np.asarray(pinn_mode["y"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for ax, field in zip(axes.flat, fields):
        ref = np.asarray(classic_mode[field], dtype=np.complex128)
        pred = np.asarray(pinn_mode[field], dtype=np.complex128)
        ax.plot(y_ref, np.real(ref), color="black", linewidth=1.3, label="Classique Re")
        ax.plot(y_ref, np.imag(ref), color="black", linewidth=1.0, linestyle=":", label="Classique Im")
        ax.plot(y_pred, np.real(pred), color="tab:orange", linewidth=1.3, linestyle="--", label="PINN Re")
        ax.plot(y_pred, np.imag(pred), color="tab:orange", linewidth=1.0, linestyle="-.", label="PINN Im")
        ax.set_title(field)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gamma_py_fields(
    classic_mode: dict[str, np.ndarray],
    pinn_mode: dict[str, np.ndarray],
    *,
    title: str,
    output_path: Path,
) -> None:
    fields = ("gamma", "p_y")
    y_ref = np.asarray(classic_mode["y"], dtype=float)
    y_pred = np.asarray(pinn_mode["y"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), constrained_layout=True)
    for row, field in enumerate(fields):
        ref = np.asarray(classic_mode[field], dtype=np.complex128)
        pred = np.asarray(pinn_mode[field], dtype=np.complex128)
        axes[row, 0].plot(y_ref, np.real(ref), color="black", linewidth=1.3, label="Classique")
        axes[row, 0].plot(y_pred, np.real(pred), color="tab:orange", linewidth=1.3, linestyle="--", label="PINN")
        axes[row, 0].set_title(f"Re {field}")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 1].plot(y_ref, np.imag(ref), color="black", linewidth=1.3, label="Classique")
        axes[row, 1].plot(y_pred, np.imag(pred), color="tab:orange", linewidth=1.3, linestyle="--", label="PINN")
        axes[row, 1].set_title(f"Im {field}")
        axes[row, 1].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_readme(output_dir: Path, *, stage1quater_dir: Path, diag_rows: list[dict[str, float]]) -> None:
    lines = [
        "# Stage 1quater pressure-first diagnostics",
        "",
        f"- source checkpoint directory: `{stage1quater_dir}`",
        "- training uses only pressure physics: pressure ODE, Robin BC, and center gauge",
        "- Stage 0 supplies the `c_i(alpha, M)` head and it is frozen by default",
        "- the classical solver is used here only for post-processing diagnostics",
        "- the classical solver is not used inside the training loss",
        "",
        "## Diagnostic points",
    ]
    for row in diag_rows:
        lines.append(
            f"- M={row['mach']:.2f}, alpha={row['alpha']:.2f}: "
            f"ci_abs_err={row['ci_abs_err']:.3e}, p_rel={row['p_rel']:.3e}, "
            f"u_rel={row['u_rel']:.3e}, v_rel={row['v_rel']:.3e}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")



def _relative_l2_masked(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != np.asarray(ref).shape[0] or not np.any(mask):
        return float("nan")
    pred_m = np.asarray(pred)[mask]
    ref_m = np.asarray(ref)[mask]
    denom = np.linalg.norm(ref_m)
    if denom <= 1.0e-14:
        return float("nan")
    return float(np.linalg.norm(pred_m - ref_m) / denom)


def _add_central_and_masked_metrics(
    metrics: dict[str, float],
    *,
    classic: dict[str, np.ndarray],
    pinn: dict[str, np.ndarray],
    central_ymax: float,
    amp_mask_frac: float,
) -> None:
    y = np.asarray(classic["y"], dtype=float)
    p_ref = np.asarray(classic["p"])
    mask_central = np.abs(y) <= float(central_ymax)
    p_amp = np.abs(p_ref)
    amp_threshold = float(amp_mask_frac) * max(float(np.max(p_amp)), 1.0e-14)
    mask_amp = p_amp >= amp_threshold
    mask_gamma = mask_central & mask_amp

    for key in ["p", "rho", "u", "v", "p_y"]:
        if key in classic and key in pinn:
            metrics[f"{key}_rel_central"] = _relative_l2_masked(
                np.asarray(pinn[key]),
                np.asarray(classic[key]),
                mask_central,
            )

    if "gamma" in classic and "gamma" in pinn:
        metrics["gamma_rel_masked"] = _relative_l2_masked(
            np.asarray(pinn["gamma"]),
            np.asarray(classic["gamma"]),
            mask_gamma,
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Stage 1quater pressure-first diagnostics.")
    parser.add_argument("--central-ymax", type=float, default=15.0)
    parser.add_argument("--amp-mask-frac", type=float, default=1.0e-3)
    parser.add_argument(
        "--stage1quater-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_2d_hybrid4ci_stage1quater_pressure"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/stage1quater_pressure_path1500"),
    )
    parser.add_argument("--num-alpha", type=int, default=61)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--modal-mach", type=float, default=0.5)
    parser.add_argument("--modal-alphas", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, _ = _load_stage1quater_model(Path(args.stage1quater_dir), device=device)
    history = pd.read_csv(Path(args.stage1quater_dir) / "history.csv")
    anchors_path = Path(args.stage1quater_dir) / "anchor_predictions_stage1quater.csv"
    anchors_df = pd.read_csv(anchors_path) if anchors_path.exists() else pd.DataFrame()

    mach_values = _parse_float_list(config.get("mach_values_json") or config.get("mach_values"))
    if not mach_values:
        mach_values = [0.1, 0.3, 0.5, 0.7]
    alpha_grid = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(args.num_alpha))
    reference_cache = config.get("reference_cache")
    if isinstance(reference_cache, float) and np.isnan(reference_cache):
        reference_cache = None
    reference_df = build_reference_surface(
        alpha_values=alpha_grid,
        mach_values=np.asarray(mach_values, dtype=float),
        reference_cache=reference_cache,
    )
    pred_df = _build_prediction_surface(model, alpha_grid, np.asarray(mach_values, dtype=float), device=device)
    grid_df = reference_df.merge(pred_df, on=["alpha", "Mach"], how="inner")
    grid_df["ci_abs_err"] = np.abs(grid_df["ci_pinn"] - grid_df["ci_reference"])
    grid_df.to_csv(output_dir / "stage1quater_ci_surface_table.csv", index=False)

    if not anchors_df.empty:
        plot_ci_curves_by_mach(
            grid_df,
            anchors_df,
            mach_values=list(mach_values),
            output_path=output_dir / "01_stage1quater_ci_vs_alpha_by_mach.png",
        )
    else:
        plot_ci_curves_by_mach(
            grid_df,
            grid_df.rename(columns={"ci_pinn": "ci_pred_stage1"}),
            mach_values=list(mach_values),
            output_path=output_dir / "01_stage1quater_ci_vs_alpha_by_mach.png",
        )
    plot_ci_error_heatmap(grid_df, output_dir / "02_stage1quater_ci_error_heatmap.png")
    plot_loss_history(history, output_dir / "03_stage1quater_loss_history.png")

    diag_rows: list[dict[str, float]] = []
    for idx, alpha in enumerate(list(args.modal_alphas), start=1):
        classic_mode, ci_classic = load_classic_full_mode_extended(float(alpha), float(args.modal_mach))
        pinn_mode, ci_pinn = load_pinn_full_mode_extended(
            model,
            float(alpha),
            float(args.modal_mach),
            n_y=801,
            device=device,
        )
        metrics = compute_extended_mode_metrics(classic_mode, pinn_mode)
        diag_rows.append(
            {
                "alpha": float(alpha),
                "mach": float(args.modal_mach),
                "ci_classic": float(ci_classic),
                "ci_pinn": float(ci_pinn),
                "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                **metrics,
            }
        )
        alpha_tag = int(round(1000.0 * float(alpha)))
        _plot_mode_fields(
            classic_mode,
            pinn_mode,
            title=f"Stage 1quater modes, M={args.modal_mach:.2f}, alpha={alpha:.2f}",
            output_path=output_dir / f"{2 * idx + 2:02d}_stage1quater_modes_M05_alpha{alpha_tag:03d}_2x2.png",
        )
        _plot_gamma_py_fields(
            classic_mode,
            pinn_mode,
            title=f"Stage 1quater gamma / p_y, M={args.modal_mach:.2f}, alpha={alpha:.2f}",
            output_path=output_dir / f"{2 * idx + 3:02d}_stage1quater_gamma_py_M05_alpha{alpha_tag:03d}.png",
        )

    diagnostics_df = pd.DataFrame(diag_rows)
    diagnostics_df.to_csv(output_dir / "diagnostics_summary.csv", index=False)
    _write_readme(output_dir, stage1quater_dir=Path(args.stage1quater_dir), diag_rows=diag_rows)
    print(f"Stage 1quater diagnostics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
