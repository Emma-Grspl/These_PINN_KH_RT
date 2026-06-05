from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.data.kh_subsonic_sampling import SubsonicReferenceCache2D
from src.models.kh_subsonic_pinn import KHSubsonicMultiMachPINN
from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    dy_dxi,
    reconstruct_pressure_p_y_from_riccati_2d,
    xi_to_y,
)


@dataclass(frozen=True)
class ModePoint:
    alpha: float
    mach: float


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
    *,
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


def normalize_full_mode(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return {
        "y": np.asarray(y, dtype=float),
        "u": np.asarray(u / scale, dtype=np.complex128),
        "v": np.asarray(v / scale, dtype=np.complex128),
        "p": np.asarray(p / scale, dtype=np.complex128),
        "rho": np.asarray(rho / scale, dtype=np.complex128),
    }


def compute_visible_xlim(
    y: np.ndarray,
    fields: list[np.ndarray],
    *,
    threshold_ratio: float = 0.02,
    min_half_width: float = 8.0,
) -> tuple[float, float]:
    envelope = np.zeros_like(y, dtype=float)
    for field in fields:
        envelope = np.maximum(envelope, np.abs(np.real(field)))
        envelope = np.maximum(envelope, np.abs(np.imag(field)))

    peak = float(np.max(envelope))
    if peak <= 0.0:
        return float(y[0]), float(y[-1])

    mask = envelope >= threshold_ratio * peak
    if not np.any(mask):
        return float(y[0]), float(y[-1])

    y_vis = y[mask]
    half_width = max(float(np.max(np.abs(y_vis))), min_half_width)
    return -half_width, half_width


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def build_reference_surface(config: pd.Series, *, num_alpha: int, num_mach: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = SubsonicReferenceCache2D.build(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        mach_min=float(config["mach_min"]),
        mach_max=float(config["mach_max"]),
        num_alpha=int(num_alpha),
        num_mach=int(num_mach),
    )
    return cache.audit_grid(num_alpha=int(num_alpha), num_mach=int(num_mach))


def build_prediction_surface(
    model: KHSubsonicMultiMachPINN,
    config: pd.Series,
    *,
    num_alpha: int,
    num_mach: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(num_alpha), dtype=float)
    mach_values = np.linspace(float(config["mach_min"]), float(config["mach_max"]), int(num_mach), dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    alpha_tensor = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_tensor = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor, mach_tensor).cpu().numpy().reshape(aa.shape)
    return aa, mm, ci_pred


def contour_levels(ci_ref: np.ndarray, ci_pred: np.ndarray, *, n_levels: int) -> np.ndarray:
    lo = max(1e-6, float(min(np.min(ci_ref), np.min(ci_pred))))
    hi = float(max(np.max(ci_ref), np.max(ci_pred)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.asarray([], dtype=float)
    return np.linspace(lo, hi, int(n_levels) + 2, dtype=float)[1:-1]


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["loss"], label="loss totale")
    for key in ("loss_pde", "loss_bc", "loss_norm", "loss_phase", "loss_ci_supervision"):
        if key in history.columns:
            axes[0].plot(history["epoch"], history[key], alpha=0.8, label=key)
    axes[0].set_yscale("log")
    axes[0].set_title("Historique des losses")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    audited = history.dropna(subset=["audit_ci_mae"]) if "audit_ci_mae" in history.columns else pd.DataFrame()
    if not audited.empty:
        axes[1].plot(audited["epoch"], audited["audit_ci_mae"], label="audit ci MAE")
        axes[1].plot(audited["epoch"], audited["audit_ci_max_abs"], label="audit ci max abs")
        axes[1].set_yscale("log")
        axes[1].legend()
    axes[1].set_title("Audit spectral 2D")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ci_surface_panels(
    aa: np.ndarray,
    mm: np.ndarray,
    ci_ref: np.ndarray,
    ci_pred: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    ci_abs_err = np.abs(ci_pred - ci_ref)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        (ci_ref, r"Référence classique $c_i(\alpha, M)$", "viridis"),
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


def plot_ci_error_heatmap(
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


def plot_ci_isolines_overlay(
    aa: np.ndarray,
    mm: np.ndarray,
    ci_ref: np.ndarray,
    ci_pred: np.ndarray,
    *,
    levels: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    cs_ref = ax.contour(aa, mm, ci_ref, levels=levels, colors="black", linewidths=1.5)
    cs_pred = ax.contour(aa, mm, ci_pred, levels=levels, colors="tab:orange", linestyles="--", linewidths=1.5)
    ax.clabel(cs_ref, inline=True, fontsize=8, fmt="%.03f")
    ax.clabel(cs_pred, inline=True, fontsize=8, fmt="%.03f")
    ax.plot([], [], color="black", linewidth=1.5, label="Classique")
    ax.plot([], [], color="tab:orange", linestyle="--", linewidth=1.5, label="PINN")
    ax.set_title(r"Isolignes $c_i$ : classique vs PINN")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ci_isolines_with_error(
    aa: np.ndarray,
    mm: np.ndarray,
    ci_ref: np.ndarray,
    ci_pred: np.ndarray,
    *,
    ci_abs_err: np.ndarray,
    levels: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    pcm = ax.pcolormesh(aa, mm, ci_abs_err, shading="auto", cmap="magma")
    ax.contour(aa, mm, ci_ref, levels=levels, colors="white", linewidths=1.1)
    ax.contour(aa, mm, ci_pred, levels=levels, colors="cyan", linestyles="--", linewidths=1.1)
    ax.set_title(r"Isolignes $c_i$ avec erreur absolue")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.grid(True, alpha=0.2)
    fig.colorbar(pcm, ax=ax, label=r"$|c_i^{PINN}-c_i^{ref}|$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_classic_full_mode(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
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
    c = -1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)

    return normalize_full_mode(y, u, v, p, rho), float(result.ci)


def load_pinn_full_mode(
    model: KHSubsonicMultiMachPINN,
    *,
    alpha: float,
    mach: float,
    n_y: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    mach_tensor = torch.full_like(xi, float(mach))
    xi.requires_grad_(True)

    if model.mode_representation == "riccati":
        pr, pi, p_y, _, y_t = reconstruct_pressure_p_y_from_riccati_2d(
            model,
            xi,
            alpha_tensor,
            mach_tensor,
            anchor_xi=0.0,
        )
    else:
        pred = model(xi, alpha_tensor, mach_tensor)
        pr = pred[:, 0:1]
        pi = pred[:, 1:2]
        y_t = xi_to_y(xi, model.get_mapping_scale().detach())
        p_r_xi = torch.autograd.grad(pr, xi, grad_outputs=torch.ones_like(pr), create_graph=False, retain_graph=True)[0]
        p_i_xi = torch.autograd.grad(pi, xi, grad_outputs=torch.ones_like(pi), create_graph=False, retain_graph=True)[0]
        p_xi = torch.complex(p_r_xi, p_i_xi)

        mapping_scale = model.get_mapping_scale().detach()
        y_xi = dy_dxi(xi, mapping_scale)
        p_y = p_xi / y_xi

    p = torch.complex(pr, pi)

    ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device), torch.tensor([[mach]], dtype=torch.float32, device=device)).item())
    c = -1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (float(mach) ** 2)

    fields = normalize_full_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    )
    return fields, ci


def compute_mode_metrics(
    classic: dict[str, np.ndarray],
    pinn: dict[str, np.ndarray],
    *,
    n_common: int,
    phase_threshold: float,
) -> dict[str, float]:
    y_min = max(float(np.min(classic["y"])), float(np.min(pinn["y"])))
    y_max = min(float(np.max(classic["y"])), float(np.max(pinn["y"])))
    y_common = np.linspace(y_min, y_max, int(n_common), dtype=float)

    p_c = interp_complex(classic["y"], classic["p"], y_common)
    p_p = interp_complex(pinn["y"], pinn["p"], y_common)
    rho_c = interp_complex(classic["y"], classic["rho"], y_common)
    rho_p = interp_complex(pinn["y"], pinn["rho"], y_common)
    u_c = interp_complex(classic["y"], classic["u"], y_common)
    u_p = interp_complex(pinn["y"], pinn["u"], y_common)
    v_c = interp_complex(classic["y"], classic["v"], y_common)
    v_p = interp_complex(pinn["y"], pinn["v"], y_common)

    def rel(field_a: np.ndarray, field_b: np.ndarray) -> float:
        return float(np.linalg.norm(field_b - field_a) / max(np.linalg.norm(field_a), 1e-12))

    amp_c = np.abs(p_c)
    amp_p = np.abs(p_p)
    amp_rel = float(np.linalg.norm(amp_p - amp_c) / max(np.linalg.norm(amp_c), 1e-12))

    phase_c = np.unwrap(np.angle(p_c))
    phase_p = np.unwrap(np.angle(p_p))
    phase_c -= phase_c[np.argmax(amp_c)]
    phase_p -= phase_p[np.argmax(amp_p)]
    mask = np.maximum(amp_c, amp_p) > float(phase_threshold)
    if np.any(mask):
        phase_diff = np.angle(np.exp(1j * (phase_p[mask] - phase_c[mask])))
        phase_rmse = float(np.sqrt(np.mean(phase_diff**2)))
    else:
        phase_rmse = float("nan")

    return {
        "p_rel": rel(p_c, p_p),
        "rho_rel": rel(rho_c, rho_p),
        "u_rel": rel(u_c, u_p),
        "v_rel": rel(v_c, v_p),
        "amp_rel": amp_rel,
        "phase_rmse": phase_rmse,
    }


def default_mode_points(config: pd.Series) -> list[ModePoint]:
    alpha_min = float(config["alpha_min"])
    alpha_max = float(config["alpha_max"])
    mach_min = float(config["mach_min"])
    mach_max = float(config["mach_max"])
    da = alpha_max - alpha_min
    dm = mach_max - mach_min
    return [
        ModePoint(alpha=alpha_min + 0.05 * da, mach=mach_min + 0.20 * dm),
        ModePoint(alpha=alpha_min + 0.20 * da, mach=mach_min + 0.40 * dm),
        ModePoint(alpha=alpha_min + 0.55 * da, mach=mach_min + 0.60 * dm),
        ModePoint(alpha=alpha_min + 0.90 * da, mach=mach_min + 0.90 * dm),
    ]


def parse_mode_points(config: pd.Series, raw_points: list[str] | None) -> list[ModePoint]:
    if not raw_points:
        return default_mode_points(config)
    points: list[ModePoint] = []
    for item in raw_points:
        alpha_s, mach_s = item.split(":")
        points.append(ModePoint(alpha=float(alpha_s), mach=float(mach_s)))
    return points


def plot_selected_modes_pdf(
    model: KHSubsonicMultiMachPINN,
    *,
    points: list[ModePoint],
    n_y: int,
    device: torch.device,
    phase_threshold: float,
    n_common: int,
    output_pdf: Path,
    output_csv: Path,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    titles = [
        ("rho", r"Density Perturbation $\hat{\rho}$"),
        ("u", r"Streamwise Velocity $\hat{u}$"),
        ("v", r"Vertical Velocity $\hat{v}$"),
        ("p", r"Pressure Perturbation $\hat{p}$"),
    ]

    with PdfPages(output_pdf) as pdf:
        for point in points:
            classic, ci_classic = load_classic_full_mode(point.alpha, point.mach)
            pinn, ci_pinn = load_pinn_full_mode(
                model,
                alpha=point.alpha,
                mach=point.mach,
                n_y=n_y,
                device=device,
            )
            metrics = compute_mode_metrics(
                classic,
                pinn,
                n_common=n_common,
                phase_threshold=phase_threshold,
            )
            rows.append(
                {
                    "alpha": float(point.alpha),
                    "mach": float(point.mach),
                    "ci_classic": float(ci_classic),
                    "ci_pinn": float(ci_pinn),
                    "ci_abs_err": abs(float(ci_pinn) - float(ci_classic)),
                    **metrics,
                }
            )

            x_limits_classic = compute_visible_xlim(classic["y"], [classic["rho"], classic["u"], classic["v"], classic["p"]])
            x_limits_pinn = compute_visible_xlim(pinn["y"], [pinn["rho"], pinn["u"], pinn["v"], pinn["p"]])
            x_limits = (
                min(x_limits_classic[0], x_limits_pinn[0]),
                max(x_limits_classic[1], x_limits_pinn[1]),
            )

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=False)
            for ax, (field_name, title) in zip(axes.flat, titles):
                field_c = classic[field_name]
                field_p = pinn[field_name]
                ax.plot(classic["y"], np.real(field_c), color="tab:blue", linewidth=1.6, label="Classic Re")
                ax.plot(classic["y"], np.imag(field_c), color="tab:orange", linewidth=1.6, label="Classic Im")
                ax.plot(pinn["y"], np.real(field_p), "--", color="tab:blue", linewidth=1.6, label="PINN Re")
                ax.plot(pinn["y"], np.imag(field_p), "--", color="tab:orange", linewidth=1.6, label="PINN Im")
                ax.set_title(title)
                ax.set_xlim(*x_limits)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7)

            fig.suptitle(
                fr"$\alpha={point.alpha:.3f}$, $M={point.mach:.3f}$"
                "\n"
                fr"$c_i^{{classic}}={ci_classic:.5f}$ | $c_i^{{PINN}}={ci_pinn:.5f}$ | "
                fr"$p_{{rel}}={metrics['p_rel']:.3e}$ | phase RMSE={metrics['phase_rmse']:.3e}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def build_mode_error_heatmaps(
    model: KHSubsonicMultiMachPINN,
    config: pd.Series,
    *,
    num_alpha: int,
    num_mach: int,
    n_y: int,
    n_common: int,
    phase_threshold: float,
    device: torch.device,
    output_png: Path,
    output_csv: Path,
) -> pd.DataFrame:
    alpha_values = np.linspace(float(config["alpha_min"]), float(config["alpha_max"]), int(num_alpha), dtype=float)
    mach_values = np.linspace(float(config["mach_min"]), float(config["mach_max"]), int(num_mach), dtype=float)
    aa, mm = np.meshgrid(alpha_values, mach_values)
    p_rel_field = np.full_like(aa, np.nan, dtype=float)
    amp_rel_field = np.full_like(aa, np.nan, dtype=float)
    phase_field = np.full_like(aa, np.nan, dtype=float)
    rows: list[dict[str, float]] = []

    total = aa.size
    counter = 0
    for j in range(mm.shape[0]):
        for i in range(aa.shape[1]):
            alpha = float(aa[j, i])
            mach = float(mm[j, i])
            classic, ci_classic = load_classic_full_mode(alpha, mach)
            pinn, ci_pinn = load_pinn_full_mode(model, alpha=alpha, mach=mach, n_y=n_y, device=device)
            metrics = compute_mode_metrics(classic, pinn, n_common=n_common, phase_threshold=phase_threshold)
            p_rel_field[j, i] = metrics["p_rel"]
            amp_rel_field[j, i] = metrics["amp_rel"]
            phase_field[j, i] = metrics["phase_rmse"]
            rows.append(
                {
                    "alpha": alpha,
                    "mach": mach,
                    "ci_classic": ci_classic,
                    "ci_pinn": ci_pinn,
                    "ci_abs_err": abs(ci_pinn - ci_classic),
                    **metrics,
                }
            )
            counter += 1
            print(
                f"[mode {counter}/{total}] alpha={alpha:.4f} mach={mach:.4f} "
                f"p_rel={metrics['p_rel']:.3e} amp_rel={metrics['amp_rel']:.3e} phase={metrics['phase_rmse']:.3e}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    panels = [
        (p_rel_field, r"$p_{rel}$", "magma"),
        (amp_rel_field, r"Erreur relative d'amplitude", "magma"),
        (phase_field, "RMSE phase", "viridis"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        pcm = ax.pcolormesh(aa, mm, field, shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(pcm, ax=ax)

    fig.suptitle("Subsonique PINN 2D : erreurs de mode vs classique")
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return df


def summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    if "audit_ci_mae" not in history.columns:
        return pd.DataFrame([{"best_epoch": -1, "best_audit_ci_mae": np.nan, "last_epoch": -1, "last_audit_ci_mae": np.nan}])
    audited = history.dropna(subset=["audit_ci_mae"]).copy()
    if audited.empty:
        return pd.DataFrame([{"best_epoch": -1, "best_audit_ci_mae": np.nan, "last_epoch": int(history["epoch"].iloc[-1]) if not history.empty else -1, "last_audit_ci_mae": np.nan}])
    best_row = audited.loc[int(pd.to_numeric(audited["audit_ci_mae"], errors="coerce").idxmin())]
    last_row = audited.iloc[-1]
    return pd.DataFrame(
        [
            {
                "best_epoch": int(best_row["epoch"]),
                "best_audit_ci_mae": float(best_row["audit_ci_mae"]),
                "best_audit_ci_max_abs": float(best_row["audit_ci_max_abs"]),
                "best_audit_ci_mean_rel": float(best_row["audit_ci_mean_rel"]),
                "last_epoch": int(last_row["epoch"]),
                "last_audit_ci_mae": float(last_row["audit_ci_mae"]),
                "last_audit_ci_max_abs": float(last_row["audit_ci_max_abs"]),
                "last_audit_ci_mean_rel": float(last_row["audit_ci_mean_rel"]),
            }
        ]
    )


def evaluate_run(
    run_dir: Path,
    *,
    device: str = "cpu",
    checkpoint: Path | None = None,
    ci_num_alpha: int = 41,
    ci_num_mach: int = 11,
    ci_n_levels: int = 8,
    mode_num_alpha: int = 7,
    mode_num_mach: int = 5,
    mode_points: list[str] | None = None,
    mode_n_y: int = 1001,
    mode_n_common: int = 1200,
    phase_threshold: float = 1e-2,
) -> dict[str, Path]:
    dev = torch.device(device)
    model, config, history = load_model(run_dir, device=dev, checkpoint_path=checkpoint)

    history_png = run_dir / "history_diagnostics_2d.png"
    ci_surface_png = run_dir / "subsonic_pinn_alphamach_ci_surface.png"
    ci_surface_csv = run_dir / "subsonic_pinn_alphamach_ci_surface.csv"
    ci_top_errors_csv = run_dir / "subsonic_pinn_alphamach_ci_top_errors.csv"
    ci_error_png = run_dir / "subsonic_pinn_alphamach_ci_error_heatmap.png"
    ci_overlay_png = run_dir / "subsonic_pinn_alphamach_ci_isolines_overlay.png"
    ci_overlay_error_png = run_dir / "subsonic_pinn_alphamach_ci_isolines_with_error.png"
    mode_heatmap_png = run_dir / "subsonic_pinn_alphamach_mode_error_heatmaps.png"
    mode_heatmap_csv = run_dir / "subsonic_pinn_alphamach_mode_error_heatmaps.csv"
    mode_points_pdf = run_dir / "subsonic_pinn_alphamach_modes.pdf"
    mode_points_csv = run_dir / "subsonic_pinn_alphamach_mode_points.csv"
    training_summary_csv = run_dir / "subsonic_pinn_alphamach_training_summary.csv"

    plot_history(history, history_png)
    aa_ref, mm_ref, ci_ref = build_reference_surface(config, num_alpha=ci_num_alpha, num_mach=ci_num_mach)
    aa_pred, mm_pred, ci_pred = build_prediction_surface(model, config, num_alpha=ci_num_alpha, num_mach=ci_num_mach, device=dev)
    if aa_ref.shape != aa_pred.shape or mm_ref.shape != mm_pred.shape:
        raise RuntimeError("Reference and prediction grids do not match.")

    ci_df = plot_ci_surface_panels(aa_ref, mm_ref, ci_ref, ci_pred, ci_surface_png)
    ci_df.to_csv(ci_surface_csv, index=False)
    ci_top = ci_df.sort_values("ci_abs_err", ascending=False).head(25)
    ci_top.to_csv(ci_top_errors_csv, index=False)

    ci_abs_err = np.abs(ci_pred - ci_ref)
    plot_ci_error_heatmap(aa_ref, mm_ref, ci_abs_err, ci_error_png)
    levels = contour_levels(ci_ref, ci_pred, n_levels=ci_n_levels)
    if levels.size > 0:
        plot_ci_isolines_overlay(aa_ref, mm_ref, ci_ref, ci_pred, levels=levels, output_path=ci_overlay_png)
        plot_ci_isolines_with_error(
            aa_ref,
            mm_ref,
            ci_ref,
            ci_pred,
            ci_abs_err=ci_abs_err,
            levels=levels,
            output_path=ci_overlay_error_png,
        )

    build_mode_error_heatmaps(
        model,
        config,
        num_alpha=mode_num_alpha,
        num_mach=mode_num_mach,
        n_y=mode_n_y,
        n_common=mode_n_common,
        phase_threshold=phase_threshold,
        device=dev,
        output_png=mode_heatmap_png,
        output_csv=mode_heatmap_csv,
    )
    parsed_points = parse_mode_points(config, mode_points)
    plot_selected_modes_pdf(
        model,
        points=parsed_points,
        n_y=mode_n_y,
        device=dev,
        phase_threshold=phase_threshold,
        n_common=mode_n_common,
        output_pdf=mode_points_pdf,
        output_csv=mode_points_csv,
    )
    summarize_history(history).to_csv(training_summary_csv, index=False)

    return {
        "history_png": history_png,
        "ci_surface_png": ci_surface_png,
        "ci_surface_csv": ci_surface_csv,
        "ci_top_errors_csv": ci_top_errors_csv,
        "ci_error_png": ci_error_png,
        "ci_overlay_png": ci_overlay_png,
        "ci_overlay_error_png": ci_overlay_error_png,
        "mode_heatmap_png": mode_heatmap_png,
        "mode_heatmap_csv": mode_heatmap_csv,
        "mode_points_pdf": mode_points_pdf,
        "mode_points_csv": mode_points_csv,
        "training_summary_csv": training_summary_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation 2D subsonique PINN: surface ci, isolignes, heatmaps et modes.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ci-num-alpha", type=int, default=41)
    parser.add_argument("--ci-num-mach", type=int, default=11)
    parser.add_argument("--ci-n-levels", type=int, default=8)
    parser.add_argument("--mode-num-alpha", type=int, default=7)
    parser.add_argument("--mode-num-mach", type=int, default=5)
    parser.add_argument("--mode-points", type=str, nargs="*", default=None)
    parser.add_argument("--mode-n-y", type=int, default=1001)
    parser.add_argument("--mode-n-common", type=int, default=1200)
    parser.add_argument("--phase-threshold", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = evaluate_run(
        args.run_dir,
        device=args.device,
        checkpoint=args.checkpoint,
        ci_num_alpha=args.ci_num_alpha,
        ci_num_mach=args.ci_num_mach,
        ci_n_levels=args.ci_n_levels,
        mode_num_alpha=args.mode_num_alpha,
        mode_num_mach=args.mode_num_mach,
        mode_points=args.mode_points,
        mode_n_y=args.mode_n_y,
        mode_n_common=args.mode_n_common,
        phase_threshold=args.phase_threshold,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
