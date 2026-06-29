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

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative
from src.physics.kh_supersonic_pressure_first_1d import (
    build_supersonic_pressure_first_model_from_config,
    supersonic_pressure_value_and_derivatives,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render diagnostics for the supersonic single-case pressure-first PINN.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-y", type=int, default=1201)
    parser.add_argument("--classical-ln-p-start-right", type=float, default=-5.0)
    parser.add_argument("--classical-match-y", type=float, default=1.0)
    parser.add_argument("--classical-use-mapping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classical-mapping-scale", type=float, default=5.0)
    parser.add_argument("--classical-min-y-limit", type=float, default=10.0)
    parser.add_argument("--classical-max-y-limit", type=float, default=2000.0)
    parser.add_argument("--classical-y-limit-factor", type=float, default=18.0)
    return parser


def _load_model(checkpoint_dir: Path, *, device: torch.device):
    checkpoint = torch.load(checkpoint_dir / "model_best.pt", map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_dir / 'model_best.pt'}.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError(f"Missing model_config in {checkpoint_dir / 'model_best.pt'}.")
    model = build_supersonic_pressure_first_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    history = pd.read_csv(checkpoint_dir / "history.csv")
    config_df = pd.read_csv(checkpoint_dir / "config.csv")
    config = config_df.iloc[0].to_dict()
    return model, config, history


def _safe_complex_div(num: np.ndarray, den: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    mask = np.abs(den) > eps
    out = np.empty_like(num, dtype=np.complex128)
    out[mask] = num[mask] / den[mask]
    out[~mask] = num[~mask] / (den[~mask] + eps)
    return out


def _normalize_mode_by_center(
    y: np.ndarray,
    *,
    p: np.ndarray,
    p_y: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, np.ndarray]:
    idx0 = int(np.argmin(np.abs(y)))
    p0 = complex(p[idx0])
    if abs(p0) <= 1.0e-12:
        phase = 1.0 + 0.0j
        scale = 1.0
    else:
        phase = np.exp(-1j * np.angle(p0))
        p0_phase = p0 * phase
        if np.real(p0_phase) < 0.0:
            phase = -phase
            p0_phase = -p0_phase
        scale = max(float(np.real(p0_phase)), 1.0e-12)

    p_n = p * phase / scale
    p_y_n = p_y * phase / scale
    rho_n = rho * phase / scale
    u_n = u * phase / scale
    v_n = v * phase / scale
    gamma_n = _safe_complex_div(p_y_n, p_n)
    return {
        "y": np.asarray(y, dtype=float),
        "p": np.asarray(p_n, dtype=np.complex128),
        "p_y": np.asarray(p_y_n, dtype=np.complex128),
        "rho": np.asarray(rho_n, dtype=np.complex128),
        "u": np.asarray(u_n, dtype=np.complex128),
        "v": np.asarray(v_n, dtype=np.complex128),
        "gamma": np.asarray(gamma_n, dtype=np.complex128),
    }


def load_pinn_fields(model, *, n_y: int, device: torch.device) -> dict[str, np.ndarray]:
    y = torch.linspace(-float(model.ymax), float(model.ymax), int(n_y), dtype=torch.float32, device=device).view(-1, 1)
    y.requires_grad_(True)
    with torch.enable_grad():
        fields = supersonic_pressure_value_and_derivatives(model, y)

    p = fields["p"]
    p_y = fields["p_y"]
    alpha_t = fields["alpha"]
    mach_t = fields["mach"]
    c = torch.complex(fields["cr"], fields["ci"])
    u_bar = torch.complex(base_velocity(y), torch.zeros_like(y))
    du_bar = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    i_alpha = torch.complex(torch.zeros_like(alpha_t), alpha_t)
    denom = i_alpha * (u_bar - c)
    denom = torch.where(
        torch.abs(denom) > 1.0e-8,
        denom,
        denom + torch.complex(torch.full_like(denom.real, 1.0e-8), torch.zeros_like(denom.real)),
    )
    gamma = p_y / torch.where(
        torch.abs(p) > 1.0e-8,
        p,
        p + torch.complex(torch.full_like(p.real, 1.0e-8), torch.zeros_like(p.real)),
    )
    v = -p_y / denom
    u = -(du_bar * v + i_alpha * p) / denom
    rho = mach_t.pow(2) * p

    return _normalize_mode_by_center(
        y.detach().cpu().numpy().reshape(-1),
        p=p.detach().cpu().numpy().reshape(-1),
        p_y=p_y.detach().cpu().numpy().reshape(-1),
        rho=rho.detach().cpu().numpy().reshape(-1),
        u=u.detach().cpu().numpy().reshape(-1),
        v=v.detach().cpu().numpy().reshape(-1),
    )


def load_classical_fields(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    match_y: float,
    use_mapping: bool,
    mapping_scale: float,
    min_y_limit: float,
    max_y_limit: float,
    y_limit_factor: float,
) -> dict[str, np.ndarray]:
    solver = Mstab17SupersonicSolver(
        alpha=float(alpha),
        Mach=float(mach),
        match_y=float(match_y),
        use_mapping=bool(use_mapping),
        mapping_scale=float(mapping_scale),
        min_y_limit=float(min_y_limit),
        max_y_limit=float(max_y_limit),
        y_limit_factor=float(y_limit_factor),
    )
    sol_left, _, sol_right_full, _ = solver.get_trajectories(float(cr), float(ci), ln_p_start_right=float(ln_p_start_right))
    if not (sol_left.success and sol_right_full.success):
        raise RuntimeError("Classical supersonic trajectory reconstruction failed.")

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right_full.t)
    k_left, q_left, ln_p_left, phi_left = sol_left.y
    k_right, q_right, ln_p_right, phi_right = sol_right_full.y

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right_full, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    left_mask = y_left < 0.0
    y = np.concatenate([y_left[left_mask], y_right[::-1]])
    p = np.concatenate([p_left[left_mask], p_right[::-1]])
    gamma = np.concatenate([gamma_left[left_mask], gamma_right[::-1]])

    u_bar = solver.base_velocity(y)
    du_bar = solver.base_velocity_derivative(y)
    c = complex(float(cr), float(ci))
    i_alpha = 1j * float(alpha)
    p_y = gamma * p
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = (float(mach) ** 2) * p

    return _normalize_mode_by_center(y, p=p, p_y=p_y, rho=rho, u=u, v=v)


def _interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def compute_relative_metrics(
    reference: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
) -> dict[str, float]:
    y_ref = np.asarray(reference["y"], dtype=float)
    metrics: dict[str, float] = {}
    for field in ("p", "rho", "u", "v", "gamma", "p_y"):
        ref = np.asarray(reference[field], dtype=np.complex128)
        pred = _interp_complex(np.asarray(prediction["y"], dtype=float), np.asarray(prediction[field], dtype=np.complex128), y_ref)
        denom = max(np.linalg.norm(ref), 1.0e-12)
        metrics[f"{field}_rel"] = float(np.linalg.norm(pred - ref) / denom)
    return metrics


def plot_loss_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for key in ("loss_total", "loss_pde_pressure", "loss_bc_robin", "loss_gauge", "loss_center_pde"):
        if key in history.columns:
            ax.plot(history["epoch"], history[key], label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title("Supersonic single-case loss history")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pressure(
    pinn: dict[str, np.ndarray],
    classical: dict[str, np.ndarray] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), constrained_layout=True, sharex=True)
    y_p = pinn["y"]
    axes[0].plot(y_p, np.real(pinn["p"]), color="tab:orange", linewidth=1.4, label="PINN")
    axes[1].plot(y_p, np.imag(pinn["p"]), color="tab:orange", linewidth=1.4, label="PINN")
    if classical is not None:
        axes[0].plot(classical["y"], np.real(classical["p"]), color="black", linewidth=1.2, label="Classique")
        axes[1].plot(classical["y"], np.imag(classical["p"]), color="black", linewidth=1.2, label="Classique")
    axes[0].set_title("Re p")
    axes[1].set_title("Im p")
    axes[0].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Amplitude")
    axes[1].set_xlabel("y")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_py(
    pinn: dict[str, np.ndarray],
    classical: dict[str, np.ndarray] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True, sharex=True)
    fields = ("gamma", "p_y")
    for row, field in enumerate(fields):
        axes[row, 0].plot(pinn["y"], np.real(pinn[field]), color="tab:orange", linewidth=1.4, label="PINN")
        axes[row, 1].plot(pinn["y"], np.imag(pinn[field]), color="tab:orange", linewidth=1.4, label="PINN")
        if classical is not None:
            axes[row, 0].plot(classical["y"], np.real(classical[field]), color="black", linewidth=1.2, label="Classique")
            axes[row, 1].plot(classical["y"], np.imag(classical[field]), color="black", linewidth=1.2, label="Classique")
        axes[row, 0].set_title(f"Re {field}")
        axes[row, 1].set_title(f"Im {field}")
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("y")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_modes(
    pinn: dict[str, np.ndarray],
    classical: dict[str, np.ndarray] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    fields = ("rho", "u", "v", "p")
    for ax, field in zip(axes.flat, fields):
        ax.plot(pinn["y"], np.real(pinn[field]), color="tab:orange", linewidth=1.4, label="PINN Re")
        ax.plot(pinn["y"], np.imag(pinn[field]), color="tab:orange", linewidth=1.0, linestyle="--", label="PINN Im")
        if classical is not None:
            ax.plot(classical["y"], np.real(classical[field]), color="black", linewidth=1.2, label="Classique Re")
            ax.plot(classical["y"], np.imag(classical[field]), color="black", linewidth=1.0, linestyle=":", label="Classique Im")
        ax.set_title(field)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    config: dict[str, object],
    classical_error: str | None,
    summary: dict[str, float | str],
) -> None:
    lines = [
        "# Supersonic Single-Case Pressure-First Diagnostics",
        "",
        f"- checkpoint_dir: `{checkpoint_dir}`",
        f"- alpha={float(config['alpha'])}",
        f"- Mach={float(config['mach'])}",
        f"- cr={float(config['cr'])}",
        f"- ci={float(config['ci'])}",
        "",
        "This is a post-training diagnostic. The classical solver is used only for comparison after training. No classical modal field is used in the training loss.",
        "",
    ]
    if classical_error is None:
        lines.extend(
            [
                "## Classical comparison",
                "",
                f"- p_rel={summary.get('p_rel', float('nan')):.6e}",
                f"- rho_rel={summary.get('rho_rel', float('nan')):.6e}",
                f"- u_rel={summary.get('u_rel', float('nan')):.6e}",
                f"- v_rel={summary.get('v_rel', float('nan')):.6e}",
                f"- gamma_rel={summary.get('gamma_rel', float('nan')):.6e}",
                f"- p_y_rel={summary.get('p_y_rel', float('nan')):.6e}",
            ]
        )
    else:
        lines.extend(["## Classical comparison failed", "", f"- error: {classical_error}"])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if str(args.device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, history = _load_model(Path(args.checkpoint_dir), device=device)
    plot_loss_history(history, output_dir / "01_supersonic_singlecase_loss_history.png")

    pinn_fields = load_pinn_fields(model, n_y=int(args.n_y), device=device)
    classical_fields: dict[str, np.ndarray] | None = None
    classical_error: str | None = None

    try:
        classical_fields = load_classical_fields(
            alpha=float(config["alpha"]),
            mach=float(config["mach"]),
            cr=float(config["cr"]),
            ci=float(config["ci"]),
            ln_p_start_right=float(args.classical_ln_p_start_right),
            match_y=float(args.classical_match_y),
            use_mapping=bool(args.classical_use_mapping),
            mapping_scale=float(args.classical_mapping_scale),
            min_y_limit=float(args.classical_min_y_limit),
            max_y_limit=float(args.classical_max_y_limit),
            y_limit_factor=float(args.classical_y_limit_factor),
        )
    except Exception as exc:  # noqa: BLE001
        classical_error = str(exc)

    plot_pressure(pinn_fields, classical_fields, output_dir / "02_supersonic_singlecase_pressure.png")
    plot_gamma_py(pinn_fields, classical_fields, output_dir / "03_supersonic_singlecase_gamma_py.png")
    plot_modes(pinn_fields, classical_fields, output_dir / "04_supersonic_singlecase_modes_rho_u_v_p.png")

    summary: dict[str, float | str] = {
        "alpha": float(config["alpha"]),
        "Mach": float(config["mach"]),
        "cr": float(config["cr"]),
        "ci": float(config["ci"]),
        "classical_status": "ok" if classical_error is None else "failed",
    }
    if classical_fields is not None:
        summary.update(compute_relative_metrics(classical_fields, pinn_fields))
    else:
        for key in ("p_rel", "rho_rel", "u_rel", "v_rel", "gamma_rel", "p_y_rel"):
            summary[key] = np.nan
        summary["classical_error"] = classical_error or ""

    pd.DataFrame([summary]).to_csv(output_dir / "diagnostics_summary.csv", index=False)
    write_readme(
        output_dir,
        checkpoint_dir=Path(args.checkpoint_dir),
        config=config,
        classical_error=classical_error,
        summary=summary,
    )
    print(f"Supersonic single-case diagnostics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
