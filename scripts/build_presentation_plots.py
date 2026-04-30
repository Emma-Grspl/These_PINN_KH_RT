from __future__ import annotations

import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN, build_fixed_mach_model_from_config
from src.physics.kh_subsonic_residual import reconstruct_pressure_from_riccati, xi_to_y


PRESENTATION_DIR = ROOT_DIR / "plot_presentation"
SUBSONIC_CLASSIC_DIR = PRESENTATION_DIR / "subsonic_classique"
SUBSONIC_PINN_DIR = PRESENTATION_DIR / "subsonic_pinn"
SUPERSONIC_CLASSIC_DIR = PRESENTATION_DIR / "supersonic_classique"


def ensure_dirs() -> None:
    for path in (SUBSONIC_CLASSIC_DIR, SUBSONIC_PINN_DIR, SUPERSONIC_CLASSIC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_pinn_model(config: pd.Series) -> KHSubsonicFixedMachPINN:
    return build_fixed_mach_model_from_config(config)


def normalize_pressure(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
    if np.max(np.real(p)) < abs(np.min(np.real(p))):
        p = -p
    scale = max(np.max(np.abs(p)), 1e-12)
    return y, p / scale


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def load_classic_pressure_mode(alpha: float, mach: float) -> tuple[np.ndarray, np.ndarray, float]:
    solver = Mstab17SubsonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    ln_p_left, phi_left = sol_left.y[2], sol_left.y[3]
    ln_p_right, phi_right = sol_right.y[2], sol_right.y[3]

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))

    y = np.concatenate([y_left[y_left < 0.0], y_right[::-1]])
    p = np.concatenate([p_left[y_left < 0.0], p_right[::-1]])
    y, p = normalize_pressure(y, p)
    return y, p, float(result.ci)


def load_pinn_pressure_mode(run_dir: Path, checkpoint: Path, alpha: float, n_y: int = 1001, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, float]:
    dev = torch.device(device)
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    model = build_pinn_model(config)
    state_dict = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    xi = torch.linspace(-0.98, 0.98, n_y, device=dev).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    with torch.no_grad():
        if model.mode_representation == "riccati":
            pr, pi, y_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
            y = y_t.cpu().numpy().reshape(-1)
            p = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor)
            y = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
        ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=dev)).item())
    y, p = normalize_pressure(y, p)
    return y, p, ci


def build_common_grid(curves: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    y_min = max(float(np.min(y)) for y, _ in curves)
    y_max = min(float(np.max(y)) for y, _ in curves)
    return np.linspace(y_min, y_max, 1200, dtype=float)


def plot_classic_three_modes(output: Path, *, mach: float, alphas: list[float]) -> None:
    fig, axes = plt.subplots(len(alphas), 2, figsize=(11.5, 10.0), sharex=False)
    if len(alphas) == 1:
        axes = np.asarray([axes])
    for row_idx, alpha in enumerate(alphas):
        y, p, ci = load_classic_pressure_mode(alpha, mach)
        amp = np.abs(p)
        phase = np.unwrap(np.angle(p))
        phase -= phase[np.argmax(amp)]

        ax_reim = axes[row_idx, 0]
        ax_amp = axes[row_idx, 1]

        ax_reim.plot(y, np.real(p), label="Re")
        ax_reim.plot(y, np.imag(p), "--", label="Im")
        ax_reim.set_title(fr"$\alpha={alpha:.2f}$, $M={mach:.2f}$, $c_i={ci:.4f}$")
        ax_reim.set_ylabel(r"$\hat{p}$")
        ax_reim.grid(True, alpha=0.3)
        ax_reim.legend()

        ax_amp.plot(y, amp, color="tab:blue", label=r"$|\hat{p}|$")
        ax_phase = ax_amp.twinx()
        ax_phase.plot(y, phase, color="tab:orange", linestyle="--", label=r"$\arg(\hat{p})$")
        ax_amp.set_title("Amplitude / phase")
        ax_amp.set_ylabel("Amplitude")
        ax_phase.set_ylabel("Phase")
        ax_amp.grid(True, alpha=0.3)

        lines1, labels1 = ax_amp.get_legend_handles_labels()
        lines2, labels2 = ax_phase.get_legend_handles_labels()
        ax_amp.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle("Subsonique classique : trois modes de pression")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pinn_vs_classic_three_modes(output: Path, *, run_dir: Path, checkpoint: Path, mach: float, alphas: list[float]) -> None:
    fig, axes = plt.subplots(len(alphas), 2, figsize=(12.5, 10.0), sharex=False)
    if len(alphas) == 1:
        axes = np.asarray([axes])

    for row_idx, alpha in enumerate(alphas):
        y_c, p_c, ci_c = load_classic_pressure_mode(alpha, mach)
        y_p, p_p, ci_p = load_pinn_pressure_mode(run_dir, checkpoint, alpha)
        y_common = build_common_grid([(y_c, p_c), (y_p, p_p)])

        p_c_i = interp_complex(y_c, p_c, y_common)
        p_p_i = interp_complex(y_p, p_p, y_common)

        amp_c = np.abs(p_c_i)
        amp_p = np.abs(p_p_i)
        phase_c = np.unwrap(np.angle(p_c_i))
        phase_p = np.unwrap(np.angle(p_p_i))
        phase_c -= phase_c[np.argmax(amp_c)]
        phase_p -= phase_p[np.argmax(amp_p)]

        ax_amp = axes[row_idx, 0]
        ax_phase = axes[row_idx, 1]

        ax_amp.plot(y_common, amp_c, color="black", linewidth=2.0, label="Classique")
        ax_amp.plot(y_common, amp_p, "--", color="tab:blue", linewidth=2.0, label="PINN")
        ax_amp.set_title(fr"Amplitude | alpha={alpha:.2f}, M={mach:.2f}")
        ax_amp.set_ylabel("Amplitude")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.legend()

        ax_phase.plot(y_common, phase_c, color="black", linewidth=2.0, label="Classique")
        ax_phase.plot(y_common, phase_p, "--", color="tab:orange", linewidth=2.0, label="PINN")
        ax_phase.set_title(fr"Phase | $c_i^{{classic}}={ci_c:.4f}$, $c_i^{{PINN}}={ci_p:.4f}$")
        ax_phase.set_ylabel("Phase")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend()

    axes[-1, 0].set_xlabel("y")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle("Subsonique PINN : modes de pression superposes au classique")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    # Existing figures copied into the presentation tree.
    copy_if_exists(
        ROOT_DIR / "subsonique_classique" / "assets_blumen" / "subsonic_shooting_vs_blumen.png",
        SUBSONIC_CLASSIC_DIR / "01_subsonic_isolines_blumen_vs_classique.png",
    )
    copy_if_exists(
        ROOT_DIR / "subsonique_classique" / "assets_blumen" / "subsonic_shooting_error_map.png",
        SUBSONIC_CLASSIC_DIR / "02_subsonic_error_map_classique.png",
    )
    copy_if_exists(
        ROOT_DIR / "kh_subsonic_fixed_mach_M05_riccati_multibranch" / "ci_error_heatmap_epoch4000.png",
        SUBSONIC_PINN_DIR / "01_ci_error_heatmap_multibranch_epoch4000.png",
    )

    copy_if_exists(
        ROOT_DIR / "supersonique_classique" / "assets_blumen" / "scan_gep_validity_frontier_pin3_xi099_m110_200_frontier.png",
        SUPERSONIC_CLASSIC_DIR / "01_supersonic_validity_frontier.png",
    )
    copy_if_exists(
        ROOT_DIR / "supersonique_classique" / "assets_blumen" / "supersonic_gep_local_4x4_vs_blumen.png",
        SUPERSONIC_CLASSIC_DIR / "02_supersonic_isolines_vs_blumen.png",
    )
    copy_if_exists(
        ROOT_DIR / "supersonique_classique" / "assets_blumen" / "supersonic_gep_local_4x4_cont_vs_blumen.png",
        SUPERSONIC_CLASSIC_DIR / "03_supersonic_error_or_contours.png",
    )
    copy_if_exists(
        ROOT_DIR / "supersonique_classique" / "mode" / "test_reconstructed_eigenmodes_fast.pdf",
        SUPERSONIC_CLASSIC_DIR / "04_supersonic_reconstructed_mode.pdf",
    )

    # Newly generated presentation figures.
    alphas = [0.30, 0.50, 0.70]
    mach = 0.50
    plot_classic_three_modes(
        SUBSONIC_CLASSIC_DIR / "03_subsonic_three_modes_reim_ampphase_m050.png",
        mach=mach,
        alphas=alphas,
    )
    plot_pinn_vs_classic_three_modes(
        SUBSONIC_PINN_DIR / "02_subsonic_pinn_vs_classic_three_modes_ampphase_m050.png",
        run_dir=ROOT_DIR / "kh_subsonic_fixed_mach_M05_riccati_multibranch",
        checkpoint=ROOT_DIR / "kh_subsonic_fixed_mach_M05_riccati_multibranch" / "checkpoint_epoch_4000.pt",
        mach=mach,
        alphas=alphas,
    )

    readme = PRESENTATION_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Plot Presentation",
                "",
                "- `subsonic_classique/`: solveur classique subsonique",
                "- `subsonic_pinn/`: comparaison PINN subsonique vs classique",
                "- `supersonic_classique/`: solveur classique supersonique",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Presentation directory: {PRESENTATION_DIR}")


if __name__ == "__main__":
    main()
