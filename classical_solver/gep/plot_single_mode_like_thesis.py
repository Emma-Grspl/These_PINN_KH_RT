from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver


def normalize_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = vector[0:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2

    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def compute_visible_xlim(y: np.ndarray, fields: list[np.ndarray], *, threshold_ratio: float = 0.02, min_half_width: float = 8.0) -> tuple[float, float]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace une figure de mode propre au format de la these/stagiaire.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--xi-max", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(args.alpha),
        Mach=float(args.mach),
        n_points=int(args.n_points),
        mapping_scale=float(args.mapping_scale),
        xi_max=float(args.xi_max),
    )
    mode, selection_source, n_modes = solver.get_selected_mode()
    if mode is None:
        raise RuntimeError("Aucun mode selectionne.")

    u, v, p, rho = normalize_mode(mode["vector"], solver.n_points, solver.Mach)
    fields = [rho, u, v, p]
    x_limits = compute_visible_xlim(solver.y, fields)
    titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, field, title in zip(axes.flat, fields, titles):
        ax.plot(solver.y, np.real(field), label="Real")
        ax.plot(solver.y, np.imag(field), "--", label="Imag")
        ax.set_title(title)
        ax.set_xlim(*x_limits)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        fr"Most unstable eigenmode for $\alpha={args.alpha:.3f}$ and $M={args.mach:.3f}$"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(
        f"alpha={args.alpha:.6f} Mach={args.mach:.6f} "
        f"cr={mode['cr']:.6e} ci={mode['ci']:.6e} omega_i={mode['omega_i']:.6e} "
        f"selection_source={selection_source} n_finite_modes={n_modes}"
    )
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
