from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


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
    half_width = max(float(np.max(np.abs(y_vis))), float(min_half_width))
    return -half_width, half_width


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace les modes GEP supersoniques classes par croissance c_i."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points", type=int, default=301)
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--positive-cr-only", action="store_true")
    parser.add_argument("--full-domain", action="store_true")
    parser.add_argument("--output-stem", type=str, default="supersonic_gep_modes_by_ci")
    return parser


def plot_mode_page(
    pdf: PdfPages,
    *,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
    title: str,
    full_domain: bool,
) -> None:
    fields = [rho, u, v, p]
    titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]
    x_limits = (float(y[0]), float(y[-1])) if full_domain else compute_visible_xlim(y, fields)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, field, field_title in zip(axes.flat, fields, titles):
        ax.plot(
            y,
            np.real(field),
            linestyle="None",
            marker="o",
            markersize=2.8,
            alpha=0.85,
            label="Re",
        )
        ax.plot(
            y,
            np.imag(field),
            linestyle="None",
            marker="x",
            markersize=2.8,
            alpha=0.85,
            label="Im",
        )
        ax.set_title(field_title)
        ax.set_xlim(*x_limits)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(args.alpha),
        Mach=float(args.mach),
        n_points=int(args.n_points),
        mapping_kind=args.mapping_kind,
        mapping_scale=float(args.mapping_scale),
        cubic_delta=float(args.cubic_delta),
        xi_max=float(args.xi_max),
    )

    modes = solver.finite_modes()
    if args.positive_cr_only:
        modes = [mode for mode in modes if mode["cr"] >= -1e-10]
    if not modes:
        raise RuntimeError("Aucun mode fini positif en croissance n'a ete trouve.")

    modes = sorted(modes, key=lambda mode: mode["ci"], reverse=True)[: max(1, int(args.top_k))]

    summary_rows = []
    pdf_path = OUTPUT_DIR / f"{args.output_stem}_a{args.alpha:.3f}_m{args.mach:.3f}.pdf"
    csv_path = OUTPUT_DIR / f"{args.output_stem}_a{args.alpha:.3f}_m{args.mach:.3f}.csv"

    with PdfPages(pdf_path) as pdf:
        for rank, mode in enumerate(modes, start=1):
            u, v, p, rho = normalize_mode(mode["vector"], solver.n_points, solver.Mach)
            summary_rows.append(
                {
                    "rank_by_ci": rank,
                    "alpha": float(args.alpha),
                    "Mach": float(args.mach),
                    "n_points": int(args.n_points),
                    "mapping_kind": args.mapping_kind,
                    "mapping_scale": float(args.mapping_scale),
                    "xi_max": float(args.xi_max),
                    "cr": float(mode["cr"]),
                    "ci": float(mode["ci"]),
                    "omega_i": float(mode["omega_i"]),
                    "abs_cr": float(mode["abs_cr"]),
                }
            )
            plot_mode_page(
                pdf,
                y=solver.y,
                u=u,
                v=v,
                p=p,
                rho=rho,
                title=(
                    f"GEP supersonique classe par c_i | rank={rank}, "
                    f"alpha={args.alpha:.3f}, M={args.mach:.3f}, "
                    f"c={mode['cr']:.5f}+i{mode['ci']:.5f}"
                ),
                full_domain=bool(args.full_domain),
            )

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"PDF: {pdf_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
