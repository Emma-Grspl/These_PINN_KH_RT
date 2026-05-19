from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Superpose les modes GEP rank 1 et rank 2 classes par c_i sur rho, u, v, p."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points", type=int, default=602)
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--positive-cr-only", action="store_true")
    parser.add_argument("--u-zoom-half-width", type=float, default=2.0)
    parser.add_argument("--output-stem", type=str, default="supersonic_gep_top2_overlay")
    return parser


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
    modes = sorted(modes, key=lambda mode: mode["ci"], reverse=True)
    if len(modes) < 2:
        raise RuntimeError("Moins de deux modes disponibles apres filtrage.")

    top_modes = modes[:2]
    mode_fields = []
    for mode in top_modes:
        u, v, p, rho = normalize_mode(mode["vector"], solver.n_points, solver.Mach)
        mode_fields.append(
            {
                "cr": float(mode["cr"]),
                "ci": float(mode["ci"]),
                "omega_i": float(mode["omega_i"]),
                "u": u,
                "v": v,
                "p": p,
                "rho": rho,
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    field_keys = ["rho", "u", "v", "p"]
    field_titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]
    styles = [
        {"color": "tab:blue", "marker": "o", "label": "rank 1 Re", "linestyle": "-", "alpha": 0.90},
        {"color": "tab:cyan", "marker": "o", "label": "rank 1 Im", "linestyle": "--", "alpha": 0.80},
        {"color": "tab:red", "marker": "s", "label": "rank 2 Re", "linestyle": "-", "alpha": 0.90},
        {"color": "tab:orange", "marker": "s", "label": "rank 2 Im", "linestyle": "--", "alpha": 0.80},
    ]

    for ax, field_key, title in zip(axes.flat, field_keys, field_titles):
        for rank, payload in enumerate(mode_fields, start=1):
            re_field = np.real(payload[field_key])
            im_field = np.imag(payload[field_key])
            re_style = styles[0] if rank == 1 else styles[2]
            im_style = styles[1] if rank == 1 else styles[3]
            ax.plot(
                solver.y,
                re_field,
                color=re_style["color"],
                linestyle=re_style["linestyle"],
                linewidth=1.1,
                marker=re_style["marker"],
                markersize=2.8,
                markevery=max(1, len(solver.y) // 70),
                alpha=re_style["alpha"],
                label=re_style["label"],
            )
            ax.plot(
                solver.y,
                im_field,
                color=im_style["color"],
                linestyle=im_style["linestyle"],
                linewidth=1.0,
                marker=im_style["marker"],
                markersize=2.6,
                markevery=max(1, len(solver.y) // 70),
                alpha=im_style["alpha"],
                label=im_style["label"],
            )
        ax.set_title(title)
        if field_key == "u":
            ax.set_xlim(-float(args.u_zoom_half_width), float(args.u_zoom_half_width))
        else:
            ax.set_xlim(float(solver.y[0]), float(solver.y[-1]))
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "GEP supersonique top 2 par c_i | "
        f"alpha={args.alpha:.3f}, M={args.mach:.3f}, "
        f"N={args.n_points}, mapping={args.mapping_kind}, L={args.mapping_scale}"
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{args.output_stem}_a{args.alpha:.3f}_m{args.mach:.3f}.png"
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    print(
        f"rank1: cr={mode_fields[0]['cr']:.6f}, ci={mode_fields[0]['ci']:.6f}, omega_i={mode_fields[0]['omega_i']:.6f}"
    )
    print(
        f"rank2: cr={mode_fields[1]['cr']:.6f}, ci={mode_fields[1]['ci']:.6f}, omega_i={mode_fields[1]['omega_i']:.6f}"
    )
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()
