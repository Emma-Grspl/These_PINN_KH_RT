from __future__ import annotations

"""
Sweep en alpha inspire du notebook de la stagiaire, mais branche sur le solveur
GEP notebook_style du repo et sur le mapping de reference du projet.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def normalize_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = vector[0:n_points]
    v = vector[n_points:2 * n_points]
    p = vector[2 * n_points:3 * n_points]
    rho = p * mach**2

    idx = int(np.argmax(np.abs(rho)))
    phase = np.exp(-1j * np.angle(rho[idx]))
    u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.real(rho)), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep alpha GEP notebook_style a Mach fixe.")
    parser.add_argument("--mach", type=float, default=1.10)
    parser.add_argument("--alpha-min", type=float, default=0.18)
    parser.add_argument("--alpha-max", type=float, default=0.26)
    parser.add_argument("--num-points", type=int, default=7)
    parser.add_argument("--n-points", type=int, default=301)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--cr-window", type=float, default=0.2)
    parser.add_argument("--output-stem", type=str, default="sweep_alpha_gep")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.num_points)
    rows = []
    all_data: dict[str, np.ndarray | float | int | str] = {}
    previous_guess: tuple[float, float] | None = None

    pdf_path = OUTPUT_DIR / f"{args.output_stem}_modes.pdf"
    txt_path = OUTPUT_DIR / f"{args.output_stem}_summary.txt"
    npz_path = OUTPUT_DIR / f"{args.output_stem}_all_data.npz"
    png_path = OUTPUT_DIR / f"{args.output_stem}_growth_curve.png"

    with open(txt_path, "w") as f_txt, PdfPages(pdf_path) as pdf:
        f_txt.write("# Mach   alpha   c_i   c_r   selection_source\n")

        for idx, alpha in enumerate(alpha_values):
            solver = NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=float(args.mach),
                n_points=args.n_points,
                mapping_scale=args.mapping_scale,
                xi_max=args.xi_max,
            )
            chosen_mode, selection_source, n_modes = solver.get_selected_mode(
                target_guess=previous_guess,
                cr_window=args.cr_window,
            )

            if chosen_mode is None:
                rows.append(
                    {
                        "Mach": float(args.mach),
                        "alpha": float(alpha),
                        "cr": np.nan,
                        "ci": np.nan,
                        "omega_i": np.nan,
                        "selection_source": selection_source,
                        "n_finite_modes": n_modes,
                        "success": False,
                    }
                )
                continue

            previous_guess = (chosen_mode["cr"], chosen_mode["ci"])
            c = chosen_mode["c"]
            f_txt.write(
                f"{args.mach:.6f}  {alpha:.8e}  {chosen_mode['ci']:.8e}  {chosen_mode['cr']:.8e}  {selection_source}\n"
            )

            rows.append(
                {
                    "Mach": float(args.mach),
                    "alpha": float(alpha),
                    "cr": chosen_mode["cr"],
                    "ci": chosen_mode["ci"],
                    "omega_i": chosen_mode["omega_i"],
                    "selection_source": selection_source,
                    "n_finite_modes": n_modes,
                    "success": True,
                }
            )

            u, v, p, rho = normalize_mode(chosen_mode["vector"], solver.n_points, solver.Mach)
            key = f"alpha_{idx}"
            all_data[f"{key}_alpha"] = float(alpha)
            all_data[f"{key}_Mach"] = float(args.mach)
            all_data[f"{key}_c_real"] = float(np.real(c))
            all_data[f"{key}_c_imag"] = float(np.imag(c))
            all_data[f"{key}_y"] = solver.y
            all_data[f"{key}_u"] = u
            all_data[f"{key}_v"] = v
            all_data[f"{key}_p"] = p
            all_data[f"{key}_rho"] = rho
            all_data[f"{key}_selection_source"] = selection_source

            fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
            fields = [rho, u, v, p]
            titles = [
                r"Density Perturbation $\hat{\rho}$",
                r"Streamwise Velocity $\hat{u}$",
                r"Vertical Velocity $\hat{v}$",
                r"Pressure Perturbation $\hat{p}$",
            ]
            for ax, field, title in zip(axes.flat, fields, titles):
                ax.plot(solver.y, field.real, label="Real")
                ax.plot(solver.y, field.imag, "--", label="Imag")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(
                f"GEP sweep | M={args.mach:.3f}, alpha={alpha:.3f}, "
                f"c={chosen_mode['cr']:.4f}+i{chosen_mode['ci']:.4f}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    np.savez(npz_path, **all_data)

    rows_df = np.array(rows, dtype=object)
    # Save a simple CSV-like text table through numpy/pandas-free path is overkill;
    # keep the human-readable txt and the npz, and add a compact CSV with numpy.
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{args.output_stem}_growth_map.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    successful = df[df["success"]]
    failed = df[~df["success"]]
    if not successful.empty:
        ax.plot(successful["alpha"], successful["ci"], "ro-", label=r"Unstable eigenvalue $c_i$")
    if not failed.empty:
        ax.plot(failed["alpha"], np.zeros(len(failed)), "b+", label="Failure / neutral")
    ax.set_title(f"GEP alpha sweep (M={args.mach:.3f})")
    ax.set_xlabel(r"Wavenumber $\alpha$")
    ax.set_ylabel(r"Selected $c_i$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nSummary txt: {txt_path}")
    print(f"Modes pdf: {pdf_path}")
    print(f"All data npz: {npz_path}")
    print(f"Growth curve png: {png_path}")
    print(f"Growth map csv: {csv_path}")


if __name__ == "__main__":
    main()
