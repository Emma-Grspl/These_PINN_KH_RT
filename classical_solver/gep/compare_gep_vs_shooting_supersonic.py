from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convergence locale GEP vs tir en regime supersonique.")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "supersonic_gep_vs_shooting.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_points = [
        {"alpha": 0.18, "Mach": 1.10, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.24, "Mach": 1.20, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
        {"alpha": 0.18, "Mach": 1.30, "cr_min": 0.03, "cr_max": 0.35, "ci_min": 0.01, "ci_max": 0.12, "max_iter": 8},
    ]
    resolutions = [121, 181, 241, 301]

    rows = []
    for point in test_points:
        alpha = float(point["alpha"])
        mach = float(point["Mach"])
        shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
            cr_min=point["cr_min"],
            cr_max=point["cr_max"],
            ci_min=point["ci_min"],
            ci_max=point["ci_max"],
            max_iter=point["max_iter"],
        )
        rows.append(
            {
                "method": "shooting",
                "alpha": alpha,
                "Mach": mach,
                "N": None,
                "cr": shooting.cr,
                "ci": shooting.ci,
                "omega_i": shooting.omega_i,
                "selection_source": "mstab17",
                "distance_to_shooting": 0.0,
                "success": shooting.success,
            }
        )

        for n_points in resolutions:
            solver = NotebookStyleDenseGEPSolver(
                alpha=alpha,
                Mach=mach,
                n_points=n_points,
                mapping_scale=args.mapping_scale,
                xi_max=args.xi_max,
            )

            modes = solver.finite_modes()
            positive_modes = [mode for mode in modes if mode["cr"] >= -1e-10]
            if not positive_modes:
                rows.append(
                    {
                        "method": "gep",
                        "alpha": alpha,
                        "Mach": mach,
                        "N": n_points,
                        "cr": None,
                        "ci": None,
                        "omega_i": None,
                        "selection_source": "no_positive_mode",
                        "distance_to_shooting": None,
                        "success": False,
                    }
                )
                continue

            def distance(mode: dict) -> float:
                return float(((mode["cr"] - shooting.cr) ** 2 + 4.0 * (mode["ci"] - shooting.ci) ** 2) ** 0.5)

            nearest = min(positive_modes, key=distance)
            rows.append(
                {
                    "method": "gep",
                    "alpha": alpha,
                    "Mach": mach,
                    "N": n_points,
                    "cr": nearest["cr"],
                    "ci": nearest["ci"],
                    "omega_i": nearest["omega_i"],
                    "selection_source": "nearest_to_shooting",
                    "distance_to_shooting": distance(nearest),
                    "success": True,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nResultats enregistres dans {args.output}")


if __name__ == "__main__":
    main()
