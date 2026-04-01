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
    parser = argparse.ArgumentParser(description="GEP adaptatif guide par le tir en regime supersonique.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-values", type=int, nargs="+", default=[181, 241, 301, 361, 401])
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--prefer-positive-cr", action="store_true", default=True)
    parser.add_argument("--output-stem", type=str, default="adaptive_gep_guided")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shooting = Mstab17SupersonicSolver(alpha=args.alpha, Mach=args.mach).solve(
        cr_min=0.03,
        cr_max=0.35,
        ci_min=0.01,
        ci_max=0.12,
        max_iter=8,
    )
    target = (shooting.cr, shooting.ci)

    rows: list[dict] = []
    chosen_row: dict | None = None

    for n_points in args.n_values:
        solver = NotebookStyleDenseGEPSolver(
            alpha=args.alpha,
            Mach=args.mach,
            n_points=n_points,
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
            target_guess=target,
            prefer_positive_cr=args.prefer_positive_cr,
            ci_weight=args.ci_weight,
        )
        if mode is None:
            rows.append(
                {
                    "alpha": args.alpha,
                    "Mach": args.mach,
                    "N": n_points,
                    "shooting_cr": shooting.cr,
                    "shooting_ci": shooting.ci,
                    "gep_cr": None,
                    "gep_ci": None,
                    "gep_omega_i": None,
                    "distance_to_shooting": None,
                    "selection_source": selection_source,
                    "n_finite_modes": n_modes,
                    "success": False,
                    "accepted": False,
                }
            )
            continue

        distance = solver.spectral_distance(mode, target, ci_weight=args.ci_weight)
        accepted = distance <= args.distance_tol
        row = {
            "alpha": args.alpha,
            "Mach": args.mach,
            "N": n_points,
            "shooting_cr": shooting.cr,
            "shooting_ci": shooting.ci,
            "gep_cr": mode["cr"],
            "gep_ci": mode["ci"],
            "gep_omega_i": mode["omega_i"],
            "distance_to_shooting": distance,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": True,
            "accepted": accepted,
        }
        rows.append(row)
        if accepted and chosen_row is None:
            chosen_row = row
            break

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{args.output_stem}_a_{args.alpha:.3f}_m_{args.mach:.3f}.csv"
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))
    if chosen_row is not None:
        print(
            "\nAccepted resolution: "
            f"N={chosen_row['N']} "
            f"(distance={chosen_row['distance_to_shooting']:.6f}, "
            f"gep=({chosen_row['gep_cr']:.6f}, {chosen_row['gep_ci']:.6f}))"
        )
    else:
        print("\nNo resolution met the requested tolerance.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
