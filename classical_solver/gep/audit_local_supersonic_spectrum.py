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
    parser = argparse.ArgumentParser(description="Audit local du spectre GEP autour d'un point supersonique.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points", type=int, default=481)
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-stem", type=str, default="audit_local_supersonic_spectrum")
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

    solver = NotebookStyleDenseGEPSolver(
        alpha=args.alpha,
        Mach=args.mach,
        n_points=args.n_points,
        mapping_kind=args.mapping_kind,
        mapping_scale=args.mapping_scale,
        cubic_delta=args.cubic_delta,
        xi_max=args.xi_max,
    )
    modes = solver.finite_modes()
    positive_modes = [mode for mode in modes if mode["cr"] >= -1e-10]

    rows: list[dict] = []
    for idx, mode in enumerate(positive_modes):
        distance = solver.spectral_distance(mode, target, ci_weight=args.ci_weight)
        rows.append(
            {
                "rank_by_growth": idx + 1,
                "mapping_kind": args.mapping_kind,
                "mapping_scale": args.mapping_scale,
                "cubic_delta": args.cubic_delta,
                "xi_max": args.xi_max,
                "cr": mode["cr"],
                "ci": mode["ci"],
                "omega_i": mode["omega_i"],
                "distance_to_shooting": distance,
            }
        )

    df = pd.DataFrame(rows).sort_values("distance_to_shooting").reset_index(drop=True)
    top_df = df.head(args.top_k).copy()
    top_df.insert(0, "shooting_cr", shooting.cr)
    top_df.insert(1, "shooting_ci", shooting.ci)
    top_df.insert(2, "shooting_omega_i", shooting.omega_i)

    csv_path = OUTPUT_DIR / (
        f"{args.output_stem}_{args.mapping_kind}"
        f"_a_{args.alpha:.3f}_m_{args.mach:.3f}_N_{args.n_points}.csv"
    )
    top_df.to_csv(csv_path, index=False)

    print(
        f"Shooting target: cr={shooting.cr:.6f}, ci={shooting.ci:.6f}, "
        f"omega_i={shooting.omega_i:.6f}, spectral_success={shooting.spectral_success}"
    )
    print(top_df.to_string(index=False))
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
