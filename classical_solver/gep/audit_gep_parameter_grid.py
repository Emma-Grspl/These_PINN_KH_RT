from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit parametrique GEP sur un point supersonique fixe.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-values", type=_parse_int_list, default=[481])
    parser.add_argument("--mapping-kinds", type=str, default="pin,cubic")
    parser.add_argument("--mapping-scales", type=_parse_float_list, default=[3.0, 5.0, 8.0])
    parser.add_argument("--xi-max-values", type=_parse_float_list, default=[0.95, 0.98, 0.99])
    parser.add_argument("--cubic-deltas", type=_parse_float_list, default=[0.1, 0.2, 0.4])
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-stem", type=str, default="audit_gep_parameter_grid")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping_kinds = [item.strip() for item in args.mapping_kinds.split(",") if item.strip()]

    shooting = Mstab17SupersonicSolver(alpha=args.alpha, Mach=args.mach).solve(
        cr_min=0.03,
        cr_max=0.35,
        ci_min=0.01,
        ci_max=0.12,
        max_iter=8,
    )
    target = (shooting.cr, shooting.ci)

    rows: list[dict] = []
    for n_points, mapping_kind, mapping_scale, xi_max in product(
        args.n_values,
        mapping_kinds,
        args.mapping_scales,
        args.xi_max_values,
    ):
        cubic_deltas = args.cubic_deltas if mapping_kind == "cubic" else [0.2]
        for cubic_delta in cubic_deltas:
            solver = NotebookStyleDenseGEPSolver(
                alpha=args.alpha,
                Mach=args.mach,
                n_points=n_points,
                mapping_kind=mapping_kind,
                mapping_scale=mapping_scale,
                cubic_delta=cubic_delta,
                xi_max=xi_max,
            )
            mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
                target_guess=target,
                prefer_positive_cr=True,
                ci_weight=args.ci_weight,
            )
            if mode is None:
                rows.append(
                    {
                        "alpha": args.alpha,
                        "Mach": args.mach,
                        "N": n_points,
                        "mapping_kind": mapping_kind,
                        "mapping_scale": mapping_scale,
                        "cubic_delta": cubic_delta,
                        "xi_max": xi_max,
                        "shooting_cr": shooting.cr,
                        "shooting_ci": shooting.ci,
                        "gep_cr": None,
                        "gep_ci": None,
                        "gep_omega_i": None,
                        "distance_to_shooting": None,
                        "selection_source": selection_source,
                        "n_finite_modes": n_modes,
                        "success": False,
                    }
                )
                continue

            rows.append(
                {
                    "alpha": args.alpha,
                    "Mach": args.mach,
                    "N": n_points,
                    "mapping_kind": mapping_kind,
                    "mapping_scale": mapping_scale,
                    "cubic_delta": cubic_delta,
                    "xi_max": xi_max,
                    "shooting_cr": shooting.cr,
                    "shooting_ci": shooting.ci,
                    "gep_cr": mode["cr"],
                    "gep_ci": mode["ci"],
                    "gep_omega_i": mode["omega_i"],
                    "distance_to_shooting": solver.spectral_distance(mode, target, ci_weight=args.ci_weight),
                    "selection_source": selection_source,
                    "n_finite_modes": n_modes,
                    "success": True,
                }
            )

    df = pd.DataFrame(rows).sort_values(["success", "distance_to_shooting"], ascending=[False, True]).reset_index(drop=True)
    top_df = df.head(args.top_k)

    csv_path = OUTPUT_DIR / f"{args.output_stem}_a_{args.alpha:.3f}_m_{args.mach:.3f}.csv"
    top_df.to_csv(csv_path, index=False)

    print(
        f"Shooting target: cr={shooting.cr:.6f}, ci={shooting.ci:.6f}, "
        f"omega_i={shooting.omega_i:.6f}, spectral_success={shooting.spectral_success}"
    )
    print(top_df.to_string(index=False))
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
