from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep alpha GEP adaptatif avec continuation locale.")
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha-min", type=float, required=True)
    parser.add_argument("--alpha-max", type=float, required=True)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--n-values", type=int, nargs="+", default=[241, 301, 361, 401])
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--previous-weight", type=float, default=0.6)
    parser.add_argument("--output-stem", type=str, default="adaptive_continuation_sweep_gep")
    return parser


def run_point(
    *,
    alpha: float,
    mach: float,
    target_guess: tuple[float, float],
    shooting_guess: tuple[float, float],
    previous_signature: np.ndarray | None,
    n_values: list[int],
    mapping_kind: str,
    mapping_scale: float,
    cubic_delta: float,
    xi_max: float,
    distance_tol: float,
    ci_weight: float,
) -> tuple[dict, np.ndarray | None, list[dict]]:
    attempts: list[dict] = []
    accepted_row: dict | None = None
    accepted_signature: np.ndarray | None = None

    for n_points in n_values:
        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=n_points,
            mapping_kind=mapping_kind,
            mapping_scale=mapping_scale,
            cubic_delta=cubic_delta,
            xi_max=xi_max,
        )
        mode, selection_source, n_modes = solver.get_branch_mode(
            target_guess=target_guess,
            previous_signature=previous_signature,
            prefer_positive_cr=True,
            ci_weight=ci_weight,
        )
        if mode is None:
            row = {
                "alpha": alpha,
                "Mach": mach,
                "N": n_points,
                "gep_cr": np.nan,
                "gep_ci": np.nan,
                "gep_omega_i": np.nan,
                "distance_to_target": np.nan,
                "overlap_to_previous": np.nan,
                "selection_source": selection_source,
                "n_finite_modes": n_modes,
                "success": False,
                "accepted": False,
            }
            attempts.append(row)
            continue

        distance = solver.spectral_distance(mode, target_guess, ci_weight=ci_weight)
        distance_to_shooting = solver.spectral_distance(mode, shooting_guess, ci_weight=ci_weight)
        overlap_to_previous = (
            np.nan if previous_signature is None else solver.signature_overlap(mode, previous_signature)
        )
        row = {
            "alpha": alpha,
            "Mach": mach,
            "N": n_points,
            "gep_cr": mode["cr"],
            "gep_ci": mode["ci"],
            "gep_omega_i": mode["omega_i"],
            "distance_to_target": distance,
            "distance_to_shooting": distance_to_shooting,
            "overlap_to_previous": overlap_to_previous,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": True,
            "accepted": distance_to_shooting <= distance_tol,
        }
        attempts.append(row)
        if row["accepted"]:
            accepted_row = row
            accepted_signature = mode.get("signature")
            break

    if accepted_row is None and attempts:
        successful = [row for row in attempts if row["success"]]
        if successful:
            accepted_row = min(successful, key=lambda row: row["distance_to_target"])
            # Recompute the matching signature from the last attempted resolution.
            for n_points in n_values:
                if n_points != int(accepted_row["N"]):
                    continue
                solver = NotebookStyleDenseGEPSolver(
                    alpha=alpha,
                    Mach=mach,
                    n_points=n_points,
                    mapping_kind=mapping_kind,
                    mapping_scale=mapping_scale,
                    cubic_delta=cubic_delta,
                    xi_max=xi_max,
                )
                mode, _, _ = solver.get_branch_mode(
                    target_guess=target_guess,
                    previous_signature=previous_signature,
                    prefer_positive_cr=True,
                    ci_weight=ci_weight,
                )
                if mode is not None:
                    accepted_signature = mode.get("signature")
                break
        else:
            accepted_row = attempts[-1]
    return accepted_row, accepted_signature, attempts


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_points)
    summary_rows: list[dict] = []
    attempts_rows: list[dict] = []
    previous_gep: tuple[float, float] | None = None
    previous_signature: np.ndarray | None = None

    for idx, alpha in enumerate(alphas):
        shooting = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(args.mach)).solve(
            cr_min=0.03,
            cr_max=0.35,
            ci_min=0.01,
            ci_max=0.12,
            max_iter=8,
        )
        shooting_guess = (shooting.cr, shooting.ci)
        if idx == 0 or previous_gep is None:
            target_guess = shooting_guess
            target_source = "shooting_anchor"
        else:
            target_guess = (
                args.previous_weight * previous_gep[0] + (1.0 - args.previous_weight) * shooting_guess[0],
                args.previous_weight * previous_gep[1] + (1.0 - args.previous_weight) * shooting_guess[1],
            )
            target_source = "blended_continuation"

        chosen, chosen_signature, attempts = run_point(
            alpha=float(alpha),
            mach=float(args.mach),
            target_guess=target_guess,
            shooting_guess=shooting_guess,
            previous_signature=previous_signature,
            n_values=list(args.n_values),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
            distance_tol=args.distance_tol,
            ci_weight=args.ci_weight,
        )
        for row in attempts:
            row["target_cr"] = target_guess[0]
            row["target_ci"] = target_guess[1]
            row["target_source"] = target_source
            row["shooting_cr"] = shooting.cr
            row["shooting_ci"] = shooting.ci
            row["shooting_omega_i"] = shooting.omega_i
            row["shooting_spectral_success"] = shooting.spectral_success
        attempts_rows.extend(attempts)

        summary_row = dict(chosen)
        summary_row["target_cr"] = target_guess[0]
        summary_row["target_ci"] = target_guess[1]
        summary_row["target_source"] = target_source
        summary_row["shooting_cr"] = shooting.cr
        summary_row["shooting_ci"] = shooting.ci
        summary_row["shooting_omega_i"] = shooting.omega_i
        summary_row["shooting_spectral_success"] = shooting.spectral_success
        summary_rows.append(summary_row)

        if chosen["success"]:
            previous_gep = (chosen["gep_cr"], chosen["gep_ci"])
            previous_signature = chosen_signature

    summary_df = pd.DataFrame(summary_rows)
    attempts_df = pd.DataFrame(attempts_rows)

    summary_csv = OUTPUT_DIR / f"{args.output_stem}_mach_{args.mach:.3f}_summary.csv"
    attempts_csv = OUTPUT_DIR / f"{args.output_stem}_mach_{args.mach:.3f}_attempts.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}_mach_{args.mach:.3f}.png"

    summary_df.to_csv(summary_csv, index=False)
    attempts_df.to_csv(attempts_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(summary_df["alpha"], summary_df["shooting_ci"], "o-", label="Shooting $c_i$")
    axes[0].plot(summary_df["alpha"], summary_df["gep_ci"], "s--", label="Adaptive continuation GEP $c_i$")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].set_title(f"$c_i$ comparison at M={args.mach:.3f}")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(summary_df["alpha"], summary_df["shooting_cr"], "o-", label="Shooting $c_r$")
    axes[1].plot(summary_df["alpha"], summary_df["gep_cr"], "s--", label="Adaptive continuation GEP $c_r$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$c_r$")
    axes[1].set_title(f"$c_r$ comparison at M={args.mach:.3f}")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Attempts CSV: {attempts_csv}")
    print(f"Figure: {png_path}")


if __name__ == "__main__":
    main()
