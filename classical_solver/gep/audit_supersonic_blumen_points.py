from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import solve_point
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classical_solver.supersonic.reconstruct_blumen_supersonic_shooting import parse_reference_level


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def load_digitized_curves() -> list[dict]:
    curves: list[dict] = []
    for csv_file in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        level, label, family = parse_reference_level(csv_file)
        df = (
            pd.read_csv(
                csv_file,
                header=None,
                names=["Mach", "alpha"],
                sep=";",
                decimal=",",
                engine="python",
            )
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .reset_index(drop=True)
        )
        curves.append(
            {
                "csv_path": csv_file,
                "stem": Path(csv_file).stem,
                "level": None if level is None else float(level),
                "label": label,
                "family": family,
                "data": df,
            }
        )
    return curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit spectral GEP sur les points digitises de Blumen.")
    parser.add_argument("--families", type=str, default="ci_level,ci_special,cr_special")
    parser.add_argument("--n-values", type=int, nargs="+", default=[561])
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--previous-weight", type=float, default=0.6)
    parser.add_argument("--cr-min", type=float, default=0.03)
    parser.add_argument("--cr-max-scale", type=float, default=0.5)
    parser.add_argument("--cr-max-floor", type=float, default=0.35)
    parser.add_argument("--cr-max-cap", type=float, default=0.7)
    parser.add_argument("--ci-min", type=float, default=0.001)
    parser.add_argument("--ci-max", type=float, default=0.12)
    parser.add_argument("--shooting-max-iter", type=int, default=10)
    parser.add_argument("--output-stem", type=str, default="audit_supersonic_blumen_points")
    return parser


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _reference_from_curve(curve: dict, shooting: Mstab17SupersonicSolver | object) -> tuple[float | None, float | None, str]:
    family = curve["family"]
    level = curve["level"]
    if family == "ci_level":
        return None, level, "curve_ci_level"
    if family == "ci_special":
        return 0.0, level, "curve_ci_special"
    if family == "cr_special":
        return level, 0.0, "curve_cr_special"
    return None, None, "curve_unknown"


def plot_audit_maps(points_df: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    ci_df = points_df[np.isfinite(points_df["ci_abs_err"])].copy()
    pcm0 = axes[0].scatter(
        ci_df["Mach"],
        ci_df["alpha"],
        c=ci_df["ci_abs_err"],
        cmap="magma",
        s=36,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0].set_title(r"Erreur absolue sur $c_i$")
    axes[0].set_xlabel(r"$M$")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(pcm0, ax=axes[0], label=r"$|c_i^{pred} - c_i^{ref}|$")

    cr_df = points_df[np.isfinite(points_df["cr_abs_err"])].copy()
    if cr_df.empty:
        axes[1].text(0.5, 0.5, "Aucun point avec $c_r$ de reference", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        pcm1 = axes[1].scatter(
            cr_df["Mach"],
            cr_df["alpha"],
            c=cr_df["cr_abs_err"],
            cmap="viridis",
            s=40,
            edgecolors="black",
            linewidths=0.3,
        )
        axes[1].set_title(r"Erreur absolue sur $c_r$")
        axes[1].set_xlabel(r"$M$")
        axes[1].set_ylabel(r"$\alpha$")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        fig.colorbar(pcm1, ax=axes[1], label=r"$|c_r^{pred} - c_r^{ref}|$")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    allowed_families = {item.strip() for item in args.families.split(",") if item.strip()}
    curves = [curve for curve in load_digitized_curves() if curve["family"] in allowed_families]

    point_rows: list[dict] = []
    for curve in curves:
        previous_gep: tuple[float, float] | None = None
        previous_signature: np.ndarray | None = None

        for point_index, row in curve["data"].iterrows():
            alpha = float(row["alpha"])
            mach = float(row["Mach"])

            shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
                cr_min=args.cr_min,
                cr_max=min(args.cr_max_cap, max(args.cr_max_floor, args.cr_max_scale * mach)),
                ci_min=args.ci_min,
                ci_max=args.ci_max,
                max_iter=args.shooting_max_iter,
            )
            shooting_guess = (shooting.cr, shooting.ci)

            ref_cr, ref_ci, ref_source = _reference_from_curve(curve, shooting)
            if previous_gep is not None:
                target_guess = previous_gep
                target_source = "curve_continuation"
            elif ref_cr is not None and ref_ci is not None:
                target_guess = (ref_cr, ref_ci)
                target_source = ref_source
            elif ref_ci is not None:
                target_guess = (shooting.cr, ref_ci)
                target_source = "shooting_cr_plus_curve_ci"
            else:
                target_guess = shooting_guess
                target_source = "shooting_anchor"

            chosen, _, mode = solve_point(
                alpha=alpha,
                mach=mach,
                n_values=list(args.n_values),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
                ci_weight=args.ci_weight,
                distance_tol=args.distance_tol,
                target_guess=target_guess,
                shooting_guess=shooting_guess,
                previous_signature=previous_signature,
            )

            gep_cr = chosen["gep_cr"]
            gep_ci = chosen["gep_ci"]
            ci_abs_err = abs(float(gep_ci) - float(ref_ci)) if np.isfinite(_nan_if_none(ref_ci)) and np.isfinite(float(gep_ci)) else np.nan
            cr_abs_err = abs(float(gep_cr) - float(ref_cr)) if np.isfinite(_nan_if_none(ref_cr)) and np.isfinite(float(gep_cr)) else np.nan

            point_rows.append(
                {
                    "curve_stem": curve["stem"],
                    "curve_label": curve["label"],
                    "curve_family": curve["family"],
                    "curve_level": _nan_if_none(curve["level"]),
                    "point_index": int(point_index),
                    "alpha": alpha,
                    "Mach": mach,
                    "ref_cr": _nan_if_none(ref_cr),
                    "ref_ci": _nan_if_none(ref_ci),
                    "ref_source": ref_source,
                    "target_cr": float(target_guess[0]),
                    "target_ci": float(target_guess[1]),
                    "target_source": target_source,
                    "shooting_cr": float(shooting.cr),
                    "shooting_ci": float(shooting.ci),
                    "shooting_omega_i": float(shooting.omega_i),
                    "shooting_spectral_success": bool(shooting.spectral_success),
                    "gep_cr": gep_cr,
                    "gep_ci": gep_ci,
                    "gep_omega_i": chosen["gep_omega_i"],
                    "distance_to_target": chosen["distance_to_target"],
                    "distance_to_shooting": chosen["distance_to_shooting"],
                    "selection_source": chosen["selection_source"],
                    "n_finite_modes": chosen["n_finite_modes"],
                    "success": bool(chosen["success"]),
                    "accepted": bool(chosen["accepted"]),
                    "ci_abs_err": ci_abs_err,
                    "cr_abs_err": cr_abs_err,
                }
            )

            if chosen["success"] and mode is not None:
                previous_gep = (float(chosen["gep_cr"]), float(chosen["gep_ci"]))
                previous_signature = mode.get("signature")

    points_df = pd.DataFrame(point_rows).sort_values(["curve_family", "curve_stem", "point_index"]).reset_index(drop=True)
    summary_df = (
        points_df.groupby(["curve_family", "curve_stem", "curve_label"], dropna=False)
        .agg(
            n_points=("alpha", "size"),
            n_success=("success", "sum"),
            n_accepted=("accepted", "sum"),
            ci_mae=("ci_abs_err", "mean"),
            ci_max=("ci_abs_err", "max"),
            cr_mae=("cr_abs_err", "mean"),
            cr_max=("cr_abs_err", "max"),
            shooting_dist_mean=("distance_to_shooting", "mean"),
        )
        .reset_index()
        .sort_values(["curve_family", "curve_stem"])
    )

    points_csv = OUTPUT_DIR / f"{args.output_stem}_points.csv"
    summary_csv = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}_error_maps.png"

    points_df.to_csv(points_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_audit_maps(points_df, png_path)

    print(summary_df.to_string(index=False))
    print(f"\nPoints CSV: {points_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {png_path}")


if __name__ == "__main__":
    main()
