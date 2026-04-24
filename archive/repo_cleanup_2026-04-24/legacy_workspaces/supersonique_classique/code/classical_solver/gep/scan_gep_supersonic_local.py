from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classical_solver.supersonic.reconstruct_blumen_supersonic_shooting import parse_reference_level


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def load_digitized_curves() -> list[dict]:
    curves = []
    for csv_file in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        level, label, family = parse_reference_level(csv_file)
        if family != "ci_level" or level is None:
            continue
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
        )
        curves.append({"level": float(level), "label": label, "data": df})
    return curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan local GEP dense en regime supersonique.")
    parser.add_argument("--mach-min", type=float, default=1.10)
    parser.add_argument("--mach-max", type=float, default=1.20)
    parser.add_argument("--alpha-min", type=float, default=0.18)
    parser.add_argument("--alpha-max", type=float, default=0.22)
    parser.add_argument("--num-mach", type=int, default=4)
    parser.add_argument("--num-alpha", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=301)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--guide-with-shooting", action="store_true")
    parser.add_argument("--cr-window", type=float, default=0.2)
    parser.add_argument("--output-stem", type=str, default="supersonic_gep_local")
    return parser


def sample_local_map(
    alphas: np.ndarray,
    machs: np.ndarray,
    *,
    n_points: int,
    mapping_scale: float,
    xi_max: float,
    guide_with_shooting: bool,
    cr_window: float,
) -> pd.DataFrame:
    rows = []
    previous_mach_gep: dict[int, tuple[float, float] | None] = {}
    for mach in machs:
        current_mach_gep: dict[int, tuple[float, float] | None] = {}
        previous_alpha_gep: tuple[float, float] | None = None
        for alpha_index, alpha in enumerate(alphas):
            target_guess = None
            shooting_result = None
            if guide_with_shooting:
                shooting_result = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(mach)).solve(
                    cr_min=0.03,
                    cr_max=0.35,
                    ci_min=0.01,
                    ci_max=0.12,
                    max_iter=8,
                )
                if shooting_result.spectral_success:
                    target_guess = (shooting_result.cr, shooting_result.ci)

            gep_anchor = None
            if previous_alpha_gep is not None:
                gep_anchor = previous_alpha_gep
            elif alpha_index in previous_mach_gep and previous_mach_gep[alpha_index] is not None:
                gep_anchor = previous_mach_gep[alpha_index]

            if gep_anchor is not None:
                target_guess = gep_anchor

            solver = NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=float(mach),
                n_points=n_points,
                mapping_scale=mapping_scale,
                xi_max=xi_max,
            )
            result = solver.solve_most_unstable(
                target_guess=target_guess,
                cr_window=cr_window,
            )
            if result.success:
                previous_alpha_gep = (result.cr, result.ci)
                current_mach_gep[alpha_index] = previous_alpha_gep
            else:
                current_mach_gep[alpha_index] = None
            rows.append(
                {
                    "alpha": result.alpha,
                    "Mach": result.Mach,
                    "cr": result.cr,
                    "ci": result.ci,
                    "omega_i": result.omega_i,
                    "n_finite_modes": result.n_finite_modes,
                    "selection_source": result.selection_source,
                    "shooting_cr": None if shooting_result is None else shooting_result.cr,
                    "shooting_ci": None if shooting_result is None else shooting_result.ci,
                    "shooting_omega_i": None if shooting_result is None else shooting_result.omega_i,
                    "shooting_spectral_success": None if shooting_result is None else shooting_result.spectral_success,
                    "target_cr": None if target_guess is None else target_guess[0],
                    "target_ci": None if target_guess is None else target_guess[1],
                    "target_source": (
                        "gep_continuation"
                        if gep_anchor is not None
                        else ("shooting_anchor" if target_guess is not None else "none")
                    ),
                    "success": result.success,
                }
            )
        previous_mach_gep = current_mach_gep
    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def plot_local_map(
    df: pd.DataFrame,
    curves: list[dict],
    *,
    mach_min: float,
    mach_max: float,
    alpha_min: float,
    alpha_max: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_context, ax_zoom = axes

    context_margin_m = 0.18
    context_margin_a = 0.08

    local_curves = []
    context_curves = []
    for curve in curves:
        local_df = curve["data"][
            (curve["data"]["Mach"] >= mach_min)
            & (curve["data"]["Mach"] <= mach_max)
            & (curve["data"]["alpha"] >= alpha_min)
            & (curve["data"]["alpha"] <= alpha_max)
        ]
        context_df = curve["data"][
            (curve["data"]["Mach"] >= mach_min - context_margin_m)
            & (curve["data"]["Mach"] <= mach_max + context_margin_m)
            & (curve["data"]["alpha"] >= max(0.0, alpha_min - context_margin_a))
            & (curve["data"]["alpha"] <= alpha_max + context_margin_a)
        ]
        if not local_df.empty:
            local_curves.append({"level": curve["level"], "label": curve["label"], "data": local_df})
        if not context_df.empty:
            context_curves.append({"level": curve["level"], "label": curve["label"], "data": context_df})

    successful = df[df["success"]].copy()

    for curve in context_curves:
        ax_context.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=14,
            alpha=0.45,
            label=f"Blumen {curve['label']}",
        )

    if not successful.empty:
        scatter_context = ax_context.scatter(
            successful["Mach"],
            successful["alpha"],
            c=successful["ci"],
            cmap="viridis",
            s=90,
            edgecolor="black",
            linewidth=0.5,
            label="GEP (colored by $c_i$)",
        )
        for _, row in successful.iterrows():
            ax_context.annotate(
                f"{row['ci']:.3f}",
                (row["Mach"], row["alpha"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
        fig.colorbar(scatter_context, ax=ax_context, fraction=0.046, pad=0.04, label=r"$c_i$")

    ax_context.set_title("Contexte Blumen + points GEP")
    ax_context.set_xlabel("Nombre de Mach (M)")
    ax_context.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax_context.set_xlim(mach_min - context_margin_m, mach_max + context_margin_m)
    ax_context.set_ylim(max(0.0, alpha_min - context_margin_a), alpha_max + context_margin_a)
    ax_context.grid(True, linestyle="--", alpha=0.35)
    ax_context.legend(fontsize=7, ncol=2)

    for curve in local_curves:
        ax_zoom.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=20,
            alpha=0.65,
            label=f"Blumen {curve['label']}",
        )

    if not successful.empty:
        scatter_zoom = ax_zoom.scatter(
            successful["Mach"],
            successful["alpha"],
            c=successful["ci"],
            cmap="viridis",
            s=110,
            edgecolor="black",
            linewidth=0.6,
            label="GEP",
        )
        for _, row in successful.iterrows():
            ax_zoom.annotate(
                f"cr={row['cr']:.3f}\nci={row['ci']:.3f}",
                (row["Mach"], row["alpha"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
            )
        fig.colorbar(scatter_zoom, ax=ax_zoom, fraction=0.046, pad=0.04, label=r"$c_i$")

    ax_zoom.set_title("Zoom local GEP")
    ax_zoom.set_xlabel("Nombre de Mach (M)")
    ax_zoom.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax_zoom.set_xlim(mach_min, mach_max)
    ax_zoom.set_ylim(alpha_min, alpha_max)
    ax_zoom.grid(True, linestyle="--", alpha=0.35)
    ax_zoom.legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)
    curves = load_digitized_curves()

    print("Echantillonnage local GEP supersonique...")
    df = sample_local_map(
        alphas,
        machs,
        n_points=args.n_points,
        mapping_scale=args.mapping_scale,
        xi_max=args.xi_max,
        guide_with_shooting=args.guide_with_shooting,
        cr_window=args.cr_window,
    )
    csv_path = OUTPUT_DIR / f"{args.output_stem}_growth_map.csv"
    fig_path = OUTPUT_DIR / f"{args.output_stem}_vs_blumen.png"
    df.to_csv(csv_path, index=False)
    plot_local_map(
        df,
        curves,
        mach_min=args.mach_min,
        mach_max=args.mach_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        output_path=fig_path,
    )
    print(df.to_string(index=False))
    print(f"\nResultats enregistres dans {csv_path}")
    print(f"Figure enregistree dans {fig_path}")


if __name__ == "__main__":
    main()
