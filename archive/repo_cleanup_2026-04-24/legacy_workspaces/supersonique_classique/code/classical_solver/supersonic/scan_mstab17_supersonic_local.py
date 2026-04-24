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

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classical_solver.supersonic.reconstruct_blumen_supersonic_shooting import parse_reference_level


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_shooting_supersonic"


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
    parser = argparse.ArgumentParser(description="Scan local supersonique avec le solveur mstab17.")
    parser.add_argument("--mach-min", type=float, default=1.10)
    parser.add_argument("--mach-max", type=float, default=1.30)
    parser.add_argument("--alpha-min", type=float, default=0.18)
    parser.add_argument("--alpha-max", type=float, default=0.30)
    parser.add_argument("--num-mach", type=int, default=5)
    parser.add_argument("--num-alpha", type=int, default=5)
    parser.add_argument("--cr-min", type=float, default=0.03)
    parser.add_argument("--cr-max", type=float, default=0.35)
    parser.add_argument("--ci-min", type=float, default=0.01)
    parser.add_argument("--ci-max", type=float, default=0.12)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-continuation", action="store_true")
    parser.add_argument("--continuation-cr-halfwidth", type=float, default=0.05)
    parser.add_argument("--continuation-ci-halfwidth", type=float, default=0.03)
    parser.add_argument("--output-stem", type=str, default="mstab17_supersonic_local")
    return parser


def _local_box_from_guess(
    guess_cr: float,
    guess_ci: float,
    *,
    global_cr_min: float,
    global_cr_max: float,
    global_ci_min: float,
    global_ci_max: float,
    cr_halfwidth: float,
    ci_halfwidth: float,
) -> tuple[float, float, float, float]:
    return (
        max(global_cr_min, guess_cr - cr_halfwidth),
        min(global_cr_max, guess_cr + cr_halfwidth),
        max(global_ci_min, guess_ci - ci_halfwidth),
        min(global_ci_max, guess_ci + ci_halfwidth),
    )


def sample_local_map(
    alphas: np.ndarray,
    machs: np.ndarray,
    *,
    cr_min: float,
    cr_max: float,
    ci_min: float,
    ci_max: float,
    max_iter: int,
    match_y: float,
    use_continuation: bool,
    continuation_cr_halfwidth: float,
    continuation_ci_halfwidth: float,
) -> pd.DataFrame:
    rows = []
    previous_mach_results: dict[int, dict | None] = {}
    for mach in machs:
        current_mach_results: dict[int, dict | None] = {}
        for alpha_index, alpha in enumerate(alphas):
            solver = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(mach), match_y=match_y)
            result = None

            local_boxes = []
            if use_continuation:
                neighbor_candidates = []
                if alpha_index > 0 and current_mach_results.get(alpha_index - 1) is not None:
                    neighbor_candidates.append(current_mach_results[alpha_index - 1])
                if alpha_index in previous_mach_results and previous_mach_results[alpha_index] is not None:
                    neighbor_candidates.append(previous_mach_results[alpha_index])

                for neighbor in neighbor_candidates:
                    local_boxes.append(
                        _local_box_from_guess(
                            neighbor["cr"],
                            neighbor["ci"],
                            global_cr_min=cr_min,
                            global_cr_max=cr_max,
                            global_ci_min=ci_min,
                            global_ci_max=ci_max,
                            cr_halfwidth=continuation_cr_halfwidth,
                            ci_halfwidth=continuation_ci_halfwidth,
                        )
                    )

            for local_cr_min, local_cr_max, local_ci_min, local_ci_max in local_boxes:
                trial = solver.solve(
                    cr_min=local_cr_min,
                    cr_max=local_cr_max,
                    ci_min=local_ci_min,
                    ci_max=local_ci_max,
                    max_iter=max_iter,
                )
                result = trial
                if trial.success:
                    break

            if result is None or not result.success:
                result = solver.solve(
                    cr_min=cr_min,
                    cr_max=cr_max,
                    ci_min=ci_min,
                    ci_max=ci_max,
                    max_iter=max_iter,
                )

            if result.success:
                current_mach_results[alpha_index] = {"cr": result.cr, "ci": result.ci}
            else:
                current_mach_results[alpha_index] = None
            rows.append(
                {
                    "alpha": result.alpha,
                    "Mach": result.Mach,
                    "cr": result.cr,
                    "ci": result.ci,
                    "omega_i": result.omega_i,
                    "stage1_mismatch": result.stage1_mismatch,
                    "stage2_mismatch": result.stage2_mismatch,
                    "y_limit": result.y_limit,
                    "ln_p_start_right": result.ln_p_start_right,
                    "success": result.success,
                }
            )
        previous_mach_results = current_mach_results
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
    fig, ax = plt.subplots(figsize=(10, 7))

    local_curves = []
    for curve in curves:
        local_df = curve["data"][
            (curve["data"]["Mach"] >= mach_min)
            & (curve["data"]["Mach"] <= mach_max)
            & (curve["data"]["alpha"] >= alpha_min)
            & (curve["data"]["alpha"] <= alpha_max)
        ]
        if not local_df.empty:
            local_curves.append({"level": curve["level"], "label": curve["label"], "data": local_df})

    successful = df[df["success"]].copy()
    if len(successful) >= 3:
        triangulation = mtri.Triangulation(successful["Mach"], successful["alpha"])
        contour_levels = sorted({curve["level"] for curve in local_curves})
        if contour_levels:
            contour = ax.tricontour(
                triangulation,
                successful["ci"],
                levels=contour_levels,
                cmap="plasma",
                linewidths=1.8,
            )
            ax.clabel(contour, fmt="%0.2f", fontsize=8)

    for curve in local_curves:
        ax.scatter(
            curve["data"]["Mach"],
            curve["data"]["alpha"],
            s=16,
            alpha=0.55,
            label=f"Blumen {curve['label']}",
        )

    if not successful.empty:
        ax.scatter(successful["Mach"], successful["alpha"], s=70, color="black", marker="x", label="mstab17 success")
    failed = df[~df["success"]]
    if not failed.empty:
        ax.scatter(failed["Mach"], failed["alpha"], s=70, color="red", marker="x", label="mstab17 fail")

    ax.set_title("Scan local supersonique mstab17 vs Blumen")
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(mach_min, mach_max)
    ax.set_ylim(alpha_min, alpha_max)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)
    curves = load_digitized_curves()

    print("Echantillonnage local supersonique mstab17...")
    df = sample_local_map(
        alphas,
        machs,
        cr_min=args.cr_min,
        cr_max=args.cr_max,
        ci_min=args.ci_min,
        ci_max=args.ci_max,
        max_iter=args.max_iter,
        match_y=args.match_y,
        use_continuation=args.use_continuation,
        continuation_cr_halfwidth=args.continuation_cr_halfwidth,
        continuation_ci_halfwidth=args.continuation_ci_halfwidth,
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
