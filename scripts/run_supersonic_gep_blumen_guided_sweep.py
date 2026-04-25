from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    build_blumen_targets,
    load_digitized_long,
)


def parse_float_list(values: list[float] | None, start: float | None, stop: float | None, count: int | None) -> list[float]:
    if values:
        return [float(v) for v in values]
    if start is None or stop is None or count is None:
        raise ValueError("Provide explicit values or start/stop/count.")
    return [float(v) for v in np.linspace(float(start), float(stop), int(count))]


def solve_default_shooting(alpha: float, mach: float) -> dict:
    solver = Mstab17SupersonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.14,
        max_iter=10,
    )
    return {
        "solver": solver,
        "cr": float(result.cr),
        "ci": float(result.ci),
        "omega_i": float(result.omega_i),
        "stage1_mismatch": float(result.stage1_mismatch),
        "stage2_mismatch": float(result.stage2_mismatch),
        "spectral_success": bool(result.spectral_success),
        "success": bool(result.success),
    }


def mode_stats(solver: NotebookStyleDenseGEPSolver, mode: dict) -> dict:
    p = np.asarray(mode["vector"][2 * solver.n_points : 3 * solver.n_points], dtype=complex)
    amp = np.abs(p)
    if float(np.max(amp)) > 0.0:
        p = p / np.max(amp)
        amp = np.abs(p)
    amp_sum = float(np.sum(amp))
    peak_idx = int(np.argmax(amp))
    centroid = float(np.sum(solver.y * amp) / amp_sum) if amp_sum > 0.0 else np.nan
    spread = float(np.sqrt(np.sum(((solver.y - centroid) ** 2) * amp) / amp_sum)) if amp_sum > 0.0 else np.nan
    phase = np.unwrap(np.angle(p))
    return {
        "peak_y": float(solver.y[peak_idx]),
        "centroid_y": centroid,
        "spread_y": spread,
        "phase_span": float(np.max(phase) - np.min(phase)),
    }


def candidate_score(
    mode: dict,
    *,
    blumen_cr: float,
    blumen_ci: float,
    shooting: dict | None,
    args: argparse.Namespace,
) -> float:
    score = 0.0
    if np.isfinite(blumen_cr):
        score += args.w_blumen_cr * abs(float(mode["cr"]) - float(blumen_cr)) / max(args.cr_scale, 1e-8)
    if np.isfinite(blumen_ci):
        score += args.w_blumen_ci * abs(float(mode["ci"]) - float(blumen_ci)) / max(args.ci_scale, 1e-8)
    if shooting is not None and bool(shooting["spectral_success"]):
        score += args.w_shooting * np.hypot(
            float(mode["cr"]) - float(shooting["cr"]),
            args.ci_weight * (float(mode["ci"]) - float(shooting["ci"])),
        )
    return float(score)


def select_mode(
    solver: NotebookStyleDenseGEPSolver,
    modes: list[dict],
    *,
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    shooting: dict | None,
    args: argparse.Namespace,
) -> tuple[dict | None, list[dict]]:
    filtered = [
        mode
        for mode in modes
        if mode["ci"] > args.ci_min
        and mode["ci"] <= args.ci_max
        and args.cr_min <= mode["cr"] <= args.cr_max
    ]
    if not filtered:
        return None, []

    preliminary = sorted(
        filtered,
        key=lambda mode: candidate_score(
            mode,
            blumen_cr=blumen_cr,
            blumen_ci=blumen_ci,
            shooting=shooting,
            args=args,
        ),
    )[: args.audit_top_k]

    candidate_rows: list[dict] = []
    for rank, mode in enumerate(preliminary, start=1):
        row = {
            "alpha": float(alpha),
            "Mach": float(mach),
            "rank": int(rank),
            "cand_cr": float(mode["cr"]),
            "cand_ci": float(mode["ci"]),
            "cand_omega_i": float(mode["omega_i"]),
            "blumen_cr": float(blumen_cr) if np.isfinite(blumen_cr) else np.nan,
            "blumen_ci": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
            "err_cr_blumen": abs(float(mode["cr"]) - float(blumen_cr)) if np.isfinite(blumen_cr) else np.nan,
            "err_ci_blumen": abs(float(mode["ci"]) - float(blumen_ci)) if np.isfinite(blumen_ci) else np.nan,
            "score": candidate_score(
                mode,
                blumen_cr=blumen_cr,
                blumen_ci=blumen_ci,
                shooting=shooting,
                args=args,
            ),
            **mode_stats(solver, mode),
        }
        if shooting is not None:
            row.update(
                {
                    "shooting_cr": shooting["cr"],
                    "shooting_ci": shooting["ci"],
                    "shooting_omega_i": shooting["omega_i"],
                    "shooting_spectral_success": shooting["spectral_success"],
                    "distance_to_shooting": np.hypot(
                        float(mode["cr"]) - float(shooting["cr"]),
                        args.ci_weight * (float(mode["ci"]) - float(shooting["ci"])),
                    ),
                }
            )
        candidate_rows.append(row)

    chosen_idx = int(np.argmin([row["score"] for row in candidate_rows]))
    return preliminary[chosen_idx], candidate_rows


def plot_surfaces(surface: pd.DataFrame, output: Path) -> None:
    if surface.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for ax, value_col, title in [
        (axes[0], "gep_cr", r"Selected $c_r$"),
        (axes[1], "gep_ci", r"Selected $c_i$"),
    ]:
        if surface["alpha"].nunique() >= 2 and surface["Mach"].nunique() >= 2:
            pivot = surface.pivot(index="alpha", columns="Mach", values=value_col).sort_index().sort_index(axis=1)
            mach_grid, alpha_grid = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
            values = pivot.to_numpy(dtype=float)
            mesh = ax.pcolormesh(mach_grid, alpha_grid, values, shading="auto", cmap="viridis")
            fig.colorbar(mesh, ax=ax, shrink=0.9)
            try:
                contours = ax.contour(mach_grid, alpha_grid, values, colors="black", linewidths=0.8)
                ax.clabel(contours, fontsize=8)
            except Exception:
                pass
        else:
            sc = ax.scatter(surface["Mach"], surface["alpha"], c=surface[value_col], cmap="viridis", s=70)
            fig.colorbar(sc, ax=ax, shrink=0.9)
        ax.scatter(surface["Mach"], surface["alpha"], s=12, color="white", edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("Mach")
        ax.grid(True, linestyle=":", alpha=0.3)
    axes[0].set_ylabel(r"$\alpha$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep GEP supersonique: resolution modale puis selection de branche auditee par Blumen."
    )
    parser.add_argument("--alpha-values", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--alpha-count", type=int, default=None)
    parser.add_argument("--mach-values", type=float, nargs="*", default=None)
    parser.add_argument("--mach-min", type=float, default=None)
    parser.add_argument("--mach-max", type=float, default=None)
    parser.add_argument("--mach-count", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--cr-min", type=float, default=0.0)
    parser.add_argument("--cr-max", type=float, default=0.9)
    parser.add_argument("--ci-min", type=float, default=1e-5)
    parser.add_argument("--ci-max", type=float, default=0.25)
    parser.add_argument("--cr-scale", type=float, default=0.05)
    parser.add_argument("--ci-scale", type=float, default=0.01)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--w-blumen-cr", type=float, default=1.0)
    parser.add_argument("--w-blumen-ci", type=float, default=1.4)
    parser.add_argument("--w-shooting", type=float, default=0.15)
    parser.add_argument("--audit-top-k", type=int, default=12)
    parser.add_argument("--output-stem", type=str, default="supersonic_gep_blumen_guided_sweep")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    alphas = parse_float_list(args.alpha_values, args.alpha_min, args.alpha_max, args.alpha_count)
    machs = parse_float_list(args.mach_values, args.mach_min, args.mach_max, args.mach_count)
    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)

    surface_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for alpha in alphas:
        targets = build_blumen_targets(machs, alpha, cr_points, ci_points)
        for target in targets.itertuples(index=False):
            mach = float(target.Mach)
            solver = NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=mach,
                n_points=int(args.n_points),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
            )
            modes = solver.finite_modes()
            shooting = solve_default_shooting(float(alpha), mach)
            chosen, rows = select_mode(
                solver,
                modes,
                alpha=float(alpha),
                mach=mach,
                blumen_cr=float(target.blumen_cr),
                blumen_ci=float(target.blumen_ci),
                shooting=shooting,
                args=args,
            )
            candidate_rows.extend(rows)
            if chosen is None:
                surface_rows.append(
                    {
                        "alpha": float(alpha),
                        "Mach": mach,
                        "N": int(args.n_points),
                        "success": False,
                        "selection_source": "no_candidate",
                        "blumen_cr": float(target.blumen_cr),
                        "blumen_ci": float(target.blumen_ci),
                        "shooting_cr": shooting["cr"],
                        "shooting_ci": shooting["ci"],
                    }
                )
                continue
            stats = mode_stats(solver, chosen)
            surface_rows.append(
                {
                    "alpha": float(alpha),
                    "Mach": mach,
                    "N": int(args.n_points),
                    "gep_cr": float(chosen["cr"]),
                    "gep_ci": float(chosen["ci"]),
                    "gep_omega_i": float(chosen["omega_i"]),
                    "blumen_cr": float(target.blumen_cr),
                    "blumen_ci": float(target.blumen_ci),
                    "err_cr_blumen": abs(float(chosen["cr"]) - float(target.blumen_cr)),
                    "err_ci_blumen": abs(float(chosen["ci"]) - float(target.blumen_ci)),
                    "shooting_cr": shooting["cr"],
                    "shooting_ci": shooting["ci"],
                    "shooting_omega_i": shooting["omega_i"],
                    "shooting_spectral_success": shooting["spectral_success"],
                    "distance_to_shooting": np.hypot(
                        float(chosen["cr"]) - shooting["cr"],
                        args.ci_weight * (float(chosen["ci"]) - shooting["ci"]),
                    ),
                    "success": True,
                    "selection_source": "gep_blumen_guided",
                    **stats,
                }
            )
            print(
                f"alpha={alpha:.4f} M={mach:.4f} "
                f"GEP=({chosen['cr']:.6f},{chosen['ci']:.6f}) "
                f"Blumen=({target.blumen_cr:.6f},{target.blumen_ci:.6f}) "
                f"Shooting=({shooting['cr']:.6f},{shooting['ci']:.6f})"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    surface = pd.DataFrame(surface_rows)
    candidates = pd.DataFrame(candidate_rows)
    surface_path = args.output_dir / f"{args.output_stem}_surface.csv"
    candidates_path = args.output_dir / f"{args.output_stem}_candidates.csv"
    figure_path = args.output_dir / f"{args.output_stem}_surface.png"
    surface.to_csv(surface_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    plot_surfaces(surface[surface["success"] == True].copy(), figure_path)
    print(f"Surface CSV: {surface_path}")
    print(f"Candidates CSV: {candidates_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
