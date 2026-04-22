from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit physique comparatif des familles supersoniques issues du clustering local."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--reference-mach", type=float, default=1.30)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--cluster-stem", type=str, required=True)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def normalize_pressure_profile(y: np.ndarray, vector: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(vector[2 * n_points : 3 * n_points], dtype=complex).copy()
    idx = int(np.argmax(np.abs(p)))
    if np.abs(p[idx]) > 0.0:
        p = p * np.exp(-1j * np.angle(p[idx]))
        amp = float(np.max(np.abs(p)))
        if amp > 0.0:
            p = p / amp
    return y.copy(), p


def profile_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    amp = np.abs(p)
    amp_sum = float(np.sum(amp))
    peak_idx = int(np.argmax(amp))
    centroid = float(np.sum(y * amp) / amp_sum) if amp_sum > 0.0 else np.nan
    spread = float(np.sqrt(np.sum(((y - centroid) ** 2) * amp) / amp_sum)) if amp_sum > 0.0 else np.nan
    phase = np.unwrap(np.angle(p))
    return {
        "peak_y": float(y[peak_idx]),
        "peak_amp": float(amp[peak_idx]),
        "centroid_y": centroid,
        "spread_y": spread,
        "phase_span": float(np.max(phase) - np.min(phase)),
    }


def resample_profile(y: np.ndarray, p: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    pr = np.interp(y_ref, y, np.real(p))
    pi = np.interp(y_ref, y, np.imag(p))
    return pr + 1j * pi


def overlap_between_profiles(y: np.ndarray, p: np.ndarray, y_ref: np.ndarray, p_ref: np.ndarray) -> float:
    p_interp = resample_profile(y, p, y_ref)
    num = abs(np.vdot(p_interp, p_ref))
    den = max(float(np.linalg.norm(p_interp) * np.linalg.norm(p_ref)), 1e-12)
    return float(num / den)


def solve_default_shooting(alpha: float, mach: float) -> tuple[Mstab17SupersonicSolver, object]:
    solver = Mstab17SupersonicSolver(alpha=alpha, Mach=mach)
    result = solver.solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=10,
    )
    return solver, result


def optimize_stage2_for_candidate(solver: Mstab17SupersonicSolver, cr: float, ci: float) -> tuple[float, float]:
    opt = minimize_scalar(
        lambda ln_p_right: solver.stage2_objective(ln_p_right, cr, ci),
        bounds=(-15.0, 5.0),
        method="bounded",
    )
    return float(opt.x), float(opt.fun)


def load_cluster_inputs(cluster_stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = pd.read_csv(OUTPUT_DIR / f"{cluster_stem}_clusters.csv")
    matches = pd.read_csv(OUTPUT_DIR / f"{cluster_stem}_matches.csv")
    return clusters, matches


def build_path_table(
    clusters_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    mach_values: list[float],
    reference_mach: float,
) -> pd.DataFrame:
    ref_clusters = clusters_df[np.isclose(clusters_df["Mach"], reference_mach)].copy()
    ref_clusters = ref_clusters.sort_values("rep_cr", ascending=False).reset_index(drop=True)

    rows: list[dict] = []
    for family_rank, ref_row in enumerate(ref_clusters.itertuples(index=False), start=1):
        current_cluster_id = int(ref_row.cluster_id)
        current_mach = float(reference_mach)
        rows.append(
            {
                "family_rank": family_rank,
                "family_source_cluster": int(ref_row.cluster_id),
                "Mach": current_mach,
                "cluster_id": current_cluster_id,
            }
        )
        for next_mach in [m for m in mach_values if m < current_mach]:
            step = matches_df[
                np.isclose(matches_df["prev_mach"], current_mach)
                & np.isclose(matches_df["next_mach"], next_mach)
                & (matches_df["prev_cluster_id"] == current_cluster_id)
            ]
            if step.empty:
                break
            next_cluster_id = int(step.iloc[0]["next_cluster_id"])
            rows.append(
                {
                    "family_rank": family_rank,
                    "family_source_cluster": int(ref_row.cluster_id),
                    "Mach": float(next_mach),
                    "cluster_id": next_cluster_id,
                }
            )
            current_mach = float(next_mach)
            current_cluster_id = next_cluster_id

    return pd.DataFrame(rows).sort_values(["family_rank", "Mach"], ascending=[True, False]).reset_index(drop=True)


def pick_mode_for_cluster(
    solver: NotebookStyleDenseGEPSolver,
    *,
    rep_cr: float,
    rep_ci: float,
    ci_weight: float,
) -> dict:
    modes = [mode for mode in solver.finite_modes() if mode["cr"] >= -1e-10]
    if not modes:
        raise RuntimeError("Aucun mode GEP fini disponible.")
    return min(
        modes,
        key=lambda mode: abs(float(mode["cr"]) - float(rep_cr)) + float(ci_weight) * abs(float(mode["ci"]) - float(rep_ci)),
    )


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def plot_family_paths(audit_df: pd.DataFrame, output_png: Path) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    metrics = [
        ("cand_cr", r"$c_r$"),
        ("cand_ci", r"$c_i$"),
        ("distance_to_shooting", "Distance to shooting"),
        ("overlap_to_reference_start", "Overlap to start"),
    ]

    for family_rank, family_df in audit_df.groupby("family_rank"):
        label = f"Family {int(family_rank)}"
        for ax, (column, ylabel) in zip(axes.flat, metrics):
            ax.plot(family_df["Mach"], family_df[column], marker="o", linewidth=2.0, label=label)
            ax.set_ylabel(ylabel)

    for ax in axes[-1, :]:
        ax.set_xlabel("Mach")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    mach_values = sorted(set(float(m) for m in args.mach_values), reverse=True)
    if float(args.reference_mach) not in mach_values:
        mach_values = sorted([float(args.reference_mach), *mach_values], reverse=True)

    clusters_df, matches_df = load_cluster_inputs(args.cluster_stem)
    paths_df = build_path_table(clusters_df, matches_df, mach_values, float(args.reference_mach))

    reference_profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    previous_profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audit_rows: list[dict] = []

    for row in paths_df.itertuples(index=False):
        cluster_row = clusters_df[
            np.isclose(clusters_df["Mach"], row.Mach) & (clusters_df["cluster_id"] == row.cluster_id)
        ]
        if cluster_row.empty:
            continue
        cluster_row = cluster_row.iloc[0]

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=float(row.Mach),
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        mode = pick_mode_for_cluster(
            solver,
            rep_cr=float(cluster_row["rep_cr"]),
            rep_ci=float(cluster_row["rep_ci"]),
            ci_weight=float(args.ci_weight),
        )
        y, p = normalize_pressure_profile(solver.y, mode["vector"], solver.n_points)
        stats = profile_stats(y, p)

        shooting_solver, shooting_result = solve_default_shooting(alpha, float(row.Mach))
        stage1_mismatch = float(shooting_solver.stage1_mismatch(float(mode["cr"]), float(mode["ci"])))
        stage2_ln_p_right, stage2_mismatch = optimize_stage2_for_candidate(
            shooting_solver,
            float(mode["cr"]),
            float(mode["ci"]),
        )

        family_rank = int(row.family_rank)
        if family_rank not in reference_profiles:
            reference_profiles[family_rank] = (y, p)
        y_ref, p_ref = reference_profiles[family_rank]
        overlap_to_reference = overlap_between_profiles(y, p, y_ref, p_ref)

        overlap_to_previous = np.nan
        if family_rank in previous_profiles:
            y_prev, p_prev = previous_profiles[family_rank]
            overlap_to_previous = overlap_between_profiles(y, p, y_prev, p_prev)
        previous_profiles[family_rank] = (y, p)

        audit_rows.append(
            {
                "alpha": alpha,
                "family_rank": family_rank,
                "family_source_cluster": int(row.family_source_cluster),
                "Mach": float(row.Mach),
                "cluster_id": int(row.cluster_id),
                "cand_cr": float(mode["cr"]),
                "cand_ci": float(mode["ci"]),
                "cand_omega_i": float(mode["omega_i"]),
                "distance_to_shooting": float(
                    solver.spectral_distance(mode, (float(shooting_result.cr), float(shooting_result.ci)), ci_weight=args.ci_weight)
                ),
                "stage1_mismatch_at_candidate": stage1_mismatch,
                "stage2_mismatch_at_candidate": stage2_mismatch,
                "stage2_ln_p_start_right_at_candidate": stage2_ln_p_right,
                "shooting_cr": float(shooting_result.cr),
                "shooting_ci": float(shooting_result.ci),
                "shooting_omega_i": float(shooting_result.omega_i),
                "shooting_spectral_success": bool(shooting_result.spectral_success),
                "shooting_mode_success": bool(shooting_result.mode_success),
                "overlap_to_reference_start": float(overlap_to_reference),
                "overlap_to_previous": float(overlap_to_previous) if np.isfinite(overlap_to_previous) else np.nan,
                **stats,
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values(["family_rank", "Mach"], ascending=[True, False]).reset_index(drop=True)
    summary_df = (
        audit_df.groupby("family_rank", as_index=False)
        .agg(
            family_source_cluster=("family_source_cluster", "first"),
            n_points=("Mach", "size"),
            mach_min=("Mach", "min"),
            mach_max=("Mach", "max"),
            cr_min=("cand_cr", "min"),
            cr_max=("cand_cr", "max"),
            ci_mean=("cand_ci", "mean"),
            omega_i_mean=("cand_omega_i", "mean"),
            distance_to_shooting_mean=("distance_to_shooting", "mean"),
            distance_to_shooting_max=("distance_to_shooting", "max"),
            overlap_to_reference_start_min=("overlap_to_reference_start", "min"),
            overlap_to_previous_min=("overlap_to_previous", "min"),
            stage1_mismatch_mean=("stage1_mismatch_at_candidate", "mean"),
            stage2_mismatch_mean=("stage2_mismatch_at_candidate", "mean"),
            centroid_y_mean=("centroid_y", "mean"),
            spread_y_mean=("spread_y", "mean"),
            phase_span_mean=("phase_span", "mean"),
        )
        .sort_values("family_rank")
        .reset_index(drop=True)
    )

    audit_csv = OUTPUT_DIR / f"{args.output_stem}_families.csv"
    summary_csv = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    plot_png = OUTPUT_DIR / f"{args.output_stem}_paths.png"

    audit_df.to_csv(audit_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_family_paths(audit_df, plot_png)

    print("Family audit:")
    print(audit_df.to_string(index=False))
    print("\nFamily summary:")
    print(summary_df.to_string(index=False))
    print(f"\nFamilies CSV: {audit_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Paths figure: {plot_png}")


if __name__ == "__main__":
    main()
