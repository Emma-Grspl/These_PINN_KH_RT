from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR, plot_isolines  # noqa: E402
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver  # noqa: E402
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402


@dataclass
class Cluster:
    mach: float
    cluster_id: int
    modes: list[dict]
    representative: dict
    stats: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clustering local des familles modales supersoniques puis matching entre Mach."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--reference-mach", type=float, default=1.30)
    parser.add_argument("--n-points", type=int, default=561)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--high-cr-threshold", type=float, default=0.60)
    parser.add_argument("--valid-cr-max", type=float, default=1.20)
    parser.add_argument("--cluster-cr-window", type=float, default=0.08)
    parser.add_argument("--cluster-ci-window", type=float, default=0.01)
    parser.add_argument("--cluster-overlap-threshold", type=float, default=0.92)
    parser.add_argument("--cluster-centroid-window", type=float, default=0.40)
    parser.add_argument("--cluster-phase-window", type=float, default=0.35)
    parser.add_argument("--w-match-overlap", type=float, default=1.40)
    parser.add_argument("--w-match-cr", type=float, default=0.90)
    parser.add_argument("--w-match-ci", type=float, default=0.45)
    parser.add_argument("--w-match-centroid", type=float, default=0.35)
    parser.add_argument("--w-match-spread", type=float, default=0.20)
    parser.add_argument("--w-match-phase", type=float, default=0.20)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def solve_default_shooting(alpha: float, mach: float) -> tuple[float, float, float, bool]:
    shooting = Mstab17SupersonicSolver(alpha=alpha, Mach=mach).solve(
        cr_min=0.03,
        cr_max=min(0.7, max(0.35, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=10,
    )
    return float(shooting.cr), float(shooting.ci), float(shooting.omega_i), bool(shooting.spectral_success)


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


def build_mode_row(solver: NotebookStyleDenseGEPSolver, mode: dict) -> dict:
    y, p = normalize_pressure_profile(solver.y, mode["vector"], solver.n_points)
    return {
        "mode": mode,
        "cr": float(mode["cr"]),
        "ci": float(mode["ci"]),
        "omega_i": float(mode["omega_i"]),
        **profile_stats(y, p),
    }


def rows_should_link(row_a: dict, row_b: dict, solver: NotebookStyleDenseGEPSolver, args: argparse.Namespace) -> bool:
    overlap = solver.signature_overlap(row_a["mode"], row_b["mode"]["signature"])
    return (
        abs(row_a["cr"] - row_b["cr"]) <= args.cluster_cr_window
        and abs(row_a["ci"] - row_b["ci"]) <= args.cluster_ci_window
        and abs(row_a["centroid_y"] - row_b["centroid_y"]) <= args.cluster_centroid_window
        and abs(row_a["phase_span"] - row_b["phase_span"]) <= args.cluster_phase_window
        and overlap >= args.cluster_overlap_threshold
    )


def cluster_modes(mach: float, rows: list[dict], solver: NotebookStyleDenseGEPSolver, args: argparse.Namespace) -> list[Cluster]:
    unassigned = list(range(len(rows)))
    clusters: list[Cluster] = []
    cluster_id = 0

    while unassigned:
        seed_idx = max(unassigned, key=lambda idx: rows[idx]["cr"])
        members = [seed_idx]
        unassigned.remove(seed_idx)
        changed = True
        while changed:
            changed = False
            for idx in list(unassigned):
                if any(rows_should_link(rows[idx], rows[m], solver, args) for m in members):
                    members.append(idx)
                    unassigned.remove(idx)
                    changed = True

        cluster_rows = [rows[idx] for idx in members]
        rep = min(
            cluster_rows,
            key=lambda row: (
                abs(row["cr"] - np.median([r["cr"] for r in cluster_rows]))
                + 2.0 * abs(row["ci"] - np.median([r["ci"] for r in cluster_rows]))
            ),
        )
        stats = {
            "cr_mean": float(np.mean([r["cr"] for r in cluster_rows])),
            "ci_mean": float(np.mean([r["ci"] for r in cluster_rows])),
            "omega_i_max": float(np.max([r["omega_i"] for r in cluster_rows])),
            "centroid_y_mean": float(np.mean([r["centroid_y"] for r in cluster_rows])),
            "spread_y_mean": float(np.mean([r["spread_y"] for r in cluster_rows])),
            "phase_span_mean": float(np.mean([r["phase_span"] for r in cluster_rows])),
            "n_modes": int(len(cluster_rows)),
        }
        clusters.append(
            Cluster(
                mach=mach,
                cluster_id=cluster_id,
                modes=[row["mode"] for row in cluster_rows],
                representative=rep["mode"],
                stats=stats,
            )
        )
        cluster_id += 1

    return sorted(clusters, key=lambda item: item.stats["cr_mean"], reverse=True)


def cluster_to_row(cluster: Cluster, shooting_guess: tuple[float, float], solver: NotebookStyleDenseGEPSolver) -> dict:
    rep = cluster.representative
    y, p = normalize_pressure_profile(solver.y, rep["vector"], solver.n_points)
    stats = profile_stats(y, p)
    return {
        "Mach": cluster.mach,
        "cluster_id": cluster.cluster_id,
        "cluster_cr": cluster.stats["cr_mean"],
        "cluster_ci": cluster.stats["ci_mean"],
        "cluster_omega_i_max": cluster.stats["omega_i_max"],
        "n_modes": cluster.stats["n_modes"],
        "rep_cr": float(rep["cr"]),
        "rep_ci": float(rep["ci"]),
        "rep_omega_i": float(rep["omega_i"]),
        "distance_to_shooting": float(solver.spectral_distance(rep, shooting_guess, ci_weight=2.0)),
        **stats,
    }


def match_score(prev_cluster: Cluster, next_cluster: Cluster, solver: NotebookStyleDenseGEPSolver, args: argparse.Namespace) -> float:
    prev_mode = prev_cluster.representative
    next_mode = next_cluster.representative
    overlap = solver.signature_overlap(next_mode, prev_mode["signature"])
    cr_term = abs(float(next_mode["cr"]) - float(prev_mode["cr"]))
    ci_term = abs(float(next_mode["ci"]) - float(prev_mode["ci"]))
    centroid_term = abs(next_cluster.stats["centroid_y_mean"] - prev_cluster.stats["centroid_y_mean"])
    spread_term = abs(next_cluster.stats["spread_y_mean"] - prev_cluster.stats["spread_y_mean"])
    phase_term = abs(next_cluster.stats["phase_span_mean"] - prev_cluster.stats["phase_span_mean"])
    return (
        args.w_match_overlap * (1.0 - float(overlap))
        + args.w_match_cr * cr_term
        + args.w_match_ci * ci_term
        + args.w_match_centroid * centroid_term
        + args.w_match_spread * spread_term
        + args.w_match_phase * phase_term
    )


def greedy_match(prev_clusters: list[Cluster], next_clusters: list[Cluster], solver: NotebookStyleDenseGEPSolver, args: argparse.Namespace) -> list[dict]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for i, prev_cluster in enumerate(prev_clusters):
        for j, next_cluster in enumerate(next_clusters):
            candidate_pairs.append((match_score(prev_cluster, next_cluster, solver, args), i, j))
    candidate_pairs.sort(key=lambda item: item[0])

    used_prev: set[int] = set()
    used_next: set[int] = set()
    matches: list[dict] = []
    for score, i, j in candidate_pairs:
        if i in used_prev or j in used_next:
            continue
        used_prev.add(i)
        used_next.add(j)
        matches.append(
            {
                "prev_mach": prev_clusters[i].mach,
                "prev_cluster_id": prev_clusters[i].cluster_id,
                "prev_cr": float(prev_clusters[i].representative["cr"]),
                "prev_ci": float(prev_clusters[i].representative["ci"]),
                "next_mach": next_clusters[j].mach,
                "next_cluster_id": next_clusters[j].cluster_id,
                "next_cr": float(next_clusters[j].representative["cr"]),
                "next_ci": float(next_clusters[j].representative["ci"]),
                "match_score": float(score),
            }
        )
    return matches


def trace_reference_family(
    clusters_by_mach: dict[float, list[Cluster]],
    matches_by_step: dict[tuple[float, float], list[dict]],
    reference_mach: float,
    high_cr_threshold: float,
) -> list[tuple[float, Cluster]]:
    ref_clusters = clusters_by_mach[reference_mach]
    reference_cluster = None
    for cluster in ref_clusters:
        if float(cluster.representative["cr"]) >= high_cr_threshold:
            reference_cluster = cluster
            break
    if reference_cluster is None:
        reference_cluster = max(ref_clusters, key=lambda item: float(item.representative["cr"]))

    tracked = [(reference_mach, reference_cluster)]
    mach_sequence = sorted(clusters_by_mach.keys(), reverse=True)
    current_cluster_id = reference_cluster.cluster_id
    current_mach = reference_mach
    for next_mach in [m for m in mach_sequence if m < reference_mach]:
        step_matches = matches_by_step.get((current_mach, next_mach), [])
        match = next(
            (item for item in step_matches if item["prev_cluster_id"] == current_cluster_id),
            None,
        )
        if match is None:
            break
        next_cluster = next(
            cluster for cluster in clusters_by_mach[next_mach] if cluster.cluster_id == match["next_cluster_id"]
        )
        tracked.append((next_mach, next_cluster))
        current_mach = next_mach
        current_cluster_id = next_cluster.cluster_id
    return tracked


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = float(args.alpha)
    mach_values = sorted(set(float(m) for m in args.mach_values), reverse=True)
    if float(args.reference_mach) not in mach_values:
        mach_values = sorted([float(args.reference_mach), *mach_values], reverse=True)

    clusters_by_mach: dict[float, list[Cluster]] = {}
    cluster_rows: list[dict] = []
    all_match_rows: list[dict] = []

    for mach in mach_values:
        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        rows = [
            build_mode_row(solver, mode)
            for mode in solver.finite_modes()
            if mode["cr"] >= -1e-10 and mode["cr"] <= float(args.valid_cr_max)
        ]
        clusters = cluster_modes(mach, rows, solver, args)
        clusters_by_mach[mach] = clusters

        shooting_guess = solve_default_shooting(alpha, mach)[:2]
        for cluster in clusters:
            cluster_rows.append(cluster_to_row(cluster, shooting_guess, solver))

    matches_by_step: dict[tuple[float, float], list[dict]] = {}
    for prev_mach, next_mach in zip(mach_values[:-1], mach_values[1:]):
        solver_next = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=next_mach,
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        matches = greedy_match(clusters_by_mach[prev_mach], clusters_by_mach[next_mach], solver_next, args)
        matches_by_step[(prev_mach, next_mach)] = matches
        all_match_rows.extend(matches)

    tracked = trace_reference_family(
        clusters_by_mach,
        matches_by_step,
        float(args.reference_mach),
        float(args.high_cr_threshold),
    )

    surface_rows: list[dict] = []
    for mach, cluster in tracked:
        shooting_cr, shooting_ci, shooting_omega_i, shooting_ok = solve_default_shooting(alpha, mach)
        rep = cluster.representative
        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        y, p = normalize_pressure_profile(solver.y, rep["vector"], solver.n_points)
        stats = profile_stats(y, p)
        surface_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "N": int(args.n_points),
                "family_cluster_id": cluster.cluster_id,
                "gep_cr": float(rep["cr"]),
                "gep_ci": float(rep["ci"]),
                "gep_omega_i": float(rep["omega_i"]),
                "distance_to_shooting": float(solver.spectral_distance(rep, (shooting_cr, shooting_ci), ci_weight=2.0)),
                "peak_y": stats["peak_y"],
                "centroid_y": stats["centroid_y"],
                "spread_y": stats["spread_y"],
                "phase_span": stats["phase_span"],
                "selection_source": "family_clustering_matching",
                "success": True,
                "accepted": False,
                "shooting_cr": shooting_cr,
                "shooting_ci": shooting_ci,
                "shooting_omega_i": shooting_omega_i,
                "shooting_spectral_success": shooting_ok,
            }
        )

    clusters_df = pd.DataFrame(cluster_rows).sort_values(["Mach", "cluster_cr"], ascending=[False, False]).reset_index(drop=True)
    matches_df = pd.DataFrame(all_match_rows).sort_values(
        ["prev_mach", "match_score"], ascending=[False, True]
    ).reset_index(drop=True)
    surface_df = pd.DataFrame(surface_rows).sort_values("Mach", ascending=False).reset_index(drop=True)

    clusters_csv = OUTPUT_DIR / f"{args.output_stem}_clusters.csv"
    matches_csv = OUTPUT_DIR / f"{args.output_stem}_matches.csv"
    surface_csv = OUTPUT_DIR / f"{args.output_stem}_surface.csv"
    png_path = OUTPUT_DIR / f"{args.output_stem}_isolines.png"

    clusters_df.to_csv(clusters_csv, index=False)
    matches_df.to_csv(matches_csv, index=False)
    surface_df.to_csv(surface_csv, index=False)
    plot_isolines(surface_df, png_path)

    print("Tracked family:")
    print(surface_df.to_string(index=False))
    print("\nLocal clusters:")
    print(clusters_df.to_string(index=False))
    print("\nMatches:")
    print(matches_df.to_string(index=False))
    print(f"\nClusters CSV: {clusters_csv}")
    print(f"Matches CSV: {matches_csv}")
    print(f"Surface CSV: {surface_csv}")
    print(f"Isoline figure: {png_path}")


if __name__ == "__main__":
    main()
