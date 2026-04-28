from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comparaison directe des candidats GEP supersoniques avec le mode shooting voisin de Blumen."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--n-points", type=int, default=1001)
    parser.add_argument("--mapping-kind", choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--max-abs-c", type=float, default=10.0)
    parser.add_argument("--positive-ci-only", action="store_true", default=True)
    parser.add_argument("--top-k-cr", type=int, default=6)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--shoot-cr-window", type=float, default=0.08)
    parser.add_argument("--shoot-ci-window", type=float, default=0.03)
    parser.add_argument("--shooting-max-iter", type=int, default=10)
    parser.add_argument("--shooting-grid-size", type=int, default=4)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def extract_raw_modes_with_vectors(
    solver: NotebookStyleDenseGEPSolver,
    *,
    max_abs_c: float,
    positive_ci_only: bool,
) -> list[dict]:
    modes = solver.finite_modes()
    rows: list[dict] = []
    for mode in modes:
        c = complex(mode["c"])
        if not np.isfinite(c.real) or not np.isfinite(c.imag):
            continue
        if abs(c) >= max_abs_c:
            continue
        if positive_ci_only and float(mode["ci"]) <= 0.0:
            continue
        rows.append(
            {
                "c": c,
                "cr": float(mode["cr"]),
                "ci": float(mode["ci"]),
                "omega_i": float(mode["omega_i"]),
                "abs_c": float(abs(c)),
                "vector": np.asarray(mode["vector"]),
            }
        )
    return rows


def normalize_gep_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = vector[0:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2

    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def extract_shooting_mode(
    alpha: float,
    mach: float,
    *,
    target_cr: float,
    target_ci: float,
    cr_window: float,
    ci_window: float,
    max_iter: int,
    grid_size: int,
) -> dict:
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        use_mapping=True,
        mapping_scale=5.0,
    )
    result = solver.solve(
        cr_min=max(0.0, target_cr - cr_window),
        cr_max=target_cr + cr_window,
        ci_min=max(1e-4, target_ci - ci_window),
        ci_max=target_ci + ci_window,
        max_iter=max_iter,
        grid_size=grid_size,
    )
    if not result.spectral_success:
        raise RuntimeError(
            f"Shooting spectral failure for alpha={alpha:.3f}, M={mach:.3f} "
            f"around target ({target_cr:.5f},{target_ci:.5f})."
        )

    sol_left, _, sol_right_full, _ = solver.get_trajectories(
        result.cr,
        result.ci,
        ln_p_start_right=result.ln_p_start_right,
    )
    if not (sol_left.success and sol_right_full.success):
        raise RuntimeError(
            f"Shooting mode reconstruction failure for alpha={alpha:.3f}, M={mach:.3f}."
        )

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right_full.t)
    ln_p_left, phi_left = sol_left.y[2], sol_left.y[3]
    ln_p_right, phi_right = sol_right_full.y[2], sol_right_full.y[3]

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right_full, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))

    y = np.concatenate([y_left[y_left < 0.0], y_right[::-1]])
    p = np.concatenate([p_left[y_left < 0.0], p_right[::-1]])
    scale = max(np.max(np.abs(np.real(p))), np.max(np.abs(np.imag(p))), 1e-12)
    p = p / scale
    rho = (mach**2) * p
    return {
        "cr": float(result.cr),
        "ci": float(result.ci),
        "omega_i": float(result.omega_i),
        "y": y,
        "p": p,
        "rho": rho,
        "spectral_success": bool(result.spectral_success),
        "mode_success": bool(result.mode_success),
    }


def profile_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p_abs = np.abs(p)
    norm = max(trapezoid_compat(p_abs, y), 1e-12)
    centroid_abs_y = float(trapezoid_compat(np.abs(y) * p_abs, y) / norm)
    spread_abs_y = float(np.sqrt(max(trapezoid_compat((y**2) * p_abs, y) / norm, 0.0)))
    peak_y = float(y[int(np.argmax(p_abs))])
    return {
        "centroid_abs_y": centroid_abs_y,
        "spread_abs_y": spread_abs_y,
        "peak_y": peak_y,
    }


def candidate_score(mode: dict, target_cr: float, target_ci: float, ci_weight: float) -> float:
    return float(np.sqrt((float(mode["cr"]) - target_cr) ** 2 + (float(ci_weight) * (float(mode["ci"]) - target_ci)) ** 2))


def plot_comparison_pages(
    summary_df: pd.DataFrame,
    shooting_profiles_df: pd.DataFrame,
    gep_profiles_df: pd.DataFrame,
    output_path: Path,
) -> None:
    with PdfPages(output_path) as pdf:
        for mach in sorted(summary_df["Mach"].unique()):
            summary_row = summary_df[summary_df["Mach"] == mach].iloc[0]
            shoot = shooting_profiles_df[shooting_profiles_df["Mach"] == mach].copy()
            gep = gep_profiles_df[gep_profiles_df["Mach"] == mach].copy()

            fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
            axes[0].plot(shoot["y"], shoot["p_real"], color="black", linewidth=2.0, label="shooting")
            axes[1].plot(shoot["y"], shoot["p_abs"], color="black", linewidth=2.0, label="shooting")
            axes[2].plot(shoot["y"], shoot["rho_real"], color="black", linewidth=2.0, label="shooting")

            for rank, group in gep.groupby("rank_cr", sort=True):
                label = (
                    f"GEP #{int(rank)} "
                    f"c=({float(group['candidate_cr'].iloc[0]):.4f},{float(group['candidate_ci'].iloc[0]):.4f})"
                )
                axes[0].plot(group["y"], group["p_real"], linewidth=1.3, alpha=0.9, label=label)
                axes[1].plot(group["y"], group["p_abs"], linewidth=1.3, alpha=0.9, label=label)
                axes[2].plot(group["y"], group["rho_real"], linewidth=1.3, alpha=0.9, label=label)

            axes[0].set_title(r"Re($\hat{p}$)")
            axes[1].set_title(r"$|\hat{p}|$")
            axes[2].set_title(r"Re($\hat{\rho}$)")
            axes[2].set_xlabel("y")
            for ax in axes:
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7, frameon=False)

            fig.suptitle(
                f"alpha={float(summary_row['alpha']):.3f}, M={float(mach):.3f}\n"
                f"Blumen=({float(summary_row['blumen_cr']):.5f},{float(summary_row['blumen_ci']):.5f}) | "
                f"shooting=({float(summary_row['shooting_cr']):.5f},{float(summary_row['shooting_ci']):.5f})"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    targets_df = build_blumen_targets(args.mach_values, args.alpha, cr_points, ci_points)

    summary_rows: list[dict] = []
    shooting_profile_rows: list[dict] = []
    gep_candidate_rows: list[dict] = []
    gep_profile_rows: list[dict] = []

    for mach in args.mach_values:
        target = targets_df[targets_df["Mach"] == float(mach)].iloc[0]
        target_cr = float(target["blumen_cr"])
        target_ci = float(target["blumen_ci"])

        shooting = extract_shooting_mode(
            float(args.alpha),
            float(mach),
            target_cr=target_cr,
            target_ci=target_ci,
            cr_window=float(args.shoot_cr_window),
            ci_window=float(args.shoot_ci_window),
            max_iter=int(args.shooting_max_iter),
            grid_size=int(args.shooting_grid_size),
        )
        shooting_diag = profile_diagnostics(np.asarray(shooting["y"]), np.asarray(shooting["p"]))
        for y_value, p_value, rho_value in zip(shooting["y"], shooting["p"], shooting["rho"]):
            shooting_profile_rows.append(
                {
                    "alpha": float(args.alpha),
                    "Mach": float(mach),
                    "y": float(y_value),
                    "p_real": float(np.real(p_value)),
                    "p_imag": float(np.imag(p_value)),
                    "p_abs": float(np.abs(p_value)),
                    "rho_real": float(np.real(rho_value)),
                    "rho_imag": float(np.imag(rho_value)),
                    "rho_abs": float(np.abs(rho_value)),
                }
            )

        solver = NotebookStyleDenseGEPSolver(
            alpha=float(args.alpha),
            Mach=float(mach),
            n_points=int(args.n_points),
            mapping_kind=args.mapping_kind,
            mapping_scale=args.mapping_scale,
            cubic_delta=args.cubic_delta,
            xi_max=args.xi_max,
        )
        raw_modes = extract_raw_modes_with_vectors(
            solver,
            max_abs_c=float(args.max_abs_c),
            positive_ci_only=bool(args.positive_ci_only),
        )
        ranked = sorted(raw_modes, key=lambda mode: abs(float(mode["cr"]) - target_cr))[: max(1, int(args.top_k_cr))]

        best_rank1 = ranked[0]
        best_to_shooting = min(
            ranked,
            key=lambda mode: candidate_score(mode, float(shooting["cr"]), float(shooting["ci"]), float(args.ci_weight)),
        )

        summary_rows.append(
            {
                "alpha": float(args.alpha),
                "Mach": float(mach),
                "n_points": int(args.n_points),
                "blumen_cr": target_cr,
                "blumen_ci": target_ci,
                "shooting_cr": float(shooting["cr"]),
                "shooting_ci": float(shooting["ci"]),
                "shooting_err_cr_vs_blumen": abs(float(shooting["cr"]) - target_cr),
                "shooting_err_ci_vs_blumen": abs(float(shooting["ci"]) - target_ci),
                "shooting_centroid_abs_y": shooting_diag["centroid_abs_y"],
                "shooting_spread_abs_y": shooting_diag["spread_abs_y"],
                "shooting_peak_y": shooting_diag["peak_y"],
                "best_rank1_gep_cr": float(best_rank1["cr"]),
                "best_rank1_gep_ci": float(best_rank1["ci"]),
                "best_rank1_err_cr_vs_blumen": abs(float(best_rank1["cr"]) - target_cr),
                "best_rank1_err_ci_vs_blumen": abs(float(best_rank1["ci"]) - target_ci),
                "best_rank1_err_cr_vs_shooting": abs(float(best_rank1["cr"]) - float(shooting["cr"])),
                "best_rank1_err_ci_vs_shooting": abs(float(best_rank1["ci"]) - float(shooting["ci"])),
                "best_to_shooting_gep_cr": float(best_to_shooting["cr"]),
                "best_to_shooting_gep_ci": float(best_to_shooting["ci"]),
                "best_to_shooting_score": candidate_score(best_to_shooting, float(shooting["cr"]), float(shooting["ci"]), float(args.ci_weight)),
            }
        )

        for rank, mode in enumerate(ranked, start=1):
            _, _, p_gep, rho_gep = normalize_gep_mode(np.asarray(mode["vector"]), solver.n_points, solver.Mach)
            diag = profile_diagnostics(np.asarray(solver.y), np.asarray(p_gep))
            gep_candidate_rows.append(
                {
                    "alpha": float(args.alpha),
                    "Mach": float(mach),
                    "n_points": int(args.n_points),
                    "rank_cr": int(rank),
                    "candidate_cr": float(mode["cr"]),
                    "candidate_ci": float(mode["ci"]),
                    "candidate_omega_i": float(mode["omega_i"]),
                    "err_cr_vs_blumen": abs(float(mode["cr"]) - target_cr),
                    "err_ci_vs_blumen": abs(float(mode["ci"]) - target_ci),
                    "err_cr_vs_shooting": abs(float(mode["cr"]) - float(shooting["cr"])),
                    "err_ci_vs_shooting": abs(float(mode["ci"]) - float(shooting["ci"])),
                    "score_to_shooting": candidate_score(mode, float(shooting["cr"]), float(shooting["ci"]), float(args.ci_weight)),
                    "centroid_abs_y": diag["centroid_abs_y"],
                    "spread_abs_y": diag["spread_abs_y"],
                    "peak_y": diag["peak_y"],
                }
            )
            for y_value, p_value, rho_value in zip(solver.y, p_gep, rho_gep):
                gep_profile_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "rank_cr": int(rank),
                        "candidate_cr": float(mode["cr"]),
                        "candidate_ci": float(mode["ci"]),
                        "y": float(y_value),
                        "p_real": float(np.real(p_value)),
                        "p_imag": float(np.imag(p_value)),
                        "p_abs": float(np.abs(p_value)),
                        "rho_real": float(np.real(rho_value)),
                        "rho_imag": float(np.imag(rho_value)),
                        "rho_abs": float(np.abs(rho_value)),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values("Mach").reset_index(drop=True)
    shooting_profiles_df = pd.DataFrame(shooting_profile_rows).sort_values(["Mach", "y"]).reset_index(drop=True)
    gep_candidates_df = pd.DataFrame(gep_candidate_rows).sort_values(["Mach", "rank_cr"]).reset_index(drop=True)
    gep_profiles_df = pd.DataFrame(gep_profile_rows).sort_values(["Mach", "rank_cr", "y"]).reset_index(drop=True)

    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    shooting_profiles_path = OUTPUT_DIR / f"{args.output_stem}_shooting_profiles.csv"
    candidates_path = OUTPUT_DIR / f"{args.output_stem}_gep_candidates.csv"
    gep_profiles_path = OUTPUT_DIR / f"{args.output_stem}_gep_profiles.csv"
    pdf_path = OUTPUT_DIR / f"{args.output_stem}_comparison.pdf"

    summary_df.to_csv(summary_path, index=False)
    shooting_profiles_df.to_csv(shooting_profiles_path, index=False)
    gep_candidates_df.to_csv(candidates_path, index=False)
    gep_profiles_df.to_csv(gep_profiles_path, index=False)
    plot_comparison_pages(summary_df, shooting_profiles_df, gep_profiles_df, pdf_path)

    print("Comparison summary:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {shooting_profiles_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {gep_profiles_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
