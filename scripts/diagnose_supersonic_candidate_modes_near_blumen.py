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
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    build_blumen_targets,
    load_digitized_long,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local des modes GEP supersoniques proches de Blumen en c_r."
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
    parser.add_argument("--top-k-cr", type=int, default=8)
    parser.add_argument("--cr-window", type=float, default=0.08)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--cr-match-tol", type=float, default=0.01)
    parser.add_argument("--ci-match-tol", type=float, default=0.015)
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


def classify_candidate(err_cr: float, err_ci: float, *, cr_match_tol: float, ci_match_tol: float) -> str:
    if err_cr <= cr_match_tol and err_ci <= ci_match_tol:
        return "matched_box"
    if err_cr <= cr_match_tol and err_ci > ci_match_tol:
        return "good_cr_low_ci"
    if err_cr <= 2.0 * cr_match_tol:
        return "near_cr_off_ci"
    return "off_family"


def plot_local_spectrum(candidates_df: pd.DataFrame, all_modes_df: pd.DataFrame, targets_df: pd.DataFrame, output_path: Path, *, cr_window: float) -> None:
    mach_values = sorted(targets_df["Mach"].unique())
    ncols = 2
    nrows = int(np.ceil(len(mach_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, mach in zip(axes_flat, mach_values):
        target = targets_df[targets_df["Mach"] == mach].iloc[0]
        sub_all = all_modes_df[all_modes_df["Mach"] == mach].copy()
        sub_all = sub_all[np.abs(sub_all["cr"] - float(target["blumen_cr"])) <= cr_window].copy()
        sub_cand = candidates_df[candidates_df["Mach"] == mach].copy().sort_values("rank_cr")

        ax.scatter(sub_all["cr"], sub_all["ci"], s=16, alpha=0.55, color="#94A3B8", label="raw modes")
        ax.scatter([target["blumen_cr"]], [target["blumen_ci"]], s=120, marker="*", color="black", label="Blumen", zorder=5)

        for _, row in sub_cand.iterrows():
            ax.scatter([row["candidate_cr"]], [row["candidate_ci"]], s=48, color="#DC2626", zorder=6)
            ax.text(
                float(row["candidate_cr"]),
                float(row["candidate_ci"]),
                f"#{int(row['rank_cr'])}",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        ax.set_title(
            f"alpha={float(target['alpha']):.3f}, M={float(mach):.3f}\n"
            f"Blumen=({float(target['blumen_cr']):.4f},{float(target['blumen_ci']):.4f})"
        )
        ax.set_xlabel("c_r")
        ax.set_ylabel("c_i")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(mach_values):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_pages(candidates_df: pd.DataFrame, profile_df: pd.DataFrame, targets_df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        for mach in sorted(targets_df["Mach"].unique()):
            target = targets_df[targets_df["Mach"] == mach].iloc[0]
            cand = candidates_df[candidates_df["Mach"] == mach].copy().sort_values("rank_cr")
            prof = profile_df[profile_df["Mach"] == mach].copy()

            fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

            for _, row in cand.iterrows():
                rank = int(row["rank_cr"])
                sub = prof[prof["rank_cr"] == rank].copy()
                label = (
                    f"#{rank} c=({float(row['candidate_cr']):.4f},{float(row['candidate_ci']):.4f}) "
                    f"{str(row['classification'])}"
                )
                axes[0].plot(sub["y"], sub["p_real"], linewidth=1.4, label=label)
                axes[1].plot(sub["y"], sub["p_abs"], linewidth=1.4, label=label)
                axes[2].plot(sub["y"], sub["rho_real"], linewidth=1.4, label=label)

            axes[0].set_title(r"Re($\hat{p}$)")
            axes[1].set_title(r"$|\hat{p}|$")
            axes[2].set_title(r"Re($\hat{\rho}$)")
            axes[2].set_xlabel("y")
            for ax in axes:
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7, frameon=False, ncol=1)

            fig.suptitle(
                f"Modes candidats pres de Blumen | alpha={float(target['alpha']):.3f}, M={float(mach):.3f}\n"
                f"Blumen=({float(target['blumen_cr']):.5f},{float(target['blumen_ci']):.5f})"
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

    candidate_rows: list[dict] = []
    profile_rows: list[dict] = []
    all_mode_rows: list[dict] = []
    summary_rows: list[dict] = []

    for mach in args.mach_values:
        target = targets_df[targets_df["Mach"] == float(mach)].iloc[0]
        target_cr = float(target["blumen_cr"])
        target_ci = float(target["blumen_ci"])

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

        for mode in raw_modes:
            all_mode_rows.append(
                {
                    "alpha": float(args.alpha),
                    "Mach": float(mach),
                    "n_points": int(args.n_points),
                    "cr": float(mode["cr"]),
                    "ci": float(mode["ci"]),
                    "err_cr": abs(float(mode["cr"]) - target_cr),
                    "err_ci": abs(float(mode["ci"]) - target_ci),
                    "distance_to_blumen": float(
                        np.sqrt((float(mode["cr"]) - target_cr) ** 2 + (float(args.ci_weight) * (float(mode["ci"]) - target_ci)) ** 2)
                    ),
                }
            )

        ranked = sorted(raw_modes, key=lambda mode: (abs(float(mode["cr"]) - target_cr), abs(float(mode["ci"]) - target_ci)))
        selected = ranked[: max(1, int(args.top_k_cr))]

        for rank, mode in enumerate(selected, start=1):
            err_cr = abs(float(mode["cr"]) - target_cr)
            err_ci = abs(float(mode["ci"]) - target_ci)
            distance = float(np.sqrt((float(mode["cr"]) - target_cr) ** 2 + (float(args.ci_weight) * (float(mode["ci"]) - target_ci)) ** 2))
            classification = classify_candidate(
                err_cr,
                err_ci,
                cr_match_tol=float(args.cr_match_tol),
                ci_match_tol=float(args.ci_match_tol),
            )

            candidate_rows.append(
                {
                    "alpha": float(args.alpha),
                    "Mach": float(mach),
                    "n_points": int(args.n_points),
                    "rank_cr": int(rank),
                    "candidate_cr": float(mode["cr"]),
                    "candidate_ci": float(mode["ci"]),
                    "candidate_omega_i": float(mode["omega_i"]),
                    "err_cr": float(err_cr),
                    "err_ci": float(err_ci),
                    "distance_to_blumen": float(distance),
                    "classification": classification,
                    "blumen_cr": target_cr,
                    "blumen_ci": target_ci,
                }
            )

            _, _, p, rho = normalize_gep_mode(np.asarray(mode["vector"]), solver.n_points, solver.Mach)
            for y_value, p_value, rho_value in zip(solver.y, p, rho):
                profile_rows.append(
                    {
                        "alpha": float(args.alpha),
                        "Mach": float(mach),
                        "n_points": int(args.n_points),
                        "rank_cr": int(rank),
                        "y": float(y_value),
                        "p_real": float(np.real(p_value)),
                        "p_imag": float(np.imag(p_value)),
                        "p_abs": float(np.abs(p_value)),
                        "rho_real": float(np.real(rho_value)),
                        "rho_imag": float(np.imag(rho_value)),
                        "rho_abs": float(np.abs(rho_value)),
                    }
                )

        best_cr = selected[0] if selected else None
        best_dist = min(raw_modes, key=lambda mode: np.sqrt((float(mode["cr"]) - target_cr) ** 2 + (float(args.ci_weight) * (float(mode["ci"]) - target_ci)) ** 2)) if raw_modes else None
        summary_rows.append(
            {
                "alpha": float(args.alpha),
                "Mach": float(mach),
                "n_points": int(args.n_points),
                "n_raw_modes": int(len(raw_modes)),
                "blumen_cr": target_cr,
                "blumen_ci": target_ci,
                "best_cr_candidate_cr": np.nan if best_cr is None else float(best_cr["cr"]),
                "best_cr_candidate_ci": np.nan if best_cr is None else float(best_cr["ci"]),
                "best_cr_err_cr": np.nan if best_cr is None else abs(float(best_cr["cr"]) - target_cr),
                "best_cr_err_ci": np.nan if best_cr is None else abs(float(best_cr["ci"]) - target_ci),
                "nearest_full_candidate_cr": np.nan if best_dist is None else float(best_dist["cr"]),
                "nearest_full_candidate_ci": np.nan if best_dist is None else float(best_dist["ci"]),
                "nearest_full_err_cr": np.nan if best_dist is None else abs(float(best_dist["cr"]) - target_cr),
                "nearest_full_err_ci": np.nan if best_dist is None else abs(float(best_dist["ci"]) - target_ci),
            }
        )

    candidates_df = pd.DataFrame(candidate_rows).sort_values(["Mach", "rank_cr"]).reset_index(drop=True)
    profiles_df = pd.DataFrame(profile_rows).sort_values(["Mach", "rank_cr", "y"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach"]).reset_index(drop=True)
    all_modes_df = pd.DataFrame(all_mode_rows).sort_values(["Mach", "distance_to_blumen", "cr", "ci"]).reset_index(drop=True)

    candidates_path = OUTPUT_DIR / f"{args.output_stem}_candidates.csv"
    profiles_path = OUTPUT_DIR / f"{args.output_stem}_profiles.csv"
    summary_path = OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    zoom_path = OUTPUT_DIR / f"{args.output_stem}_local_spectrum.png"
    pdf_path = OUTPUT_DIR / f"{args.output_stem}_candidate_modes.pdf"

    candidates_df.to_csv(candidates_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_local_spectrum(candidates_df, all_modes_df, targets_df, zoom_path, cr_window=float(args.cr_window))
    plot_candidate_pages(candidates_df, profiles_df, targets_df, pdf_path)

    print("Blumen targets:")
    print(targets_df.to_string(index=False))
    print("\nCandidate summary:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {candidates_path}")
    print(f"Wrote {profiles_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {zoom_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
