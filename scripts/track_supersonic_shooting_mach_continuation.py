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

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    DEFAULT_OUTPUT_DIR,
    build_blumen_targets,
    load_digitized_long,
)


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuation pure shooting en Mach pour verrouiller une branche supersonique."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--anchor-mach", type=float, default=None)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--anchor-cr-half-window", type=float, default=0.08)
    parser.add_argument("--anchor-ci-half-window", type=float, default=0.03)
    parser.add_argument("--cont-cr-half-window", type=float, default=0.03)
    parser.add_argument("--cont-ci-half-window", type=float, default=0.015)
    parser.add_argument("--retry-growth", type=float, default=1.75)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def extract_shooting_profile(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
    ln_p_start_right: float,
) -> dict[str, np.ndarray | float]:
    sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
        cr,
        ci,
        ln_p_start_right=ln_p_start_right,
    )
    if not (sol_left.success and sol_right_full.success):
        raise RuntimeError("Echec reconstruction du mode shooting.")

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
    rho = (solver.Mach**2) * p
    return {
        "y": y,
        "p": p,
        "rho": rho,
        "y_limit": float(y_limit),
    }


def profile_diagnostics(y: np.ndarray, p: np.ndarray, *, center_window: float = 8.0) -> dict[str, float]:
    p_abs = np.abs(p)
    norm = max(trapezoid_compat(p_abs, y), 1e-12)
    centroid_abs_y = float(trapezoid_compat(np.abs(y) * p_abs, y) / norm)
    spread_abs_y = float(np.sqrt(max(trapezoid_compat((y**2) * p_abs, y) / norm, 0.0)))
    peak_y = float(y[int(np.argmax(p_abs))])

    center_mask = np.abs(y) <= center_window
    if np.any(center_mask):
        y_c = y[center_mask]
        p_c = p_abs[center_mask]
        norm_c = max(trapezoid_compat(p_c, y_c), 1e-12)
        centroid_abs_y_c = float(trapezoid_compat(np.abs(y_c) * p_c, y_c) / norm_c)
        spread_abs_y_c = float(np.sqrt(max(trapezoid_compat((y_c**2) * p_c, y_c) / norm_c, 0.0)))
        peak_y_c = float(y_c[int(np.argmax(p_c))])
    else:
        centroid_abs_y_c = np.nan
        spread_abs_y_c = np.nan
        peak_y_c = np.nan

    return {
        "centroid_abs_y": centroid_abs_y,
        "spread_abs_y": spread_abs_y,
        "peak_y": peak_y,
        "centroid_abs_y_center8": centroid_abs_y_c,
        "spread_abs_y_center8": spread_abs_y_c,
        "peak_y_center8": peak_y_c,
    }


def solve_in_box(
    *,
    alpha: float,
    mach: float,
    match_y: float,
    use_mapping: bool,
    mapping_scale: float,
    cr_center: float,
    ci_center: float,
    cr_half_window: float,
    ci_half_window: float,
    max_iter: int,
    grid_size: int,
) -> tuple[Mstab17SupersonicSolver, object]:
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=match_y,
        use_mapping=use_mapping,
        mapping_scale=mapping_scale,
    )
    result = solver.solve(
        cr_min=max(0.0, cr_center - cr_half_window),
        cr_max=cr_center + cr_half_window,
        ci_min=max(1e-4, ci_center - ci_half_window),
        ci_max=ci_center + ci_half_window,
        max_iter=max_iter,
        grid_size=grid_size,
    )
    return solver, result


def continuation_solve(
    *,
    alpha: float,
    mach: float,
    match_y: float,
    use_mapping: bool,
    mapping_scale: float,
    cr_center: float,
    ci_center: float,
    cr_half_window: float,
    ci_half_window: float,
    retry_growth: float,
    max_retries: int,
    max_iter: int,
    grid_size: int,
) -> tuple[Mstab17SupersonicSolver, object, int, float, float]:
    current_cr_half = float(cr_half_window)
    current_ci_half = float(ci_half_window)
    best_solver = None
    best_result = None
    best_score = np.inf

    for retry_idx in range(max_retries + 1):
        solver, result = solve_in_box(
            alpha=alpha,
            mach=mach,
            match_y=match_y,
            use_mapping=use_mapping,
            mapping_scale=mapping_scale,
            cr_center=cr_center,
            ci_center=ci_center,
            cr_half_window=current_cr_half,
            ci_half_window=current_ci_half,
            max_iter=max_iter,
            grid_size=grid_size,
        )
        score = float(result.stage1_mismatch + result.stage2_mismatch)
        if score < best_score:
            best_solver = solver
            best_result = result
            best_score = score
        if bool(result.spectral_success and result.mode_success):
            return solver, result, retry_idx, current_cr_half, current_ci_half
        current_cr_half *= retry_growth
        current_ci_half *= retry_growth

    if best_solver is None or best_result is None:
        raise RuntimeError("Aucun resultat shooting retourne.")
    return best_solver, best_result, max_retries, current_cr_half / retry_growth, current_ci_half / retry_growth


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    mach = summary_df["Mach"].to_numpy(dtype=float)
    axes[0, 0].plot(mach, summary_df["blumen_cr"], marker="o", label="Blumen")
    axes[0, 0].plot(mach, summary_df["shooting_cr"], marker="s", label="shooting")
    axes[0, 0].set_title("c_r")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(mach, summary_df["blumen_ci"], marker="o", label="Blumen")
    axes[0, 1].plot(mach, summary_df["shooting_ci"], marker="s", label="shooting")
    axes[0, 1].set_title("c_i")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(mach, summary_df["shooting_err_cr_vs_blumen"], marker="o")
    axes[1, 0].set_title("|Δ c_r| vs Blumen")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(mach, summary_df["shooting_err_ci_vs_blumen"], marker="o")
    axes[1, 1].set_title("|Δ c_i| vs Blumen")
    axes[1, 1].grid(True, alpha=0.25)

    for ax in axes.ravel():
        ax.set_xlabel("Mach")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_profiles_pdf(summary_df: pd.DataFrame, profiles_df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        for mach in sorted(summary_df["Mach"].unique()):
            summary_row = summary_df[summary_df["Mach"] == mach].iloc[0]
            prof = profiles_df[profiles_df["Mach"] == mach].copy()

            fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
            axes[0].plot(prof["y"], prof["p_real"], color="black", linewidth=2.0)
            axes[0].set_title(r"Re($\hat{p}$)")
            axes[1].plot(prof["y"], prof["p_abs"], color="black", linewidth=2.0)
            axes[1].set_title(r"$|\hat{p}|$")
            axes[2].plot(prof["y"], prof["rho_real"], color="black", linewidth=2.0)
            axes[2].set_title(r"Re($\hat{\rho}$)")
            axes[2].set_xlabel("y")

            for ax in axes:
                ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                ax.grid(True, alpha=0.25)

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
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mach_values = [float(m) for m in args.mach_values]
    if args.anchor_mach is None:
        anchor_mach = float(mach_values[0])
    else:
        anchor_mach = float(args.anchor_mach)
    ordered_mach_values = sorted(mach_values)
    if anchor_mach not in ordered_mach_values:
        raise ValueError("anchor-mach doit appartenir a mach-values")
    if ordered_mach_values[0] != anchor_mach:
        raise ValueError("Cette continuation suppose que le premier Mach est l'ancre.")

    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    targets_df = build_blumen_targets(ordered_mach_values, float(args.alpha), cr_points, ci_points)

    summary_rows: list[dict] = []
    profile_rows: list[dict] = []

    previous_cr = None
    previous_ci = None

    for idx, mach in enumerate(ordered_mach_values):
        target = targets_df[targets_df["Mach"] == float(mach)].iloc[0]
        blumen_cr = float(target["blumen_cr"])
        blumen_ci = float(target["blumen_ci"])

        if idx == 0:
            cr_center = blumen_cr
            ci_center = blumen_ci
            cr_half = float(args.anchor_cr_half_window)
            ci_half = float(args.anchor_ci_half_window)
            source = "anchor_blumen"
        else:
            cr_center = float(previous_cr)
            ci_center = float(previous_ci)
            cr_half = float(args.cont_cr_half_window)
            ci_half = float(args.cont_ci_half_window)
            source = "continuation_previous"

        solver, result, retry_idx, used_cr_half, used_ci_half = continuation_solve(
            alpha=float(args.alpha),
            mach=float(mach),
            match_y=float(args.match_y),
            use_mapping=bool(args.use_mapping),
            mapping_scale=float(args.mapping_scale),
            cr_center=cr_center,
            ci_center=ci_center,
            cr_half_window=cr_half,
            ci_half_window=ci_half,
            retry_growth=float(args.retry_growth),
            max_retries=int(args.max_retries),
            max_iter=int(args.max_iter),
            grid_size=int(args.grid_size),
        )

        profile = extract_shooting_profile(
            solver,
            cr=float(result.cr),
            ci=float(result.ci),
            ln_p_start_right=float(result.ln_p_start_right),
        )
        diag = profile_diagnostics(np.asarray(profile["y"]), np.asarray(profile["p"]))

        summary_rows.append(
            {
                "alpha": float(args.alpha),
                "Mach": float(mach),
                "blumen_cr": blumen_cr,
                "blumen_ci": blumen_ci,
                "seed_source": source,
                "seed_cr_center": cr_center,
                "seed_ci_center": ci_center,
                "used_cr_half_window": used_cr_half,
                "used_ci_half_window": used_ci_half,
                "retry_index": int(retry_idx),
                "shooting_cr": float(result.cr),
                "shooting_ci": float(result.ci),
                "shooting_omega_i": float(result.omega_i),
                "shooting_err_cr_vs_blumen": abs(float(result.cr) - blumen_cr),
                "shooting_err_ci_vs_blumen": abs(float(result.ci) - blumen_ci),
                "stage1_mismatch": float(result.stage1_mismatch),
                "stage2_mismatch": float(result.stage2_mismatch),
                "spectral_success": bool(result.spectral_success),
                "mode_success": bool(result.mode_success),
                "success": bool(result.success),
                "y_limit": float(profile["y_limit"]),
                **diag,
            }
        )

        y = np.asarray(profile["y"])
        p = np.asarray(profile["p"])
        rho = np.asarray(profile["rho"])
        for y_i, p_i, rho_i in zip(y, p, rho):
            profile_rows.append(
                {
                    "alpha": float(args.alpha),
                    "Mach": float(mach),
                    "y": float(y_i),
                    "p_real": float(np.real(p_i)),
                    "p_imag": float(np.imag(p_i)),
                    "p_abs": float(np.abs(p_i)),
                    "rho_real": float(np.real(rho_i)),
                    "rho_imag": float(np.imag(rho_i)),
                    "rho_abs": float(np.abs(rho_i)),
                }
            )

        previous_cr = float(result.cr)
        previous_ci = float(result.ci)

    summary_df = pd.DataFrame(summary_rows)
    profiles_df = pd.DataFrame(profile_rows)

    print("Shooting continuation summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df.to_string(index=False))

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    profiles_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_profiles.csv"
    fig_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.png"
    pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_profiles.pdf"

    summary_df.to_csv(summary_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    plot_summary(summary_df, fig_path)
    plot_profiles_pdf(summary_df, profiles_df, pdf_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {profiles_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
