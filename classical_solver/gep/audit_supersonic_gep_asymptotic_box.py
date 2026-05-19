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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def normalize_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit GEP supersonique avec boite finie calibree par kappa/q asymptotiques."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--n-points", type=int, default=401)
    parser.add_argument("--auto-n-points", action="store_true")
    parser.add_argument("--max-n-points", type=int, default=4001)
    parser.add_argument("--mapping-L", type=float, default=3.0)
    parser.add_argument("--xi-max", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--positive-cr-only", action="store_true")
    parser.add_argument("--decay-box-factor", type=float, default=4.0)
    parser.add_argument("--oscillation-box-factor", type=float, default=4.0)
    parser.add_argument("--max-wavelength-to-decay-ratio-for-q-box", type=float, default=8.0)
    parser.add_argument("--min-half-box", type=float, default=8.0)
    parser.add_argument("--max-half-box", type=float, default=120.0)
    parser.add_argument("--force-symmetric-half-box", action="store_true")
    parser.add_argument("--q-floor", type=float, default=1e-4)
    parser.add_argument("--core-half-width", type=float, default=1.0)
    parser.add_argument("--min-points-core", type=int, default=120)
    parser.add_argument("--center-half-width", type=float, default=4.0)
    parser.add_argument("--min-points-center", type=int, default=300)
    parser.add_argument("--tail-points-per-wavelength", type=float, default=24.0)
    parser.add_argument("--tail-fraction-for-spacing", type=float, default=0.15)
    parser.add_argument("--target-cr", type=float, default=None)
    parser.add_argument("--target-ci", type=float, default=None)
    parser.add_argument("--shooting-cr-max", type=float, default=0.70)
    parser.add_argument("--shooting-ci-max", type=float, default=0.12)
    parser.add_argument("--shooting-max-iter", type=int, default=12)
    parser.add_argument("--shooting-grid-size", type=int, default=4)
    parser.add_argument("--u-zoom-half-width", type=float, default=2.0)
    parser.add_argument("--output-stem", type=str, default="supersonic_gep_asymptotic_box")
    return parser


def get_target_mode(args: argparse.Namespace) -> tuple[float, float, str, Mstab17SupersonicSolver | None]:
    if args.target_cr is not None and args.target_ci is not None:
        return float(args.target_cr), float(args.target_ci), "user", None

    shooting_solver = Mstab17SupersonicSolver(alpha=float(args.alpha), Mach=float(args.mach))
    shooting = shooting_solver.solve(
        cr_min=0.0,
        cr_max=float(args.shooting_cr_max),
        ci_min=1e-3,
        ci_max=float(args.shooting_ci_max),
        max_iter=int(args.shooting_max_iter),
        grid_size=int(args.shooting_grid_size),
    )
    return float(shooting.cr), float(shooting.ci), "shooting_mstab17", shooting_solver


def build_box_from_asymptotics(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    decay_box_factor: float,
    oscillation_box_factor: float,
    max_wavelength_to_decay_ratio_for_q_box: float,
    min_half_box: float,
    max_half_box: float,
    force_symmetric_half_box: bool,
    q_floor: float,
) -> dict[str, float]:
    solver = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(mach))
    gamma_left, gamma_right = solver.asymptotic_gammas(float(cr), float(ci))

    kappa_left = max(float(gamma_left.real), 1e-10)
    kappa_right = max(float(-gamma_right.real), 1e-10)
    q_left = float(gamma_left.imag)
    q_right = float(gamma_right.imag)
    q_left_abs = abs(q_left)
    q_right_abs = abs(q_right)

    left_decay_length = 1.0 / kappa_left
    right_decay_length = 1.0 / kappa_right
    left_decay_extent = float(decay_box_factor) * left_decay_length
    right_decay_extent = float(decay_box_factor) * right_decay_length

    left_wavelength = np.inf if q_left_abs < q_floor else (2.0 * np.pi / q_left_abs)
    right_wavelength = np.inf if q_right_abs < q_floor else (2.0 * np.pi / q_right_abs)

    left_q_box_enabled = bool(
        np.isfinite(left_wavelength)
        and left_wavelength <= float(max_wavelength_to_decay_ratio_for_q_box) * left_decay_length
    )
    right_q_box_enabled = bool(
        np.isfinite(right_wavelength)
        and right_wavelength <= float(max_wavelength_to_decay_ratio_for_q_box) * right_decay_length
    )

    left_osc_extent = 0.0 if not left_q_box_enabled else float(oscillation_box_factor) * left_wavelength
    right_osc_extent = 0.0 if not right_q_box_enabled else float(oscillation_box_factor) * right_wavelength

    left_uncapped = max(float(min_half_box), left_decay_extent, left_osc_extent)
    right_uncapped = max(float(min_half_box), right_decay_extent, right_osc_extent)
    left_half_box = min(left_uncapped, float(max_half_box))
    right_half_box = min(right_uncapped, float(max_half_box))
    symmetric_half_box = max(left_half_box, right_half_box)
    if force_symmetric_half_box:
        left_half_box = symmetric_half_box
        right_half_box = symmetric_half_box

    def dominant_source(decay_extent: float, osc_extent: float, min_half: float) -> str:
        dominant = max(min_half, decay_extent, osc_extent)
        if dominant == osc_extent and osc_extent > 0.0:
            return "q"
        if dominant == decay_extent:
            return "kappa"
        return "min_half_box"

    return {
        "gamma_left_real": float(gamma_left.real),
        "gamma_left_imag": float(gamma_left.imag),
        "gamma_right_real": float(gamma_right.real),
        "gamma_right_imag": float(gamma_right.imag),
        "kappa_left": float(kappa_left),
        "kappa_right": float(kappa_right),
        "q_left": float(q_left),
        "q_right": float(q_right),
        "left_decay_length": float(left_decay_length),
        "right_decay_length": float(right_decay_length),
        "left_wavelength": float(left_wavelength) if np.isfinite(left_wavelength) else np.nan,
        "right_wavelength": float(right_wavelength) if np.isfinite(right_wavelength) else np.nan,
        "left_box_driver": dominant_source(left_decay_extent, left_osc_extent, float(min_half_box)),
        "right_box_driver": dominant_source(right_decay_extent, right_osc_extent, float(min_half_box)),
        "left_q_box_enabled": bool(left_q_box_enabled),
        "right_q_box_enabled": bool(right_q_box_enabled),
        "force_symmetric_half_box": bool(force_symmetric_half_box),
        "symmetric_half_box": float(symmetric_half_box),
        "left_box_capped": bool(left_half_box < left_uncapped),
        "right_box_capped": bool(right_half_box < right_uncapped),
        "left_half_box": float(left_half_box),
        "right_half_box": float(right_half_box),
        "y_min": float(-left_half_box),
        "y_max": float(right_half_box),
    }


def compute_mesh_diagnostics(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    center_idx = int(np.argmin(np.abs(y)))
    if 0 < center_idx < len(y) - 1:
        dy_center = 0.5 * (y[center_idx + 1] - y[center_idx - 1])
    elif len(dy) > 0:
        dy_center = float(np.min(np.abs(dy)))
    else:
        dy_center = np.nan
    return {
        "dy_center": float(dy_center),
        "n_points_abs_y_le_1": int(np.sum(np.abs(y) <= 1.0)),
        "n_points_abs_y_le_2": int(np.sum(np.abs(y) <= 2.0)),
        "n_points_abs_y_le_4": int(np.sum(np.abs(y) <= 4.0)),
        "n_points_abs_y_le_8": int(np.sum(np.abs(y) <= 8.0)),
    }


def compute_resolution_targets(
    *,
    y: np.ndarray,
    left_wavelength: float,
    right_wavelength: float,
    core_half_width: float,
    min_points_core: int,
    center_half_width: float,
    min_points_center: int,
    tail_points_per_wavelength: float,
    tail_fraction_for_spacing: float,
) -> dict[str, float | int | bool]:
    y = np.asarray(y, dtype=float)
    dy = np.abs(np.diff(y))
    n = len(y)
    tail_count = max(1, int(np.ceil(float(tail_fraction_for_spacing) * max(n - 1, 1))))
    left_tail_dy_max = float(np.max(dy[:tail_count])) if len(dy) > 0 else np.nan
    right_tail_dy_max = float(np.max(dy[-tail_count:])) if len(dy) > 0 else np.nan

    n_core = int(np.sum(np.abs(y) <= float(core_half_width)))
    n_center = int(np.sum(np.abs(y) <= float(center_half_width)))

    left_tail_dy_target = (
        np.nan if not np.isfinite(left_wavelength) else float(left_wavelength) / float(tail_points_per_wavelength)
    )
    right_tail_dy_target = (
        np.nan if not np.isfinite(right_wavelength) else float(right_wavelength) / float(tail_points_per_wavelength)
    )

    left_tail_ok = True if not np.isfinite(left_tail_dy_target) else left_tail_dy_max <= left_tail_dy_target
    right_tail_ok = True if not np.isfinite(right_tail_dy_target) else right_tail_dy_max <= right_tail_dy_target
    core_ok = n_core >= int(min_points_core)
    center_ok = n_center >= int(min_points_center)

    return {
        "left_tail_dy_max": left_tail_dy_max,
        "right_tail_dy_max": right_tail_dy_max,
        "left_tail_dy_target": float(left_tail_dy_target) if np.isfinite(left_tail_dy_target) else np.nan,
        "right_tail_dy_target": float(right_tail_dy_target) if np.isfinite(right_tail_dy_target) else np.nan,
        "n_points_abs_y_le_core": n_core,
        "n_points_abs_y_le_center": n_center,
        "core_ok": bool(core_ok),
        "center_ok": bool(center_ok),
        "left_tail_ok": bool(left_tail_ok),
        "right_tail_ok": bool(right_tail_ok),
        "all_resolution_constraints_ok": bool(core_ok and center_ok and left_tail_ok and right_tail_ok),
    }


def choose_n_points(
    *,
    alpha: float,
    mach: float,
    mapping_L: float,
    xi_max: float,
    box: dict[str, float],
    initial_n_points: int,
    max_n_points: int,
    core_half_width: float,
    min_points_core: int,
    center_half_width: float,
    min_points_center: int,
    tail_points_per_wavelength: float,
    tail_fraction_for_spacing: float,
) -> tuple[int, dict[str, float | int | bool]]:
    n_points = max(11, int(initial_n_points))
    if n_points % 2 == 0:
        n_points += 1

    last_diag: dict[str, float | int | bool] | None = None
    while True:
        solver = NotebookStyleDenseGEPSolver(
            alpha=float(alpha),
            Mach=float(mach),
            n_points=int(n_points),
            mapping_kind="finite_box",
            mapping_scale=float(mapping_L),
            xi_max=float(xi_max),
            box_y_min=float(box["y_min"]),
            box_y_max=float(box["y_max"]),
        )
        last_diag = compute_resolution_targets(
            y=solver.y,
            left_wavelength=float(box["left_wavelength"]),
            right_wavelength=float(box["right_wavelength"]),
            core_half_width=float(core_half_width),
            min_points_core=int(min_points_core),
            center_half_width=float(center_half_width),
            min_points_center=int(min_points_center),
            tail_points_per_wavelength=float(tail_points_per_wavelength),
            tail_fraction_for_spacing=float(tail_fraction_for_spacing),
        )
        if bool(last_diag["all_resolution_constraints_ok"]) or n_points >= int(max_n_points):
            last_diag["n_points_selected"] = int(n_points)
            last_diag["n_points_hit_cap"] = bool(n_points >= int(max_n_points))
            return int(n_points), last_diag

        next_n = min(int(max_n_points), int(np.ceil(1.35 * n_points)))
        if next_n <= n_points:
            next_n = n_points + 2
        if next_n % 2 == 0:
            next_n += 1
        n_points = next_n


def plot_top_modes_pdf(
    *,
    solver: NotebookStyleDenseGEPSolver,
    modes: list[dict],
    output_path: Path,
) -> None:
    field_titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]
    with PdfPages(output_path) as pdf:
        for rank, mode in enumerate(modes, start=1):
            u, v, p, rho = normalize_mode(mode["vector"], solver.n_points, solver.Mach)
            fields = [rho, u, v, p]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            for ax, field, title in zip(axes.flat, fields, field_titles):
                ax.plot(solver.y, np.real(field), color="tab:blue", linewidth=1.0, marker="o", markersize=2.0, markevery=max(1, len(solver.y) // 80), label="Re")
                ax.plot(solver.y, np.imag(field), color="tab:orange", linewidth=1.0, linestyle="--", marker="x", markersize=2.0, markevery=max(1, len(solver.y) // 80), label="Im")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(frameon=False, fontsize=8)
            fig.suptitle(
                f"GEP asymptotic-box | rank={rank}, alpha={solver.alpha:.3f}, M={solver.Mach:.3f}, "
                f"c={mode['cr']:.5f}+i{mode['ci']:.5f}"
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def plot_u_zoom_pdf(
    *,
    solver: NotebookStyleDenseGEPSolver,
    modes: list[dict],
    half_width: float,
    output_path: Path,
) -> None:
    with PdfPages(output_path) as pdf:
        for rank, mode in enumerate(modes, start=1):
            u, _, p, rho = normalize_mode(mode["vector"], solver.n_points, solver.Mach)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(solver.y, np.real(u), color="tab:blue", linewidth=1.0, marker="o", markersize=2.0, markevery=max(1, len(solver.y) // 80), label="Re")
            ax.plot(solver.y, np.imag(u), color="tab:orange", linewidth=1.0, linestyle="--", marker="x", markersize=2.0, markevery=max(1, len(solver.y) // 80), label="Im")
            ax.set_xlim(-float(half_width), float(half_width))
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
            ax.set_title(
                f"Zoom on u-hat | rank={rank}, alpha={solver.alpha:.3f}, M={solver.Mach:.3f}, "
                f"c={mode['cr']:.5f}+i{mode['ci']:.5f}"
            )
            ax.set_xlabel("y")
            ax.set_ylabel("u-hat normalized")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_cr, target_ci, target_source, _ = get_target_mode(args)
    box = build_box_from_asymptotics(
        alpha=float(args.alpha),
        mach=float(args.mach),
        cr=target_cr,
        ci=target_ci,
        decay_box_factor=float(args.decay_box_factor),
        oscillation_box_factor=float(args.oscillation_box_factor),
        max_wavelength_to_decay_ratio_for_q_box=float(args.max_wavelength_to_decay_ratio_for_q_box),
        min_half_box=float(args.min_half_box),
        max_half_box=float(args.max_half_box),
        force_symmetric_half_box=bool(args.force_symmetric_half_box),
        q_floor=float(args.q_floor),
    )

    auto_n_diag: dict[str, float | int | bool] = {}
    n_points = int(args.n_points)
    if args.auto_n_points:
        n_points, auto_n_diag = choose_n_points(
            alpha=float(args.alpha),
            mach=float(args.mach),
            mapping_L=float(args.mapping_L),
            xi_max=float(args.xi_max),
            box=box,
            initial_n_points=int(args.n_points),
            max_n_points=int(args.max_n_points),
            core_half_width=float(args.core_half_width),
            min_points_core=int(args.min_points_core),
            center_half_width=float(args.center_half_width),
            min_points_center=int(args.min_points_center),
            tail_points_per_wavelength=float(args.tail_points_per_wavelength),
            tail_fraction_for_spacing=float(args.tail_fraction_for_spacing),
        )

    solver = NotebookStyleDenseGEPSolver(
        alpha=float(args.alpha),
        Mach=float(args.mach),
        n_points=int(n_points),
        mapping_kind="finite_box",
        mapping_scale=float(args.mapping_L),
        xi_max=float(args.xi_max),
        box_y_min=float(box["y_min"]),
        box_y_max=float(box["y_max"]),
    )

    mesh_diag = compute_mesh_diagnostics(solver.y)
    if not auto_n_diag:
        auto_n_diag = compute_resolution_targets(
            y=solver.y,
            left_wavelength=float(box["left_wavelength"]),
            right_wavelength=float(box["right_wavelength"]),
            core_half_width=float(args.core_half_width),
            min_points_core=int(args.min_points_core),
            center_half_width=float(args.center_half_width),
            min_points_center=int(args.min_points_center),
            tail_points_per_wavelength=float(args.tail_points_per_wavelength),
            tail_fraction_for_spacing=float(args.tail_fraction_for_spacing),
        )
        auto_n_diag["n_points_selected"] = int(n_points)
        auto_n_diag["n_points_hit_cap"] = False

    modes = solver.finite_modes()
    if args.positive_cr_only:
        modes = [mode for mode in modes if mode["cr"] >= -1e-10]
    modes = sorted(modes, key=lambda mode: mode["ci"], reverse=True)
    top_modes = modes[: max(1, int(args.top_k))]

    summary_rows = []
    for rank, mode in enumerate(top_modes, start=1):
        summary_rows.append(
            {
                "rank_by_ci": rank,
                "alpha": float(args.alpha),
                "Mach": float(args.mach),
                "n_points": int(n_points),
                "mapping_kind": "finite_box",
                "mapping_L": float(args.mapping_L),
                "xi_max": float(args.xi_max),
                "target_source": target_source,
                "target_cr": float(target_cr),
                "target_ci": float(target_ci),
                "cr": float(mode["cr"]),
                "ci": float(mode["ci"]),
                "omega_i": float(mode["omega_i"]),
                "abs_cr": float(mode["abs_cr"]),
                "distance_to_target": float(np.hypot(mode["cr"] - target_cr, mode["ci"] - target_ci)),
            }
        )

    box_row = {
        "alpha": float(args.alpha),
        "Mach": float(args.mach),
        "n_points": int(n_points),
        "mapping_kind": "finite_box",
        "mapping_L": float(args.mapping_L),
        "xi_max": float(args.xi_max),
        "auto_n_points": bool(args.auto_n_points),
        "initial_n_points": int(args.n_points),
        "max_n_points": int(args.max_n_points),
        "core_half_width": float(args.core_half_width),
        "min_points_core": int(args.min_points_core),
        "center_half_width": float(args.center_half_width),
        "min_points_center": int(args.min_points_center),
        "tail_points_per_wavelength": float(args.tail_points_per_wavelength),
        "tail_fraction_for_spacing": float(args.tail_fraction_for_spacing),
        "target_source": target_source,
        "target_cr": float(target_cr),
        "target_ci": float(target_ci),
        **box,
        **mesh_diag,
        **auto_n_diag,
    }

    stem = f"{args.output_stem}_a{args.alpha:.3f}_m{args.mach:.3f}_N{n_points}_L{args.mapping_L:.2f}"
    summary_csv = OUTPUT_DIR / f"{stem}_summary.csv"
    box_csv = OUTPUT_DIR / f"{stem}_box.csv"
    top_modes_pdf = OUTPUT_DIR / f"{stem}_modes.pdf"
    u_zoom_pdf = OUTPUT_DIR / f"{stem}_u_zoom.pdf"

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame([box_row]).to_csv(box_csv, index=False)
    plot_top_modes_pdf(solver=solver, modes=top_modes, output_path=top_modes_pdf)
    plot_u_zoom_pdf(solver=solver, modes=top_modes, half_width=float(args.u_zoom_half_width), output_path=u_zoom_pdf)

    print("Asymptotic box calibration:")
    print(pd.DataFrame([box_row]).to_string(index=False))
    print("\nTop GEP modes ranked by c_i:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Box CSV: {box_csv}")
    print(f"Modes PDF: {top_modes_pdf}")
    print(f"u-zoom PDF: {u_zoom_pdf}")


if __name__ == "__main__":
    main()
