from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


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


DEFAULT_REFERENCE_SUMMARY = DEFAULT_OUTPUT_DIR / "supersonic_shooting_multistart_a020_m120_130_summary.csv"


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit des conditions d'eigenvaleur et des asymptotiques du shooting supersonique contre Blumen."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--amplitude-match-y", type=float, default=None)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--map-cr-half-window", type=float, default=0.05)
    parser.add_argument("--map-ci-half-window", type=float, default=0.02)
    parser.add_argument("--map-n-cr", type=int, default=25)
    parser.add_argument("--map-n-ci", type=int, default=21)
    parser.add_argument("--tail-fraction", type=float, default=0.80)
    parser.add_argument("--match-band-width", type=float, default=0.20)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def infer_reference_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidate_pairs = [
        ("best_shooting_cr", "best_shooting_ci"),
        ("shooting_cr", "shooting_ci"),
        ("cr", "ci"),
    ]
    for cr_col, ci_col in candidate_pairs:
        if {cr_col, ci_col}.issubset(df.columns):
            return cr_col, ci_col
    raise ValueError(
        "Impossible d'identifier les colonnes de branche de reference dans le summary CSV. "
        "Colonnes attendues parmi best_shooting_cr/best_shooting_ci, shooting_cr/shooting_ci ou cr/ci."
    )


def load_reference_branch(summary_path: Path | None) -> pd.DataFrame:
    if summary_path is None or not summary_path.exists():
        return pd.DataFrame(columns=["Mach", "reference_cr", "reference_ci"])
    df = pd.read_csv(summary_path)
    if "Mach" not in df.columns:
        raise ValueError(f"{summary_path} doit contenir une colonne Mach.")
    cr_col, ci_col = infer_reference_columns(df)
    out = df[["Mach", cr_col, ci_col]].copy()
    out = out.rename(columns={cr_col: "reference_cr", ci_col: "reference_ci"})
    return out.dropna(subset=["Mach", "reference_cr", "reference_ci"]).reset_index(drop=True)


def classify_relative_regime(mach: float, u_inf: float, c: complex, *, tol: float = 1e-6) -> tuple[float, str]:
    rel_mach = float(mach * abs(u_inf - c))
    if rel_mach < 1.0 - tol:
        return rel_mach, "subsonic_relative"
    if rel_mach > 1.0 + tol:
        return rel_mach, "supersonic_relative"
    return rel_mach, "sonic_relative"


def fit_tail_slopes(y: np.ndarray, ln_p: np.ndarray, phi: np.ndarray, *, side: str, y_limit: float, tail_fraction: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    ln_p = np.asarray(ln_p, dtype=float)
    phi = np.unwrap(np.asarray(phi, dtype=float))
    if side == "left":
        mask = y <= (-tail_fraction * y_limit)
    else:
        mask = y >= (tail_fraction * y_limit)
    if int(np.count_nonzero(mask)) < 8:
        if side == "left":
            mask = y <= (-0.6 * y_limit)
        else:
            mask = y >= (0.6 * y_limit)
    if int(np.count_nonzero(mask)) < 8:
        return np.nan, np.nan
    y_fit = y[mask]
    ln_fit = ln_p[mask]
    phi_fit = phi[mask]
    order = np.argsort(y_fit)
    y_fit = y_fit[order]
    ln_fit = ln_fit[order]
    phi_fit = phi_fit[order]
    slope_ln = float(np.polyfit(y_fit, ln_fit, deg=1)[0])
    slope_phi = float(np.polyfit(y_fit, phi_fit, deg=1)[0])
    return slope_ln, slope_phi


def fast_stage1_mismatch(solver: Mstab17SupersonicSolver, cr: float, ci: float) -> float:
    if cr < 0.0 or ci <= 0.0:
        return 1e6

    gamma_left_inf, gamma_right_inf = solver.asymptotic_gammas(cr, ci)
    y_limit = solver.estimate_y_limit(cr, ci)
    init_left = [gamma_left_inf.real, gamma_left_inf.imag, solver.ln_p_start_left, 0.0]
    init_right = [gamma_right_inf.real, gamma_right_inf.imag, solver.ln_p_start_left, 0.0]

    if solver.use_mapping:
        xi_left = solver.y_to_xi(-y_limit)
        xi_right = solver.y_to_xi(y_limit)
        xi_match = solver.y_to_xi(solver.match_y)
        sol_left = solve_ivp(
            solver.riccati_system_real_split_xi,
            (xi_left, xi_match),
            init_left,
            t_eval=np.array([xi_match]),
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
        sol_right = solve_ivp(
            solver.riccati_system_real_split_xi,
            (xi_right, xi_match),
            init_right,
            t_eval=np.array([xi_match]),
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
    else:
        sol_left = solve_ivp(
            solver.riccati_system_real_split,
            (-y_limit, solver.match_y),
            init_left,
            t_eval=np.array([solver.match_y]),
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
        sol_right = solve_ivp(
            solver.riccati_system_real_split,
            (y_limit, solver.match_y),
            init_right,
            t_eval=np.array([solver.match_y]),
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )

    if not (sol_left.success and sol_right.success):
        return 1e6
    delta_k = float(sol_left.y[0, -1] - sol_right.y[0, -1])
    delta_q = float(sol_left.y[1, -1] - sol_right.y[1, -1])
    return float(np.hypot(delta_k, delta_q))


def integrate_branches_to_match(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
    ln_p_start_right: float,
) -> tuple[object, object, float]:
    gamma_left_inf, gamma_right_inf = solver.asymptotic_gammas(cr, ci)
    y_limit = solver.estimate_y_limit(cr, ci)

    init_left = [gamma_left_inf.real, gamma_left_inf.imag, solver.ln_p_start_left, 0.0]
    init_right = [gamma_right_inf.real, gamma_right_inf.imag, ln_p_start_right, 0.0]

    if solver.use_mapping:
        xi_left = solver.y_to_xi(-y_limit)
        xi_right = solver.y_to_xi(y_limit)
        xi_match = solver.y_to_xi(solver.match_y)
        xi_eval_left = np.linspace(xi_left, xi_match, 2500)
        xi_eval_right = np.linspace(xi_right, xi_match, 2500)

        sol_left = solve_ivp(
            solver.riccati_system_real_split_xi,
            (xi_left, xi_match),
            init_left,
            t_eval=xi_eval_left,
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
        sol_right = solve_ivp(
            solver.riccati_system_real_split_xi,
            (xi_right, xi_match),
            init_right,
            t_eval=xi_eval_right,
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
        sol_left.t = solver.xi_to_y(sol_left.t)
        sol_right.t = solver.xi_to_y(sol_right.t)
    else:
        y_eval_left = np.linspace(-y_limit, solver.match_y, 2500)
        y_eval_right = np.linspace(y_limit, solver.match_y, 2500)
        sol_left = solve_ivp(
            solver.riccati_system_real_split,
            (-y_limit, solver.match_y),
            init_left,
            t_eval=y_eval_left,
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
        sol_right = solve_ivp(
            solver.riccati_system_real_split,
            (y_limit, solver.match_y),
            init_right,
            t_eval=y_eval_right,
            args=(cr, ci),
            method="RK45",
            rtol=solver.rtol,
            atol=solver.atol,
        )
    return sol_left, sol_right, float(y_limit)


def reconstruct_branches(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
    ln_p_start_right: float,
) -> dict[str, np.ndarray | float]:
    sol_left, sol_right, y_limit = integrate_branches_to_match(
        solver,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_p_start_right,
    )
    if not (sol_left.success and sol_right.success):
        raise RuntimeError("Echec de reconstruction des branches jusqu'au point de raccord.")

    y_left = np.asarray(sol_left.t, dtype=float)
    y_right = np.asarray(sol_right.t, dtype=float)
    k_left, q_left, ln_p_left, phi_left = [np.asarray(comp, dtype=float) for comp in sol_left.y]
    k_right, q_right, ln_p_right, phi_right = [np.asarray(comp, dtype=float) for comp in sol_right.y]

    phase_shift = float(phi_left[-1] - phi_right[-1])
    p_left = np.exp(ln_p_left + 1j * phi_left)
    p_right = np.exp(ln_p_right + 1j * (phi_right + phase_shift))

    y_right_asc = y_right[::-1]
    p_right_asc = p_right[::-1]
    k_right_asc = k_right[::-1]
    q_right_asc = q_right[::-1]

    y_joined = np.concatenate([y_left, y_right_asc[1:]])
    p_joined = np.concatenate([p_left, p_right_asc[1:]])
    k_joined = np.concatenate([k_left, k_right_asc[1:]])
    q_joined = np.concatenate([q_left, q_right_asc[1:]])

    scale = max(np.max(np.abs(np.real(p_joined))), np.max(np.abs(np.imag(p_joined))), 1e-12)
    p_joined = p_joined / scale
    p_left = p_left / scale
    p_right = p_right / scale

    return {
        "y_left": y_left,
        "y_right": y_right,
        "k_left": k_left,
        "q_left": q_left,
        "ln_p_left": ln_p_left,
        "phi_left": phi_left,
        "k_right": k_right,
        "q_right": q_right,
        "ln_p_right": ln_p_right,
        "phi_right": phi_right + phase_shift,
        "p_left": p_left,
        "p_right": p_right,
        "y_joined": y_joined,
        "p_joined": p_joined,
        "k_joined": k_joined,
        "q_joined": q_joined,
        "y_limit": y_limit,
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
        center_mass_fraction = float(trapezoid_compat(p_c, y_c) / norm)
    else:
        centroid_abs_y_c = np.nan
        spread_abs_y_c = np.nan
        peak_y_c = np.nan
        center_mass_fraction = np.nan

    return {
        "centroid_abs_y": centroid_abs_y,
        "spread_abs_y": spread_abs_y,
        "peak_y": peak_y,
        "centroid_abs_y_center8": centroid_abs_y_c,
        "spread_abs_y_center8": spread_abs_y_c,
        "peak_y_center8": peak_y_c,
        "center8_mass_fraction": center_mass_fraction,
    }


def relative_ode_residual(
    *,
    y: np.ndarray,
    p: np.ndarray,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    match_y: float,
    match_band_width: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=complex)
    p_y = np.gradient(p, y, edge_order=2)
    p_yy = np.gradient(p_y, y, edge_order=2)
    c = float(cr) + 1j * float(ci)
    u = np.tanh(y)
    up = 1.0 - u**2
    residual = (u - c) * p_yy - 2.0 * up * p_y - (alpha**2) * (u - c) * (1.0 - mach**2 * (u - c) ** 2) * p
    scale = (
        np.abs((u - c) * p_yy)
        + np.abs(2.0 * up * p_y)
        + np.abs((alpha**2) * (u - c) * (1.0 - mach**2 * (u - c) ** 2) * p)
        + 1e-12
    )
    rel = np.abs(residual) / scale
    match_mask = np.abs(y - match_y) <= match_band_width
    return {
        "max_rel_ode_residual": float(np.nanmax(rel)),
        "match_band_max_rel_ode_residual": float(np.nanmax(rel[match_mask])) if np.any(match_mask) else np.nan,
    }


def evaluate_candidate(
    *,
    solver: Mstab17SupersonicSolver,
    alpha: float,
    mach: float,
    name: str,
    cr: float,
    ci: float,
    tail_fraction: float,
    match_band_width: float,
) -> tuple[dict[str, float | str], dict[str, np.ndarray | float]]:
    stage1 = fast_stage1_mismatch(solver, cr, ci)
    amp_opt = minimize_scalar(
        lambda ln_p_right: solver.stage2_objective(ln_p_right, cr, ci),
        bounds=(-15.0, 5.0),
        method="bounded",
    )
    stage2 = float(amp_opt.fun)
    branches = reconstruct_branches(
        solver,
        cr=cr,
        ci=ci,
        ln_p_start_right=float(amp_opt.x),
    )

    gamma_left_match = complex(branches["k_left"][-1], branches["q_left"][-1])
    gamma_right_match = complex(branches["k_right"][-1], branches["q_right"][-1])
    gamma_jump = gamma_left_match - gamma_right_match
    p_left_match = complex(branches["p_left"][-1])
    p_right_match = complex(branches["p_right"][-1])
    p_jump = p_left_match - p_right_match

    c = float(cr) + 1j * float(ci)
    left_relative_mach, left_regime = classify_relative_regime(mach, -1.0, c)
    right_relative_mach, right_regime = classify_relative_regime(mach, +1.0, c)
    gamma_left_inf, gamma_right_inf = solver.asymptotic_gammas(cr, ci)
    left_log_slope, left_phase_slope = fit_tail_slopes(
        np.asarray(branches["y_left"], dtype=float),
        np.asarray(branches["ln_p_left"], dtype=float),
        np.asarray(branches["phi_left"], dtype=float),
        side="left",
        y_limit=float(branches["y_limit"]),
        tail_fraction=tail_fraction,
    )
    right_log_slope, right_phase_slope = fit_tail_slopes(
        np.asarray(branches["y_right"], dtype=float),
        np.asarray(branches["ln_p_right"], dtype=float),
        np.asarray(branches["phi_right"], dtype=float),
        side="right",
        y_limit=float(branches["y_limit"]),
        tail_fraction=tail_fraction,
    )

    ode_diag = relative_ode_residual(
        y=np.asarray(branches["y_joined"], dtype=float),
        p=np.asarray(branches["p_joined"], dtype=complex),
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        match_y=solver.match_y,
        match_band_width=match_band_width,
    )
    prof_diag = profile_diagnostics(
        np.asarray(branches["y_joined"], dtype=float),
        np.asarray(branches["p_joined"], dtype=complex),
    )

    row = {
        "alpha": float(alpha),
        "Mach": float(mach),
        "candidate_name": name,
        "candidate_cr": float(cr),
        "candidate_ci": float(ci),
        "candidate_omega_i": float(alpha * ci),
        "stage1_mismatch": float(stage1),
        "stage2_mismatch": float(stage2),
        "ln_p_start_right_opt": float(amp_opt.x),
        "gamma_jump_abs": float(abs(gamma_jump)),
        "gamma_jump_real": float(np.real(gamma_jump)),
        "gamma_jump_imag": float(np.imag(gamma_jump)),
        "pressure_jump_abs": float(abs(p_jump)),
        "pressure_jump_real": float(np.real(p_jump)),
        "pressure_jump_imag": float(np.imag(p_jump)),
        "left_relative_mach": left_relative_mach,
        "right_relative_mach": right_relative_mach,
        "left_regime": left_regime,
        "right_regime": right_regime,
        "left_gamma_inf_real": float(np.real(gamma_left_inf)),
        "left_gamma_inf_imag": float(np.imag(gamma_left_inf)),
        "right_gamma_inf_real": float(np.real(gamma_right_inf)),
        "right_gamma_inf_imag": float(np.imag(gamma_right_inf)),
        "left_tail_logamp_slope": float(left_log_slope),
        "left_tail_phase_slope": float(left_phase_slope),
        "right_tail_logamp_slope": float(right_log_slope),
        "right_tail_phase_slope": float(right_phase_slope),
        "left_tail_logamp_slope_err": float(left_log_slope - np.real(gamma_left_inf)) if np.isfinite(left_log_slope) else np.nan,
        "left_tail_phase_slope_err": float(left_phase_slope - np.imag(gamma_left_inf)) if np.isfinite(left_phase_slope) else np.nan,
        "right_tail_logamp_slope_err": float(right_log_slope - np.real(gamma_right_inf)) if np.isfinite(right_log_slope) else np.nan,
        "right_tail_phase_slope_err": float(right_phase_slope - np.imag(gamma_right_inf)) if np.isfinite(right_phase_slope) else np.nan,
        "left_decay_length": float(1.0 / max(np.real(gamma_left_inf), 1e-12)),
        "right_decay_length": float(1.0 / max(-np.real(gamma_right_inf), 1e-12)),
        "match_y": float(solver.match_y),
        "y_limit": float(branches["y_limit"]),
    }
    row.update(ode_diag)
    row.update(prof_diag)
    return row, branches


def evaluate_local_map(
    *,
    solver: Mstab17SupersonicSolver,
    alpha: float,
    mach: float,
    center_cr: float,
    center_ci: float,
    cr_half_window: float,
    ci_half_window: float,
    n_cr: int,
    n_ci: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cr_values = np.linspace(max(0.0, center_cr - cr_half_window), center_cr + cr_half_window, n_cr)
    ci_values = np.linspace(max(1e-4, center_ci - ci_half_window), center_ci + ci_half_window, n_ci)
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    for ci in ci_values:
        for cr in cr_values:
            mismatch = fast_stage1_mismatch(solver, float(cr), float(ci))
            row = {
                "alpha": float(alpha),
                "Mach": float(mach),
                "cr": float(cr),
                "ci": float(ci),
                "stage1_mismatch": float(mismatch),
                "distance_to_center": float(np.hypot(float(cr - center_cr), float(ci - center_ci))),
            }
            rows.append(row)
            if best_row is None or row["stage1_mismatch"] < best_row["stage1_mismatch"]:
                best_row = row
    if best_row is None:
        raise RuntimeError("Carte locale vide.")
    return pd.DataFrame(rows), best_row


def plot_mismatch_maps(
    *,
    map_df: pd.DataFrame,
    map_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    mach_values = list(map_summary_df["Mach"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(mach_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.2 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, mach in enumerate(mach_values):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_visible(True)
        sub = map_df[map_df["Mach"] == mach].copy()
        summary_row = map_summary_df[map_summary_df["Mach"] == mach].iloc[0]
        cr_grid = np.sort(sub["cr"].unique())
        ci_grid = np.sort(sub["ci"].unique())
        pivot = sub.pivot(index="ci", columns="cr", values="stage1_mismatch").sort_index().sort_index(axis=1)
        values = np.asarray(pivot, dtype=float)
        finite = values[np.isfinite(values) & (values > 0.0)]
        vmin = float(np.min(finite)) if finite.size else 1e-8
        vmax = float(np.max(finite)) if finite.size else 1.0
        mesh = ax.pcolormesh(
            cr_grid,
            ci_grid,
            values,
            shading="auto",
            norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=max(vmax, vmin * 1.01)),
            cmap="viridis",
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="stage1 mismatch")
        ax.scatter(
            [summary_row["blumen_cr"]],
            [summary_row["blumen_ci"]],
            color="white",
            edgecolor="black",
            marker="*",
            s=180,
            label="Blumen",
            zorder=5,
        )
        if np.isfinite(summary_row.get("reference_cr", np.nan)) and np.isfinite(summary_row.get("reference_ci", np.nan)):
            ax.scatter(
                [summary_row["reference_cr"]],
                [summary_row["reference_ci"]],
                color="red",
                marker="x",
                s=100,
                label="Branche tiree",
                zorder=5,
            )
        ax.scatter(
            [summary_row["map_best_cr"]],
            [summary_row["map_best_ci"]],
            color="orange",
            marker="o",
            s=70,
            label="Minimum local",
            zorder=5,
        )
        ax.set_title(
            f"M={mach:.3f} | mismatch Blumen={summary_row['stage1_at_blumen']:.3e}\n"
            f"best=({summary_row['map_best_cr']:.4f},{summary_row['map_best_ci']:.4f})"
        )
        ax.set_xlabel(r"$c_r$")
        ax.set_ylabel(r"$c_i$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_profiles_pdf(
    *,
    profile_payloads: list[tuple[dict[str, float | str], dict[str, np.ndarray | float]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for row, payload in profile_payloads:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            y_left = np.asarray(payload["y_left"], dtype=float)
            y_right = np.asarray(payload["y_right"], dtype=float)
            p_left = np.asarray(payload["p_left"], dtype=complex)
            p_right = np.asarray(payload["p_right"], dtype=complex)
            y_joined = np.asarray(payload["y_joined"], dtype=float)
            p_joined = np.asarray(payload["p_joined"], dtype=complex)

            axes[0, 0].plot(y_joined, np.real(p_joined), label=r"$\Re(p)$")
            axes[0, 0].plot(y_joined, np.imag(p_joined), "--", label=r"$\Im(p)$")
            axes[0, 0].axvline(float(row["match_y"]), color="black", linestyle=":", alpha=0.6)
            axes[0, 0].set_title("Profil reconstruit (raccord au point de match)")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            axes[0, 1].plot(y_left, np.exp(np.asarray(payload["ln_p_left"], dtype=float)), label="Branche gauche")
            axes[0, 1].plot(y_right, np.exp(np.asarray(payload["ln_p_right"], dtype=float)), "--", label="Branche droite")
            axes[0, 1].axvline(float(row["match_y"]), color="black", linestyle=":", alpha=0.6)
            axes[0, 1].set_yscale("log")
            axes[0, 1].set_title(r"Amplitude $|p|$")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            axes[1, 0].plot(y_left, np.asarray(payload["k_left"], dtype=float), label=r"$\kappa$ gauche")
            axes[1, 0].plot(y_left, np.asarray(payload["q_left"], dtype=float), ":", label=r"$q$ gauche")
            axes[1, 0].plot(y_right, np.asarray(payload["k_right"], dtype=float), "--", label=r"$\kappa$ droite")
            axes[1, 0].plot(y_right, np.asarray(payload["q_right"], dtype=float), "-.", label=r"$q$ droite")
            axes[1, 0].axvline(float(row["match_y"]), color="black", linestyle=":", alpha=0.6)
            axes[1, 0].set_title("Variables de Riccati")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

            axes[1, 1].axis("off")
            text = "\n".join(
                [
                    f"alpha={float(row['alpha']):.3f}  M={float(row['Mach']):.3f}",
                    f"c=({float(row['candidate_cr']):.6f}, {float(row['candidate_ci']):.6f})",
                    f"candidate={row['candidate_name']}",
                    f"stage1={float(row['stage1_mismatch']):.3e}",
                    f"stage2={float(row['stage2_mismatch']):.3e}",
                    f"|Δgamma|={float(row['gamma_jump_abs']):.3e}",
                    f"|Δp|={float(row['pressure_jump_abs']):.3e}",
                    f"max rel ODE={float(row['max_rel_ode_residual']):.3e}",
                    f"match-band rel ODE={float(row['match_band_max_rel_ode_residual']):.3e}",
                    f"left regime={row['left_regime']}  Mrel={float(row['left_relative_mach']):.4f}",
                    f"right regime={row['right_regime']}  Mrel={float(row['right_relative_mach']):.4f}",
                    f"tail slope err left={float(row['left_tail_logamp_slope_err']):.3e}",
                    f"tail slope err right={float(row['right_tail_logamp_slope_err']):.3e}",
                    f"center8 mass frac={float(row['center8_mass_fraction']):.4f}",
                ]
            )
            axes[1, 1].text(0.01, 0.99, text, va="top", ha="left", family="monospace", fontsize=10)

            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    targets = build_blumen_targets(args.mach_values, args.alpha, cr_points, ci_points)
    reference_branch = load_reference_branch(args.reference_summary)
    targets = targets.merge(reference_branch, on="Mach", how="left")

    point_rows: list[dict[str, float | str]] = []
    map_rows: list[pd.DataFrame] = []
    map_summary_rows: list[dict[str, float]] = []
    profile_payloads: list[tuple[dict[str, float | str], dict[str, np.ndarray | float]]] = []

    for _, target in targets.iterrows():
        mach = float(target["Mach"])
        blumen_cr = float(target["blumen_cr"])
        blumen_ci = float(target["blumen_ci"])
        reference_stage1 = np.nan
        reference_stage2 = np.nan
        solver = Mstab17SupersonicSolver(
            alpha=float(args.alpha),
            Mach=mach,
            match_y=float(args.match_y),
            amplitude_match_y=args.amplitude_match_y,
            use_mapping=bool(args.use_mapping),
            mapping_scale=float(args.mapping_scale),
        )

        blumen_row, blumen_payload = evaluate_candidate(
            solver=solver,
            alpha=float(args.alpha),
            mach=mach,
            name="blumen_target",
            cr=blumen_cr,
            ci=blumen_ci,
            tail_fraction=float(args.tail_fraction),
            match_band_width=float(args.match_band_width),
        )
        point_rows.append(blumen_row)
        profile_payloads.append((blumen_row, blumen_payload))

        reference_cr = float(target["reference_cr"]) if np.isfinite(target.get("reference_cr", np.nan)) else np.nan
        reference_ci = float(target["reference_ci"]) if np.isfinite(target.get("reference_ci", np.nan)) else np.nan
        if np.isfinite(reference_cr) and np.isfinite(reference_ci):
            ref_row, ref_payload = evaluate_candidate(
                solver=solver,
                alpha=float(args.alpha),
                mach=mach,
                name="reference_branch",
                cr=reference_cr,
                ci=reference_ci,
                tail_fraction=float(args.tail_fraction),
                match_band_width=float(args.match_band_width),
            )
            point_rows.append(ref_row)
            profile_payloads.append((ref_row, ref_payload))
            reference_stage1 = float(ref_row["stage1_mismatch"])
            reference_stage2 = float(ref_row["stage2_mismatch"])

        local_map_df, best_row = evaluate_local_map(
            solver=solver,
            alpha=float(args.alpha),
            mach=mach,
            center_cr=blumen_cr,
            center_ci=blumen_ci,
            cr_half_window=float(args.map_cr_half_window),
            ci_half_window=float(args.map_ci_half_window),
            n_cr=int(args.map_n_cr),
            n_ci=int(args.map_n_ci),
        )
        map_rows.append(local_map_df)
        map_summary_rows.append(
            {
                "alpha": float(args.alpha),
                "Mach": mach,
                "blumen_cr": blumen_cr,
                "blumen_ci": blumen_ci,
                "reference_cr": reference_cr,
                "reference_ci": reference_ci,
                "stage1_at_blumen": float(blumen_row["stage1_mismatch"]),
                "stage2_at_blumen": float(blumen_row["stage2_mismatch"]),
                "stage1_at_reference": reference_stage1,
                "stage2_at_reference": reference_stage2,
                "map_best_cr": float(best_row["cr"]),
                "map_best_ci": float(best_row["ci"]),
                "map_best_stage1_mismatch": float(best_row["stage1_mismatch"]),
                "map_best_distance_to_blumen": float(best_row["distance_to_center"]),
            }
        )

    point_df = pd.DataFrame(point_rows)
    map_df = pd.concat(map_rows, ignore_index=True) if map_rows else pd.DataFrame()
    map_summary_df = pd.DataFrame(map_summary_rows).sort_values("Mach").reset_index(drop=True)

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_point_audit.csv"
    map_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_map_grid.csv"
    map_summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_map_summary.csv"
    maps_fig_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_mismatch_maps.png"
    profiles_pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_profiles.pdf"

    point_df.to_csv(summary_path, index=False)
    map_df.to_csv(map_path, index=False)
    map_summary_df.to_csv(map_summary_path, index=False)
    plot_mismatch_maps(map_df=map_df, map_summary_df=map_summary_df, output_path=maps_fig_path)
    plot_profiles_pdf(profile_payloads=profile_payloads, output_path=profiles_pdf_path)

    print("Blumen targets:")
    print(targets.to_string(index=False))
    print("\nPoint audit summary:")
    display_cols = [
        "alpha",
        "Mach",
        "candidate_name",
        "candidate_cr",
        "candidate_ci",
        "stage1_mismatch",
        "stage2_mismatch",
        "gamma_jump_abs",
        "pressure_jump_abs",
        "left_regime",
        "right_regime",
        "center8_mass_fraction",
    ]
    print(point_df[display_cols].to_string(index=False))
    print("\nLocal mismatch map summary around Blumen:")
    print(map_summary_df.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {map_path}")
    print(f"Wrote {map_summary_path}")
    print(f"Wrote {maps_fig_path}")
    print(f"Wrote {profiles_pdf_path}")


if __name__ == "__main__":
    main()
