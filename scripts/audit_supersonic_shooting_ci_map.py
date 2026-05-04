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

from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    DEFAULT_OUTPUT_DIR,
    build_blumen_targets,
    load_digitized_long,
)
from scripts.track_supersonic_shooting_multistart import (  # noqa: E402
    extract_shooting_profile,
    multistart_single_box,
    profile_diagnostics,
)


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Carte d'erreur c_i du shooting supersonique sur une grille (alpha, Mach)."
    )
    parser.add_argument("--alpha-values", type=float, nargs="+", required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.015, 0.03, 0.06, 0.10])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.008, 0.015, 0.03])
    parser.add_argument("--retry-growth", type=float, default=1.75)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ci-weight", type=float, default=4.0)
    parser.add_argument("--cr-weight", type=float, default=0.35)
    parser.add_argument("--continuity-weight", type=float, default=0.20)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def classify_relative_regime(mach: float, u_inf: float, c: complex, *, tol: float = 1e-6) -> tuple[float, str]:
    rel_mach = float(mach * abs(u_inf - c))
    if rel_mach < 1.0 - tol:
        return rel_mach, "subsonic_relative"
    if rel_mach > 1.0 + tol:
        return rel_mach, "supersonic_relative"
    return rel_mach, "sonic_relative"


def center8_mass_fraction(y: np.ndarray, p: np.ndarray, *, center_window: float = 8.0) -> float:
    p_abs = np.abs(p)
    total = max(trapezoid_compat(p_abs, y), 1e-12)
    center_mask = np.abs(y) <= center_window
    if not np.any(center_mask):
        return np.nan
    return float(trapezoid_compat(p_abs[center_mask], y[center_mask]) / total)


def extended_profile_diagnostics(y: np.ndarray, p: np.ndarray, *, center_window: float = 8.0) -> dict[str, float]:
    diag = profile_diagnostics(y, p, center_window=center_window)
    p_abs = np.abs(p)
    total = max(trapezoid_compat(p_abs, y), 1e-12)
    left_mask = y < 0.0
    right_mask = y >= 0.0
    left_mass = trapezoid_compat(p_abs[left_mask], y[left_mask]) if np.any(left_mask) else np.nan
    right_mass = trapezoid_compat(p_abs[right_mask], y[right_mask]) if np.any(right_mask) else np.nan
    diag.update(
        {
            "center8_mass_fraction": center8_mass_fraction(y, p, center_window=center_window),
            "left_mass_fraction": np.nan if not np.isfinite(left_mass) else float(left_mass / total),
            "right_mass_fraction": np.nan if not np.isfinite(right_mass) else float(right_mass / total),
        }
    )
    return diag


def ci_primary_score(
    *,
    shooting_cr: float,
    shooting_ci: float,
    blumen_cr: float,
    blumen_ci: float,
    previous_mach: tuple[float, float] | None,
    previous_alpha: tuple[float, float] | None,
    ci_weight: float,
    cr_weight: float,
    continuity_weight: float,
) -> float:
    score = np.hypot(cr_weight * (shooting_cr - blumen_cr), ci_weight * (shooting_ci - blumen_ci))
    for neighbor in (previous_mach, previous_alpha):
        if neighbor is None:
            continue
        score += continuity_weight * np.hypot(
            0.5 * cr_weight * (shooting_cr - float(neighbor[0])),
            ci_weight * (shooting_ci - float(neighbor[1])),
        )
    return float(score)


def dedup_seeds(seeds: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[str, float, float]] = []
    for seed_name, cr_center, ci_center in seeds:
        key = (round(float(cr_center), 6), round(float(ci_center), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((seed_name, float(cr_center), float(ci_center)))
    return out


def build_seed_list(
    *,
    blumen_cr: float,
    blumen_ci: float,
    previous_mach: tuple[float, float] | None,
    previous_alpha: tuple[float, float] | None,
) -> list[tuple[str, float, float]]:
    seeds: list[tuple[str, float, float]] = [("blumen", blumen_cr, blumen_ci)]
    if previous_mach is not None:
        seeds.append(("previous_mach", float(previous_mach[0]), float(previous_mach[1])))
    if previous_alpha is not None:
        seeds.append(("previous_alpha", float(previous_alpha[0]), float(previous_alpha[1])))
    if previous_mach is not None and previous_alpha is not None:
        seeds.append(
            (
                "blend_neighbors",
                0.5 * (float(previous_mach[0]) + float(previous_alpha[0])),
                0.5 * (float(previous_mach[1]) + float(previous_alpha[1])),
            )
        )
    if previous_mach is not None:
        seeds.append(
            (
                "blend_blumen_mach",
                0.5 * (blumen_cr + float(previous_mach[0])),
                0.5 * (blumen_ci + float(previous_mach[1])),
            )
        )
    if previous_alpha is not None:
        seeds.append(
            (
                "blend_blumen_alpha",
                0.5 * (blumen_cr + float(previous_alpha[0])),
                0.5 * (blumen_ci + float(previous_alpha[1])),
            )
        )
    return dedup_seeds(seeds)


def plot_grid_panels(summary_df: pd.DataFrame, output_path: Path) -> None:
    alpha_values = np.array(sorted(summary_df["alpha"].unique()), dtype=float)
    mach_values = np.array(sorted(summary_df["Mach"].unique()), dtype=float)
    panels = [
        ("blumen_ci", r"Blumen $c_i$"),
        ("best_shooting_ci", r"Shooting $c_i$"),
        ("best_err_ci_abs", r"$| \Delta c_i |$"),
        ("best_err_ci_rel", r"$| \Delta c_i | / c_i$"),
        ("best_shooting_cr", r"Shooting $c_r$"),
        ("best_err_cr_abs", r"$| \Delta c_r |$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    axes = axes.ravel()

    if len(alpha_values) > 1 and len(mach_values) > 1:
        for ax, (column, title) in zip(axes, panels):
            pivot = summary_df.pivot(index="alpha", columns="Mach", values=column)
            data = pivot.to_numpy(dtype=float)
            image = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=[float(mach_values.min()), float(mach_values.max()), float(alpha_values.min()), float(alpha_values.max())],
            )
            ax.set_title(title)
            ax.set_xlabel("Mach")
            ax.set_ylabel("alpha")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        x_name = "Mach" if len(mach_values) > 1 else "alpha"
        x_values = summary_df[x_name].to_numpy(dtype=float)
        order = np.argsort(x_values)
        x_values = x_values[order]
        for ax, (column, title) in zip(axes, panels):
            y_values = summary_df[column].to_numpy(dtype=float)[order]
            ax.plot(x_values, y_values, marker="o")
            ax.set_title(title)
            ax.set_xlabel(x_name)
            ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = sorted(float(value) for value in args.alpha_values)
    mach_values = sorted(float(value) for value in args.mach_values)

    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)

    solution_lookup: dict[tuple[float, float], tuple[float, float]] = {}
    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for alpha_idx, alpha in enumerate(alpha_values):
        targets_df = build_blumen_targets(mach_values, alpha, cr_points, ci_points)
        for mach_idx, mach in enumerate(mach_values):
            target = targets_df[targets_df["Mach"] == mach].iloc[0]
            blumen_cr = float(target["blumen_cr"])
            blumen_ci = float(target["blumen_ci"])

            previous_mach = None
            previous_alpha = None
            if mach_idx > 0:
                previous_mach = solution_lookup.get((alpha, mach_values[mach_idx - 1]))
            if alpha_idx > 0:
                previous_alpha = solution_lookup.get((alpha_values[alpha_idx - 1], mach))

            seeds = build_seed_list(
                blumen_cr=blumen_cr,
                blumen_ci=blumen_ci,
                previous_mach=previous_mach,
                previous_alpha=previous_alpha,
            )

            local_candidates: list[dict[str, object]] = []
            for seed_name, cr_center, ci_center in seeds:
                for cr_half in args.cr_half_windows:
                    for ci_half in args.ci_half_windows:
                        solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                            alpha=alpha,
                            mach=mach,
                            match_y=float(args.match_y),
                            use_mapping=bool(args.use_mapping),
                            mapping_scale=float(args.mapping_scale),
                            cr_center=float(cr_center),
                            ci_center=float(ci_center),
                            cr_half_window=float(cr_half),
                            ci_half_window=float(ci_half),
                            retry_growth=float(args.retry_growth),
                            max_retries=int(args.max_retries),
                            max_iter=int(args.max_iter),
                            grid_size=int(args.grid_size),
                        )
                        score = ci_primary_score(
                            shooting_cr=float(result.cr),
                            shooting_ci=float(result.ci),
                            blumen_cr=blumen_cr,
                            blumen_ci=blumen_ci,
                            previous_mach=previous_mach,
                            previous_alpha=previous_alpha,
                            ci_weight=float(args.ci_weight),
                            cr_weight=float(args.cr_weight),
                            continuity_weight=float(args.continuity_weight),
                        )
                        row = {
                            "alpha": alpha,
                            "Mach": mach,
                            "seed_name": seed_name,
                            "seed_cr_center": float(cr_center),
                            "seed_ci_center": float(ci_center),
                            "requested_cr_half_window": float(cr_half),
                            "requested_ci_half_window": float(ci_half),
                            "used_cr_half_window": float(used_cr_half),
                            "used_ci_half_window": float(used_ci_half),
                            "retry_index": int(retry_idx),
                            "blumen_cr": blumen_cr,
                            "blumen_ci": blumen_ci,
                            "shooting_cr": float(result.cr),
                            "shooting_ci": float(result.ci),
                            "shooting_omega_i": float(result.omega_i),
                            "err_cr_abs": abs(float(result.cr) - blumen_cr),
                            "err_ci_abs": abs(float(result.ci) - blumen_ci),
                            "err_ci_rel": abs(float(result.ci) - blumen_ci) / max(abs(blumen_ci), 1e-12),
                            "stage1_mismatch": float(result.stage1_mismatch),
                            "stage2_mismatch": float(result.stage2_mismatch),
                            "ln_p_start_right": float(result.ln_p_start_right),
                            "spectral_success": bool(result.spectral_success),
                            "mode_success": bool(result.mode_success),
                            "success": bool(result.success),
                            "ci_primary_score": float(score),
                            "_solver": solver,
                            "_result": result,
                        }
                        local_candidates.append(row)

            ranked = sorted(
                local_candidates,
                key=lambda row: (
                    0 if bool(row["success"]) else 1,
                    float(row["ci_primary_score"]),
                    float(row["stage1_mismatch"] + row["stage2_mismatch"]),
                ),
            )
            best = ranked[0]
            solver = best.pop("_solver")
            result = best.pop("_result")
            profile = extract_shooting_profile(
                solver,
                cr=float(result.cr),
                ci=float(result.ci),
                ln_p_start_right=float(result.ln_p_start_right),
            )
            y = np.asarray(profile["y"])
            p = np.asarray(profile["p"])
            diag = extended_profile_diagnostics(y, p)
            c = complex(float(result.cr), float(result.ci))
            left_relative_mach, left_regime = classify_relative_regime(mach, -1.0, c)
            right_relative_mach, right_regime = classify_relative_regime(mach, 1.0, c)

            for rank_idx, row in enumerate(ranked, start=1):
                row_out = {k: v for k, v in row.items() if not k.startswith("_")}
                row_out["rank"] = rank_idx
                candidate_rows.append(row_out)

            summary_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "blumen_cr": blumen_cr,
                    "blumen_ci": blumen_ci,
                    "n_candidates": len(ranked),
                    "best_seed_name": str(best["seed_name"]),
                    "best_seed_cr_center": float(best["seed_cr_center"]),
                    "best_seed_ci_center": float(best["seed_ci_center"]),
                    "best_requested_cr_half_window": float(best["requested_cr_half_window"]),
                    "best_requested_ci_half_window": float(best["requested_ci_half_window"]),
                    "best_used_cr_half_window": float(best["used_cr_half_window"]),
                    "best_used_ci_half_window": float(best["used_ci_half_window"]),
                    "best_retry_index": int(best["retry_index"]),
                    "best_shooting_cr": float(best["shooting_cr"]),
                    "best_shooting_ci": float(best["shooting_ci"]),
                    "best_shooting_omega_i": float(best["shooting_omega_i"]),
                    "best_err_cr_abs": float(best["err_cr_abs"]),
                    "best_err_ci_abs": float(best["err_ci_abs"]),
                    "best_err_ci_rel": float(best["err_ci_rel"]),
                    "best_stage1_mismatch": float(best["stage1_mismatch"]),
                    "best_stage2_mismatch": float(best["stage2_mismatch"]),
                    "best_ln_p_start_right": float(best["ln_p_start_right"]),
                    "best_spectral_success": bool(best["spectral_success"]),
                    "best_mode_success": bool(best["mode_success"]),
                    "best_success": bool(best["success"]),
                    "best_ci_primary_score": float(best["ci_primary_score"]),
                    "match_y": float(args.match_y),
                    "use_mapping": bool(args.use_mapping),
                    "mapping_scale": float(args.mapping_scale),
                    "y_limit": float(profile["y_limit"]),
                    "left_relative_mach": left_relative_mach,
                    "right_relative_mach": right_relative_mach,
                    "left_regime": left_regime,
                    "right_regime": right_regime,
                    **diag,
                }
            )
            solution_lookup[(alpha, mach)] = (float(best["shooting_cr"]), float(best["shooting_ci"]))

    summary_df = pd.DataFrame(summary_rows).sort_values(["alpha", "Mach"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows).sort_values(["alpha", "Mach", "rank"]).reset_index(drop=True)

    print("Shooting c_i map summary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    candidates_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_candidates.csv"
    fig_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.png"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    plot_grid_panels(summary_df, fig_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
