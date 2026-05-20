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

from classical_solver.supersonic.blumen_reference import load_digitized_curves  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    DEFAULT_OUTPUT_DIR,
    build_blumen_targets,
    load_digitized_long,
)
from scripts.audit_supersonic_shooting_ci_map import build_seed_list, ci_primary_score  # noqa: E402
from scripts.audit_supersonic_shooting_visual_validation import (  # noqa: E402
    compute_visible_xlim,
    reconstruct_shooting_fields,
)
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


DATA_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construit un paquet de reference supersonique au tir : isolignes c_i/c_r et top modes pour quelques points."
    )
    parser.add_argument("--mach-min", type=float, default=1.0)
    parser.add_argument("--mach-max", type=float, default=2.0)
    parser.add_argument("--alpha-min", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=0.50)
    parser.add_argument("--num-mach", type=int, default=21)
    parser.add_argument("--num-alpha", type=int, default=21)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=500.0)
    parser.add_argument("--y-limit-factor", type=float, default=6.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
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
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting_reference_package")
    parser.add_argument(
        "--resume-grid-summary",
        type=Path,
        default=None,
        help="CSV de reprise pour la grille shooting. Si le fichier existe, les points deja calcules sont sautes.",
    )
    parser.add_argument(
        "--mode-points",
        type=str,
        nargs="*",
        default=["0.18:1.33", "0.20:1.20", "0.20:1.30"],
        help="Liste de points alpha:Mach pour lesquels on extrait les modes les plus instables.",
    )
    parser.add_argument("--mode-top-k", type=int, default=2)
    parser.add_argument("--mode-seed-cr-count", type=int, default=6)
    parser.add_argument("--mode-seed-ci-count", type=int, default=6)
    parser.add_argument("--mode-cr-min", type=float, default=0.0)
    parser.add_argument("--mode-cr-max", type=float, default=0.60)
    parser.add_argument("--mode-ci-min", type=float, default=0.01)
    parser.add_argument("--mode-ci-max", type=float, default=0.12)
    parser.add_argument("--mode-cr-half-windows", type=float, nargs="+", default=[0.03, 0.06])
    parser.add_argument("--mode-ci-half-windows", type=float, nargs="+", default=[0.015, 0.03])
    parser.add_argument("--mode-dedup-cr-tol", type=float, default=5e-3)
    parser.add_argument("--mode-dedup-ci-tol", type=float, default=5e-3)
    parser.add_argument("--visible-threshold-ratio", type=float, default=0.02)
    parser.add_argument("--visible-min-half-width", type=float, default=8.0)
    return parser


def parse_mode_points(values: list[str]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in values:
        alpha_raw, mach_raw = item.split(":")
        points.append((float(alpha_raw), float(mach_raw)))
    return points


def generic_seed_list() -> list[tuple[str, float, float]]:
    return [
        ("generic_00", 0.00, 0.015),
        ("generic_01", 0.05, 0.025),
        ("generic_02", 0.10, 0.040),
        ("generic_03", 0.18, 0.055),
        ("generic_04", 0.26, 0.070),
        ("generic_05", 0.34, 0.085),
    ]


def load_reference_curves(quantity: str) -> list[dict]:
    curves = load_digitized_curves(DATA_DIR)
    if quantity == "ci":
        families = {"ci_level", "ci_special", "cr_special"}
    elif quantity == "cr":
        families = {"cr_level", "ci_special", "cr_special"}
    else:
        raise ValueError(f"quantite inconnue: {quantity}")
    return [curve for curve in curves if curve["family"] in families and curve["level"] is not None]


def contour_levels(curves: list[dict], quantity: str) -> list[float]:
    family = f"{quantity}_level"
    return sorted({float(curve["level"]) for curve in curves if curve["family"] == family})


def style_for_curve(curve: dict) -> dict:
    if curve["family"] == "ci_special":
        return {"color": "black", "linewidth": 1.0, "linestyle": (0, (5, 3)), "alpha": 0.9}
    if curve["family"] == "cr_special":
        return {"color": "black", "linewidth": 1.0, "linestyle": (0, (2, 2)), "alpha": 0.9}
    return {"color": "black", "linewidth": 1.4, "linestyle": "-", "alpha": 0.95}


def label_position(df: pd.DataFrame, mach_target: float) -> tuple[float, float]:
    idx = (df["Mach"] - mach_target).abs().idxmin()
    row = df.loc[idx]
    return float(row["Mach"]), float(row["alpha"])


def plot_blumen_reference(curves: list[dict], quantity: str, output_path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for curve in curves:
        ax.plot(curve["data"]["Mach"], curve["data"]["alpha"], **style_for_curve(curve))

    if quantity == "ci":
        targets = [
            (0.10, 1.15, "0·1"),
            (0.07, 1.32, "0·07"),
            (0.05, 1.48, "0·05"),
            (0.03, 1.63, "0·03"),
            (0.01, 1.88, "0·01"),
        ]
        title = r"Blumen 1975: isolignes supersoniques de $c_i$"
    else:
        targets = [
            (0.10, 1.22, "0·10"),
            (0.20, 1.33, "0·20"),
            (0.30, 1.50, "0·30"),
            (0.40, 1.70, "0·40"),
            (0.50, 1.88, "0·50"),
        ]
        title = r"Blumen 1975: isolignes supersoniques de $c_r$"

    main_family = f"{quantity}_level"
    main_levels = [curve for curve in curves if curve["family"] == main_family]
    for level, mach_target, label in targets:
        curve = next((item for item in main_levels if abs(float(item["level"]) - level) < 1e-12), None)
        if curve is None:
            continue
        x, y = label_position(curve["data"], mach_target)
        ax.text(x + 0.02, y - 0.01, label, fontsize=9)

    special_targets = [
        ("ci_special", r"$c_i = 0$", 0.98, 0.015),
        ("cr_special", r"$c_r = 0$", 1.22, 0.010),
    ]
    for family, label, mach_target, alpha_shift in special_targets:
        curve = next((item for item in curves if item["family"] == family and item["stem"] != "ci_sup=0"), None)
        if curve is None:
            continue
        x, y = label_position(curve["data"], mach_target)
        ax.text(x - 0.01, y + alpha_shift, label, fontsize=8.5)
    curve = next((item for item in curves if item["stem"] == "ci_sup=0"), None)
    if curve is not None:
        x, y = label_position(curve["data"], 1.06)
        ax.text(x - 0.01, y + 0.012, r"$c_i^{sup} = 0$", fontsize=8.5)

    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(0.0, 0.50)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=10)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.grid(True, linestyle=":", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shooting_overlay(summary_df: pd.DataFrame, curves: list[dict], quantity: str, output_path: Path) -> None:
    value_col = "best_shooting_ci" if quantity == "ci" else "best_shooting_cr"
    cmap = "viridis" if quantity == "ci" else "magma"
    title = (
        r"Reconstruction visuelle: isolignes de $c_i$ du shooting vs Blumen"
        if quantity == "ci"
        else r"Reconstruction visuelle: isolignes de $c_r$ du shooting vs Blumen"
    )
    legend_main = r"Blumen $c_i$ (principal)" if quantity == "ci" else r"Blumen $c_r$ (principal)"
    legend_shoot = r"Shooting $c_i$" if quantity == "ci" else r"Shooting $c_r$"

    pivot = summary_df.pivot(index="alpha", columns="Mach", values=value_col).sort_index().sort_index(axis=1)
    machs = pivot.columns.to_numpy(dtype=float)
    alphas = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    mach_grid, alpha_grid = np.meshgrid(machs, alphas)

    levels = contour_levels(curves, quantity)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    contour = ax.contour(
        mach_grid,
        alpha_grid,
        values,
        levels=levels,
        cmap=cmap,
        linewidths=2.0,
    )
    ax.clabel(contour, fmt="%0.02f", fontsize=8)

    for curve in curves:
        ax.plot(curve["data"]["Mach"], curve["data"]["alpha"], **style_for_curve(curve))

    ax.set_xlim(float(np.min(machs)), float(np.max(machs)))
    ax.set_ylim(float(np.min(alphas)), float(np.max(alphas)))
    ax.set_xlabel(r"Mach $M$")
    ax.set_ylabel(r"Nombre d'onde $\alpha$")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.25)

    legend_lines = [
        plt.Line2D([], [], color="black", linewidth=1.4, label=legend_main),
        plt.Line2D([], [], color="black", linewidth=1.0, linestyle=(0, (5, 3)), label=r"Blumen $c_i=0$"),
        plt.Line2D([], [], color="black", linewidth=1.0, linestyle=(0, (2, 2)), label=r"Blumen $c_r=0$"),
        plt.Line2D([], [], color=plt.get_cmap(cmap)(0.7), linewidth=2.0, label=legend_shoot),
    ]
    ax.legend(handles=legend_lines, loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_pointwise_curve_levels(curves: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for curve in curves:
        df = curve["data"]
        for _, row in df.iterrows():
            rows.append(
                {
                    "family": str(curve["family"]),
                    "label": str(curve["label"]),
                    "level": float(curve["level"]),
                    "Mach": float(row["Mach"]),
                    "alpha": float(row["alpha"]),
                }
            )
    return pd.DataFrame(rows)


def _persist_partial_grid(summary_rows: list[dict[str, object]], resume_path: Path) -> None:
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
    pd.DataFrame(summary_rows).to_csv(tmp_path, index=False)
    tmp_path.replace(resume_path)


def compute_shooting_grid(args: argparse.Namespace, *, resume_path: Path | None = None) -> pd.DataFrame:
    alpha_values = sorted(np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.num_alpha)).tolist(), reverse=True)
    mach_values = sorted(np.linspace(float(args.mach_min), float(args.mach_max), int(args.num_mach)).tolist())
    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)

    solution_lookup: dict[tuple[float, float], tuple[float, float]] = {}
    summary_rows: list[dict[str, object]] = []
    completed: set[tuple[float, float]] = set()

    if resume_path is not None and resume_path.exists():
        existing = pd.read_csv(resume_path)
        required_cols = {"alpha", "Mach", "best_shooting_cr", "best_shooting_ci"}
        if required_cols.issubset(existing.columns):
            for row in existing.to_dict(orient="records"):
                alpha = float(row["alpha"])
                mach = float(row["Mach"])
                summary_rows.append(row)
                completed.add((alpha, mach))
                solution_lookup[(alpha, mach)] = (
                    float(row["best_shooting_cr"]),
                    float(row["best_shooting_ci"]),
                )
            print(
                f"Resume shooting grid from {resume_path} "
                f"with {len(completed)} completed points."
            )

    def fallback_score(
        *,
        shooting_cr: float,
        shooting_ci: float,
        previous_mach: tuple[float, float] | None,
        previous_alpha: tuple[float, float] | None,
    ) -> float:
        continuity_terms: list[float] = []
        for neighbor in (previous_mach, previous_alpha):
            if neighbor is None:
                continue
            continuity_terms.append(
                float(
                    np.hypot(
                        0.25 * (shooting_cr - float(neighbor[0])),
                        float(args.ci_weight) * (shooting_ci - float(neighbor[1])),
                    )
                )
            )
        continuity_penalty = 0.0 if not continuity_terms else float(np.mean(continuity_terms))
        return float(float(args.continuity_weight) * continuity_penalty - shooting_ci)

    for alpha_idx, alpha in enumerate(alpha_values):
        targets_df = build_blumen_targets(mach_values, alpha, cr_points, ci_points)
        for mach_idx, mach in enumerate(mach_values):
            if (alpha, mach) in completed:
                continue
            target = targets_df[targets_df["Mach"] == mach].iloc[0]
            blumen_cr = float(target["blumen_cr"])
            blumen_ci = float(target["blumen_ci"])
            target_available = bool(np.isfinite(blumen_cr) and np.isfinite(blumen_ci))

            previous_mach = solution_lookup.get((alpha, mach_values[mach_idx - 1])) if mach_idx > 0 else None
            previous_alpha = solution_lookup.get((alpha_values[alpha_idx - 1], mach)) if alpha_idx > 0 else None

            if target_available:
                seeds = build_seed_list(
                    blumen_cr=blumen_cr,
                    blumen_ci=blumen_ci,
                    previous_mach=previous_mach,
                    previous_alpha=previous_alpha,
                )
            else:
                seeds = []
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
                seeds.extend(generic_seed_list())

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
                            min_y_limit=float(args.min_y_limit),
                            max_y_limit=float(args.max_y_limit),
                            y_limit_factor=float(args.y_limit_factor),
                            amp_lower_bound=float(args.amp_lower_bound),
                            amp_upper_bound=float(args.amp_upper_bound),
                            cr_center=float(cr_center),
                            ci_center=float(ci_center),
                            cr_half_window=float(cr_half),
                            ci_half_window=float(ci_half),
                            retry_growth=float(args.retry_growth),
                            max_retries=int(args.max_retries),
                            max_iter=int(args.max_iter),
                            grid_size=int(args.grid_size),
                        )
                        if target_available:
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
                        else:
                            score = fallback_score(
                                shooting_cr=float(result.cr),
                                shooting_ci=float(result.ci),
                                previous_mach=previous_mach,
                                previous_alpha=previous_alpha,
                            )
                        local_candidates.append(
                            {
                                "target_available": target_available,
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
                                "y_limit": float(result.y_limit),
                            }
                        )

            ranked = sorted(
                local_candidates,
                key=lambda row: (
                    0 if bool(row["success"]) else 1,
                    float(row["ci_primary_score"]),
                    -float(row["shooting_ci"]),
                    float(row["stage1_mismatch"] + row["stage2_mismatch"]),
                ),
            )
            best = ranked[0]
            summary_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "blumen_cr": blumen_cr,
                    "blumen_ci": blumen_ci,
                    "blumen_target_available": bool(target_available),
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
                    "min_y_limit": float(args.min_y_limit),
                    "max_y_limit": float(args.max_y_limit),
                    "y_limit_factor": float(args.y_limit_factor),
                    "amp_lower_bound": float(args.amp_lower_bound),
                    "amp_upper_bound": float(args.amp_upper_bound),
                    "y_limit": float(best["y_limit"]),
                }
            )
            solution_lookup[(alpha, mach)] = (float(best["shooting_cr"]), float(best["shooting_ci"]))
            if resume_path is not None:
                _persist_partial_grid(summary_rows, resume_path)
            print(
                f"[grid] alpha={alpha:.3f} Mach={mach:.3f} "
                f"ci={float(best['shooting_ci']):.5f} cr={float(best['shooting_cr']):.5f} "
                f"success={bool(best['success'])}"
            )

    return pd.DataFrame(summary_rows).sort_values(["alpha", "Mach"]).reset_index(drop=True)


def build_mode_seed_list(
    *,
    blumen_cr: float,
    blumen_ci: float,
    args: argparse.Namespace,
) -> list[tuple[str, float, float]]:
    seeds: list[tuple[str, float, float]] = []
    if np.isfinite(blumen_cr) and np.isfinite(blumen_ci):
        seeds.append(("blumen", blumen_cr, blumen_ci))
    else:
        seeds.extend(generic_seed_list())
    cr_values = np.linspace(float(args.mode_cr_min), float(args.mode_cr_max), int(args.mode_seed_cr_count))
    ci_values = np.linspace(float(args.mode_ci_min), float(args.mode_ci_max), int(args.mode_seed_ci_count))
    for i, cr in enumerate(cr_values):
        for j, ci in enumerate(ci_values):
            seeds.append((f"grid_{i}_{j}", float(cr), float(ci)))
    seen: set[tuple[float, float]] = set()
    out: list[tuple[str, float, float]] = []
    for name, cr, ci in seeds:
        key = (round(cr, 6), round(ci, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, cr, ci))
    return out


def is_same_mode(candidate: dict[str, object], reference: dict[str, object], *, cr_tol: float, ci_tol: float) -> bool:
    return (
        abs(float(candidate["shooting_cr"]) - float(reference["shooting_cr"])) <= cr_tol
        and abs(float(candidate["shooting_ci"]) - float(reference["shooting_ci"])) <= ci_tol
    )


def extract_top_modes_for_points(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cr_points = load_digitized_long(args.cr_points)
    ci_points = load_digitized_long(args.ci_points)
    mode_points = parse_mode_points(list(args.mode_points))

    candidate_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []

    for alpha, mach in mode_points:
        target_df = build_blumen_targets([mach], alpha, cr_points, ci_points)
        target = target_df.iloc[0]
        blumen_cr = float(target["blumen_cr"])
        blumen_ci = float(target["blumen_ci"])
        target_available = bool(np.isfinite(blumen_cr) and np.isfinite(blumen_ci))
        seeds = build_mode_seed_list(blumen_cr=blumen_cr, blumen_ci=blumen_ci, args=args)

        point_candidates: list[dict[str, object]] = []
        for seed_name, cr_center, ci_center in seeds:
            for cr_half in args.mode_cr_half_windows:
                for ci_half in args.mode_ci_half_windows:
                    solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
                        alpha=float(alpha),
                        mach=float(mach),
                        match_y=float(args.match_y),
                        use_mapping=bool(args.use_mapping),
                        mapping_scale=float(args.mapping_scale),
                        min_y_limit=float(args.min_y_limit),
                        max_y_limit=float(args.max_y_limit),
                        y_limit_factor=float(args.y_limit_factor),
                        amp_lower_bound=float(args.amp_lower_bound),
                        amp_upper_bound=float(args.amp_upper_bound),
                        cr_center=float(cr_center),
                        ci_center=float(ci_center),
                        cr_half_window=float(cr_half),
                        ci_half_window=float(ci_half),
                        retry_growth=float(args.retry_growth),
                        max_retries=int(args.max_retries),
                        max_iter=int(args.max_iter),
                        grid_size=int(args.grid_size),
                    )
                    row = {
                        "alpha": float(alpha),
                        "Mach": float(mach),
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
                        "blumen_target_available": bool(target_available),
                        "shooting_cr": float(result.cr),
                        "shooting_ci": float(result.ci),
                        "shooting_omega_i": float(result.omega_i),
                        "err_cr_abs": abs(float(result.cr) - blumen_cr),
                        "err_ci_abs": abs(float(result.ci) - blumen_ci),
                        "stage1_mismatch": float(result.stage1_mismatch),
                        "stage2_mismatch": float(result.stage2_mismatch),
                        "ln_p_start_right": float(result.ln_p_start_right),
                        "spectral_success": bool(result.spectral_success),
                        "mode_success": bool(result.mode_success),
                        "success": bool(result.success),
                        "y_limit": float(result.y_limit),
                    }
                    point_candidates.append(row)

        for row in point_candidates:
            candidate_rows.append(row)

        successful = [
            row for row in point_candidates if bool(row["spectral_success"]) and bool(row["mode_success"]) and bool(row["success"])
        ]
        successful = sorted(
            successful,
            key=lambda row: (
                -float(row["shooting_ci"]),
                float(row["stage1_mismatch"] + row["stage2_mismatch"]),
            ),
        )
        unique_modes: list[dict[str, object]] = []
        for row in successful:
            if any(
                is_same_mode(
                    row,
                    existing,
                    cr_tol=float(args.mode_dedup_cr_tol),
                    ci_tol=float(args.mode_dedup_ci_tol),
                )
                for existing in unique_modes
            ):
                continue
            unique_modes.append(row)
            if len(unique_modes) >= int(args.mode_top_k):
                break

        for rank, row in enumerate(unique_modes, start=1):
            top_rows.append(
                {
                    **row,
                    "rank_by_ci": int(rank),
                    "match_y": float(args.match_y),
                    "use_mapping": bool(args.use_mapping),
                    "mapping_scale": float(args.mapping_scale),
                    "min_y_limit": float(args.min_y_limit),
                    "max_y_limit": float(args.max_y_limit),
                    "y_limit_factor": float(args.y_limit_factor),
                    "amp_lower_bound": float(args.amp_lower_bound),
                    "amp_upper_bound": float(args.amp_upper_bound),
                }
            )
            fields = reconstruct_shooting_fields(
                alpha=float(alpha),
                mach=float(mach),
                cr=float(row["shooting_cr"]),
                ci=float(row["shooting_ci"]),
                ln_p_start_right=float(row["ln_p_start_right"]),
                match_y=float(args.match_y),
                use_mapping=bool(args.use_mapping),
                mapping_scale=float(args.mapping_scale),
                min_y_limit=float(args.min_y_limit),
                max_y_limit=float(args.max_y_limit),
                y_limit_factor=float(args.y_limit_factor),
            )
            for y_value, rho_value, u_value, v_value, p_value in zip(
                fields["y"], fields["rho"], fields["u"], fields["v"], fields["p"]
            ):
                field_rows.append(
                    {
                        "alpha": float(alpha),
                        "Mach": float(mach),
                        "rank_by_ci": int(rank),
                        "y": float(y_value),
                        "rho_real": float(np.real(rho_value)),
                        "rho_imag": float(np.imag(rho_value)),
                        "u_real": float(np.real(u_value)),
                        "u_imag": float(np.imag(u_value)),
                        "v_real": float(np.real(v_value)),
                        "v_imag": float(np.imag(v_value)),
                        "p_real": float(np.real(p_value)),
                        "p_imag": float(np.imag(p_value)),
                    }
                )

    return (
        pd.DataFrame(candidate_rows).sort_values(["alpha", "Mach", "shooting_ci"], ascending=[True, True, False]).reset_index(drop=True),
        pd.DataFrame(top_rows).sort_values(["alpha", "Mach", "rank_by_ci"]).reset_index(drop=True),
        pd.DataFrame(field_rows).sort_values(["alpha", "Mach", "rank_by_ci", "y"]).reset_index(drop=True),
    )


def plot_top_modes_pdf(
    top_df: pd.DataFrame,
    fields_df: pd.DataFrame,
    *,
    threshold_ratio: float,
    min_half_width: float,
    output_path: Path,
) -> None:
    field_names = ["rho", "u", "v", "p"]
    field_titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]
    with PdfPages(output_path) as pdf:
        grouped = top_df.groupby(["alpha", "Mach"], sort=True)
        for (alpha, mach), sub in grouped:
            n_modes = len(sub)
            fig, axes = plt.subplots(n_modes, 4, figsize=(16, max(4.2, 3.6 * n_modes)), squeeze=False)
            for col, title in enumerate(field_titles):
                axes[0, col].set_title(title)
            for row_idx, top_row in enumerate(sub.sort_values("rank_by_ci").itertuples(index=False), start=0):
                prof = fields_df[
                    np.isclose(fields_df["alpha"].to_numpy(dtype=float), float(alpha))
                    & np.isclose(fields_df["Mach"].to_numpy(dtype=float), float(mach))
                    & (fields_df["rank_by_ci"].to_numpy(dtype=int) == int(top_row.rank_by_ci))
                ].copy()
                if prof.empty:
                    continue
                y = prof["y"].to_numpy(dtype=float)
                x_limits = compute_visible_xlim(
                    y,
                    [
                        prof["rho_real"].to_numpy(dtype=float) + 1j * prof["rho_imag"].to_numpy(dtype=float),
                        prof["u_real"].to_numpy(dtype=float) + 1j * prof["u_imag"].to_numpy(dtype=float),
                        prof["v_real"].to_numpy(dtype=float) + 1j * prof["v_imag"].to_numpy(dtype=float),
                        prof["p_real"].to_numpy(dtype=float) + 1j * prof["p_imag"].to_numpy(dtype=float),
                    ],
                    threshold_ratio=threshold_ratio,
                    min_half_width=min_half_width,
                )
                for col, field_name in enumerate(field_names):
                    ax = axes[row_idx, col]
                    ax.plot(y, prof[f"{field_name}_real"], color="black", linewidth=1.8, label="Real")
                    ax.plot(y, prof[f"{field_name}_imag"], color="#D97706", linestyle="--", linewidth=1.3, label="Imag")
                    ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                    ax.set_xlim(*x_limits)
                    ax.grid(True, alpha=0.25)
                    if row_idx == 0 and col == 0:
                        ax.legend(frameon=False, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(
                            f"rank {int(top_row.rank_by_ci)}\n"
                            f"$c_r$={float(top_row.shooting_cr):.4f}\n"
                            f"$c_i$={float(top_row.shooting_ci):.4f}"
                        )
                    if row_idx == n_modes - 1:
                        ax.set_xlabel("y")

            fig.suptitle(
                f"Top shooting modes by $c_i$ | alpha={float(alpha):.3f}, M={float(mach):.3f}\n"
                "Rows ordered by decreasing $c_i$"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"{args.output_stem}_grid_summary.csv"
    resume_path = Path(args.resume_grid_summary) if args.resume_grid_summary is not None else summary_csv

    print("Construction du paquet de reference supersonique au tir")
    print(f"grid Mach: {float(args.mach_min):.3f} -> {float(args.mach_max):.3f} ({int(args.num_mach)} points)")
    print(f"grid alpha: {float(args.alpha_min):.3f} -> {float(args.alpha_max):.3f} ({int(args.num_alpha)} points)")
    print(
        f"runtime box: min={float(args.min_y_limit):.1f} max={float(args.max_y_limit):.1f} "
        f"factor={float(args.y_limit_factor):.2f} amp=[{float(args.amp_lower_bound):.1f},{float(args.amp_upper_bound):.1f}]"
    )
    print(f"mode points: {' '.join(args.mode_points)}")
    print(f"output stem: {args.output_stem}")
    print(f"resume grid summary: {resume_path}")

    summary_df = compute_shooting_grid(args, resume_path=resume_path)

    ci_curves = load_reference_curves("ci")
    cr_curves = load_reference_curves("cr")

    ci_points_csv = output_dir / f"{args.output_stem}_ci_blumen_points.csv"
    cr_points_csv = output_dir / f"{args.output_stem}_cr_blumen_points.csv"
    ci_ref_png = output_dir / f"{args.output_stem}_ci_blumen_reference.png"
    ci_overlay_png = output_dir / f"{args.output_stem}_ci_shooting_overlay.png"
    cr_ref_png = output_dir / f"{args.output_stem}_cr_blumen_reference.png"
    cr_overlay_png = output_dir / f"{args.output_stem}_cr_shooting_overlay.png"

    summary_df.to_csv(summary_csv, index=False)
    summarize_pointwise_curve_levels(ci_curves).to_csv(ci_points_csv, index=False)
    summarize_pointwise_curve_levels(cr_curves).to_csv(cr_points_csv, index=False)
    plot_blumen_reference(ci_curves, "ci", ci_ref_png)
    plot_shooting_overlay(summary_df, ci_curves, "ci", ci_overlay_png)
    plot_blumen_reference(cr_curves, "cr", cr_ref_png)
    plot_shooting_overlay(summary_df, cr_curves, "cr", cr_overlay_png)

    candidates_df, top_modes_df, fields_df = extract_top_modes_for_points(args)
    mode_candidates_csv = output_dir / f"{args.output_stem}_mode_candidates.csv"
    mode_summary_csv = output_dir / f"{args.output_stem}_mode_topk_summary.csv"
    mode_fields_csv = output_dir / f"{args.output_stem}_mode_topk_fields.csv"
    mode_pdf = output_dir / f"{args.output_stem}_mode_topk_visual.pdf"

    candidates_df.to_csv(mode_candidates_csv, index=False)
    top_modes_df.to_csv(mode_summary_csv, index=False)
    fields_df.to_csv(mode_fields_csv, index=False)
    if not top_modes_df.empty:
        plot_top_modes_pdf(
            top_modes_df,
            fields_df,
            threshold_ratio=float(args.visible_threshold_ratio),
            min_half_width=float(args.visible_min_half_width),
            output_path=mode_pdf,
        )

    print("\nGrid summary:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(summary_df.head(12).to_string(index=False))
    if not top_modes_df.empty:
        print("\nTop modes summary:")
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(top_modes_df.to_string(index=False))

    print(f"Wrote {summary_csv}")
    print(f"Wrote {ci_points_csv}")
    print(f"Wrote {cr_points_csv}")
    print(f"Wrote {ci_ref_png}")
    print(f"Wrote {ci_overlay_png}")
    print(f"Wrote {cr_ref_png}")
    print(f"Wrote {cr_overlay_png}")
    print(f"Wrote {mode_candidates_csv}")
    print(f"Wrote {mode_summary_csv}")
    print(f"Wrote {mode_fields_csv}")
    if not top_modes_df.empty:
        print(f"Wrote {mode_pdf}")


if __name__ == "__main__":
    main()
