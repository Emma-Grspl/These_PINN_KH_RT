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

from scripts.audit_supersonic_blumen_local_reference import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    normalize_points,
    summarize_target_for_mach,
)
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CR_POINTS,
    estimate_level_from_isolines,
    load_digitized_long,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estime l'incertitude locale de la reference Blumen c_r par bootstrap+jitter."
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--local-mach-half-window", type=float, default=0.08)
    parser.add_argument("--local-alpha-half-window", type=float, default=0.08)
    parser.add_argument("--neighbor-levels", type=int, default=2)
    parser.add_argument("--min-points-per-level", type=int, default=4)
    parser.add_argument("--max-points-per-level", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--jitter-scale-mach", type=float, default=0.25)
    parser.add_argument("--jitter-scale-alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "supersonic_shooting_multistart_a020_m120_130_summary.csv",
    )
    parser.add_argument("--reference-column", type=str, default="best_shooting_cr")
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def parse_support_levels(value: str) -> list[float]:
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(float(token))
    return items


def load_reference_lookup(path: Path, column: str, alpha_target: float) -> dict[float, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Mach" not in df.columns or column not in df.columns:
        return {}
    if "alpha" in df.columns:
        df = df[np.isclose(df["alpha"].to_numpy(dtype=float), float(alpha_target))]
    out: dict[float, float] = {}
    for _, row in df.iterrows():
        out[float(row["Mach"])] = float(row[column])
    return out


def choose_support_pool(
    points: pd.DataFrame,
    *,
    level: float,
    mach_target: float,
    alpha_target: float,
    local_mach_half_window: float,
    local_alpha_half_window: float,
    min_points: int,
    max_points: int,
) -> pd.DataFrame:
    sub = points[np.isclose(points["level_value"].to_numpy(dtype=float), float(level))].copy()
    if sub.empty:
        return sub

    sub["mach_distance"] = np.abs(sub["Mach"] - float(mach_target))
    local = sub[
        (sub["Mach"] >= mach_target - local_mach_half_window)
        & (sub["Mach"] <= mach_target + local_mach_half_window)
        & (sub["alpha"] >= alpha_target - local_alpha_half_window)
        & (sub["alpha"] <= alpha_target + local_alpha_half_window)
    ].copy()

    selected_parts: list[pd.DataFrame] = []
    if not local.empty:
        below_local = local[local["Mach"] <= mach_target].nsmallest(max(2, max_points // 2), "mach_distance")
        above_local = local[local["Mach"] >= mach_target].nsmallest(max(2, max_points // 2), "mach_distance")
        selected_parts.extend([below_local, above_local])

    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates() if selected_parts else local.iloc[0:0].copy()
    if len(selected) < min_points:
        below = sub[sub["Mach"] <= mach_target].nsmallest(max(2, min_points // 2 + 1), "mach_distance")
        above = sub[sub["Mach"] >= mach_target].nsmallest(max(2, min_points // 2 + 1), "mach_distance")
        selected = pd.concat([selected, below, above], ignore_index=True).drop_duplicates()

    if len(selected) < min_points:
        selected = pd.concat([selected, sub.nsmallest(min_points, "mach_distance")], ignore_index=True).drop_duplicates()

    if len(selected) > max_points:
        selected = selected.nsmallest(max_points, "mach_distance")

    return selected.sort_values("Mach").reset_index(drop=True)


def estimate_jitter_scales(
    pool: pd.DataFrame,
    *,
    mach_scale: float,
    alpha_scale: float,
) -> tuple[float, float]:
    mach_values = np.sort(pool["Mach"].drop_duplicates().to_numpy(dtype=float))
    alpha_values = np.sort(pool["alpha"].to_numpy(dtype=float))

    if mach_values.size >= 2:
        mach_step = float(np.median(np.diff(mach_values)))
    else:
        mach_step = 0.01
    if alpha_values.size >= 2:
        alpha_step = float(np.median(np.abs(np.diff(alpha_values))))
    else:
        alpha_step = 0.01

    sigma_mach = max(mach_scale * mach_step, 1e-4)
    sigma_alpha = max(alpha_scale * alpha_step, 1e-4)
    return sigma_mach, sigma_alpha


def bootstrap_alpha_on_curve(
    pool: pd.DataFrame,
    *,
    mach_target: float,
    rng: np.random.Generator,
    sigma_mach: float,
    sigma_alpha: float,
    max_attempts: int = 16,
) -> float:
    if len(pool) < 2:
        return np.nan

    n = int(len(pool))
    pool_mach = pool["Mach"].to_numpy(dtype=float)
    pool_alpha = pool["alpha"].to_numpy(dtype=float)

    for _ in range(max_attempts):
        sample_idx = rng.integers(0, n, size=n)
        mach_sample = pool_mach[sample_idx] + rng.normal(0.0, sigma_mach, size=n)
        alpha_sample = pool_alpha[sample_idx] + rng.normal(0.0, sigma_alpha, size=n)
        sample = pd.DataFrame({"Mach": mach_sample, "alpha": alpha_sample})
        sample = sample.groupby("Mach", as_index=False)["alpha"].mean().sort_values("Mach")
        mach_values = sample["Mach"].to_numpy(dtype=float)
        alpha_values = sample["alpha"].to_numpy(dtype=float)
        if len(mach_values) < 2:
            continue
        if mach_target < float(np.min(mach_values)) or mach_target > float(np.max(mach_values)):
            continue
        return float(np.interp(mach_target, mach_values, alpha_values))
    return np.nan


def bootstrap_target_level(
    support_pools: dict[float, pd.DataFrame],
    *,
    mach_target: float,
    alpha_target: float,
    bootstrap_samples: int,
    jitter_scale_mach: float,
    jitter_scale_alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    level_values = sorted(float(level) for level in support_pools)
    scales = {
        float(level): estimate_jitter_scales(
            support_pools[float(level)],
            mach_scale=jitter_scale_mach,
            alpha_scale=jitter_scale_alpha,
        )
        for level in level_values
    }

    rows: list[dict[str, float]] = []
    samples = np.full(int(bootstrap_samples), np.nan, dtype=float)
    for sample_idx in range(int(bootstrap_samples)):
        anchor_pairs: list[tuple[float, float]] = []
        row: dict[str, float] = {"sample_idx": float(sample_idx)}
        for level in level_values:
            sigma_mach, sigma_alpha = scales[level]
            alpha_on_curve = bootstrap_alpha_on_curve(
                support_pools[level],
                mach_target=mach_target,
                rng=rng,
                sigma_mach=sigma_mach,
                sigma_alpha=sigma_alpha,
            )
            row[f"alpha_on_curve_level_{level:.3f}"] = alpha_on_curve
            if np.isfinite(alpha_on_curve):
                anchor_pairs.append((alpha_on_curve, level))

        if len(anchor_pairs) >= 2:
            anchor_pairs = sorted(anchor_pairs, key=lambda item: item[0])
            alpha_grid = np.array([item[0] for item in anchor_pairs], dtype=float)
            level_grid = np.array([item[1] for item in anchor_pairs], dtype=float)
            if alpha_target >= float(np.min(alpha_grid)) and alpha_target <= float(np.max(alpha_grid)):
                samples[sample_idx] = float(np.interp(alpha_target, alpha_grid, level_grid))
        row["sampled_target_level"] = samples[sample_idx]
        rows.append(row)

    return samples, pd.DataFrame(rows)


def summarize_distribution(
    *,
    mach: float,
    alpha_target: float,
    target_level: float,
    support_levels: list[float],
    support_pools: dict[float, pd.DataFrame],
    samples: np.ndarray,
    reference_cr: float | None,
) -> dict[str, float | str]:
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        raise RuntimeError(f"Aucun echantillon bootstrap valide pour Mach={mach:.6f}.")

    q05, q16, q50, q84, q95 = np.quantile(valid, [0.05, 0.16, 0.50, 0.84, 0.95])
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0

    if reference_cr is None or not np.isfinite(reference_cr):
        ref_inside_90 = np.nan
        ref_zscore = np.nan
        ref_tail_prob = np.nan
        ref_minus_target = np.nan
    else:
        ref_inside_90 = float(q05 <= reference_cr <= q95)
        ref_zscore = float((reference_cr - mean) / max(std, 1e-12))
        ref_tail_prob = float(np.mean(valid >= reference_cr)) if reference_cr >= q50 else float(np.mean(valid <= reference_cr))
        ref_minus_target = float(reference_cr - target_level)

    summary: dict[str, float | str] = {
        "alpha_target": float(alpha_target),
        "Mach": float(mach),
        "target_level": float(target_level),
        "bootstrap_valid_count": int(valid.size),
        "bootstrap_valid_fraction": float(valid.size / max(len(samples), 1)),
        "bootstrap_mean": mean,
        "bootstrap_std": std,
        "bootstrap_q05": float(q05),
        "bootstrap_q16": float(q16),
        "bootstrap_q50": float(q50),
        "bootstrap_q84": float(q84),
        "bootstrap_q95": float(q95),
        "bootstrap_ci90_width": float(q95 - q05),
        "bootstrap_ci68_width": float(q84 - q16),
        "support_levels": ",".join(f"{level:.3f}" for level in support_levels),
        "n_support_levels": int(len(support_levels)),
        "reference_cr": np.nan if reference_cr is None else float(reference_cr),
        "reference_minus_target": ref_minus_target,
        "reference_minus_q50": np.nan if reference_cr is None else float(reference_cr - q50),
        "reference_inside_ci90": ref_inside_90,
        "reference_zscore": ref_zscore,
        "reference_tail_probability": ref_tail_prob,
    }
    for idx, level in enumerate(support_levels, start=1):
        pool = support_pools[level]
        summary[f"support_level_{idx}"] = float(level)
        summary[f"support_level_{idx}_points"] = int(len(pool))
        summary[f"support_level_{idx}_mach_min"] = float(pool["Mach"].min())
        summary[f"support_level_{idx}_mach_max"] = float(pool["Mach"].max())
    return summary


def plot_pdf(
    *,
    summary_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for _, row in summary_df.sort_values("Mach").iterrows():
            mach = float(row["Mach"])
            sub = samples_df[np.isclose(samples_df["Mach"].to_numpy(dtype=float), mach)].copy()
            valid = sub["sampled_target_level"].to_numpy(dtype=float)
            valid = valid[np.isfinite(valid)]

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

            axes[0].hist(valid, bins=min(max(len(valid) // 25, 20), 60), color="tab:blue", alpha=0.75, density=True)
            axes[0].axvline(float(row["target_level"]), color="tab:green", linestyle="--", linewidth=2.0, label="target local")
            if np.isfinite(float(row["reference_cr"])):
                axes[0].axvline(float(row["reference_cr"]), color="tab:red", linestyle="-", linewidth=2.0, label="shooting")
            axes[0].axvspan(float(row["bootstrap_q05"]), float(row["bootstrap_q95"]), color="tab:blue", alpha=0.15, label="IC 90%")
            axes[0].axvline(float(row["bootstrap_q50"]), color="black", linestyle=":", linewidth=1.6, label="mediane")
            axes[0].set_xlabel(r"$c_r(\alpha=0.2)$")
            axes[0].set_ylabel("densite")
            axes[0].set_title(f"Distribution bootstrap | M={mach:.3f}")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(fontsize=8, loc="best")

            alpha_cols = [col for col in sub.columns if col.startswith("alpha_on_curve_level_")]
            labels = [float(col.split("_")[-1]) for col in alpha_cols]
            data = [sub[col].to_numpy(dtype=float) for col in alpha_cols]
            tick_labels = [f"{level:.2f}" for level in labels]
            try:
                axes[1].boxplot(data, tick_labels=tick_labels, showfliers=False)
            except TypeError:
                axes[1].boxplot(data, labels=tick_labels, showfliers=False)
            axes[1].axhline(float(row["alpha_target"]), color="tab:red", linestyle=":", linewidth=1.6, label=r"$\alpha$ cible")
            axes[1].set_xlabel(r"Niveaux $c_r$")
            axes[1].set_ylabel(r"$\alpha(M, c_r)$ bootstrap")
            axes[1].set_title(f"Dispersion des courbes de support | M={mach:.3f}")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(fontsize=8, loc="best")

            fig.suptitle(
                f"Uncertainty audit Blumen c_r | alpha={float(row['alpha_target']):.3f}, M={mach:.3f}\n"
                f"target={float(row['target_level']):.6f} | median={float(row['bootstrap_q50']):.6f} | "
                f"std={float(row['bootstrap_std']):.3e} | ref_z={float(row['reference_zscore']):.3f}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=220, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    points = normalize_points(load_digitized_long(args.cr_points))
    reference_lookup = load_reference_lookup(
        Path(args.reference_summary),
        column=str(args.reference_column),
        alpha_target=float(args.alpha),
    )
    rng = np.random.default_rng(int(args.seed))

    summary_rows: list[dict[str, float | str]] = []
    support_rows: list[dict[str, float | str]] = []
    sample_rows: list[pd.DataFrame] = []

    for mach in [float(value) for value in args.mach_values]:
        local_summary, support_anchor_rows, local_points = summarize_target_for_mach(
            points=points,
            mach=mach,
            alpha_target=float(args.alpha),
            quantity="cr",
            local_mach_half_window=float(args.local_mach_half_window),
            local_alpha_half_window=float(args.local_alpha_half_window),
            neighbor_levels=int(args.neighbor_levels),
        )
        target_level = float(local_summary["target_level"])
        support_levels = parse_support_levels(str(local_summary["support_levels"]))

        support_pools: dict[float, pd.DataFrame] = {}
        for level in support_levels:
            pool = choose_support_pool(
                points,
                level=level,
                mach_target=mach,
                alpha_target=float(args.alpha),
                local_mach_half_window=float(args.local_mach_half_window),
                local_alpha_half_window=float(args.local_alpha_half_window),
                min_points=int(args.min_points_per_level),
                max_points=int(args.max_points_per_level),
            )
            if len(pool) >= 2:
                support_pools[level] = pool
                pool = pool.copy()
                pool.insert(0, "support_level", float(level))
                pool.insert(0, "target_mach", float(mach))
                pool.insert(0, "alpha_target", float(args.alpha))
                support_rows.extend(pool.to_dict("records"))

        samples, sample_df = bootstrap_target_level(
            support_pools,
            mach_target=mach,
            alpha_target=float(args.alpha),
            bootstrap_samples=int(args.bootstrap_samples),
            jitter_scale_mach=float(args.jitter_scale_mach),
            jitter_scale_alpha=float(args.jitter_scale_alpha),
            rng=rng,
        )
        sample_df.insert(0, "Mach", float(mach))
        sample_df.insert(0, "alpha_target", float(args.alpha))
        sample_rows.append(sample_df)

        summary_rows.append(
            summarize_distribution(
                mach=mach,
                alpha_target=float(args.alpha),
                target_level=target_level,
                support_levels=list(support_pools.keys()),
                support_pools=support_pools,
                samples=samples,
                reference_cr=reference_lookup.get(float(mach)),
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Mach").reset_index(drop=True)
    support_df = pd.DataFrame(support_rows)
    samples_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    support_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_support_points.csv"
    samples_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_samples.csv"
    pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_audit.pdf"

    summary_df.to_csv(summary_path, index=False)
    support_df.to_csv(support_path, index=False)
    samples_df.to_csv(samples_path, index=False)
    plot_pdf(summary_df=summary_df, samples_df=samples_df, output_path=pdf_path)

    print("Local Blumen c_r uncertainty summary:")
    print(summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {support_path}")
    print(f"Wrote {samples_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
