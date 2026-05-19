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
from scripts.audit_supersonic_families_against_blumen import DEFAULT_OUTPUT_DIR  # noqa: E402
from scripts.track_supersonic_shooting_multistart import extract_shooting_profile  # noqa: E402


DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "supersonic_shooting_ci_map_a020_m120_130_summary.csv"


def trapezoid_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit de structure modale pour une branche shooting supersonique."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--center-window", type=float, default=8.0)
    parser.add_argument("--common-grid-size", type=int, default=801)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def infer_summary_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    candidate_triplets = [
        ("best_shooting_cr", "best_shooting_ci", "best_ln_p_start_right"),
        ("shooting_cr", "shooting_ci", "ln_p_start_right"),
        ("cr", "ci", "ln_p_start_right"),
    ]
    for cr_col, ci_col, ln_col in candidate_triplets:
        if {cr_col, ci_col, ln_col}.issubset(df.columns):
            return cr_col, ci_col, ln_col
    raise ValueError(
        "Impossible d'identifier les colonnes cr/ci/ln_p_start_right dans le summary CSV."
    )


def phase_unwrap(p: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(p))


def center_mask(y: np.ndarray, center_window: float) -> np.ndarray:
    return np.abs(y) <= center_window


def zero_crossings(signal: np.ndarray, *, tol: float = 1e-10) -> int:
    signal = np.asarray(signal, dtype=float)
    cleaned = signal.copy()
    cleaned[np.abs(cleaned) < tol] = 0.0
    signs = np.sign(cleaned)
    signs = signs[signs != 0.0]
    if signs.size <= 1:
        return 0
    return int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))


def extrema_count(signal: np.ndarray) -> int:
    values = np.asarray(signal, dtype=float)
    if values.size < 3:
        return 0
    diff = np.diff(values)
    signs = np.sign(diff)
    signs = signs[signs != 0.0]
    if signs.size <= 1:
        return 0
    return int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))


def basic_mode_metrics(y: np.ndarray, p: np.ndarray, *, center_window: float) -> dict[str, float]:
    p_abs = np.abs(p)
    total = max(trapezoid_compat(p_abs, y), 1e-12)
    c_mask = center_mask(y, center_window)
    left_mask = y < 0.0
    right_mask = y >= 0.0

    center_mass = trapezoid_compat(p_abs[c_mask], y[c_mask]) if np.any(c_mask) else np.nan
    left_mass = trapezoid_compat(p_abs[left_mask], y[left_mask]) if np.any(left_mask) else np.nan
    right_mass = trapezoid_compat(p_abs[right_mask], y[right_mask]) if np.any(right_mask) else np.nan

    left_peak = np.max(p_abs[left_mask]) if np.any(left_mask) else np.nan
    right_peak = np.max(p_abs[right_mask]) if np.any(right_mask) else np.nan
    peak_idx = int(np.argmax(p_abs))
    peak_y = float(y[peak_idx])
    peak_abs = float(p_abs[peak_idx])

    p_phase = phase_unwrap(p)
    c_y = y[c_mask]
    c_p = p[c_mask]
    c_phase = p_phase[c_mask]
    c_abs = p_abs[c_mask]

    return {
        "peak_y": peak_y,
        "peak_abs": peak_abs,
        "center8_mass_fraction": np.nan if not np.isfinite(center_mass) else float(center_mass / total),
        "left_mass_fraction": np.nan if not np.isfinite(left_mass) else float(left_mass / total),
        "right_mass_fraction": np.nan if not np.isfinite(right_mass) else float(right_mass / total),
        "left_right_peak_ratio": np.nan
        if (not np.isfinite(left_peak) or not np.isfinite(right_peak) or right_peak <= 0.0)
        else float(left_peak / right_peak),
        "zero_crossings_real_center8": zero_crossings(np.real(c_p)),
        "zero_crossings_imag_center8": zero_crossings(np.imag(c_p)),
        "abs_extrema_center8": extrema_count(c_abs),
        "phase_span_center8": np.nan if c_phase.size == 0 else float(np.max(c_phase) - np.min(c_phase)),
        "peak_y_center8": np.nan if c_abs.size == 0 else float(c_y[int(np.argmax(c_abs))]),
    }


def interpolate_complex(y: np.ndarray, values: np.ndarray, y_common: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    values = np.asarray(values)
    real = np.interp(y_common, y, np.real(values))
    imag = np.interp(y_common, y, np.imag(values))
    return real + 1j * imag


def complex_overlap_metrics(
    y_a: np.ndarray,
    p_a: np.ndarray,
    y_b: np.ndarray,
    p_b: np.ndarray,
    *,
    center_window: float,
    common_grid_size: int,
) -> tuple[float, float]:
    y_common = np.linspace(-center_window, center_window, common_grid_size)
    a_interp = interpolate_complex(y_a, p_a, y_common)
    b_interp = interpolate_complex(y_b, p_b, y_common)

    norm_a = max(trapezoid_compat(np.abs(a_interp) ** 2, y_common), 1e-12)
    norm_b = max(trapezoid_compat(np.abs(b_interp) ** 2, y_common), 1e-12)
    inner = trapezoid_compat(np.real(np.conj(a_interp) * b_interp), y_common) + 1j * trapezoid_compat(
        np.imag(np.conj(a_interp) * b_interp), y_common
    )
    overlap = float(abs(inner) / np.sqrt(norm_a * norm_b))
    phase_shift = np.angle(inner)
    b_aligned = b_interp * np.exp(-1j * phase_shift)
    rel_l2 = float(
        np.sqrt(max(trapezoid_compat(np.abs(a_interp - b_aligned) ** 2, y_common), 0.0) / norm_a)
    )
    return overlap, rel_l2


def plot_point_pdf(
    *,
    summary_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_path: Path,
    center_window: float,
) -> None:
    with PdfPages(output_path) as pdf:
        for _, row in summary_df.sort_values(["alpha", "Mach"]).iterrows():
            alpha = float(row["alpha"])
            mach = float(row["Mach"])
            prof = profiles_df[
                np.isclose(profiles_df["alpha"].to_numpy(dtype=float), alpha)
                & np.isclose(profiles_df["Mach"].to_numpy(dtype=float), mach)
            ].copy()
            if prof.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axes = axes.ravel()
            axes[0].plot(prof["y"], prof["p_real"], color="black", linewidth=2.0)
            axes[0].set_title(r"Re($\hat{p}$)")
            axes[1].plot(prof["y"], prof["p_imag"], color="black", linewidth=2.0)
            axes[1].set_title(r"Im($\hat{p}$)")
            axes[2].plot(prof["y"], prof["p_abs"], color="black", linewidth=2.0)
            axes[2].set_title(r"$|\hat{p}|$")
            axes[3].plot(prof["y"], prof["p_phase"], color="black", linewidth=2.0)
            axes[3].set_title(r"arg($\hat{p}$)")
            axes[2].set_xlabel("y")
            axes[3].set_xlabel("y")

            for ax in axes:
                ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                ax.axvline(-center_window, color="#D97706", linewidth=1.0, alpha=0.6, linestyle=":")
                ax.axvline(center_window, color="#D97706", linewidth=1.0, alpha=0.6, linestyle=":")
                ax.grid(True, alpha=0.25)

            fig.suptitle(
                f"alpha={alpha:.3f}, M={mach:.3f} | "
                f"c_r={float(row['shooting_cr']):.6f}, c_i={float(row['shooting_ci']):.6f}\n"
                f"peak_y={float(row['peak_y']):.3f} | center8_mass={float(row['center8_mass_fraction']):.3f} | "
                f"overlap_prev_mach={float(row['overlap_prev_mach_center8']) if np.isfinite(float(row['overlap_prev_mach_center8'])) else np.nan:.3f}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_overlay_pdf(
    *,
    summary_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_path: Path,
) -> None:
    with PdfPages(output_path) as pdf:
        alpha_values = sorted(float(value) for value in summary_df["alpha"].unique())
        for alpha in alpha_values:
            sub = summary_df[np.isclose(summary_df["alpha"].to_numpy(dtype=float), alpha)].copy()
            if sub.empty:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
            for _, row in sub.sort_values("Mach").iterrows():
                mach = float(row["Mach"])
                prof = profiles_df[
                    np.isclose(profiles_df["alpha"].to_numpy(dtype=float), alpha)
                    & np.isclose(profiles_df["Mach"].to_numpy(dtype=float), mach)
                ].copy()
                if prof.empty:
                    continue
                label = f"M={mach:.3f}, c_i={float(row['shooting_ci']):.4f}"
                axes[0].plot(prof["y"], prof["p_abs"], linewidth=2.0, label=label)
                axes[1].plot(prof["y"], prof["p_phase"], linewidth=2.0, label=label)
            axes[0].set_title(r"$|\hat{p}|$")
            axes[1].set_title(r"arg($\hat{p}$)")
            axes[1].set_xlabel("y")
            for ax in axes:
                ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, frameon=False)
            fig.suptitle(f"Overlays mode structure | alpha={alpha:.3f}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_summary_png(summary_df: pd.DataFrame, output_path: Path) -> None:
    alpha_values = np.array(sorted(summary_df["alpha"].unique()), dtype=float)
    mach_values = np.array(sorted(summary_df["Mach"].unique()), dtype=float)
    panels = [
        ("shooting_ci", r"$c_i$"),
        ("center8_mass_fraction", "center8 mass fraction"),
        ("peak_y", "peak y"),
        ("overlap_prev_mach_center8", "overlap prev Mach"),
        ("rel_l2_prev_mach_center8", "rel L2 prev Mach"),
        ("zero_crossings_real_center8", "zero crossings Re center8"),
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

    summary_raw = pd.read_csv(args.summary_csv)
    if not {"alpha", "Mach"}.issubset(summary_raw.columns):
        raise ValueError(f"{args.summary_csv} doit contenir au moins alpha et Mach.")
    cr_col, ci_col, ln_col = infer_summary_columns(summary_raw)

    mode_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    profile_lookup: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    for _, row in summary_raw.sort_values(["alpha", "Mach"]).iterrows():
        alpha = float(row["alpha"])
        mach = float(row["Mach"])
        solver = Mstab17SupersonicSolver(
            alpha=alpha,
            Mach=mach,
            match_y=float(row["match_y"]) if "match_y" in row.index and pd.notna(row["match_y"]) else 1.0,
            use_mapping=bool(row["use_mapping"]) if "use_mapping" in row.index and pd.notna(row["use_mapping"]) else True,
            mapping_scale=float(row["mapping_scale"]) if "mapping_scale" in row.index and pd.notna(row["mapping_scale"]) else 5.0,
            min_y_limit=float(row["min_y_limit"]) if "min_y_limit" in row.index and pd.notna(row["min_y_limit"]) else 10.0,
            max_y_limit=float(row["max_y_limit"]) if "max_y_limit" in row.index and pd.notna(row["max_y_limit"]) else 80.0,
            y_limit_factor=float(row["y_limit_factor"]) if "y_limit_factor" in row.index and pd.notna(row["y_limit_factor"]) else 4.0,
        )
        profile = extract_shooting_profile(
            solver,
            cr=float(row[cr_col]),
            ci=float(row[ci_col]),
            ln_p_start_right=float(row[ln_col]),
        )
        y = np.asarray(profile["y"], dtype=float)
        p = np.asarray(profile["p"])
        rho = np.asarray(profile["rho"])
        phase = phase_unwrap(p)
        metrics = basic_mode_metrics(y, p, center_window=float(args.center_window))

        mode_rows.append(
            {
                "alpha": alpha,
                "Mach": mach,
                "shooting_cr": float(row[cr_col]),
                "shooting_ci": float(row[ci_col]),
                "ln_p_start_right": float(row[ln_col]),
                "match_y": float(row["match_y"]) if "match_y" in row.index and pd.notna(row["match_y"]) else 1.0,
                "use_mapping": bool(row["use_mapping"]) if "use_mapping" in row.index and pd.notna(row["use_mapping"]) else True,
                "mapping_scale": float(row["mapping_scale"]) if "mapping_scale" in row.index and pd.notna(row["mapping_scale"]) else 5.0,
                "min_y_limit": float(row["min_y_limit"]) if "min_y_limit" in row.index and pd.notna(row["min_y_limit"]) else 10.0,
                "max_y_limit": float(row["max_y_limit"]) if "max_y_limit" in row.index and pd.notna(row["max_y_limit"]) else 80.0,
                "y_limit_factor": float(row["y_limit_factor"]) if "y_limit_factor" in row.index and pd.notna(row["y_limit_factor"]) else 4.0,
                **metrics,
                "overlap_prev_mach_center8": np.nan,
                "rel_l2_prev_mach_center8": np.nan,
                "peak_shift_prev_mach": np.nan,
                "overlap_prev_alpha_center8": np.nan,
                "rel_l2_prev_alpha_center8": np.nan,
                "peak_shift_prev_alpha": np.nan,
            }
        )

        for y_i, p_i, rho_i, phase_i in zip(y, p, rho, phase):
            profile_rows.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "y": float(y_i),
                    "p_real": float(np.real(p_i)),
                    "p_imag": float(np.imag(p_i)),
                    "p_abs": float(np.abs(p_i)),
                    "p_phase": float(phase_i),
                    "rho_real": float(np.real(rho_i)),
                    "rho_imag": float(np.imag(rho_i)),
                    "rho_abs": float(np.abs(rho_i)),
                }
            )
        profile_lookup[(alpha, mach)] = (y, p)

    mode_df = pd.DataFrame(mode_rows).sort_values(["alpha", "Mach"]).reset_index(drop=True)
    profiles_df = pd.DataFrame(profile_rows).sort_values(["alpha", "Mach", "y"]).reset_index(drop=True)

    for alpha in sorted(mode_df["alpha"].unique()):
        idx = mode_df[np.isclose(mode_df["alpha"].to_numpy(dtype=float), float(alpha))].sort_values("Mach").index.tolist()
        for cur_pos in range(1, len(idx)):
            prev_idx = idx[cur_pos - 1]
            cur_idx = idx[cur_pos]
            prev_row = mode_df.loc[prev_idx]
            cur_row = mode_df.loc[cur_idx]
            y_prev, p_prev = profile_lookup[(float(prev_row["alpha"]), float(prev_row["Mach"]))]
            y_cur, p_cur = profile_lookup[(float(cur_row["alpha"]), float(cur_row["Mach"]))]
            overlap, rel_l2 = complex_overlap_metrics(
                y_prev,
                p_prev,
                y_cur,
                p_cur,
                center_window=float(args.center_window),
                common_grid_size=int(args.common_grid_size),
            )
            mode_df.loc[cur_idx, "overlap_prev_mach_center8"] = overlap
            mode_df.loc[cur_idx, "rel_l2_prev_mach_center8"] = rel_l2
            mode_df.loc[cur_idx, "peak_shift_prev_mach"] = abs(float(cur_row["peak_y"]) - float(prev_row["peak_y"]))

    for mach in sorted(mode_df["Mach"].unique()):
        idx = mode_df[np.isclose(mode_df["Mach"].to_numpy(dtype=float), float(mach))].sort_values("alpha").index.tolist()
        for cur_pos in range(1, len(idx)):
            prev_idx = idx[cur_pos - 1]
            cur_idx = idx[cur_pos]
            prev_row = mode_df.loc[prev_idx]
            cur_row = mode_df.loc[cur_idx]
            y_prev, p_prev = profile_lookup[(float(prev_row["alpha"]), float(prev_row["Mach"]))]
            y_cur, p_cur = profile_lookup[(float(cur_row["alpha"]), float(cur_row["Mach"]))]
            overlap, rel_l2 = complex_overlap_metrics(
                y_prev,
                p_prev,
                y_cur,
                p_cur,
                center_window=float(args.center_window),
                common_grid_size=int(args.common_grid_size),
            )
            mode_df.loc[cur_idx, "overlap_prev_alpha_center8"] = overlap
            mode_df.loc[cur_idx, "rel_l2_prev_alpha_center8"] = rel_l2
            mode_df.loc[cur_idx, "peak_shift_prev_alpha"] = abs(float(cur_row["peak_y"]) - float(prev_row["peak_y"]))

    print("Shooting mode audit summary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(mode_df.to_string(index=False))

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    profiles_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_profiles.csv"
    fig_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.png"
    point_pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_profiles.pdf"
    overlay_pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_overlays.pdf"

    mode_df.to_csv(summary_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    plot_summary_png(mode_df, fig_path)
    plot_point_pdf(
        summary_df=mode_df,
        profiles_df=profiles_df,
        output_path=point_pdf_path,
        center_window=float(args.center_window),
    )
    plot_overlay_pdf(
        summary_df=mode_df,
        profiles_df=profiles_df,
        output_path=overlay_pdf_path,
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {profiles_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {point_pdf_path}")
    print(f"Wrote {overlay_pdf_path}")


if __name__ == "__main__":
    main()
