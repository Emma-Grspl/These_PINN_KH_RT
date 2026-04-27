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

from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep alpha subsonique avec selection hybride du profil PINN selon le regime."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-values", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--n-alpha", type=int, default=17)
    parser.add_argument("--low-alpha-max", type=float, default=0.35)
    parser.add_argument("--mid-alpha-max", type=float, default=0.65)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_hybrid_alpha_sweep"),
    )
    return parser


def resolve_alpha_values(args: argparse.Namespace) -> list[float]:
    if args.alpha_values:
        return [float(alpha) for alpha in args.alpha_values]
    if args.n_alpha <= 1:
        return [float(args.alpha_min)]
    return np.linspace(args.alpha_min, args.alpha_max, int(args.n_alpha)).tolist()


def choose_profile(alpha: float, *, low_alpha_max: float, mid_alpha_max: float) -> dict[str, object]:
    if alpha <= low_alpha_max:
        return {
            "regime": "low_alpha",
            "representation": "cartesian",
            "anchor_strategy": "point_max",
            "w_integral_norm": 0.5,
        }
    if alpha <= mid_alpha_max:
        return {
            "regime": "mid_alpha",
            "representation": "amplitude_phase",
            "anchor_strategy": "max",
            "w_integral_norm": 0.5,
        }
    return {
        "regime": "high_alpha",
        "representation": "amplitude_phase",
        "anchor_strategy": "max",
        "w_integral_norm": 2.0,
    }


def build_singlecase_config(
    *,
    mach: float,
    alpha: float,
    representation: str,
    anchor_strategy: str,
    w_integral_norm: float,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    device: str,
    output_dir: Path,
) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=mach,
        alpha_min=alpha,
        alpha_max=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        mode_depth=4,
        ci_depth=2,
        activation="tanh",
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=8,
        n_anchor_alpha=64,
        n_norm_interior=384,
        n_reference_alpha=1,
        n_audit_alpha=1,
        n_mode_audit_alpha=1,
        n_mode_audit_y=1201,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        focus_fraction=0.0,
        focus_half_width=0.0,
        neutral_fraction=0.0,
        neutral_half_width=0.0,
        error_threshold=0.0,
        mode_error_threshold=0.0,
        max_focus_points=0,
        anchor_strategy=anchor_strategy,
        anchor_half_width=0.10,
        anchor_max_candidates=257,
        mode_center_fraction=0.9,
        mode_center_half_width=0.18,
        w_pde=1.0,
        w_bc=10.0,
        w_norm=1.5,
        w_integral_norm=w_integral_norm,
        w_phase=4.0,
        w_ci_supervision=2.0,
        audit_ci_weight=5.0,
        audit_mode_weight=1.0,
        audit_env_weight=1.0,
        audit_phase_weight=0.5,
        audit_peak_weight=0.25,
        phase_mask_fraction=0.15,
        classic_n_points=561,
        classic_mapping_scale=3.0,
        classic_xi_max=0.99,
        mode_representation=representation,
        output_dir=str(output_dir),
        device=device,
    )


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    audited = history.dropna(subset=["audit_ci_mae"]).copy()
    if audited.empty:
        return {
            "best_epoch": -1,
            "best_ci_mae": np.nan,
            "best_p_rel": np.nan,
            "best_env": np.nan,
            "best_phase": np.nan,
            "last_epoch": -1,
            "last_ci_mae": np.nan,
            "last_p_rel": np.nan,
            "last_env": np.nan,
            "last_phase": np.nan,
        }

    metric = pd.to_numeric(audited["audit_checkpoint_metric"], errors="coerce")
    best_idx = int(metric.idxmin())
    best_row = audited.loc[best_idx]
    last_row = audited.iloc[-1]
    return {
        "best_epoch": int(best_row["epoch"]),
        "best_ci_mae": float(best_row["audit_ci_mae"]),
        "best_p_rel": float(best_row["audit_p_rel_l2_mean"]),
        "best_env": float(best_row["audit_env_rel_mean"]),
        "best_phase": float(best_row["audit_phase_rel_mean"]),
        "last_epoch": int(last_row["epoch"]),
        "last_ci_mae": float(last_row["audit_ci_mae"]),
        "last_p_rel": float(last_row["audit_p_rel_l2_mean"]),
        "last_env": float(last_row["audit_env_rel_mean"]),
        "last_phase": float(last_row["audit_phase_rel_mean"]),
    }


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("best_p_rel", r"Best $p_{rel}$"),
        ("best_env", "Best envelope error"),
        ("best_phase", "Best phase error"),
        ("best_ci_mae", r"Best $c_i$ MAE"),
    ]
    colors = {
        "low_alpha": "#1D4ED8",
        "mid_alpha": "#D97706",
        "high_alpha": "#059669",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    ordered = summary.sort_values("alpha").reset_index(drop=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for regime, group in ordered.groupby("regime", sort=False):
            ax.plot(
                group["alpha"].to_numpy(dtype=float),
                group[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                color=colors.get(str(regime), "#111827"),
                label=str(regime),
            )
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(True, alpha=0.25)
        if metric == "best_phase":
            ax.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = resolve_alpha_values(args)
    rows: list[dict[str, object]] = []
    total = len(alpha_values)

    for idx, alpha in enumerate(alpha_values, start=1):
        profile = choose_profile(
            float(alpha),
            low_alpha_max=float(args.low_alpha_max),
            mid_alpha_max=float(args.mid_alpha_max),
        )
        run_dir = (
            args.output_dir
            / f"alpha_{alpha:.3f}_{profile['regime']}_{profile['representation']}_{profile['anchor_strategy']}_wint_{profile['w_integral_norm']:g}"
        )
        cfg = build_singlecase_config(
            mach=float(args.mach),
            alpha=float(alpha),
            representation=str(profile["representation"]),
            anchor_strategy=str(profile["anchor_strategy"]),
            w_integral_norm=float(profile["w_integral_norm"]),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            hidden_dim=int(args.hidden_dim),
            device=str(args.device),
            output_dir=run_dir,
        )
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

        row = {
            "mach": float(args.mach),
            "alpha": float(alpha),
            "regime": str(profile["regime"]),
            "representation": str(profile["representation"]),
            "anchor_strategy": str(profile["anchor_strategy"]),
            "w_integral_norm": float(profile["w_integral_norm"]),
            "run_dir": str(run_dir),
            **summarize_history(history),
        }
        rows.append(row)
        print(
            f"[{idx}/{total}] alpha={alpha:.3f} regime={row['regime']} "
            f"rep={row['representation']} anchor={row['anchor_strategy']} "
            f"w_int={row['w_integral_norm']:g} best_ci={row['best_ci_mae']:.3e} "
            f"best_p_rel={row['best_p_rel']:.3e} best_env={row['best_env']:.3e} "
            f"best_phase={row['best_phase']:.3e}"
        )

    summary_df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    summary_path = args.output_dir / "hybrid_alpha_sweep_summary.csv"
    fig_path = args.output_dir / "hybrid_alpha_sweep_summary.png"
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, fig_path)
    print(f"Summary written to {summary_path}")
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
