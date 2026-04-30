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


REPRESENTATION_LABELS = {
    "cartesian": "Re/Im",
    "amplitude_phase": "Amp/Phase",
    "log_amplitude_phase": "LogAmp/Phase",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation locale des representations de sortie pour le mode subsonique a Mach fixe."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        default=["cartesian", "amplitude_phase", "log_amplitude_phase"],
        choices=["cartesian", "amplitude_phase", "log_amplitude_phase"],
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/subsonic_pinn_representation_ablation_local"),
    )
    return parser


def build_singlecase_config(
    *,
    mach: float,
    alpha: float,
    representation: str,
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
        anchor_strategy="max",
        anchor_half_width=0.10,
        anchor_max_candidates=257,
        mode_center_fraction=0.9,
        mode_center_half_width=0.18,
        w_pde=1.0,
        w_bc=10.0,
        w_norm=1.5,
        w_integral_norm=2.0,
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
        ("best_ci_mae", r"Best $c_i$ MAE"),
        ("best_p_rel", r"Best $p_{rel}$"),
        ("best_env", "Best envelope error"),
        ("best_phase", "Best phase error"),
    ]
    representations = list(dict.fromkeys(summary["representation"].tolist()))
    alpha_values = sorted(summary["alpha"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        matrix = np.full((len(representations), len(alpha_values)), np.nan, dtype=float)
        for i, rep in enumerate(representations):
            for j, alpha in enumerate(alpha_values):
                sub = summary[(summary["representation"] == rep) & (summary["alpha"] == alpha)]
                if not sub.empty:
                    matrix[i, j] = float(sub.iloc[0][metric])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(alpha_values)), [f"{alpha:.2f}" for alpha in alpha_values])
        ax.set_yticks(range(len(representations)), [REPRESENTATION_LABELS.get(rep, rep) for rep in representations])
        ax.set_xlabel(r"$\alpha$")
        for i in range(len(representations)):
            for j in range(len(alpha_values)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2e}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total = len(args.alpha_values) * len(args.representations)
    counter = 0

    for alpha in args.alpha_values:
        for representation in args.representations:
            counter += 1
            run_dir = args.output_dir / f"alpha_{alpha:.3f}_{representation}"
            cfg = build_singlecase_config(
                mach=float(args.mach),
                alpha=float(alpha),
                representation=str(representation),
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                hidden_dim=int(args.hidden_dim),
                device=str(args.device),
                output_dir=run_dir,
            )
            model, history = train_fixed_mach_subsonic_pinn(cfg)
            save_training_artifacts(model, history, cfg)
            summary = summarize_history(history)
            row = {
                "mach": float(args.mach),
                "alpha": float(alpha),
                "representation": str(representation),
                "representation_label": REPRESENTATION_LABELS.get(str(representation), str(representation)),
                "run_dir": str(run_dir),
                **summary,
            }
            rows.append(row)
            print(
                f"[{counter}/{total}] alpha={alpha:.3f} rep={representation} "
                f"best_ci={row['best_ci_mae']:.3e} best_p_rel={row['best_p_rel']:.3e} "
                f"best_env={row['best_env']:.3e} best_phase={row['best_phase']:.3e}"
            )

    summary_df = pd.DataFrame(rows).sort_values(["alpha", "representation"]).reset_index(drop=True)
    summary_path = args.output_dir / "representation_ablation_summary.csv"
    fig_path = args.output_dir / "representation_ablation_summary.png"
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, fig_path)
    print(f"Summary written to {summary_path}")
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
