from __future__ import annotations

import argparse
from dataclasses import dataclass
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
}

ANCHOR_LABELS = {
    "point": "point",
    "max": "max",
    "point_max": "point_max",
    "band": "band",
}


@dataclass(frozen=True)
class RepairProfile:
    family: str
    representation: str
    anchor_strategy: str
    w_integral_norm: float
    w_phase: float

    @property
    def label(self) -> str:
        return (
            f"{self.family} | "
            f"{REPRESENTATION_LABELS.get(self.representation, self.representation)} | "
            f"{ANCHOR_LABELS.get(self.anchor_strategy, self.anchor_strategy)} | "
            f"w_int={self.w_integral_norm:g} | "
            f"w_phase={self.w_phase:g}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini-sweep de reparation des bords alpha pour le PINN KH subsonique."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.05, 0.85])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_edge_repair_sweep"),
    )
    return parser


def profiles_for_alpha(alpha: float) -> list[RepairProfile]:
    if alpha <= 0.10:
        return [
            RepairProfile("low_edge", "cartesian", "point", 0.25, 4.0),
            RepairProfile("low_edge", "cartesian", "point", 0.50, 4.0),
            RepairProfile("low_edge", "cartesian", "point_max", 0.25, 4.0),
            RepairProfile("low_edge", "cartesian", "point_max", 0.50, 4.0),
            RepairProfile("low_edge", "cartesian", "point_max", 1.00, 4.0),
            RepairProfile("low_edge", "cartesian", "band", 0.25, 4.0),
            RepairProfile("low_edge", "cartesian", "band", 0.50, 4.0),
        ]
    return [
        RepairProfile("high_edge", "amplitude_phase", "max", 1.0, 2.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 1.0, 3.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 2.0, 2.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 2.0, 3.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 2.0, 4.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 4.0, 2.0),
        RepairProfile("high_edge", "amplitude_phase", "max", 4.0, 3.0),
    ]


def build_singlecase_config(
    *,
    mach: float,
    alpha: float,
    profile: RepairProfile,
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
        anchor_strategy=profile.anchor_strategy,
        anchor_half_width=0.10,
        anchor_max_candidates=257,
        mode_center_fraction=0.9,
        mode_center_half_width=0.18,
        w_pde=1.0,
        w_bc=10.0,
        w_norm=1.5,
        w_integral_norm=profile.w_integral_norm,
        w_phase=profile.w_phase,
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
        mode_representation=profile.representation,
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
    ordered = summary.sort_values(["alpha", "best_p_rel"]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for alpha, group in ordered.groupby("alpha", sort=True):
            x = np.arange(len(group))
            ax.plot(
                x,
                group[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                label=fr"$\alpha={alpha:.2f}$",
            )
            ax.set_xticks(x, group["profile_label"].tolist(), rotation=65, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if metric == "best_phase":
            ax.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_by_alpha(summary: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for alpha, group in summary.groupby("alpha"):
        best = group.loc[group["best_p_rel"].idxmin()]
        rows.append(
            {
                "alpha": float(alpha),
                "best_p_rel": float(best["best_p_rel"]),
                "best_env": float(best["best_env"]),
                "best_phase": float(best["best_phase"]),
                "profile_label": str(best["profile_label"]),
            }
        )

    best_df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), squeeze=False)
    columns = [
        ("best_p_rel", "Best p_rel"),
        ("best_env", "Best envelope"),
        ("best_phase", "Best phase"),
    ]
    for ax, (column, title) in zip(axes.ravel(), columns):
        x = np.arange(len(best_df))
        ax.bar(x, best_df[column].to_numpy(dtype=float), color="#2563EB")
        ax.set_xticks(x, [f"{alpha:.2f}" for alpha in best_df["alpha"]])
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        for idx, row in best_df.iterrows():
            ax.text(
                idx,
                float(row[column]),
                str(row["profile_label"]),
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = [float(alpha) for alpha in args.alpha_values]
    rows: list[dict[str, object]] = []
    total = sum(len(profiles_for_alpha(alpha)) for alpha in alpha_values)
    run_idx = 0

    for alpha in alpha_values:
        profiles = profiles_for_alpha(alpha)
        for profile in profiles:
            run_idx += 1
            run_dir = (
                args.output_dir
                / f"alpha_{alpha:.3f}_{profile.family}_{profile.representation}_{profile.anchor_strategy}"
                f"_wint_{profile.w_integral_norm:g}_wphase_{profile.w_phase:g}"
            )
            cfg = build_singlecase_config(
                mach=float(args.mach),
                alpha=alpha,
                profile=profile,
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                hidden_dim=int(args.hidden_dim),
                device=str(args.device),
                output_dir=run_dir,
            )
            result = train_fixed_mach_subsonic_pinn(cfg)
            save_training_artifacts(result, run_dir)
            summary = summarize_history(result.history)
            row = {
                "alpha": alpha,
                "family": profile.family,
                "representation": profile.representation,
                "anchor_strategy": profile.anchor_strategy,
                "w_integral_norm": profile.w_integral_norm,
                "w_phase": profile.w_phase,
                "profile_label": profile.label,
                **summary,
            }
            rows.append(row)
            print(
                f"[{run_idx}/{total}] alpha={alpha:.3f} family={profile.family} "
                f"rep={profile.representation} anchor={profile.anchor_strategy} "
                f"w_int={profile.w_integral_norm:g} w_phase={profile.w_phase:g} "
                f"best_ci={row['best_ci_mae']:.3e} best_p_rel={row['best_p_rel']:.3e} "
                f"best_env={row['best_env']:.3e} best_phase={row['best_phase']:.3e}"
            )

    summary_df = pd.DataFrame(rows).sort_values(
        ["alpha", "family", "representation", "anchor_strategy", "w_integral_norm", "w_phase"]
    )
    summary_path = args.output_dir / "edge_repair_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")

    summary_fig_path = args.output_dir / "edge_repair_sweep_summary.png"
    plot_summary(summary_df, summary_fig_path)
    print(f"Figure written to {summary_fig_path}")

    best_fig_path = args.output_dir / "edge_repair_sweep_best_by_alpha.png"
    plot_best_by_alpha(summary_df, best_fig_path)
    print(f"Best-by-alpha figure written to {best_fig_path}")


if __name__ == "__main__":
    main()
