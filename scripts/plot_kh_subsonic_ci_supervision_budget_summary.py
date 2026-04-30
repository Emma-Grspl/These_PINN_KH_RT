from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plots de synthese pour l'ablation du budget de supervision classique sur c_i."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget/ci_supervision_budget_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot_presentation/subsonic_pinn/ci_supervision_budget"),
    )
    parser.add_argument(
        "--comparison-summary-csv",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics/comparison_summary.csv"),
    )
    return parser


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def save_ci_mae_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    x = df["n_alpha_supervision"].to_numpy(dtype=float)
    best = df["best_audit_ci_mae"].to_numpy(dtype=float)
    last = df["last_ci_mae"].to_numpy(dtype=float)
    target = float(df["target_ci_mae"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, best, marker="o", linewidth=2.0, color="#0b6e4f", label="Best audit $c_i$ MAE")
    ax.plot(x, last, marker="s", linewidth=2.0, color="#c84c09", label="Last $c_i$ MAE")
    ax.axhline(target, color="#4a4a4a", linestyle=":", linewidth=1.6, label="Target MAE")

    min_best_idx = int(np.argmin(best))
    ax.scatter([x[min_best_idx]], [best[min_best_idx]], color="#0b6e4f", s=70, zorder=5)
    ax.annotate(
        f"best={best[min_best_idx]:.2e}\n@ {int(x[min_best_idx])} points",
        xy=(x[min_best_idx], best[min_best_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of classical supervision points")
    ax.set_ylabel("$c_i$ MAE")
    ax.set_title("$c_i$ error vs supervision budget")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = output_dir / "ci_mae_vs_supervision_budget.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_diminishing_returns_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    x = df["n_alpha_supervision"].to_numpy(dtype=float)
    best = df["best_audit_ci_mae"].to_numpy(dtype=float)

    improvement = np.empty_like(best)
    improvement[0] = np.nan
    improvement[1:] = (best[:-1] - best[1:]) / best[:-1]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.plot(x, best, marker="o", linewidth=2.0, color="#1f3a93", label="Best audit $c_i$ MAE")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Number of classical supervision points")
    ax1.set_ylabel("Best audit $c_i$ MAE", color="#1f3a93")
    ax1.tick_params(axis="y", labelcolor="#1f3a93")

    ax2 = ax1.twinx()
    ax2.bar(x[1:], improvement[1:], width=x[1:] * 0.22, color="#d1495b", alpha=0.35, label="Relative gain vs previous budget")
    ax2.set_ylabel("Relative gain", color="#d1495b")
    ax2.tick_params(axis="y", labelcolor="#d1495b")

    ax1.set_title("Diminishing returns beyond a small supervision budget")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()

    path = output_dir / "ci_mae_diminishing_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_budget_metrics_panel(df: pd.DataFrame, output_dir: Path) -> Path:
    x = df["n_alpha_supervision"].to_numpy(dtype=float)
    best = df["best_audit_ci_mae"].to_numpy(dtype=float)
    last = df["last_ci_mae"].to_numpy(dtype=float)
    p_rel = df["last_p_rel"].to_numpy(dtype=float)
    env = df["last_env"].to_numpy(dtype=float)
    phase = df["last_phase"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    axes = axes.ravel()

    axes[0].plot(x, best, marker="o", color="#0b6e4f", linewidth=2.0)
    axes[0].plot(x, last, marker="s", color="#c84c09", linewidth=2.0)
    axes[0].set_title("$c_i$ MAE")

    axes[1].plot(x, p_rel, marker="o", color="#1f3a93", linewidth=2.0)
    axes[1].set_title("Last pressure relative error")

    axes[2].plot(x, env, marker="o", color="#6c5ce7", linewidth=2.0)
    axes[2].set_title("Last envelope relative error")

    axes[3].plot(x, phase, marker="o", color="#d1495b", linewidth=2.0)
    axes[3].set_title("Last phase relative error")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Supervision points")

    fig.suptitle("Hybrid budget ablation at fixed Mach $M=0.5$", y=1.01)
    fig.tight_layout()

    path = output_dir / "ci_supervision_budget_metrics_panel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_readme(df: pd.DataFrame, output_dir: Path) -> Path:
    target = float(df["target_ci_mae"].iloc[0])
    min_best = int(df.loc[df["meets_target_best"], "n_alpha_supervision"].min())
    min_last = int(df.loc[df["meets_target_last"], "n_alpha_supervision"].min())
    path = output_dir / "README.txt"
    lines = [
        "Available plots generated from the summary only:",
        "- ci_mae_vs_supervision_budget.png",
        "- ci_mae_diminishing_returns.png",
        "- ci_supervision_budget_metrics_panel.png",
        "",
        f"Target ci_mae threshold: {target:.3e}",
        f"Minimal budget meeting target on best audit ci_mae: {min_best}",
        f"Minimal budget meeting target on last ci_mae: {min_last}",
        "",
        "Missing for alpha-resolved heatmaps and mode reconstructions:",
        "- model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget/n_alpha_supervision_*/history.csv",
        "- model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget/n_alpha_supervision_*/config.csv",
        "- model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget/n_alpha_supervision_*/model_best.pt",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def maybe_prepend_physics_only(df: pd.DataFrame, comparison_summary_csv: Path) -> pd.DataFrame:
    if not comparison_summary_csv.exists():
        return df
    comp = pd.read_csv(comparison_summary_csv)
    physics = comp.loc[comp["label"] == "physics_only"]
    if physics.empty:
        return df
    row = physics.iloc[0]
    if (df["n_alpha_supervision"] == 0).any():
        return df
    prepend = pd.DataFrame(
        [
            {
                "n_alpha_supervision": 0,
                "best_loss": np.nan,
                "best_epoch": int(row.get("best_epoch", -1)),
                "best_audit_ci_mae": float(row.get("best_audit_ci_mae", np.nan)),
                "best_audit_ci_epoch": int(row.get("best_epoch", -1)),
                "best_audit_p_rel": float(row.get("best_audit_p_rel", np.nan)),
                "last_epoch": int(row.get("last_epoch", -1)),
                "last_loss": np.nan,
                "last_ci_mae": float(row.get("last_ci_mae", np.nan)),
                "last_p_rel": float(row.get("last_p_rel", np.nan)),
                "last_env": np.nan,
                "last_phase": np.nan,
                "last_peak": np.nan,
                "target_ci_mae": float(df["target_ci_mae"].iloc[0]),
                "meets_target_best": bool(row.get("best_audit_ci_mae", np.inf) <= float(df["target_ci_mae"].iloc[0])),
                "meets_target_last": bool(row.get("last_ci_mae", np.inf) <= float(df["target_ci_mae"].iloc[0])),
            }
        ]
    )
    return pd.concat([prepend, df], ignore_index=True).sort_values("n_alpha_supervision").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.summary_csv).sort_values("n_alpha_supervision").reset_index(drop=True)
    df = maybe_prepend_physics_only(df, args.comparison_summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    paths = [
        save_ci_mae_plot(df, args.output_dir),
        save_diminishing_returns_plot(df, args.output_dir),
        save_budget_metrics_panel(df, args.output_dir),
        save_readme(df, args.output_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
