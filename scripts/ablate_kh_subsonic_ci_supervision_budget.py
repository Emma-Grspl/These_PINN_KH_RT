from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def parse_counts(raw: str) -> list[int]:
    counts = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All supervision counts must be positive.")
        counts.append(value)
    if not counts:
        raise ValueError("At least one supervision count is required.")
    return sorted(set(counts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation du nombre de points classiques utilises pour superviser c_i."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--counts", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--ci-mae-target", type=float, default=0.02)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_config(
    args: argparse.Namespace,
    n_alpha_supervision: int,
    output_dir: Path,
) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=args.mach,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=160,
        mode_depth=4,
        ci_depth=2,
        activation="tanh",
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=n_alpha_supervision,
        n_anchor_alpha=16,
        n_norm_interior=256,
        n_reference_alpha=81,
        n_audit_alpha=21,
        n_mode_audit_alpha=7,
        n_mode_audit_y=801,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        focus_fraction=0.6,
        focus_half_width=0.03,
        neutral_fraction=0.2,
        neutral_half_width=0.04,
        error_threshold=0.01,
        mode_error_threshold=0.12,
        max_focus_points=8,
        anchor_strategy="point",
        anchor_half_width=0.12,
        w_pde=1.0,
        w_bc_kappa=10.0,
        w_bc_q=20.0,
        w_ci_supervision=5.0,
        w_riccati_anchor=1.0,
        riccati_anchor_supervision=True,
        riccati_anchor_n_xi=49,
        riccati_anchor_every=20,
        riccati_anchor_alphas=(0.20, 0.65),
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=1e-3,
        mode_branch_lr=1e-3,
        audit_ci_weight=10.0,
        audit_env_weight=1.0,
        audit_phase_weight=0.5,
        audit_peak_weight=0.25,
        phase_mask_fraction=0.15,
        classic_n_points=561,
        classic_mapping_scale=3.0,
        classic_xi_max=0.99,
        mode_representation="riccati",
        mode_experts=2,
        alpha_split_threshold=0.40,
        output_dir=str(output_dir),
        device=args.device,
    )


def row_from_history(n_alpha_supervision: int, history: pd.DataFrame) -> dict:
    if history.empty:
        return {
            "n_alpha_supervision": n_alpha_supervision,
            "best_loss": np.nan,
            "best_epoch": -1,
            "best_audit_ci_mae": np.nan,
            "best_audit_ci_epoch": -1,
            "best_audit_p_rel": np.nan,
            "last_epoch": -1,
            "last_loss": np.nan,
            "last_ci_mae": np.nan,
            "last_p_rel": np.nan,
            "last_env": np.nan,
            "last_phase": np.nan,
            "last_peak": np.nan,
        }

    best_loss_idx = int(pd.to_numeric(history["loss"], errors="coerce").idxmin())
    best_ci_idx = int(pd.to_numeric(history["audit_ci_mae"], errors="coerce").idxmin())
    best_loss_row = history.loc[best_loss_idx]
    best_ci_row = history.loc[best_ci_idx]
    last_row = history.iloc[-1]

    return {
        "n_alpha_supervision": n_alpha_supervision,
        "best_loss": float(best_loss_row.get("loss", np.nan)),
        "best_epoch": int(best_loss_row.get("epoch", -1)),
        "best_audit_ci_mae": float(best_ci_row.get("audit_ci_mae", np.nan)),
        "best_audit_ci_epoch": int(best_ci_row.get("epoch", -1)),
        "best_audit_p_rel": float(best_ci_row.get("audit_p_rel_l2_mean", np.nan)),
        "last_epoch": int(last_row.get("epoch", -1)),
        "last_loss": float(last_row.get("loss", np.nan)),
        "last_ci_mae": float(last_row.get("audit_ci_mae", np.nan)),
        "last_p_rel": float(last_row.get("audit_p_rel_l2_mean", np.nan)),
        "last_env": float(last_row.get("audit_env_rel_mean", np.nan)),
        "last_phase": float(last_row.get("audit_phase_rel_mean", np.nan)),
        "last_peak": float(last_row.get("audit_peak_shift_mean", np.nan)),
    }


def main() -> None:
    args = build_parser().parse_args()
    counts = parse_counts(args.counts)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for idx, count in enumerate(counts, start=1):
        run_dir = output_dir / f"n_alpha_supervision_{count:03d}"
        cfg = build_config(args, count, run_dir)
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

        row = row_from_history(count, history)
        rows.append(row)

        print(
            f"[{idx}/{len(counts)}] n_alpha_supervision={count} "
            f"best_audit_ci_mae={row['best_audit_ci_mae']:.3e} "
            f"last_ci_mae={row['last_ci_mae']:.3e} "
            f"last_p_rel={row['last_p_rel']:.3e}"
        )

    summary = pd.DataFrame(rows).sort_values("n_alpha_supervision").reset_index(drop=True)
    summary["target_ci_mae"] = float(args.ci_mae_target)
    summary["meets_target_best"] = summary["best_audit_ci_mae"] <= float(args.ci_mae_target)
    summary["meets_target_last"] = summary["last_ci_mae"] <= float(args.ci_mae_target)

    summary_path = output_dir / "ci_supervision_budget_summary.csv"
    summary.to_csv(summary_path, index=False)

    eligible_best = summary.loc[summary["meets_target_best"]]
    eligible_last = summary.loc[summary["meets_target_last"]]
    min_best = (
        int(eligible_best["n_alpha_supervision"].iloc[0]) if not eligible_best.empty else None
    )
    min_last = (
        int(eligible_last["n_alpha_supervision"].iloc[0]) if not eligible_last.empty else None
    )

    report_path = output_dir / "ci_supervision_budget_report.txt"
    report_lines = [
        f"target_ci_mae={float(args.ci_mae_target):.12e}",
        f"counts={','.join(str(v) for v in counts)}",
        f"min_count_meeting_target_best={min_best}",
        f"min_count_meeting_target_last={min_last}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"Summary written to {summary_path}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
