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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Campagne comparee: supervision legere de c_i (8 points) vs physique pure."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_base_config(args: argparse.Namespace, output_dir: Path) -> KHSubsonicTrainingConfig:
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
        n_anchor_alpha=16,
        n_norm_interior=256,
        n_reference_alpha=81,
        n_audit_alpha=81,
        n_mode_audit_alpha=7,
        n_mode_audit_y=801,
        audit_every=100,
        checkpoint_every=500,
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


def summarize_history(label: str, history: pd.DataFrame) -> dict[str, object]:
    audited = history.dropna(subset=["audit_ci_mae"]).copy()
    if audited.empty:
        return {
            "label": label,
            "best_epoch": -1,
            "best_audit_ci_mae": np.nan,
            "best_audit_p_rel": np.nan,
            "last_epoch": int(history["epoch"].iloc[-1]) if not history.empty else -1,
            "last_ci_mae": np.nan,
            "last_p_rel": np.nan,
        }

    best_idx = int(pd.to_numeric(audited["audit_ci_mae"], errors="coerce").idxmin())
    best_row = audited.loc[best_idx]
    last_row = audited.iloc[-1]
    return {
        "label": label,
        "best_epoch": int(best_row["epoch"]),
        "best_audit_ci_mae": float(best_row["audit_ci_mae"]),
        "best_audit_p_rel": float(best_row["audit_p_rel_l2_mean"]),
        "last_epoch": int(last_row["epoch"]),
        "last_ci_mae": float(last_row["audit_ci_mae"]),
        "last_p_rel": float(last_row["audit_p_rel_l2_mean"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        {
            "label": "hybrid_8pt",
            "subdir": "hybrid_8pt",
            "n_alpha_supervision": 8,
            "enable_classic_ci_supervision": True,
            "w_ci_supervision": 5.0,
        },
        {
            "label": "physics_only",
            "subdir": "physics_only",
            "n_alpha_supervision": 0,
            "enable_classic_ci_supervision": False,
            "w_ci_supervision": 0.0,
        },
    ]

    rows: list[dict[str, object]] = []

    for idx, profile in enumerate(profiles, start=1):
        run_dir = args.output_dir / str(profile["subdir"])
        cfg = build_base_config(args, run_dir)
        cfg.n_alpha_supervision = int(profile["n_alpha_supervision"])
        cfg.enable_classic_ci_supervision = bool(profile["enable_classic_ci_supervision"])
        cfg.w_ci_supervision = float(profile["w_ci_supervision"])

        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

        row = summarize_history(str(profile["label"]), history)
        row["run_dir"] = str(run_dir)
        row["n_alpha_supervision"] = int(profile["n_alpha_supervision"])
        row["enable_classic_ci_supervision"] = bool(profile["enable_classic_ci_supervision"])
        rows.append(row)

        print(
            f"[{idx}/{len(profiles)}] {profile['label']} "
            f"best_audit_ci_mae={row['best_audit_ci_mae']:.3e} "
            f"last_ci_mae={row['last_ci_mae']:.3e} "
            f"last_p_rel={row['last_p_rel']:.3e}"
        )

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
