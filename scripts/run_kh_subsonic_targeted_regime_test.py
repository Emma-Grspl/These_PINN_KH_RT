from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.kh_subsonic_sampling import SubsonicReferenceCache  # noqa: E402
from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    PressureModeReferenceCache,
    audit_ci_and_mode,
    compute_mode_diagnostics,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test cible par regime pour le PINN KH subsonique."
    )
    parser.add_argument("--regime", choices=("low_alpha", "high_alpha"), required=True)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-target", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def resolve_alpha_window(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.regime == "low_alpha":
        alpha_target = 0.05 if args.alpha_target is None else float(args.alpha_target)
        alpha_min = 0.05 if args.alpha_min is None else float(args.alpha_min)
        alpha_max = 0.20 if args.alpha_max is None else float(args.alpha_max)
    else:
        alpha_target = 0.85 if args.alpha_target is None else float(args.alpha_target)
        alpha_min = 0.78 if args.alpha_min is None else float(args.alpha_min)
        alpha_max = 0.88 if args.alpha_max is None else float(args.alpha_max)

    if not (alpha_min <= alpha_target <= alpha_max):
        raise ValueError(
            f"Expected alpha_min <= alpha_target <= alpha_max, got "
            f"{alpha_min} <= {alpha_target} <= {alpha_max}."
        )
    return alpha_target, alpha_min, alpha_max


def build_config(args: argparse.Namespace) -> KHSubsonicTrainingConfig:
    alpha_target, alpha_min, alpha_max = resolve_alpha_window(args)

    common = dict(
        mach=float(args.mach),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        mode_depth=4,
        ci_depth=2,
        activation="tanh",
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_anchor_alpha=32,
        n_norm_interior=256,
        n_reference_alpha=61,
        n_audit_alpha=11,
        n_mode_audit_alpha=5,
        n_mode_audit_y=1201,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        error_threshold=0.0,
        mode_error_threshold=0.0,
        max_focus_points=0,
        output_dir=str(args.output_dir),
        device=str(args.device),
    )

    if args.regime == "low_alpha":
        return KHSubsonicTrainingConfig(
            **common,
            n_alpha_supervision=16,
            focus_fraction=0.55,
            focus_half_width=0.02,
            neutral_fraction=0.0,
            neutral_half_width=0.0,
            anchor_strategy="point",
            anchor_half_width=0.12,
            anchor_max_candidates=257,
            mode_center_fraction=0.6,
            mode_center_half_width=0.25,
            w_pde=1.0,
            w_bc_kappa=10.0,
            w_bc_q=20.0,
            w_norm=1.0,
            w_integral_norm=1.0,
            w_phase=3.0,
            w_ci_supervision=5.0,
            audit_ci_weight=10.0,
            audit_mode_weight=1.0,
            audit_env_weight=1.0,
            audit_phase_weight=1.5,
            audit_peak_weight=0.25,
            phase_mask_fraction=0.15,
            classic_n_points=561,
            classic_mapping_scale=3.0,
            classic_xi_max=0.99,
            mode_representation="riccati",
            mode_experts=2,
            alpha_split_threshold=0.12,
            separate_branch_optimizers=True,
            detach_ci_in_mode_branch=True,
            ci_branch_lr=float(args.learning_rate),
            mode_branch_lr=float(args.learning_rate),
            riccati_anchor_supervision=True,
            riccati_anchor_n_xi=49,
            riccati_anchor_every=20,
            riccati_anchor_alphas=(alpha_min, 0.10, 0.15, alpha_max),
            w_riccati_anchor=1.5,
            w_q_supervision=1.0,
            q_supervision_n_xi=97,
            q_supervision_every=20,
            q_supervision_alpha_count=8,
            mode_low_alpha_threshold=0.12,
            mode_low_alpha_weight=4.0,
            mode_low_alpha_audit_fraction=0.8,
        )

    return KHSubsonicTrainingConfig(
        **common,
        n_alpha_supervision=32,
        stage_split_epoch=max(int(0.5 * int(args.epochs)), 1),
        stage1_w_ci_supervision=4.0,
        stage2_w_ci_supervision=2.0,
        stage1_neutral_fraction=0.65,
        stage2_neutral_fraction=0.35,
        focus_fraction=0.35,
        focus_half_width=0.015,
        neutral_fraction=0.45,
        ci_supervision_neutral_boost=0.20,
        neutral_half_width=0.025,
        anchor_strategy="max",
        anchor_half_width=0.10,
        anchor_max_candidates=257,
        mode_center_fraction=0.9,
        mode_center_half_width=0.18,
        w_pde=1.0,
        w_bc=10.0,
        w_norm=1.5,
        w_integral_norm=2.0,
        w_phase=3.0,
        w_ci_supervision=3.0,
        w_ci_stability_outside=20.0,
        w_ci_neutrality=25.0,
        w_ci_smoothness=0.25,
        n_ci_spectral_grid=129,
        audit_ci_weight=8.0,
        audit_mode_weight=1.0,
        audit_env_weight=1.0,
        audit_phase_weight=0.75,
        audit_peak_weight=0.25,
        phase_mask_fraction=0.15,
        classic_n_points=561,
        classic_mapping_scale=3.0,
        classic_xi_max=0.99,
        mode_representation="amplitude_phase",
        mode_experts=1,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=float(args.learning_rate),
        mode_branch_lr=float(args.learning_rate),
    )


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    audited = history.dropna(subset=["audit_checkpoint_metric"]).copy()
    if audited.empty:
        return {
            "best_epoch": -1,
            "best_loss": float("nan"),
            "best_ci_mae": float("nan"),
            "best_p_rel": float("nan"),
            "best_env": float("nan"),
            "best_phase": float("nan"),
        }
    metric = pd.to_numeric(audited["audit_checkpoint_metric"], errors="coerce")
    best_idx = int(metric.idxmin())
    best = audited.loc[best_idx]
    return {
        "best_epoch": int(best["epoch"]),
        "best_loss": float(best["audit_checkpoint_metric"]),
        "best_ci_mae": float(best["audit_ci_mae"]),
        "best_p_rel": float(best["audit_p_rel_l2_mean"]),
        "best_env": float(best["audit_env_rel_mean"]),
        "best_phase": float(best["audit_phase_rel_mean"]),
    }


def evaluate_target(
    *,
    model,
    cfg: KHSubsonicTrainingConfig,
    alpha_target: float,
    device: torch.device,
) -> dict[str, float]:
    ref_cache = SubsonicReferenceCache.build(
        mach=cfg.mach,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        num_alpha=max(int(cfg.n_reference_alpha), 11),
    )
    mode_ref_cache = PressureModeReferenceCache(
        mach=cfg.mach,
        n_points=cfg.classic_n_points,
        mapping_scale=cfg.classic_mapping_scale,
        xi_max=cfg.classic_xi_max,
    )

    band_metrics, _, _, _, _ = audit_ci_and_mode(
        model,
        ref_cache,
        mode_ref_cache,
        cfg,
        device=device,
    )

    alpha_t = torch.tensor([[float(alpha_target)]], dtype=torch.float32, device=device)
    with torch.no_grad():
        ci_pred = float(model.get_ci(alpha_t).item())
        ci_true = float(ref_cache.interpolate(alpha_t).item())

    target_diag = compute_mode_diagnostics(
        model,
        alpha=float(alpha_target),
        device=device,
        n_y=cfg.n_mode_audit_y,
        reference_cache=mode_ref_cache,
        phase_mask_fraction=cfg.phase_mask_fraction,
    )

    neutral_alpha = float(np.sqrt(max(1.0 - cfg.mach**2, 0.0))) if cfg.mach < 1.0 else float("nan")
    with torch.no_grad():
        ci_neutral_pred = (
            float(model.get_ci(torch.tensor([[neutral_alpha]], dtype=torch.float32, device=device)).item())
            if np.isfinite(neutral_alpha)
            else float("nan")
        )

    outside_mean_abs_ci = float("nan")
    if np.isfinite(neutral_alpha) and cfg.alpha_max > neutral_alpha:
        alpha_out = torch.linspace(
            neutral_alpha,
            float(cfg.alpha_max),
            17,
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)
        with torch.no_grad():
            outside_mean_abs_ci = float(torch.mean(torch.abs(model.get_ci(alpha_out))).item())

    return {
        "alpha_target": float(alpha_target),
        "target_ci_true": ci_true,
        "target_ci_pred": ci_pred,
        "target_ci_abs_err": abs(ci_pred - ci_true),
        "target_p_rel": float(target_diag["p_rel"]),
        "target_env_rel": float(target_diag["env_rel"]),
        "target_phase_rel": float(target_diag["phase_rel"]),
        "target_peak_shift": float(target_diag["peak_shift"]),
        "neutral_alpha": neutral_alpha,
        "ci_neutral_pred": ci_neutral_pred,
        "outside_mean_abs_ci": outside_mean_abs_ci,
        "band_ci_mae": float(band_metrics["audit_ci_mae"]),
        "band_p_rel": float(band_metrics["audit_p_rel_l2_mean"]),
        "band_env_rel": float(band_metrics["audit_env_rel_mean"]),
        "band_phase_rel": float(band_metrics["audit_phase_rel_mean"]),
        "band_peak_shift": float(band_metrics["audit_peak_shift_mean"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_target, alpha_min, alpha_max = resolve_alpha_window(args)
    cfg = build_config(args)
    model, history = train_fixed_mach_subsonic_pinn(cfg)
    save_training_artifacts(model, history, cfg)

    device = torch.device(cfg.device)
    summary = {
        "regime": str(args.regime),
        "mach": float(args.mach),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        **summarize_history(history),
        **evaluate_target(model=model, cfg=cfg, alpha_target=alpha_target, device=device),
    }

    summary_path = args.output_dir / "targeted_regime_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(
        f"regime={args.regime} alpha_target={alpha_target:.3f} "
        f"target_ci_err={summary['target_ci_abs_err']:.3e} "
        f"target_p_rel={summary['target_p_rel']:.3e} "
        f"target_env={summary['target_env_rel']:.3e} "
        f"target_phase={summary['target_phase_rel']:.3e}"
    )
    print(
        f"band_ci_mae={summary['band_ci_mae']:.3e} band_p_rel={summary['band_p_rel']:.3e} "
        f"ci_neutral_pred={summary['ci_neutral_pred']:.3e} "
        f"outside_mean_abs_ci={summary['outside_mean_abs_ci']:.3e}"
    )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
