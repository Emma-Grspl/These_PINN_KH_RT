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
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config  # noqa: E402
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
        description="Test high-alpha v2 par continuation chaude depuis un checkpoint alpha~0.80."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-target", type=float, default=0.85)
    parser.add_argument("--alpha-min", type=float, default=0.80)
    parser.add_argument("--alpha-max", type=float, default=0.86)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--warmstart-run-dir",
        type=Path,
        default=Path(
            "model_saved/kh_subsonic_fixed_mach_M05_hybrid_alpha_sweep/"
            "alpha_0.800_high_alpha_amplitude_phase_max_wint_2"
        ),
    )
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def build_config(args: argparse.Namespace, initial_model_path: Path) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=float(args.mach),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        mode_depth=4,
        ci_depth=2,
        activation="tanh",
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=24,
        n_anchor_alpha=32,
        n_norm_interior=256,
        n_reference_alpha=41,
        n_audit_alpha=11,
        n_mode_audit_alpha=5,
        n_mode_audit_y=1201,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        stage_split_epoch=0,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=float(args.learning_rate),
        mode_branch_lr=float(args.learning_rate),
        focus_fraction=0.45,
        focus_half_width=0.01,
        neutral_fraction=0.0,
        ci_supervision_neutral_boost=0.0,
        neutral_half_width=0.0,
        error_threshold=0.005,
        mode_error_threshold=0.08,
        max_focus_points=4,
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
        w_ci_supervision=8.0,
        w_ci_stability_outside=0.0,
        w_ci_neutrality=0.0,
        w_ci_low_alpha_zero=0.0,
        w_ci_smoothness=0.05,
        n_ci_spectral_grid=65,
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
        initial_model_path=str(initial_model_path),
        initial_model_strict=True,
        output_dir=str(args.output_dir),
        device=str(args.device),
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

    return {
        "target_ci_true": ci_true,
        "target_ci_pred": ci_pred,
        "target_ci_abs_err": abs(ci_pred - ci_true),
        "target_p_rel": float(target_diag["p_rel"]),
        "target_env_rel": float(target_diag["env_rel"]),
        "target_phase_rel": float(target_diag["phase_rel"]),
        "target_peak_shift": float(target_diag["peak_shift"]),
        "band_ci_mae": float(band_metrics["audit_ci_mae"]),
        "band_p_rel": float(band_metrics["audit_p_rel_l2_mean"]),
        "band_env_rel": float(band_metrics["audit_env_rel_mean"]),
        "band_phase_rel": float(band_metrics["audit_phase_rel_mean"]),
        "band_peak_shift": float(band_metrics["audit_peak_shift_mean"]),
    }


def load_warmstart_checkpoint(args: argparse.Namespace) -> Path:
    run_dir = Path(args.warmstart_run_dir)
    checkpoint = Path(args.warmstart_checkpoint) if args.warmstart_checkpoint is not None else run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
    return checkpoint


def evaluate_checkpoint(
    *,
    cfg: KHSubsonicTrainingConfig,
    checkpoint: Path,
    alpha_target: float,
    device: torch.device,
) -> dict[str, float]:
    model = build_fixed_mach_model_from_config(cfg).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.mach = float(cfg.mach)
    model.eval()
    return evaluate_target(model=model, cfg=cfg, alpha_target=alpha_target, device=device)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_warmstart_checkpoint(args)
    cfg = build_config(args, checkpoint)
    device = torch.device(cfg.device)

    pretrain_metrics = evaluate_checkpoint(
        cfg=cfg,
        checkpoint=checkpoint,
        alpha_target=float(args.alpha_target),
        device=device,
    )
    print(
        "Warm-start metrics "
        f"alpha_target={float(args.alpha_target):.3f} "
        f"ci_err={pretrain_metrics['target_ci_abs_err']:.3e} "
        f"p_rel={pretrain_metrics['target_p_rel']:.3e} "
        f"env={pretrain_metrics['target_env_rel']:.3e} "
        f"phase={pretrain_metrics['target_phase_rel']:.3e}"
    )

    model, history = train_fixed_mach_subsonic_pinn(cfg)
    save_training_artifacts(model, history, cfg)

    summary = {
        "regime": "high_alpha_continuation",
        "mach": float(args.mach),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "alpha_target": float(args.alpha_target),
        "warmstart_run_dir": str(Path(args.warmstart_run_dir)),
        "warmstart_checkpoint": str(checkpoint),
        **summarize_history(history),
        **{f"pre_{key}": value for key, value in pretrain_metrics.items()},
        **evaluate_target(model=model, cfg=cfg, alpha_target=float(args.alpha_target), device=device),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "targeted_regime_summary.csv", index=False)

    print(
        "regime=high_alpha_continuation "
        f"alpha_target={float(args.alpha_target):.3f} "
        f"warm_ci_err={pretrain_metrics['target_ci_abs_err']:.3e} "
        f"warm_p_rel={pretrain_metrics['target_p_rel']:.3e} "
        f"target_ci_err={summary['target_ci_abs_err']:.3e} "
        f"target_p_rel={summary['target_p_rel']:.3e} "
        f"target_env={summary['target_env_rel']:.3e} "
        f"target_phase={summary['target_phase_rel']:.3e}"
    )
    print(
        f"band_ci_mae={summary['band_ci_mae']:.3e} "
        f"band_p_rel={summary['band_p_rel']:.3e}"
    )
    print(f"Summary written to {args.output_dir / 'targeted_regime_summary.csv'}")


if __name__ == "__main__":
    main()
