from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

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
        description="Protocole high-alpha step-by-step pour le PINN subsonique."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--stage-alphas", type=float, nargs="+", default=[0.82, 0.84, 0.85])
    parser.add_argument("--anchor-alpha", type=float, default=0.80)
    parser.add_argument("--global-alpha-min", type=float, default=0.80)
    parser.add_argument("--global-alpha-max", type=float, default=0.86)
    parser.add_argument("--stage-forward-pad", type=float, default=0.005)
    parser.add_argument("--epochs-per-stage", type=int, default=1500)
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


def load_warmstart_checkpoint(args: argparse.Namespace) -> Path:
    run_dir = Path(args.warmstart_run_dir)
    checkpoint = Path(args.warmstart_checkpoint) if args.warmstart_checkpoint is not None else run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
    return checkpoint


def stage_window(
    *,
    stage_target: float,
    previous_anchor: float,
    global_alpha_min: float,
    global_alpha_max: float,
    stage_forward_pad: float,
) -> tuple[float, float]:
    alpha_min = max(float(global_alpha_min), float(previous_anchor))
    alpha_max = min(float(global_alpha_max), float(stage_target) + float(stage_forward_pad))
    if not (alpha_min <= stage_target <= alpha_max):
        raise ValueError(
            f"Invalid stage window for alpha_target={stage_target:.3f}: "
            f"[{alpha_min:.3f}, {alpha_max:.3f}]"
        )
    return float(alpha_min), float(alpha_max)


def build_stage_config(
    *,
    mach: float,
    alpha_min: float,
    alpha_max: float,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    initial_model_path: Path,
    output_dir: Path,
    device: str,
) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=float(mach),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        hidden_dim=int(hidden_dim),
        mode_depth=4,
        ci_depth=2,
        activation="tanh",
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=12,
        n_anchor_alpha=24,
        n_norm_interior=256,
        n_reference_alpha=25,
        n_audit_alpha=9,
        n_mode_audit_alpha=5,
        n_mode_audit_y=1201,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        stage_split_epoch=0,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=float(learning_rate),
        mode_branch_lr=float(learning_rate),
        focus_fraction=0.75,
        focus_half_width=0.006,
        neutral_fraction=0.0,
        ci_supervision_neutral_boost=0.0,
        neutral_half_width=0.0,
        error_threshold=0.003,
        mode_error_threshold=0.05,
        max_focus_points=3,
        anchor_strategy="max",
        anchor_half_width=0.08,
        anchor_max_candidates=257,
        mode_center_fraction=0.92,
        mode_center_half_width=0.14,
        w_pde=1.0,
        w_bc=10.0,
        w_norm=1.5,
        w_integral_norm=2.0,
        w_phase=3.0,
        w_ci_supervision=10.0,
        w_ci_stability_outside=0.0,
        w_ci_neutrality=0.0,
        w_ci_low_alpha_zero=0.0,
        w_ci_smoothness=0.02,
        n_ci_spectral_grid=49,
        audit_ci_weight=10.0,
        audit_mode_weight=1.0,
        audit_env_weight=1.0,
        audit_phase_weight=0.9,
        audit_peak_weight=0.35,
        phase_mask_fraction=0.15,
        classic_n_points=561,
        classic_mapping_scale=3.0,
        classic_xi_max=0.99,
        mode_representation="amplitude_phase",
        mode_experts=1,
        initial_model_path=str(initial_model_path),
        initial_model_strict=True,
        output_dir=str(output_dir),
        device=str(device),
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
    eval_alpha_min: float | None = None,
    eval_alpha_max: float | None = None,
) -> dict[str, float]:
    ref_cache = SubsonicReferenceCache.build(
        mach=cfg.mach,
        alpha_min=cfg.alpha_min if eval_alpha_min is None else float(eval_alpha_min),
        alpha_max=cfg.alpha_max if eval_alpha_max is None else float(eval_alpha_max),
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


def evaluate_checkpoint(
    *,
    cfg: KHSubsonicTrainingConfig,
    checkpoint: Path,
    alpha_target: float,
    device: torch.device,
    eval_alpha_min: float | None = None,
    eval_alpha_max: float | None = None,
) -> dict[str, float]:
    model = build_fixed_mach_model_from_config(cfg).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.mach = float(cfg.mach)
    model.eval()
    return evaluate_target(
        model=model,
        cfg=cfg,
        alpha_target=alpha_target,
        device=device,
        eval_alpha_min=eval_alpha_min,
        eval_alpha_max=eval_alpha_max,
    )


def stage_dir_name(stage_idx: int, stage_alpha: float) -> str:
    return f"stage_{stage_idx:02d}_alpha_{stage_alpha:.3f}".replace(".", "p")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_alphas = [float(value) for value in args.stage_alphas]
    if stage_alphas != sorted(stage_alphas):
        raise ValueError("stage-alphas doit etre trie croissant.")
    if any(alpha <= float(args.anchor_alpha) for alpha in stage_alphas):
        raise ValueError("Chaque stage-alpha doit etre strictement au-dessus de anchor-alpha.")

    initial_checkpoint = load_warmstart_checkpoint(args)
    previous_checkpoint = initial_checkpoint
    previous_anchor = float(args.anchor_alpha)
    device = torch.device(str(args.device))

    stage_rows: list[dict[str, float | str]] = []

    print(
        "Stepwise high-alpha plan "
        f"anchor_alpha={float(args.anchor_alpha):.3f} "
        f"stages={' '.join(f'{alpha:.3f}' for alpha in stage_alphas)} "
        f"global_band=[{float(args.global_alpha_min):.3f}, {float(args.global_alpha_max):.3f}]"
    )

    for stage_idx, stage_alpha in enumerate(stage_alphas, start=1):
        alpha_min, alpha_max = stage_window(
            stage_target=float(stage_alpha),
            previous_anchor=float(previous_anchor),
            global_alpha_min=float(args.global_alpha_min),
            global_alpha_max=float(args.global_alpha_max),
            stage_forward_pad=float(args.stage_forward_pad),
        )
        stage_dir = Path(args.output_dir) / stage_dir_name(stage_idx, stage_alpha)
        cfg = build_stage_config(
            mach=float(args.mach),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            epochs=int(args.epochs_per_stage),
            learning_rate=float(args.learning_rate),
            hidden_dim=int(args.hidden_dim),
            initial_model_path=previous_checkpoint,
            output_dir=stage_dir,
            device=str(args.device),
        )

        pre_metrics = evaluate_checkpoint(
            cfg=cfg,
            checkpoint=previous_checkpoint,
            alpha_target=float(stage_alpha),
            device=device,
        )
        print(
            f"[stage {stage_idx}/{len(stage_alphas)}] "
            f"alpha_target={stage_alpha:.3f} window=[{alpha_min:.3f}, {alpha_max:.3f}] "
            f"warm_ci_err={pre_metrics['target_ci_abs_err']:.3e} "
            f"warm_p_rel={pre_metrics['target_p_rel']:.3e} "
            f"warm_env={pre_metrics['target_env_rel']:.3e} "
            f"warm_phase={pre_metrics['target_phase_rel']:.3e}"
        )

        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)
        history_summary = summarize_history(history)
        post_metrics = evaluate_target(model=model, cfg=cfg, alpha_target=float(stage_alpha), device=device)

        final_target_metrics = evaluate_target(
            model=model,
            cfg=cfg,
            alpha_target=float(stage_alphas[-1]),
            device=device,
            eval_alpha_min=float(args.global_alpha_min),
            eval_alpha_max=float(args.global_alpha_max),
        )

        row = {
            "stage_index": int(stage_idx),
            "stage_alpha_target": float(stage_alpha),
            "stage_alpha_min": float(alpha_min),
            "stage_alpha_max": float(alpha_max),
            "checkpoint_in": str(previous_checkpoint),
            "checkpoint_out": str(stage_dir / "model_best.pt"),
            "warm_ci_err": float(pre_metrics["target_ci_abs_err"]),
            "warm_p_rel": float(pre_metrics["target_p_rel"]),
            "warm_env_rel": float(pre_metrics["target_env_rel"]),
            "warm_phase_rel": float(pre_metrics["target_phase_rel"]),
            **{f"stage_{key}": value for key, value in history_summary.items()},
            **{f"post_stage_{key}": value for key, value in post_metrics.items()},
            **{f"post_final_{key}": value for key, value in final_target_metrics.items()},
        }
        stage_rows.append(row)

        print(
            f"[stage {stage_idx}/{len(stage_alphas)}] "
            f"done alpha_target={stage_alpha:.3f} "
            f"post_stage_ci_err={post_metrics['target_ci_abs_err']:.3e} "
            f"post_stage_p_rel={post_metrics['target_p_rel']:.3e} "
            f"post_final_ci_err={final_target_metrics['target_ci_abs_err']:.3e} "
            f"post_final_p_rel={final_target_metrics['target_p_rel']:.3e}"
        )

        previous_checkpoint = stage_dir / "model_best.pt"
        previous_anchor = float(stage_alpha)

    summary_df = pd.DataFrame(stage_rows)
    summary_path = Path(args.output_dir) / "stepwise_highalpha_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    final_row = summary_df.iloc[-1]
    print(
        "Stepwise summary "
        f"final_stage_alpha={float(final_row['stage_alpha_target']):.3f} "
        f"post_final_ci_err={float(final_row['post_final_target_ci_abs_err']):.3e} "
        f"post_final_p_rel={float(final_row['post_final_target_p_rel']):.3e} "
        f"post_final_env={float(final_row['post_final_target_env_rel']):.3e} "
        f"post_final_phase={float(final_row['post_final_target_phase_rel']):.3e}"
    )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
