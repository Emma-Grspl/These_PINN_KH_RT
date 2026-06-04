from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_kh_subsonic_riccati_lowalpha_repair import (  # noqa: E402
    _maybe_bool,
    _maybe_float,
    _maybe_int,
    evaluate_candidate,
    load_warmstart_checkpoint,
    prepare_warmstart_eval_run_dir,
)
from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Core-only 1D subsonic Riccati training for joint ci + mode reconstruction on alpha in [0.05, 0.8]."
    )
    parser.add_argument("--warmstart-run-dir", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument("--target-mach", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ci-branch-lr", type=float, default=5e-5)
    parser.add_argument("--mode-branch-lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--num-alpha-ci", type=int, default=41)
    parser.add_argument("--num-alpha-modes", type=int, default=21)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-common", type=int, default=801)
    parser.add_argument("--overlay-alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80])
    parser.add_argument("--low-alpha-threshold", type=float, default=0.5)
    parser.add_argument("--low-alpha-weight", type=float, default=8.0)
    parser.add_argument("--low-alpha-sample-fraction", type=float, default=0.30)
    parser.add_argument("--focus-fraction", type=float, default=0.50)
    parser.add_argument("--focus-half-width", type=float, default=0.04)
    parser.add_argument("--mode-error-threshold", type=float, default=0.05)
    parser.add_argument("--ci-supervision-weight", type=float, default=5.0)
    parser.add_argument("--q-supervision-weight", type=float, default=5.0)
    parser.add_argument("--gamma-supervision-weight", type=float, default=3.0)
    parser.add_argument("--riccati-anchor-weight", type=float, default=2.0)
    parser.add_argument("--boundary-kappa-weight", type=float, default=5.0)
    parser.add_argument("--boundary-q-weight", type=float, default=10.0)
    parser.add_argument("--shooting-match-weight", type=float, default=12.0)
    parser.add_argument("--center-kappa-weight", type=float, default=0.5)
    parser.add_argument("--center-peak-weight", type=float, default=0.5)
    parser.add_argument(
        "--anchor-alphas",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80],
    )
    parser.add_argument("--q-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--gamma-supervision-alphas", type=float, nargs="*", default=None)
    return parser


def build_config(args: argparse.Namespace, warm_config: pd.Series, checkpoint: Path) -> KHSubsonicTrainingConfig:
    q_supervision_alphas = args.q_supervision_alphas
    if q_supervision_alphas is None:
        q_supervision_alphas = args.anchor_alphas
    gamma_supervision_alphas = args.gamma_supervision_alphas
    if gamma_supervision_alphas is None:
        gamma_supervision_alphas = args.anchor_alphas
    target_mach = float(warm_config["mach"]) if args.target_mach is None else float(args.target_mach)
    model_alpha_min = float(warm_config["alpha_min"])
    model_alpha_max = float(warm_config["alpha_max"])

    return KHSubsonicTrainingConfig(
        mach=float(target_mach),
        alpha_min=float(model_alpha_min),
        alpha_max=float(model_alpha_max),
        sampling_alpha_min=float(args.alpha_min),
        sampling_alpha_max=float(args.alpha_max),
        audit_alpha_min=float(args.alpha_min),
        audit_alpha_max=float(args.alpha_max),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(warm_config["hidden_dim"]),
        mode_hidden_dim=_maybe_int(warm_config.get("mode_hidden_dim")),
        ci_hidden_dim=_maybe_int(warm_config.get("ci_hidden_dim")),
        mode_depth=int(warm_config["mode_depth"]),
        ci_depth=int(warm_config["ci_depth"]),
        fixed_scalar_ci=_maybe_bool(warm_config.get("fixed_scalar_ci", False)),
        freeze_ci=False,
        activation=str(warm_config["activation"]),
        fourier_features=int(warm_config.get("fourier_features", 0)),
        fourier_scale=float(warm_config.get("fourier_scale", 2.0)),
        initial_ci=float(warm_config.get("initial_ci", 0.2)),
        mapping_scale=float(warm_config["mapping_scale"]),
        trainable_mapping_scale=False,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=64,
        n_anchor_alpha=32,
        n_norm_interior=256,
        n_reference_alpha=41,
        n_audit_alpha=int(args.num_alpha_ci),
        n_mode_audit_alpha=int(args.num_alpha_modes),
        n_mode_audit_y=801,
        audit_every=250,
        checkpoint_every=int(args.checkpoint_every),
        enable_classic_ci_supervision=True,
        enable_classic_mode_audit=True,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=float(args.ci_branch_lr),
        mode_branch_lr=float(args.mode_branch_lr),
        focus_fraction=float(args.focus_fraction),
        focus_half_width=float(args.focus_half_width),
        low_alpha_sample_fraction=float(args.low_alpha_sample_fraction),
        low_alpha_sample_threshold=float(args.low_alpha_threshold),
        neutral_fraction=0.0,
        neutral_half_width=float(warm_config.get("neutral_half_width", 0.04)),
        error_threshold=0.01,
        mode_error_threshold=float(args.mode_error_threshold),
        max_focus_points=12,
        anchor_strategy=str(warm_config.get("anchor_strategy", "point")),
        anchor_half_width=float(warm_config.get("anchor_half_width", 0.12)),
        anchor_max_candidates=int(warm_config.get("anchor_max_candidates", 257)),
        mode_center_fraction=float(warm_config.get("mode_center_fraction", 0.5)),
        mode_center_half_width=float(warm_config.get("mode_center_half_width", 0.3)),
        w_pde=float(warm_config.get("w_pde", 1.0)),
        w_bc=float(warm_config.get("w_bc", 10.0)),
        w_bc_kappa=float(warm_config.get("w_bc_kappa", 10.0)),
        w_bc_q=float(warm_config.get("w_bc_q", 20.0)),
        w_norm=float(warm_config.get("w_norm", 1.0)),
        w_integral_norm=float(warm_config.get("w_integral_norm", 1.0)),
        w_phase=float(warm_config.get("w_phase", 1.0)),
        w_peak_slope=float(warm_config.get("w_peak_slope", 0.0)),
        w_peak_curvature=float(warm_config.get("w_peak_curvature", 0.0)),
        w_loc_center=float(warm_config.get("w_loc_center", 0.0)),
        w_loc_spread=float(warm_config.get("w_loc_spread", 0.0)),
        w_ci_supervision=float(args.ci_supervision_weight),
        audit_ci_weight=1.0,
        audit_mode_weight=8.0,
        audit_env_weight=2.0,
        audit_phase_weight=2.0,
        audit_peak_weight=1.0,
        phase_mask_fraction=float(warm_config.get("phase_mask_fraction", 0.15)),
        classic_n_points=int(warm_config.get("classic_n_points", 561)),
        classic_mapping_scale=float(warm_config.get("classic_mapping_scale", 3.0)),
        classic_xi_max=float(warm_config.get("classic_xi_max", 0.99)),
        enforce_mode_symmetry=_maybe_bool(warm_config.get("enforce_mode_symmetry", False)),
        mode_representation=str(warm_config["mode_representation"]),
        mode_experts=int(warm_config.get("mode_experts", 1)),
        alpha_split_threshold=_maybe_float(warm_config.get("alpha_split_threshold")),
        riccati_anchor_supervision=True,
        riccati_anchor_n_xi=97,
        riccati_anchor_every=20,
        riccati_anchor_alphas=tuple(float(alpha) for alpha in args.anchor_alphas),
        w_riccati_anchor=float(args.riccati_anchor_weight),
        w_q_supervision=float(args.q_supervision_weight),
        q_supervision_n_xi=129,
        q_supervision_every=10,
        q_supervision_alpha_count=len(q_supervision_alphas),
        q_supervision_alphas=tuple(float(alpha) for alpha in q_supervision_alphas),
        w_riccati_gamma_supervision=float(args.gamma_supervision_weight),
        riccati_gamma_n_xi=129,
        riccati_gamma_every=10,
        riccati_gamma_alpha_count=len(gamma_supervision_alphas),
        riccati_gamma_alphas=tuple(float(alpha) for alpha in gamma_supervision_alphas),
        w_riccati_center_kappa=float(args.center_kappa_weight),
        w_riccati_center_peak=float(args.center_peak_weight),
        w_riccati_boundary_band_kappa=float(args.boundary_kappa_weight),
        w_riccati_boundary_band_q=float(args.boundary_q_weight),
        riccati_center_xi=0.0,
        riccati_boundary_band_points=64,
        riccati_boundary_band_start=0.94,
        riccati_boundary_band_end=0.995,
        w_riccati_shooting_match=float(args.shooting_match_weight),
        riccati_shooting_steps=256,
        riccati_shooting_xi_boundary=0.995,
        mode_low_alpha_threshold=float(args.low_alpha_threshold),
        mode_low_alpha_weight=float(args.low_alpha_weight),
        mode_low_alpha_audit_fraction=0.75,
        initial_model_path=str(checkpoint),
        initial_model_strict=True,
        output_dir=str(args.output_dir),
        device=str(args.device),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    checkpoint = load_warmstart_checkpoint(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    warm_config_path = Path(args.warmstart_run_dir) / "config.csv"
    if not warm_config_path.exists():
        raise FileNotFoundError(f"Warm-start config not found: {warm_config_path}")
    warm_config_df = pd.read_csv(warm_config_path)
    if warm_config_df.empty:
        raise RuntimeError(f"Warm-start config is empty: {warm_config_path}")
    warm_config = warm_config_df.iloc[0]

    cfg = build_config(args, warm_config, checkpoint)
    pd.DataFrame([asdict(cfg)]).to_csv(args.output_dir / "config.csv", index=False)
    pd.DataFrame(
        [
            {
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(checkpoint),
            }
        ]
    ).to_csv(args.output_dir / "warmstart_source.csv", index=False)

    print("Core-only 1D subsonic Riccati protocol")
    print(f"warm start={checkpoint}")
    print(f"mach={cfg.mach:.3f}")
    print(
        f"alpha-range(model)=[{cfg.alpha_min:.3f}, {cfg.alpha_max:.3f}] "
        f"active=[{args.alpha_min:.3f}, {args.alpha_max:.3f}]"
    )
    print(f"epochs={cfg.epochs} lr={cfg.learning_rate:.2e} ci_lr={cfg.ci_branch_lr:.2e} mode_lr={cfg.mode_branch_lr:.2e}")
    print(
        f"weights: ci={cfg.w_ci_supervision:.2f} q_sup={cfg.w_q_supervision:.2f} "
        f"gamma_sup={cfg.w_riccati_gamma_supervision:.2f} anchor={cfg.w_riccati_anchor:.2f} "
        f"bc_k={cfg.w_riccati_boundary_band_kappa:.2f} bc_q={cfg.w_riccati_boundary_band_q:.2f} "
        f"shoot={cfg.w_riccati_shooting_match:.2f}"
    )
    print(
        f"sampling: focus_fraction={cfg.focus_fraction:.2f} focus_half_width={cfg.focus_half_width:.3f} "
        f"low_alpha_fraction={cfg.low_alpha_sample_fraction:.2f} low_alpha_threshold={cfg.low_alpha_sample_threshold:.3f}"
    )

    if args.skip_training:
        model_best_path = Path(cfg.output_dir) / "model_best.pt"
        if not model_best_path.exists():
            raise FileNotFoundError(
                f"Skip-training requested but post-train model is missing: {model_best_path}"
            )
        print(f"Skip training enabled; reusing existing post-train model at {model_best_path}")
    else:
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

    device = torch.device(cfg.device)
    eval_root = Path(cfg.output_dir)
    warm_eval_root = eval_root / "warmstart_eval"
    post_eval_root = eval_root / "posttrain_eval"
    warm_eval_run_dir = prepare_warmstart_eval_run_dir(
        base_run_dir=Path(args.warmstart_run_dir),
        checkpoint=checkpoint if args.warmstart_checkpoint is not None else None,
        eval_root=eval_root,
        eval_config=cfg,
    )

    warm_summary, warm_regimes_df = evaluate_candidate(
        name="warmstart",
        run_dir=warm_eval_run_dir,
        device=device,
        output_root=warm_eval_root,
        num_alpha_ci=args.num_alpha_ci,
        num_alpha_modes=args.num_alpha_modes,
        overlay_alphas=list(args.overlay_alphas),
        n_y_pinn=args.n_y_pinn,
        y_max=args.y_max,
        y_common=args.y_common,
        low_alpha_threshold=args.low_alpha_threshold,
        alpha_min_override=args.alpha_min,
        alpha_max_override=args.alpha_max,
    )
    post_summary, post_regimes_df = evaluate_candidate(
        name="posttrain",
        run_dir=Path(cfg.output_dir),
        device=device,
        output_root=post_eval_root,
        num_alpha_ci=args.num_alpha_ci,
        num_alpha_modes=args.num_alpha_modes,
        overlay_alphas=list(args.overlay_alphas),
        n_y_pinn=args.n_y_pinn,
        y_max=args.y_max,
        y_common=args.y_common,
        low_alpha_threshold=args.low_alpha_threshold,
        alpha_min_override=args.alpha_min,
        alpha_max_override=args.alpha_max,
    )

    lowalpha_summary = pd.DataFrame(
        [
            {
                "regime": "riccati_core1d",
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(checkpoint),
                "output_dir": str(cfg.output_dir),
                "model_alpha_min": float(cfg.alpha_min),
                "model_alpha_max": float(cfg.alpha_max),
                "active_alpha_min": float(args.alpha_min),
                "active_alpha_max": float(args.alpha_max),
                "target_mach": float(cfg.mach),
                "low_alpha_threshold": float(cfg.mode_low_alpha_threshold),
                "low_alpha_weight": float(cfg.mode_low_alpha_weight),
                "low_alpha_sample_fraction": float(cfg.low_alpha_sample_fraction),
                "epochs": int(cfg.epochs),
                "learning_rate": float(cfg.learning_rate),
                "ci_branch_lr": float(cfg.ci_branch_lr or 0.0),
                "mode_branch_lr": float(cfg.mode_branch_lr or 0.0),
                "w_ci_supervision": float(cfg.w_ci_supervision),
                "w_q_supervision": float(cfg.w_q_supervision),
                "w_riccati_gamma_supervision": float(cfg.w_riccati_gamma_supervision),
                "w_riccati_anchor": float(cfg.w_riccati_anchor),
                "w_riccati_boundary_band_kappa": float(cfg.w_riccati_boundary_band_kappa),
                "w_riccati_boundary_band_q": float(cfg.w_riccati_boundary_band_q),
                "w_riccati_shooting_match": float(cfg.w_riccati_shooting_match),
                **{f"warm_{k}": v for k, v in warm_summary.items()},
                **{f"post_{k}": v for k, v in post_summary.items()},
            }
        ]
    )
    improvement_df = pd.DataFrame(
        [
            {
                "delta_ci_mae": float(post_summary["ci_mae"] - warm_summary["ci_mae"]),
                "delta_low_alpha_p_rel_mean": float(post_summary["low_alpha_p_rel_mean"] - warm_summary["low_alpha_p_rel_mean"]),
                "delta_low_alpha_u_rel_mean": float(post_summary["low_alpha_u_rel_mean"] - warm_summary["low_alpha_u_rel_mean"]),
                "delta_low_alpha_v_rel_mean": float(post_summary["low_alpha_v_rel_mean"] - warm_summary["low_alpha_v_rel_mean"]),
                "delta_high_alpha_p_rel_mean": float(post_summary["high_alpha_p_rel_mean"] - warm_summary["high_alpha_p_rel_mean"]),
            }
        ]
    )

    lowalpha_summary_path = eval_root / "core1d_repair_summary.csv"
    improvement_path = eval_root / "core1d_repair_improvement.csv"
    lowalpha_summary.to_csv(lowalpha_summary_path, index=False)
    improvement_df.to_csv(improvement_path, index=False)
    warm_regimes_df.to_csv(
        warm_eval_root / "modes" / "mode_regime_summary.csv",
        index=False,
    )
    post_regimes_df.to_csv(
        post_eval_root / "modes" / "mode_regime_summary.csv",
        index=False,
    )

    print(
        f"Warm summary: ci_mae={warm_summary['ci_mae']:.3e} "
        f"low_p={warm_summary['low_alpha_p_rel_mean']:.3e} "
        f"low_u={warm_summary['low_alpha_u_rel_mean']:.3e} "
        f"high_p={warm_summary['high_alpha_p_rel_mean']:.3e}"
    )
    print(
        f"Post summary: ci_mae={post_summary['ci_mae']:.3e} "
        f"low_p={post_summary['low_alpha_p_rel_mean']:.3e} "
        f"low_u={post_summary['low_alpha_u_rel_mean']:.3e} "
        f"high_p={post_summary['high_alpha_p_rel_mean']:.3e}"
    )
    print(f"Summary written to {lowalpha_summary_path}")
    print(f"Improvement written to {improvement_path}")


if __name__ == "__main__":
    main()
