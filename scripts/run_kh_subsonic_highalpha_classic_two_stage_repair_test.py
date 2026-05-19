from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
        description="Reparation high-alpha en deux etapes: balanced full-mode puis raffinement pressure+overlap."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-target", type=float, default=0.85)
    parser.add_argument("--alpha-min", type=float, default=0.85)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--stage-a-epochs", type=int, default=600)
    parser.add_argument("--stage-a-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage-a-balanced-weight", type=float, default=10.0)
    parser.add_argument("--stage-a-balanced-n-xi", type=int, default=257)
    parser.add_argument("--stage-a-balanced-y-max", type=float, default=8.0)
    parser.add_argument("--stage-a-balanced-center-y-max", type=float, default=4.0)
    parser.add_argument("--stage-a-balanced-l2-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-balanced-overlap-weight", type=float, default=2.0)
    parser.add_argument("--stage-a-balanced-rho-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-balanced-u-weight", type=float, default=0.15)
    parser.add_argument("--stage-a-balanced-v-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-balanced-p-weight", type=float, default=1.0)

    parser.add_argument("--stage-b-epochs", type=int, default=1000)
    parser.add_argument("--stage-b-learning-rate", type=float, default=8e-5)
    parser.add_argument("--stage-b-mode-supervision-weight", type=float, default=8.0)
    parser.add_argument("--stage-b-mode-supervision-n-xi", type=int, default=257)
    parser.add_argument("--stage-b-mode-supervision-y-max", type=float, default=8.0)
    parser.add_argument("--stage-b-balanced-weight", type=float, default=5.0)
    parser.add_argument("--stage-b-balanced-n-xi", type=int, default=257)
    parser.add_argument("--stage-b-balanced-y-max", type=float, default=8.0)
    parser.add_argument("--stage-b-balanced-center-y-max", type=float, default=4.0)
    parser.add_argument("--stage-b-balanced-l2-weight", type=float, default=1.0)
    parser.add_argument("--stage-b-balanced-overlap-weight", type=float, default=2.5)
    parser.add_argument("--stage-b-balanced-rho-weight", type=float, default=1.0)
    parser.add_argument("--stage-b-balanced-u-weight", type=float, default=0.15)
    parser.add_argument("--stage-b-balanced-v-weight", type=float, default=1.0)
    parser.add_argument("--stage-b-balanced-p-weight", type=float, default=1.0)

    parser.add_argument(
        "--warmstart-run-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_highalpha_classic_balanced_full_mode_supervision"),
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


def _common_cfg_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mach": float(args.mach),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "hidden_dim": int(args.hidden_dim),
        "mode_depth": 4,
        "ci_depth": 2,
        "activation": "tanh",
        "mapping_scale": 3.0,
        "n_interior": 512,
        "n_boundary": 64,
        "n_alpha_supervision": 1,
        "n_anchor_alpha": 16,
        "n_norm_interior": 256,
        "n_reference_alpha": 11,
        "n_audit_alpha": 5,
        "n_mode_audit_alpha": 1,
        "n_mode_audit_y": 1201,
        "audit_every": 100,
        "checkpoint_every": 500,
        "enable_classic_ci_supervision": True,
        "enable_classic_mode_audit": True,
        "freeze_ci": True,
        "separate_branch_optimizers": True,
        "detach_ci_in_mode_branch": True,
        "ci_branch_lr": 1e-6,
        "focus_fraction": 1.0,
        "focus_half_width": 0.0,
        "neutral_fraction": 0.0,
        "ci_supervision_neutral_boost": 0.0,
        "neutral_half_width": 0.0,
        "error_threshold": 0.002,
        "mode_error_threshold": 0.02,
        "max_focus_points": 1,
        "anchor_strategy": "max",
        "anchor_half_width": 0.06,
        "anchor_max_candidates": 257,
        "mode_center_fraction": 0.95,
        "mode_center_half_width": 0.12,
        "w_pde": 1.0,
        "w_bc": 10.0,
        "w_norm": 1.0,
        "w_integral_norm": 1.5,
        "w_phase": 1.0,
        "w_peak_slope": 0.5,
        "w_peak_curvature": 0.5,
        "w_loc_center": 1.0,
        "w_loc_spread": 1.0,
        "w_first_order_v_energy": 0.0,
        "w_first_order_amp_cap": 0.1,
        "first_order_amp_cap": 2.0,
        "w_ci_supervision": 0.0,
        "w_ci_stability_outside": 0.0,
        "w_ci_neutrality": 0.0,
        "w_ci_low_alpha_zero": 0.0,
        "w_ci_smoothness": 0.0,
        "n_ci_spectral_grid": 17,
        "audit_ci_weight": 2.0,
        "audit_mode_weight": 8.0,
        "audit_env_weight": 8.0,
        "audit_phase_weight": 6.0,
        "audit_peak_weight": 3.0,
        "phase_mask_fraction": 0.15,
        "classic_n_points": 561,
        "classic_mapping_scale": 3.0,
        "classic_xi_max": 0.99,
        "mode_representation": "amplitude_phase",
        "mode_experts": 1,
        "device": str(args.device),
    }


def build_stage_a_config(args: argparse.Namespace, initial_model_path: Path, output_dir: Path) -> KHSubsonicTrainingConfig:
    common = _common_cfg_kwargs(args)
    return KHSubsonicTrainingConfig(
        **common,
        epochs=int(args.stage_a_epochs),
        learning_rate=float(args.stage_a_learning_rate),
        mode_branch_lr=float(args.stage_a_learning_rate),
        classic_mode_supervision=False,
        classic_full_mode_supervision=False,
        classic_balanced_full_mode_supervision=True,
        classic_balanced_full_mode_supervision_alphas=(float(args.alpha_target),),
        classic_balanced_full_mode_supervision_every=1,
        classic_balanced_full_mode_supervision_n_xi=int(args.stage_a_balanced_n_xi),
        classic_balanced_full_mode_supervision_xi_min=-0.98,
        classic_balanced_full_mode_supervision_xi_max=0.98,
        classic_balanced_full_mode_supervision_y_max=float(args.stage_a_balanced_y_max),
        classic_balanced_full_mode_supervision_center_y_max=float(args.stage_a_balanced_center_y_max),
        classic_balanced_full_mode_l2_weight=float(args.stage_a_balanced_l2_weight),
        classic_balanced_full_mode_overlap_weight=float(args.stage_a_balanced_overlap_weight),
        classic_balanced_full_mode_rho_weight=float(args.stage_a_balanced_rho_weight),
        classic_balanced_full_mode_u_weight=float(args.stage_a_balanced_u_weight),
        classic_balanced_full_mode_v_weight=float(args.stage_a_balanced_v_weight),
        classic_balanced_full_mode_p_weight=float(args.stage_a_balanced_p_weight),
        w_classic_mode_supervision=0.0,
        w_classic_full_mode_supervision=0.0,
        w_classic_balanced_full_mode_supervision=float(args.stage_a_balanced_weight),
        initial_model_path=str(initial_model_path),
        initial_model_strict=True,
        output_dir=str(output_dir),
    )


def build_stage_b_config(args: argparse.Namespace, initial_model_path: Path, output_dir: Path) -> KHSubsonicTrainingConfig:
    common = _common_cfg_kwargs(args)
    return KHSubsonicTrainingConfig(
        **common,
        epochs=int(args.stage_b_epochs),
        learning_rate=float(args.stage_b_learning_rate),
        mode_branch_lr=float(args.stage_b_learning_rate),
        classic_mode_supervision=True,
        classic_mode_supervision_alphas=(float(args.alpha_target),),
        classic_mode_supervision_every=1,
        classic_mode_supervision_n_xi=int(args.stage_b_mode_supervision_n_xi),
        classic_mode_supervision_xi_min=-0.98,
        classic_mode_supervision_xi_max=0.98,
        classic_mode_supervision_y_max=float(args.stage_b_mode_supervision_y_max),
        classic_full_mode_supervision=False,
        classic_balanced_full_mode_supervision=True,
        classic_balanced_full_mode_supervision_alphas=(float(args.alpha_target),),
        classic_balanced_full_mode_supervision_every=1,
        classic_balanced_full_mode_supervision_n_xi=int(args.stage_b_balanced_n_xi),
        classic_balanced_full_mode_supervision_xi_min=-0.98,
        classic_balanced_full_mode_supervision_xi_max=0.98,
        classic_balanced_full_mode_supervision_y_max=float(args.stage_b_balanced_y_max),
        classic_balanced_full_mode_supervision_center_y_max=float(args.stage_b_balanced_center_y_max),
        classic_balanced_full_mode_l2_weight=float(args.stage_b_balanced_l2_weight),
        classic_balanced_full_mode_overlap_weight=float(args.stage_b_balanced_overlap_weight),
        classic_balanced_full_mode_rho_weight=float(args.stage_b_balanced_rho_weight),
        classic_balanced_full_mode_u_weight=float(args.stage_b_balanced_u_weight),
        classic_balanced_full_mode_v_weight=float(args.stage_b_balanced_v_weight),
        classic_balanced_full_mode_p_weight=float(args.stage_b_balanced_p_weight),
        w_classic_mode_supervision=float(args.stage_b_mode_supervision_weight),
        w_classic_full_mode_supervision=0.0,
        w_classic_balanced_full_mode_supervision=float(args.stage_b_balanced_weight),
        initial_model_path=str(initial_model_path),
        initial_model_strict=True,
        output_dir=str(output_dir),
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
            "best_peak": float("nan"),
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
        "best_peak": float(best["audit_peak_shift_mean"]),
    }


def evaluate_target(*, model, cfg: KHSubsonicTrainingConfig, alpha_target: float, device: torch.device) -> dict[str, float]:
    ref_cache = SubsonicReferenceCache.build(
        mach=cfg.mach,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        num_alpha=max(int(cfg.n_reference_alpha), 5),
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
        "band_ci_mae": float(band_metrics["audit_ci_mae"]),
        "band_p_rel": float(band_metrics["audit_p_rel_l2_mean"]),
        "band_env_rel": float(band_metrics["audit_env_rel_mean"]),
        "band_phase_rel": float(band_metrics["audit_phase_rel_mean"]),
        "band_peak_shift": float(band_metrics["audit_peak_shift_mean"]),
        "target_ci_true": ci_true,
        "target_ci_pred": ci_pred,
        "target_ci_abs_err": abs(ci_pred - ci_true),
        "target_p_rel": float(target_diag["p_rel"]),
        "target_env_rel": float(target_diag["env_rel"]),
        "target_phase_rel": float(target_diag["phase_rel"]),
        "target_peak_shift": float(target_diag["peak_shift"]),
    }


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


def run_stage(
    *,
    stage_name: str,
    cfg: KHSubsonicTrainingConfig,
    warmstart_checkpoint: Path,
    alpha_target: float,
    device: torch.device,
) -> dict[str, object]:
    pretrain_metrics = evaluate_checkpoint(
        cfg=cfg,
        checkpoint=warmstart_checkpoint,
        alpha_target=alpha_target,
        device=device,
    )
    print(
        f"[{stage_name}] warm metrics "
        f"alpha_target={alpha_target:.3f} "
        f"ci_err={pretrain_metrics['target_ci_abs_err']:.3e} "
        f"p_rel={pretrain_metrics['target_p_rel']:.3e} "
        f"env={pretrain_metrics['target_env_rel']:.3e} "
        f"phase={pretrain_metrics['target_phase_rel']:.3e} "
        f"peak={pretrain_metrics['target_peak_shift']:.3e}"
    )

    model, history = train_fixed_mach_subsonic_pinn(cfg)
    save_training_artifacts(model, history, cfg)

    summary = {
        "stage_name": stage_name,
        "warmstart_checkpoint": str(warmstart_checkpoint),
        "output_dir": str(cfg.output_dir),
        **{f"warm_{key}": value for key, value in pretrain_metrics.items()},
        **summarize_history(history),
        **evaluate_target(model=model, cfg=cfg, alpha_target=alpha_target, device=device),
    }
    summary_path = Path(cfg.output_dir) / "stage_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(
        f"[{stage_name}] done "
        f"target_ci_err={summary['target_ci_abs_err']:.3e} "
        f"target_p_rel={summary['target_p_rel']:.3e} "
        f"target_env={summary['target_env_rel']:.3e} "
        f"target_phase={summary['target_phase_rel']:.3e} "
        f"target_peak={summary['target_peak_shift']:.3e}"
    )
    print(f"[{stage_name}] summary written to {summary_path}")
    return {
        "model": model,
        "history": history,
        "summary": summary,
        "checkpoint": Path(cfg.output_dir) / "model_best.pt",
        "summary_path": summary_path,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initial_checkpoint = load_warmstart_checkpoint(args)
    device = torch.device(str(args.device))

    print("Two-stage high-alpha repair plan")
    print(f"alpha_target={float(args.alpha_target):.3f} band=[{float(args.alpha_min):.3f}, {float(args.alpha_max):.3f}]")
    print(f"warm start={initial_checkpoint}")
    print(
        f"stage_a: epochs={int(args.stage_a_epochs)} lr={float(args.stage_a_learning_rate):.2e} "
        f"balanced_w={float(args.stage_a_balanced_weight):.2f}"
    )
    print(
        f"stage_b: epochs={int(args.stage_b_epochs)} lr={float(args.stage_b_learning_rate):.2e} "
        f"mode_w={float(args.stage_b_mode_supervision_weight):.2f} "
        f"balanced_w={float(args.stage_b_balanced_weight):.2f}"
    )

    initial_ref_cfg = build_stage_b_config(
        args,
        initial_model_path=initial_checkpoint,
        output_dir=args.output_dir / "_initial_probe",
    )
    initial_metrics = evaluate_checkpoint(
        cfg=initial_ref_cfg,
        checkpoint=initial_checkpoint,
        alpha_target=float(args.alpha_target),
        device=device,
    )

    stage_a_result: dict[str, object] | None = None
    stage_b_input = initial_checkpoint
    if int(args.stage_a_epochs) > 0:
        stage_a_cfg = build_stage_a_config(
            args,
            initial_model_path=initial_checkpoint,
            output_dir=args.output_dir / "stage_a_balanced",
        )
        stage_a_result = run_stage(
            stage_name="stage_a_balanced",
            cfg=stage_a_cfg,
            warmstart_checkpoint=initial_checkpoint,
            alpha_target=float(args.alpha_target),
            device=device,
        )
        stage_b_input = Path(stage_a_result["checkpoint"])

    stage_b_cfg = build_stage_b_config(
        args,
        initial_model_path=stage_b_input,
        output_dir=args.output_dir / "stage_b_pressure_overlap",
    )
    stage_b_result = run_stage(
        stage_name="stage_b_pressure_overlap",
        cfg=stage_b_cfg,
        warmstart_checkpoint=stage_b_input,
        alpha_target=float(args.alpha_target),
        device=device,
    )

    final_summary = {
        "regime": "high_alpha_classic_two_stage_repair",
        "mach": float(args.mach),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "alpha_target": float(args.alpha_target),
        "initial_checkpoint": str(initial_checkpoint),
        "stage_a_ran": bool(int(args.stage_a_epochs) > 0),
        "stage_a_checkpoint": "" if stage_a_result is None else str(stage_a_result["checkpoint"]),
        "stage_b_checkpoint": str(stage_b_result["checkpoint"]),
        "stage_a_epochs": int(args.stage_a_epochs),
        "stage_b_epochs": int(args.stage_b_epochs),
        "stage_a_balanced_weight": float(args.stage_a_balanced_weight),
        "stage_b_mode_supervision_weight": float(args.stage_b_mode_supervision_weight),
        "stage_b_balanced_weight": float(args.stage_b_balanced_weight),
        **{f"initial_{key}": value for key, value in initial_metrics.items()},
        **({} if stage_a_result is None else {f"stage_a_{key}": value for key, value in stage_a_result["summary"].items() if key != "stage_name"}),
        **{f"stage_b_{key}": value for key, value in stage_b_result["summary"].items() if key != "stage_name"},
    }
    summary_path = Path(args.output_dir) / "classic_two_stage_repair_summary.csv"
    pd.DataFrame([final_summary]).to_csv(summary_path, index=False)
    print(
        f"regime=high_alpha_classic_two_stage_repair alpha_target={float(args.alpha_target):.3f} "
        f"initial_p_rel={final_summary['initial_target_p_rel']:.3e} "
        f"stage_b_p_rel={final_summary['stage_b_target_p_rel']:.3e} "
        f"stage_b_env={final_summary['stage_b_target_env_rel']:.3e} "
        f"stage_b_phase={final_summary['stage_b_target_phase_rel']:.3e} "
        f"stage_b_peak={final_summary['stage_b_target_peak_shift']:.3e}"
    )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
