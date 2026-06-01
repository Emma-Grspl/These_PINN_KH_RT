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
)
from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edge-focused modal repair on alpha in [0.2, 0.8] with frozen spectral branch."
    )
    parser.add_argument("--warmstart-run-dir", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mode-branch-lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--num-alpha-ci", type=int, default=41)
    parser.add_argument("--num-alpha-modes", type=int, default=25)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-common", type=int, default=801)
    parser.add_argument("--overlay-alphas", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.50, 0.70, 0.75, 0.80])
    parser.add_argument("--edge-low-threshold", type=float, default=0.30)
    parser.add_argument("--edge-high-threshold", type=float, default=0.70)
    parser.add_argument("--edge-low-weight", type=float, default=3.0)
    parser.add_argument("--edge-high-weight", type=float, default=3.0)
    parser.add_argument("--edge-low-audit-fraction", type=float, default=0.35)
    parser.add_argument("--edge-high-audit-fraction", type=float, default=0.35)
    parser.add_argument("--low-alpha-sample-fraction", type=float, default=0.30)
    parser.add_argument("--high-alpha-sample-fraction", type=float, default=0.30)
    parser.add_argument("--focus-fraction", type=float, default=0.40)
    parser.add_argument("--focus-half-width", type=float, default=0.03)
    parser.add_argument("--mode-error-threshold", type=float, default=0.05)
    parser.add_argument("--audit-ci-weight", type=float, default=0.25)
    parser.add_argument("--audit-p-weight", type=float, default=4.0)
    parser.add_argument("--audit-env-weight", type=float, default=2.0)
    parser.add_argument("--audit-phase-weight", type=float, default=2.0)
    parser.add_argument("--audit-peak-weight", type=float, default=1.0)
    parser.add_argument("--q-supervision-weight", type=float, default=5.0)
    parser.add_argument("--gamma-supervision-weight", type=float, default=3.0)
    parser.add_argument("--riccati-anchor-weight", type=float, default=2.0)
    parser.add_argument("--boundary-kappa-weight", type=float, default=5.0)
    parser.add_argument("--boundary-q-weight", type=float, default=10.0)
    parser.add_argument("--shooting-match-weight", type=float, default=12.0)
    parser.add_argument("--center-kappa-weight", type=float, default=0.5)
    parser.add_argument("--center-peak-weight", type=float, default=0.5)
    parser.add_argument("--balanced-full-mode-weight", type=float, default=1.0)
    parser.add_argument("--balanced-full-mode-u-weight", type=float, default=0.25)
    parser.add_argument("--balanced-full-mode-v-weight", type=float, default=1.0)
    parser.add_argument("--balanced-full-mode-p-weight", type=float, default=1.0)
    parser.add_argument("--balanced-full-mode-rho-weight", type=float, default=1.0)
    parser.add_argument(
        "--anchor-alphas",
        type=float,
        nargs="*",
        default=[0.20, 0.25, 0.30, 0.50, 0.70, 0.75, 0.80],
    )
    parser.add_argument("--q-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--gamma-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--balanced-full-mode-alphas", type=float, nargs="*", default=None)
    return parser


def build_config(args: argparse.Namespace, warm_config: pd.Series, checkpoint: Path) -> KHSubsonicTrainingConfig:
    q_supervision_alphas = args.q_supervision_alphas
    if q_supervision_alphas is None:
        q_supervision_alphas = args.anchor_alphas
    gamma_supervision_alphas = args.gamma_supervision_alphas
    if gamma_supervision_alphas is None:
        gamma_supervision_alphas = args.anchor_alphas
    balanced_full_mode_alphas = args.balanced_full_mode_alphas
    if balanced_full_mode_alphas is None:
        balanced_full_mode_alphas = args.anchor_alphas

    return KHSubsonicTrainingConfig(
        mach=float(warm_config["mach"]),
        alpha_min=float(warm_config["alpha_min"]),
        alpha_max=float(warm_config["alpha_max"]),
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
        freeze_ci=True,
        activation=str(warm_config["activation"]),
        fourier_features=int(warm_config.get("fourier_features", 0)),
        fourier_scale=float(warm_config.get("fourier_scale", 2.0)),
        initial_ci=float(warm_config.get("initial_ci", 0.2)),
        mapping_scale=float(warm_config["mapping_scale"]),
        trainable_mapping_scale=False,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=0,
        n_anchor_alpha=32,
        n_norm_interior=256,
        n_reference_alpha=41,
        n_audit_alpha=int(args.num_alpha_ci),
        n_mode_audit_alpha=int(args.num_alpha_modes),
        n_mode_audit_y=801,
        audit_every=int(args.audit_every),
        checkpoint_every=int(args.checkpoint_every),
        enable_classic_ci_supervision=False,
        enable_classic_mode_audit=True,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=0.0,
        mode_branch_lr=float(args.mode_branch_lr),
        focus_fraction=float(args.focus_fraction),
        focus_half_width=float(args.focus_half_width),
        low_alpha_sample_fraction=float(args.low_alpha_sample_fraction),
        low_alpha_sample_threshold=float(args.edge_low_threshold),
        high_alpha_sample_fraction=float(args.high_alpha_sample_fraction),
        high_alpha_sample_threshold=float(args.edge_high_threshold),
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
        w_ci_supervision=0.0,
        audit_ci_weight=float(args.audit_ci_weight),
        audit_p_weight=float(args.audit_p_weight),
        audit_mode_weight=1.0,
        audit_env_weight=float(args.audit_env_weight),
        audit_phase_weight=float(args.audit_phase_weight),
        audit_peak_weight=float(args.audit_peak_weight),
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
        riccati_anchor_every=10,
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
        mode_low_alpha_threshold=float(args.edge_low_threshold),
        mode_low_alpha_weight=float(args.edge_low_weight),
        mode_low_alpha_audit_fraction=float(args.edge_low_audit_fraction),
        mode_high_alpha_threshold=float(args.edge_high_threshold),
        mode_high_alpha_weight=float(args.edge_high_weight),
        mode_high_alpha_audit_fraction=float(args.edge_high_audit_fraction),
        classic_balanced_full_mode_supervision=bool(args.balanced_full_mode_weight > 0.0),
        classic_balanced_full_mode_supervision_alphas=tuple(float(alpha) for alpha in balanced_full_mode_alphas),
        classic_balanced_full_mode_supervision_every=10,
        classic_balanced_full_mode_supervision_n_xi=257,
        classic_balanced_full_mode_supervision_xi_min=float(warm_config.get("classic_balanced_full_mode_supervision_xi_min", -0.98)),
        classic_balanced_full_mode_supervision_xi_max=float(warm_config.get("classic_balanced_full_mode_supervision_xi_max", 0.98)),
        classic_balanced_full_mode_supervision_y_max=float(warm_config.get("classic_balanced_full_mode_supervision_y_max", 8.0)),
        classic_balanced_full_mode_supervision_center_y_max=float(
            warm_config.get("classic_balanced_full_mode_supervision_center_y_max", 4.0)
        ),
        classic_balanced_full_mode_l2_weight=float(warm_config.get("classic_balanced_full_mode_l2_weight", 1.0)),
        classic_balanced_full_mode_overlap_weight=float(warm_config.get("classic_balanced_full_mode_overlap_weight", 2.0)),
        classic_balanced_full_mode_rho_weight=float(args.balanced_full_mode_rho_weight),
        classic_balanced_full_mode_u_weight=float(args.balanced_full_mode_u_weight),
        classic_balanced_full_mode_v_weight=float(args.balanced_full_mode_v_weight),
        classic_balanced_full_mode_p_weight=float(args.balanced_full_mode_p_weight),
        w_classic_balanced_full_mode_supervision=float(args.balanced_full_mode_weight),
        initial_model_path=str(checkpoint),
        initial_model_strict=True,
        output_dir=str(args.output_dir),
        device=str(args.device),
    )


def summarize_edge_bands(mode_summary_df: pd.DataFrame, edge_low_max: float, edge_high_min: float) -> pd.DataFrame:
    metrics = ("ci_abs_err", "p_rel", "rho_rel", "u_rel", "v_rel", "amp_rel", "phase_rmse")
    masks = [
        ("full_range", np.ones(len(mode_summary_df), dtype=bool)),
        ("lower_edge", mode_summary_df["alpha"] <= float(edge_low_max)),
        ("core", (mode_summary_df["alpha"] > float(edge_low_max)) & (mode_summary_df["alpha"] < float(edge_high_min))),
        ("upper_edge", mode_summary_df["alpha"] >= float(edge_high_min)),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, mask in masks:
        sub = mode_summary_df.loc[mask].copy()
        row: dict[str, float | int | str] = {"band": label, "n_alpha": int(mask.sum())}
        for key in metrics:
            row[f"{key}_mean"] = float(sub[key].mean()) if not sub.empty else float("nan")
            row[f"{key}_max"] = float(sub[key].max()) if not sub.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def band_metric_row(df: pd.DataFrame, band: str) -> dict[str, float]:
    sub = df.loc[df["band"] == band]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {f"{band}_{col}": float(row[col]) for col in df.columns if col not in {"band", "n_alpha"}} | {
        f"{band}_n_alpha": int(row["n_alpha"])
    }


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

    print("Edge-focused subsonic Riccati mode repair")
    print(f"warm start={checkpoint}")
    print(
        f"alpha-range(model)=[{cfg.alpha_min:.3f}, {cfg.alpha_max:.3f}] "
        f"active=[{float(args.alpha_min):.3f}, {float(args.alpha_max):.3f}]"
    )
    print(
        f"epochs={cfg.epochs} lr={cfg.learning_rate:.2e} mode_lr={cfg.mode_branch_lr:.2e} "
        f"freeze_ci={int(cfg.freeze_ci)}"
    )
    print(
        f"edges: low<={cfg.mode_low_alpha_threshold:.3f} (weight={cfg.mode_low_alpha_weight:.2f}) "
        f"high>={cfg.mode_high_alpha_threshold:.3f} (weight={cfg.mode_high_alpha_weight:.2f})"
    )
    print(
        f"sampling: focus_fraction={cfg.focus_fraction:.2f} low_edge_fraction={cfg.low_alpha_sample_fraction:.2f} "
        f"high_edge_fraction={cfg.high_alpha_sample_fraction:.2f}"
    )
    print(
        f"losses: q_sup={cfg.w_q_supervision:.2f} gamma_sup={cfg.w_riccati_gamma_supervision:.2f} "
        f"anchor={cfg.w_riccati_anchor:.2f} full_mode={cfg.w_classic_balanced_full_mode_supervision:.2f}"
    )
    print(
        f"audit metric: ci={cfg.audit_ci_weight:.2f} p={cfg.audit_p_weight:.2f} "
        f"env={cfg.audit_env_weight:.2f} phase={cfg.audit_phase_weight:.2f} peak={cfg.audit_peak_weight:.2f}"
    )

    if args.skip_training:
        model_best_path = Path(cfg.output_dir) / "model_best.pt"
        if not model_best_path.exists():
            raise FileNotFoundError(f"Skip-training requested but post-train model is missing: {model_best_path}")
        print(f"Skip training enabled; reusing existing post-train model at {model_best_path}")
    else:
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

    device = torch.device(cfg.device)
    eval_root = Path(cfg.output_dir)
    warm_eval_root = eval_root / "warmstart_eval"
    post_eval_root = eval_root / "posttrain_eval"

    warm_summary, _ = evaluate_candidate(
        name="warmstart",
        run_dir=Path(args.warmstart_run_dir),
        device=device,
        output_root=warm_eval_root,
        num_alpha_ci=args.num_alpha_ci,
        num_alpha_modes=args.num_alpha_modes,
        overlay_alphas=list(args.overlay_alphas),
        n_y_pinn=args.n_y_pinn,
        y_max=args.y_max,
        y_common=args.y_common,
        low_alpha_threshold=0.5 * (float(args.alpha_min) + float(args.alpha_max)),
        alpha_min_override=float(args.alpha_min),
        alpha_max_override=float(args.alpha_max),
    )
    post_summary, _ = evaluate_candidate(
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
        low_alpha_threshold=0.5 * (float(args.alpha_min) + float(args.alpha_max)),
        alpha_min_override=float(args.alpha_min),
        alpha_max_override=float(args.alpha_max),
    )

    warm_mode_summary = pd.read_csv(warm_eval_root / "modes" / "mode_field_error_summary.csv")
    post_mode_summary = pd.read_csv(post_eval_root / "modes" / "mode_field_error_summary.csv")
    warm_bands_df = summarize_edge_bands(warm_mode_summary, float(args.edge_low_threshold), float(args.edge_high_threshold))
    post_bands_df = summarize_edge_bands(post_mode_summary, float(args.edge_low_threshold), float(args.edge_high_threshold))
    warm_bands_df.to_csv(warm_eval_root / "modes" / "mode_edge_band_summary.csv", index=False)
    post_bands_df.to_csv(post_eval_root / "modes" / "mode_edge_band_summary.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "regime": "riccati_mode_repair_edges",
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(checkpoint),
                "output_dir": str(cfg.output_dir),
                "alpha_min": float(cfg.alpha_min),
                "alpha_max": float(cfg.alpha_max),
                "edge_low_threshold": float(args.edge_low_threshold),
                "edge_high_threshold": float(args.edge_high_threshold),
                "epochs": int(cfg.epochs),
                "learning_rate": float(cfg.learning_rate),
                "mode_branch_lr": float(cfg.mode_branch_lr or 0.0),
                "low_alpha_sample_fraction": float(cfg.low_alpha_sample_fraction),
                "high_alpha_sample_fraction": float(cfg.high_alpha_sample_fraction),
                "w_q_supervision": float(cfg.w_q_supervision),
                "w_riccati_gamma_supervision": float(cfg.w_riccati_gamma_supervision),
                "w_riccati_anchor": float(cfg.w_riccati_anchor),
                "w_classic_balanced_full_mode_supervision": float(cfg.w_classic_balanced_full_mode_supervision),
                **{f"warm_{k}": v for k, v in warm_summary.items()},
                **{f"post_{k}": v for k, v in post_summary.items()},
                **{f"warm_{k}": v for k, v in band_metric_row(warm_bands_df, "lower_edge").items()},
                **{f"warm_{k}": v for k, v in band_metric_row(warm_bands_df, "core").items()},
                **{f"warm_{k}": v for k, v in band_metric_row(warm_bands_df, "upper_edge").items()},
                **{f"post_{k}": v for k, v in band_metric_row(post_bands_df, "lower_edge").items()},
                **{f"post_{k}": v for k, v in band_metric_row(post_bands_df, "core").items()},
                **{f"post_{k}": v for k, v in band_metric_row(post_bands_df, "upper_edge").items()},
            }
        ]
    )
    improvement_df = pd.DataFrame(
        [
            {
                "delta_ci_mae": float(post_summary["ci_mae"] - warm_summary["ci_mae"]),
                "delta_lower_edge_p_rel_mean": float(
                    post_bands_df.loc[post_bands_df["band"] == "lower_edge", "p_rel_mean"].iloc[0]
                    - warm_bands_df.loc[warm_bands_df["band"] == "lower_edge", "p_rel_mean"].iloc[0]
                ),
                "delta_lower_edge_u_rel_mean": float(
                    post_bands_df.loc[post_bands_df["band"] == "lower_edge", "u_rel_mean"].iloc[0]
                    - warm_bands_df.loc[warm_bands_df["band"] == "lower_edge", "u_rel_mean"].iloc[0]
                ),
                "delta_core_p_rel_mean": float(
                    post_bands_df.loc[post_bands_df["band"] == "core", "p_rel_mean"].iloc[0]
                    - warm_bands_df.loc[warm_bands_df["band"] == "core", "p_rel_mean"].iloc[0]
                ),
                "delta_upper_edge_p_rel_mean": float(
                    post_bands_df.loc[post_bands_df["band"] == "upper_edge", "p_rel_mean"].iloc[0]
                    - warm_bands_df.loc[warm_bands_df["band"] == "upper_edge", "p_rel_mean"].iloc[0]
                ),
                "delta_upper_edge_u_rel_mean": float(
                    post_bands_df.loc[post_bands_df["band"] == "upper_edge", "u_rel_mean"].iloc[0]
                    - warm_bands_df.loc[warm_bands_df["band"] == "upper_edge", "u_rel_mean"].iloc[0]
                ),
            }
        ]
    )

    summary_path = eval_root / "mode_repair_edges_summary.csv"
    improvement_path = eval_root / "mode_repair_edges_improvement.csv"
    summary_df.to_csv(summary_path, index=False)
    improvement_df.to_csv(improvement_path, index=False)

    warm_lower = warm_bands_df.loc[warm_bands_df["band"] == "lower_edge"].iloc[0]
    warm_core = warm_bands_df.loc[warm_bands_df["band"] == "core"].iloc[0]
    warm_upper = warm_bands_df.loc[warm_bands_df["band"] == "upper_edge"].iloc[0]
    post_lower = post_bands_df.loc[post_bands_df["band"] == "lower_edge"].iloc[0]
    post_core = post_bands_df.loc[post_bands_df["band"] == "core"].iloc[0]
    post_upper = post_bands_df.loc[post_bands_df["band"] == "upper_edge"].iloc[0]

    print(
        f"Warm summary: ci_mae={warm_summary['ci_mae']:.3e} "
        f"lower_p={warm_lower['p_rel_mean']:.3e} core_p={warm_core['p_rel_mean']:.3e} "
        f"upper_p={warm_upper['p_rel_mean']:.3e}"
    )
    print(
        f"Post summary: ci_mae={post_summary['ci_mae']:.3e} "
        f"lower_p={post_lower['p_rel_mean']:.3e} core_p={post_core['p_rel_mean']:.3e} "
        f"upper_p={post_upper['p_rel_mean']:.3e}"
    )
    print(f"Summary written to {summary_path}")
    print(f"Improvement written to {improvement_path}")


if __name__ == "__main__":
    main()
