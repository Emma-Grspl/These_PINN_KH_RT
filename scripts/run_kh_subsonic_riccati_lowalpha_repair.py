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

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
from scripts.compare_kh_subsonic_fixed_mach_modal_candidates import (  # noqa: E402
    Candidate,
    load_candidate,
    load_classic_full_mode,
    plot_ci_outputs,
    plot_mode_heatmaps,
    plot_overlay_pdf,
)
from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-alpha Riccati repair a partir de la base riccati_multibranch."
    )
    parser.add_argument("--warmstart-run-dir", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mode-branch-lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-alpha-ci", type=int, default=41)
    parser.add_argument("--num-alpha-modes", type=int, default=11)
    parser.add_argument("--n-y-pinn", type=int, default=1001)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-common", type=int, default=601)
    parser.add_argument("--overlay-alphas", type=float, nargs="+", default=[0.05, 0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--low-alpha-threshold", type=float, default=0.5)
    parser.add_argument("--low-alpha-weight", type=float, default=6.0)
    parser.add_argument("--mode-error-threshold", type=float, default=0.05)
    parser.add_argument("--q-supervision-weight", type=float, default=4.0)
    parser.add_argument("--gamma-supervision-weight", type=float, default=0.0)
    parser.add_argument("--riccati-anchor-weight", type=float, default=2.0)
    parser.add_argument("--boundary-kappa-weight", type=float, default=5.0)
    parser.add_argument("--boundary-q-weight", type=float, default=8.0)
    parser.add_argument("--shooting-match-weight", type=float, default=10.0)
    parser.add_argument("--center-kappa-weight", type=float, default=0.5)
    parser.add_argument("--center-peak-weight", type=float, default=0.5)
    parser.add_argument("--anchor-alphas", type=float, nargs="*", default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--q-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--gamma-supervision-alphas", type=float, nargs="*", default=None)
    return parser


def _maybe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _maybe_bool(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_warmstart_checkpoint(args: argparse.Namespace) -> Path:
    run_dir = Path(args.warmstart_run_dir)
    checkpoint = Path(args.warmstart_checkpoint) if args.warmstart_checkpoint is not None else run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
    return checkpoint


def build_config(args: argparse.Namespace, warm_config: pd.Series, checkpoint: Path) -> KHSubsonicTrainingConfig:
    q_supervision_alphas = args.q_supervision_alphas
    if q_supervision_alphas is None:
        q_supervision_alphas = args.anchor_alphas
    gamma_supervision_alphas = args.gamma_supervision_alphas
    if gamma_supervision_alphas is None:
        gamma_supervision_alphas = args.anchor_alphas

    return KHSubsonicTrainingConfig(
        mach=float(warm_config["mach"]),
        alpha_min=float(warm_config["alpha_min"]),
        alpha_max=float(warm_config["alpha_max"]),
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
        n_alpha_supervision=32,
        n_anchor_alpha=24,
        n_norm_interior=256,
        n_reference_alpha=41,
        n_audit_alpha=41,
        n_mode_audit_alpha=int(args.num_alpha_modes),
        n_mode_audit_y=801,
        audit_every=250,
        checkpoint_every=500,
        enable_classic_ci_supervision=False,
        enable_classic_mode_audit=True,
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=1e-6,
        mode_branch_lr=float(args.mode_branch_lr or args.learning_rate),
        focus_fraction=0.85,
        focus_half_width=0.04,
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
        mode_low_alpha_audit_fraction=0.85,
        initial_model_path=str(checkpoint),
        initial_model_strict=True,
        output_dir=str(args.output_dir),
        device=str(args.device),
    )


def summarize_regimes(mode_summary_df: pd.DataFrame, low_alpha_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label, mask in [
        ("low_alpha", mode_summary_df["alpha"] <= float(low_alpha_threshold)),
        ("high_alpha", mode_summary_df["alpha"] > float(low_alpha_threshold)),
    ]:
        sub = mode_summary_df.loc[mask].copy()
        row: dict[str, float | int | str] = {"regime": label, "n_alpha": int(mask.sum())}
        for key in ("ci_abs_err", "p_rel", "rho_rel", "u_rel", "v_rel", "amp_rel", "phase_rmse"):
            row[f"{key}_mean"] = float(sub[key].mean())
            row[f"{key}_max"] = float(sub[key].max())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_candidate(
    *,
    name: str,
    run_dir: Path,
    device: torch.device,
    output_root: Path,
    num_alpha_ci: int,
    num_alpha_modes: int,
    overlay_alphas: list[float],
    n_y_pinn: int,
    y_max: float,
    y_common: int,
    low_alpha_threshold: float,
    alpha_min_override: float | None = None,
    alpha_max_override: float | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    candidate = load_candidate(name, run_dir, device)
    alpha_min = float(candidate.config["alpha_min"]) if alpha_min_override is None else float(alpha_min_override)
    alpha_max = float(candidate.config["alpha_max"]) if alpha_max_override is None else float(alpha_max_override)
    alpha_values_ci = np.linspace(alpha_min, alpha_max, int(num_alpha_ci), dtype=float)
    alpha_values_modes = np.linspace(alpha_min, alpha_max, int(num_alpha_modes), dtype=float)

    ci_ref = np.array(
        [RobustSubsonicShootingSolver(alpha=float(alpha), Mach=candidate.mach).solve().ci for alpha in alpha_values_ci],
        dtype=float,
    )
    classic_mode_cache = {float(alpha): load_classic_full_mode(float(alpha), candidate.mach) for alpha in alpha_values_modes}
    for alpha in overlay_alphas:
        if float(alpha) not in classic_mode_cache:
            classic_mode_cache[float(alpha)] = load_classic_full_mode(float(alpha), candidate.mach)

    ci_dir = output_root / "ci"
    modes_dir = output_root / "modes"
    ci_dir.mkdir(parents=True, exist_ok=True)
    modes_dir.mkdir(parents=True, exist_ok=True)

    alpha_tensor = torch.tensor(alpha_values_ci, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = candidate.model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    ci_mae, ci_max = plot_ci_outputs(ci_dir, alpha_values_ci, ci_ref, ci_pred)

    overlay_df = plot_overlay_pdf(
        modes_dir,
        candidate,
        overlay_alphas,
        classic_mode_cache,
        n_y_pinn=int(n_y_pinn),
        device=device,
    )
    mode_summary_df = plot_mode_heatmaps(
        modes_dir,
        candidate,
        alpha_values_modes,
        classic_mode_cache,
        y_max=float(y_max),
        y_common_n=int(y_common),
        n_y_pinn=int(n_y_pinn),
        device=device,
    )
    regime_df = summarize_regimes(mode_summary_df, float(low_alpha_threshold))
    regime_df.to_csv(modes_dir / "mode_regime_summary.csv", index=False)

    summary = {
        "candidate": name,
        "mach": float(candidate.mach),
        "ci_mae": float(ci_mae),
        "ci_max_abs": float(ci_max),
        "mode_p_rel_mean": float(mode_summary_df["p_rel"].mean()),
        "mode_p_rel_max": float(mode_summary_df["p_rel"].max()),
        "mode_u_rel_mean": float(mode_summary_df["u_rel"].mean()),
        "mode_u_rel_max": float(mode_summary_df["u_rel"].max()),
        "mode_v_rel_mean": float(mode_summary_df["v_rel"].mean()),
        "mode_v_rel_max": float(mode_summary_df["v_rel"].max()),
        "mode_phase_rmse_mean": float(mode_summary_df["phase_rmse"].mean()),
        "overlay_p_rel_mean": float(overlay_df["p_rel"].mean()),
        "overlay_phase_rmse_mean": float(overlay_df["phase_rmse"].mean()),
        "low_alpha_p_rel_mean": float(regime_df.loc[regime_df["regime"] == "low_alpha", "p_rel_mean"].iloc[0]),
        "low_alpha_u_rel_mean": float(regime_df.loc[regime_df["regime"] == "low_alpha", "u_rel_mean"].iloc[0]),
        "low_alpha_v_rel_mean": float(regime_df.loc[regime_df["regime"] == "low_alpha", "v_rel_mean"].iloc[0]),
        "high_alpha_p_rel_mean": float(regime_df.loc[regime_df["regime"] == "high_alpha", "p_rel_mean"].iloc[0]),
        "high_alpha_u_rel_mean": float(regime_df.loc[regime_df["regime"] == "high_alpha", "u_rel_mean"].iloc[0]),
        "high_alpha_v_rel_mean": float(regime_df.loc[regime_df["regime"] == "high_alpha", "v_rel_mean"].iloc[0]),
    }
    return summary, regime_df


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_warmstart_checkpoint(args)
    warm_config = pd.read_csv(Path(args.warmstart_run_dir) / "config.csv").iloc[0]
    cfg = build_config(args, warm_config, checkpoint)
    device = torch.device(cfg.device)

    print("Low-alpha Riccati repair plan")
    print(f"warm start={checkpoint}")
    print(f"epochs={cfg.epochs} lr={cfg.learning_rate:.2e} mode_lr={cfg.mode_branch_lr:.2e}")
    print(
        "weights: "
        f"q_sup={cfg.w_q_supervision:.2f} "
        f"gamma_sup={cfg.w_riccati_gamma_supervision:.2f} "
        f"anchor={cfg.w_riccati_anchor:.2f} "
        f"bc_k={cfg.w_riccati_boundary_band_kappa:.2f} "
        f"bc_q={cfg.w_riccati_boundary_band_q:.2f} "
        f"shoot={cfg.w_riccati_shooting_match:.2f}"
    )
    print(
        f"low-alpha focus: threshold={cfg.mode_low_alpha_threshold:.3f} "
        f"weight={cfg.mode_low_alpha_weight:.2f}"
    )

    warm_eval_dir = Path(args.output_dir) / "warmstart_eval"
    warm_summary, warm_regimes = evaluate_candidate(
        name="riccati_multibranch_warmstart",
        run_dir=Path(args.warmstart_run_dir),
        device=device,
        output_root=warm_eval_dir,
        num_alpha_ci=int(args.num_alpha_ci),
        num_alpha_modes=int(args.num_alpha_modes),
        overlay_alphas=[float(alpha) for alpha in args.overlay_alphas],
        n_y_pinn=int(args.n_y_pinn),
        y_max=float(args.y_max),
        y_common=int(args.y_common),
        low_alpha_threshold=float(args.low_alpha_threshold),
    )
    print(
        "Warm summary: "
        f"ci_mae={warm_summary['ci_mae']:.3e} "
        f"low_p={warm_summary['low_alpha_p_rel_mean']:.3e} "
        f"low_u={warm_summary['low_alpha_u_rel_mean']:.3e} "
        f"high_p={warm_summary['high_alpha_p_rel_mean']:.3e}"
    )

    model, history = train_fixed_mach_subsonic_pinn(cfg)
    save_training_artifacts(model, history, cfg)

    post_eval_dir = Path(args.output_dir) / "posttrain_eval"
    post_summary, post_regimes = evaluate_candidate(
        name="riccati_lowalpha_repair",
        run_dir=Path(args.output_dir),
        device=device,
        output_root=post_eval_dir,
        num_alpha_ci=int(args.num_alpha_ci),
        num_alpha_modes=int(args.num_alpha_modes),
        overlay_alphas=[float(alpha) for alpha in args.overlay_alphas],
        n_y_pinn=int(args.n_y_pinn),
        y_max=float(args.y_max),
        y_common=int(args.y_common),
        low_alpha_threshold=float(args.low_alpha_threshold),
    )
    print(
        "Post summary: "
        f"ci_mae={post_summary['ci_mae']:.3e} "
        f"low_p={post_summary['low_alpha_p_rel_mean']:.3e} "
        f"low_u={post_summary['low_alpha_u_rel_mean']:.3e} "
        f"high_p={post_summary['high_alpha_p_rel_mean']:.3e}"
    )

    summary = {
        "regime": "riccati_lowalpha_repair",
        "warmstart_run_dir": str(args.warmstart_run_dir),
        "warmstart_checkpoint": str(checkpoint),
        "output_dir": str(args.output_dir),
        "low_alpha_threshold": float(args.low_alpha_threshold),
        "low_alpha_weight": float(args.low_alpha_weight),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "mode_branch_lr": float(cfg.mode_branch_lr or cfg.learning_rate),
        "w_q_supervision": float(cfg.w_q_supervision),
        "w_riccati_gamma_supervision": float(cfg.w_riccati_gamma_supervision),
        "w_riccati_anchor": float(cfg.w_riccati_anchor),
        "w_riccati_boundary_band_kappa": float(cfg.w_riccati_boundary_band_kappa),
        "w_riccati_boundary_band_q": float(cfg.w_riccati_boundary_band_q),
        "w_riccati_shooting_match": float(cfg.w_riccati_shooting_match),
        **{f"warm_{key}": value for key, value in warm_summary.items() if key != "candidate"},
        **{f"post_{key}": value for key, value in post_summary.items() if key != "candidate"},
    }
    summary_path = Path(args.output_dir) / "lowalpha_repair_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    improvement = {
        "delta_ci_mae": float(summary["post_ci_mae"] - summary["warm_ci_mae"]),
        "delta_low_alpha_p_rel_mean": float(summary["post_low_alpha_p_rel_mean"] - summary["warm_low_alpha_p_rel_mean"]),
        "delta_low_alpha_u_rel_mean": float(summary["post_low_alpha_u_rel_mean"] - summary["warm_low_alpha_u_rel_mean"]),
        "delta_low_alpha_v_rel_mean": float(summary["post_low_alpha_v_rel_mean"] - summary["warm_low_alpha_v_rel_mean"]),
        "delta_high_alpha_p_rel_mean": float(summary["post_high_alpha_p_rel_mean"] - summary["warm_high_alpha_p_rel_mean"]),
    }
    improvement_path = Path(args.output_dir) / "lowalpha_repair_improvement.csv"
    pd.DataFrame([improvement]).to_csv(improvement_path, index=False)

    print(
        f"regime=riccati_lowalpha_repair "
        f"warm_ci_mae={summary['warm_ci_mae']:.3e} "
        f"post_ci_mae={summary['post_ci_mae']:.3e} "
        f"warm_low_p={summary['warm_low_alpha_p_rel_mean']:.3e} "
        f"post_low_p={summary['post_low_alpha_p_rel_mean']:.3e} "
        f"warm_low_u={summary['warm_low_alpha_u_rel_mean']:.3e} "
        f"post_low_u={summary['post_low_alpha_u_rel_mean']:.3e}"
    )
    print(f"Summary written to {summary_path}")
    print(f"Improvement written to {improvement_path}")


if __name__ == "__main__":
    main()
