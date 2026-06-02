from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
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


def build_mlp(in_dim: int, out_dim: int, *, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    net = nn.Sequential(*layers)
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return net


class GuideWindowNet(nn.Module):
    def __init__(self, *, alpha_min: float, alpha_max: float, hidden_dim: int, depth: int, min_width: float):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.min_width = float(min_width)
        self.net = build_mlp(1, 2, hidden_dim=hidden_dim, depth=depth)

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1.0e-8)
        return 2.0 * (alpha - self.alpha_min) / span - 1.0

    def forward(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(self.normalize_alpha(alpha))
        mu = F.softplus(raw[:, 0:1]) + 1.0e-6
        width = self.min_width + F.softplus(raw[:, 1:2])
        return mu, width


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoothness_loss(values: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(values.sum(), alpha, create_graph=True)[0]
    return torch.mean(grad.pow(2))


def build_reference_ci_grid(alpha_values: np.ndarray, mach: float) -> np.ndarray:
    return np.asarray(
        [RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve().ci for alpha in alpha_values],
        dtype=float,
    )


def build_anchor_indices(n_points: int, n_anchors: int) -> np.ndarray:
    if n_anchors < 2:
        raise ValueError("n-guide-anchor-alpha must be at least 2.")
    return np.linspace(0, n_points - 1, n_anchors, dtype=int)


def train_guide(
    *,
    alpha_ref_t: torch.Tensor,
    alpha_anchor_t: torch.Tensor,
    ci_anchor_t: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    hidden_dim: int,
    depth: int,
    epochs: int,
    lr: float,
    min_width: float,
    sigma_penalty: float,
    mu_smoothness: float,
    width_smoothness: float,
    device: torch.device,
) -> tuple[GuideWindowNet, pd.DataFrame]:
    guide = GuideWindowNet(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        hidden_dim=hidden_dim,
        depth=depth,
        min_width=min_width,
    ).to(device)
    optimizer = torch.optim.Adam(guide.parameters(), lr=lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        mu_anchor, width_anchor = guide(alpha_anchor_t)
        loss_anchor = torch.mean((mu_anchor - ci_anchor_t).pow(2))

        alpha_smooth = alpha_ref_t.detach().clone().requires_grad_(True)
        mu_full, width_full = guide(alpha_smooth)
        loss_mu_smooth = smoothness_loss(mu_full, alpha_smooth)
        loss_width_smooth = smoothness_loss(width_full, alpha_smooth)
        loss_width_penalty = torch.mean(width_full.pow(2))

        loss = loss_anchor + sigma_penalty * loss_width_penalty + mu_smoothness * loss_mu_smooth + width_smoothness * loss_width_smooth
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.item()),
                    "anchor_mse": float(loss_anchor.item()),
                    "width_mean": float(width_full.mean().item()),
                    "mu_smoothness": float(loss_mu_smooth.item()),
                    "width_smoothness": float(loss_width_smooth.item()),
                }
            )

    return guide, pd.DataFrame(history)


def plot_guide_window(
    *,
    output_path: Path,
    alpha_ref: np.ndarray,
    ci_ref: np.ndarray,
    guide_mu: np.ndarray,
    guide_width: np.ndarray,
    anchor_alpha: np.ndarray,
    anchor_ci: np.ndarray,
) -> None:
    low = np.clip(guide_mu - guide_width, a_min=0.0, a_max=None)
    high = guide_mu + guide_width
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(alpha_ref, ci_ref, label="Classique dense", linewidth=2.0, color="black")
    axes[0].plot(alpha_ref, guide_mu, label="Guide mu", linewidth=2.0, color="#1f77b4")
    axes[0].fill_between(alpha_ref, low, high, alpha=0.18, color="#1f77b4", label="Fenetre guide")
    axes[0].scatter(anchor_alpha, anchor_ci, label="Ancres", color="#ff7f0e", zorder=5)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(alpha_ref, np.abs(guide_mu - ci_ref), linewidth=2.0, color="#d62728")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$| \mu - c_i^{ref} |$")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Core-only 1D subsonic Riccati training with sparse spectral guide-window: "
            "few classical ci anchors build a guide curve, then the main PINN trains "
            "against the guide instead of dense classical ci supervision."
        )
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
    parser.add_argument("--guide-supervision-weight", type=float, default=1.0)
    parser.add_argument("--guide-window-weight", type=float, default=8.0)
    parser.add_argument("--q-supervision-weight", type=float, default=5.0)
    parser.add_argument("--gamma-supervision-weight", type=float, default=3.0)
    parser.add_argument("--riccati-anchor-weight", type=float, default=2.0)
    parser.add_argument("--boundary-kappa-weight", type=float, default=5.0)
    parser.add_argument("--boundary-q-weight", type=float, default=10.0)
    parser.add_argument("--shooting-match-weight", type=float, default=12.0)
    parser.add_argument("--center-kappa-weight", type=float, default=0.5)
    parser.add_argument("--center-peak-weight", type=float, default=0.5)
    parser.add_argument("--anchor-alphas", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80])
    parser.add_argument("--q-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--gamma-supervision-alphas", type=float, nargs="*", default=None)
    parser.add_argument("--n-guide-reference-alpha", type=int, default=81)
    parser.add_argument("--n-guide-anchor-alpha", type=int, default=9)
    parser.add_argument("--guide-hidden-dim", type=int, default=48)
    parser.add_argument("--guide-depth", type=int, default=2)
    parser.add_argument("--guide-epochs", type=int, default=3000)
    parser.add_argument("--guide-lr", type=float, default=2.0e-3)
    parser.add_argument("--guide-min-width", type=float, default=2.5e-3)
    parser.add_argument("--guide-sigma-penalty", type=float, default=3.0e-2)
    parser.add_argument("--guide-mu-smoothness", type=float, default=1.0e-2)
    parser.add_argument("--guide-width-smoothness", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def build_config(
    args: argparse.Namespace,
    warm_config: pd.Series,
    checkpoint: Path,
    guide_supervision_csv: Path,
    guide_window_csv: Path,
) -> KHSubsonicTrainingConfig:
    q_supervision_alphas = args.q_supervision_alphas if args.q_supervision_alphas is not None else args.anchor_alphas
    gamma_supervision_alphas = args.gamma_supervision_alphas if args.gamma_supervision_alphas is not None else args.anchor_alphas
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
        enable_classic_ci_supervision=False,
        external_ci_supervision_csv=str(guide_supervision_csv),
        ci_window_csv=str(guide_window_csv),
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
        w_ci_supervision=float(args.guide_supervision_weight),
        w_ci_window=float(args.guide_window_weight),
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
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    checkpoint = load_warmstart_checkpoint(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    warm_config_path = Path(args.warmstart_run_dir) / "config.csv"
    if not warm_config_path.exists():
        raise FileNotFoundError(f"Warm-start config not found: {warm_config_path}")
    warm_config_df = pd.read_csv(warm_config_path)
    if warm_config_df.empty:
        raise RuntimeError(f"Warm-start config is empty: {warm_config_path}")
    warm_config = warm_config_df.iloc[0]

    target_mach = float(warm_config["mach"]) if args.target_mach is None else float(args.target_mach)
    device = torch.device(args.device)
    guide_root = args.output_dir / "guide"
    guide_root.mkdir(parents=True, exist_ok=True)

    alpha_ref = np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.n_guide_reference_alpha), dtype=float)
    ci_ref = build_reference_ci_grid(alpha_ref, target_mach)
    anchor_indices = build_anchor_indices(len(alpha_ref), int(args.n_guide_anchor_alpha))
    anchor_alpha = alpha_ref[anchor_indices]
    anchor_ci = ci_ref[anchor_indices]

    alpha_ref_t = torch.tensor(alpha_ref, dtype=torch.float32, device=device).view(-1, 1)
    alpha_anchor_t = torch.tensor(anchor_alpha, dtype=torch.float32, device=device).view(-1, 1)
    ci_anchor_t = torch.tensor(anchor_ci, dtype=torch.float32, device=device).view(-1, 1)

    guide_model, guide_history = train_guide(
        alpha_ref_t=alpha_ref_t,
        alpha_anchor_t=alpha_anchor_t,
        ci_anchor_t=ci_anchor_t,
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        hidden_dim=int(args.guide_hidden_dim),
        depth=int(args.guide_depth),
        epochs=int(args.guide_epochs),
        lr=float(args.guide_lr),
        min_width=float(args.guide_min_width),
        sigma_penalty=float(args.guide_sigma_penalty),
        mu_smoothness=float(args.guide_mu_smoothness),
        width_smoothness=float(args.guide_width_smoothness),
        device=device,
    )
    guide_model.eval()
    with torch.no_grad():
        guide_mu_t, guide_width_t = guide_model(alpha_ref_t)
    guide_mu = guide_mu_t.cpu().numpy().reshape(-1)
    guide_width = guide_width_t.cpu().numpy().reshape(-1)
    guide_curve_csv = guide_root / "guide_curve.csv"
    guide_df = pd.DataFrame(
        {
            "alpha": alpha_ref,
            "ci_reference": ci_ref,
            "guide_mu": guide_mu,
            "guide_width": guide_width,
            "ci_window_low": np.clip(guide_mu - guide_width, a_min=0.0, a_max=None),
            "ci_window_high": guide_mu + guide_width,
            "is_anchor": np.isin(np.arange(len(alpha_ref)), anchor_indices),
        }
    )
    guide_df.to_csv(guide_curve_csv, index=False)
    guide_supervision_csv = guide_root / "guide_supervision_curve.csv"
    pd.DataFrame({"alpha": alpha_ref, "ci_reference": guide_mu}).to_csv(guide_supervision_csv, index=False)
    pd.DataFrame({"alpha": anchor_alpha, "ci_reference": anchor_ci}).to_csv(guide_root / "guide_anchor_points.csv", index=False)
    guide_history.to_csv(guide_root / "guide_history.csv", index=False)
    plot_guide_window(
        output_path=guide_root / "guide_window_vs_reference.png",
        alpha_ref=alpha_ref,
        ci_ref=ci_ref,
        guide_mu=guide_mu,
        guide_width=guide_width,
        anchor_alpha=anchor_alpha,
        anchor_ci=anchor_ci,
    )
    torch.save(guide_model.state_dict(), guide_root / "guide_model.pt", _use_new_zipfile_serialization=False)

    cfg = build_config(args, warm_config, checkpoint, guide_supervision_csv, guide_curve_csv)
    pd.DataFrame([asdict(cfg)]).to_csv(args.output_dir / "config.csv", index=False)
    pd.DataFrame(
        [
            {
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(checkpoint),
                "guide_curve_csv": str(guide_curve_csv),
                "guide_supervision_csv": str(guide_supervision_csv),
                "target_mach": float(cfg.mach),
            }
        ]
    ).to_csv(args.output_dir / "warmstart_source.csv", index=False)

    print("Core-only 1D subsonic Riccati protocol (windowed sparse)")
    print(f"warm start={checkpoint}")
    print(f"mach={cfg.mach:.3f}")
    print(
        f"alpha-range(model)=[{cfg.alpha_min:.3f}, {cfg.alpha_max:.3f}] "
        f"active=[{args.alpha_min:.3f}, {args.alpha_max:.3f}]"
    )
    print(
        f"guide: n_ref={int(args.n_guide_reference_alpha)} n_anchor={int(args.n_guide_anchor_alpha)} "
        f"epochs={int(args.guide_epochs)}"
    )
    print(
        f"weights: guide_sup={cfg.w_ci_supervision:.2f} ci_window={cfg.w_ci_window:.2f} "
        f"q_sup={cfg.w_q_supervision:.2f} gamma_sup={cfg.w_riccati_gamma_supervision:.2f} "
        f"anchor={cfg.w_riccati_anchor:.2f} bc_k={cfg.w_riccati_boundary_band_kappa:.2f} "
        f"bc_q={cfg.w_riccati_boundary_band_q:.2f} shoot={cfg.w_riccati_shooting_match:.2f}"
    )

    if args.skip_training:
        model_best_path = Path(cfg.output_dir) / "model_best.pt"
        if not model_best_path.exists():
            raise FileNotFoundError(f"Skip-training requested but post-train model is missing: {model_best_path}")
        print(f"Skip training enabled; reusing existing post-train model at {model_best_path}")
    else:
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

    eval_root = Path(cfg.output_dir)
    warm_eval_root = eval_root / "warmstart_eval"
    post_eval_root = eval_root / "posttrain_eval"

    warm_summary, warm_regimes_df = evaluate_candidate(
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

    summary_df = pd.DataFrame(
        [
            {
                "regime": "riccati_core1d_windowed_sparse",
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(checkpoint),
                "guide_curve_csv": str(guide_curve_csv),
                "output_dir": str(cfg.output_dir),
                "model_alpha_min": float(cfg.alpha_min),
                "model_alpha_max": float(cfg.alpha_max),
                "active_alpha_min": float(args.alpha_min),
                "active_alpha_max": float(args.alpha_max),
                "target_mach": float(cfg.mach),
                "n_guide_reference_alpha": int(args.n_guide_reference_alpha),
                "n_guide_anchor_alpha": int(args.n_guide_anchor_alpha),
                "guide_epochs": int(args.guide_epochs),
                "guide_hidden_dim": int(args.guide_hidden_dim),
                "guide_depth": int(args.guide_depth),
                "guide_supervision_weight": float(cfg.w_ci_supervision),
                "guide_window_weight": float(cfg.w_ci_window),
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

    summary_path = eval_root / "core1d_windowed_sparse_summary.csv"
    improvement_path = eval_root / "core1d_windowed_sparse_improvement.csv"
    summary_df.to_csv(summary_path, index=False)
    improvement_df.to_csv(improvement_path, index=False)
    warm_regimes_df.to_csv(warm_eval_root / "modes" / "mode_regime_summary.csv", index=False)
    post_regimes_df.to_csv(post_eval_root / "modes" / "mode_regime_summary.csv", index=False)

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
    print(f"Guide window written to {guide_curve_csv}")
    print(f"Guide supervision written to {guide_supervision_csv}")
    print(f"Summary written to {summary_path}")
    print(f"Improvement written to {improvement_path}")


if __name__ == "__main__":
    main()
