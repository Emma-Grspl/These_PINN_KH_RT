from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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


class WindowedSpectralNet(nn.Module):
    def __init__(self, *, alpha_min: float, alpha_max: float, hidden_dim: int, depth: int):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.net = build_mlp(1, 1, hidden_dim=hidden_dim, depth=depth)

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1.0e-8)
        return 2.0 * (alpha - self.alpha_min) / span - 1.0

    def forward(self, alpha: torch.Tensor, guide_mu: torch.Tensor, guide_width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.net(self.normalize_alpha(alpha))
        ci = guide_mu + guide_width * torch.tanh(latent)
        return ci, latent


class DenseBaselineNet(nn.Module):
    def __init__(self, *, alpha_min: float, alpha_max: float, hidden_dim: int, depth: int):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.net = build_mlp(1, 1, hidden_dim=hidden_dim, depth=depth)

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1.0e-8)
        return 2.0 * (alpha - self.alpha_min) / span - 1.0

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.normalize_alpha(alpha))
        return F.softplus(raw) + 1.0e-6


@dataclass
class AblationConfig:
    mach: float = 0.5
    alpha_min: float = 0.05
    alpha_max: float = 0.8
    n_reference_alpha: int = 81
    n_anchor_alpha: int = 9
    guide_hidden_dim: int = 48
    guide_depth: int = 2
    guide_epochs: int = 3000
    guide_lr: float = 2.0e-3
    guide_min_width: float = 2.5e-3
    guide_sigma_penalty: float = 3.0e-2
    guide_mu_smoothness: float = 1.0e-2
    guide_width_smoothness: float = 1.0e-2
    main_hidden_dim: int = 64
    main_depth: int = 2
    main_epochs: int = 4000
    main_lr: float = 2.0e-3
    main_anchor_weight: float = 6.0
    main_smoothness_weight: float = 2.0e-2
    main_latent_weight: float = 2.0e-3
    baseline_hidden_dim: int = 64
    baseline_depth: int = 2
    baseline_epochs: int = 4000
    baseline_lr: float = 2.0e-3
    seed: int = 0
    output_dir: str = "model_saved/kh_subsonic_fixed_mach_M05_ci_window_ablation"
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation spectrale subsonique : un reseau guide apprend une fenetre sur c_i(alpha), "
            "puis un second reseau apprend c_i borne dans cette fenetre. Compare ensuite au classique."
        )
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--n-reference-alpha", type=int, default=81)
    parser.add_argument("--n-anchor-alpha", type=int, default=9)
    parser.add_argument("--guide-hidden-dim", type=int, default=48)
    parser.add_argument("--guide-depth", type=int, default=2)
    parser.add_argument("--guide-epochs", type=int, default=3000)
    parser.add_argument("--guide-lr", type=float, default=2.0e-3)
    parser.add_argument("--guide-min-width", type=float, default=2.5e-3)
    parser.add_argument("--guide-sigma-penalty", type=float, default=3.0e-2)
    parser.add_argument("--guide-mu-smoothness", type=float, default=1.0e-2)
    parser.add_argument("--guide-width-smoothness", type=float, default=1.0e-2)
    parser.add_argument("--main-hidden-dim", type=int, default=64)
    parser.add_argument("--main-depth", type=int, default=2)
    parser.add_argument("--main-epochs", type=int, default=4000)
    parser.add_argument("--main-lr", type=float, default=2.0e-3)
    parser.add_argument("--main-anchor-weight", type=float, default=6.0)
    parser.add_argument("--main-smoothness-weight", type=float, default=2.0e-2)
    parser.add_argument("--main-latent-weight", type=float, default=2.0e-3)
    parser.add_argument("--baseline-hidden-dim", type=int, default=64)
    parser.add_argument("--baseline-depth", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=4000)
    parser.add_argument("--baseline-lr", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/kh_subsonic_fixed_mach_M05_ci_window_ablation"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser


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
        raise ValueError("n_anchor_alpha must be at least 2.")
    return np.linspace(0, n_points - 1, n_anchors, dtype=int)


def train_guide(
    cfg: AblationConfig,
    *,
    alpha_ref_t: torch.Tensor,
    alpha_anchor_t: torch.Tensor,
    ci_anchor_t: torch.Tensor,
    device: torch.device,
) -> tuple[GuideWindowNet, pd.DataFrame]:
    guide = GuideWindowNet(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.guide_hidden_dim,
        depth=cfg.guide_depth,
        min_width=cfg.guide_min_width,
    ).to(device)
    optimizer = torch.optim.Adam(guide.parameters(), lr=cfg.guide_lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.guide_epochs + 1):
        optimizer.zero_grad()
        mu_anchor, width_anchor = guide(alpha_anchor_t)
        loss_anchor = torch.mean((mu_anchor - ci_anchor_t).pow(2))

        alpha_smooth = alpha_ref_t.detach().clone().requires_grad_(True)
        mu_full, width_full = guide(alpha_smooth)
        loss_mu_smooth = smoothness_loss(mu_full, alpha_smooth)
        loss_width_smooth = smoothness_loss(width_full, alpha_smooth)
        loss_width_penalty = torch.mean(width_full.pow(2))

        loss = (
            loss_anchor
            + cfg.guide_sigma_penalty * loss_width_penalty
            + cfg.guide_mu_smoothness * loss_mu_smooth
            + cfg.guide_width_smoothness * loss_width_smooth
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 250 == 0 or epoch == cfg.guide_epochs:
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


def train_windowed_main(
    cfg: AblationConfig,
    *,
    guide: GuideWindowNet,
    alpha_ref_t: torch.Tensor,
    alpha_anchor_t: torch.Tensor,
    ci_anchor_t: torch.Tensor,
    device: torch.device,
) -> tuple[WindowedSpectralNet, pd.DataFrame]:
    main = WindowedSpectralNet(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.main_hidden_dim,
        depth=cfg.main_depth,
    ).to(device)
    optimizer = torch.optim.Adam(main.parameters(), lr=cfg.main_lr)
    history: list[dict[str, float | int]] = []

    guide.eval()
    for param in guide.parameters():
        param.requires_grad_(False)

    for epoch in range(1, cfg.main_epochs + 1):
        optimizer.zero_grad()

        with torch.no_grad():
            mu_anchor, width_anchor = guide(alpha_anchor_t)
        ci_anchor_pred, latent_anchor = main(alpha_anchor_t, mu_anchor, width_anchor)
        loss_anchor = torch.mean((ci_anchor_pred - ci_anchor_t).pow(2))

        alpha_smooth = alpha_ref_t.detach().clone().requires_grad_(True)
        with torch.no_grad():
            mu_full, width_full = guide(alpha_smooth)
        ci_full, latent_full = main(alpha_smooth, mu_full, width_full)
        loss_smooth = smoothness_loss(ci_full, alpha_smooth)
        loss_latent = torch.mean(latent_full.pow(2))

        loss = (
            cfg.main_anchor_weight * loss_anchor
            + cfg.main_smoothness_weight * loss_smooth
            + cfg.main_latent_weight * loss_latent
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 250 == 0 or epoch == cfg.main_epochs:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.item()),
                    "anchor_mse": float(loss_anchor.item()),
                    "smoothness": float(loss_smooth.item()),
                    "latent_l2": float(loss_latent.item()),
                    "latent_abs_mean": float(torch.mean(torch.abs(latent_full)).item()),
                }
            )

    return main, pd.DataFrame(history)


def train_dense_baseline(
    cfg: AblationConfig,
    *,
    alpha_ref_t: torch.Tensor,
    ci_ref_t: torch.Tensor,
    device: torch.device,
) -> tuple[DenseBaselineNet, pd.DataFrame]:
    baseline = DenseBaselineNet(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.baseline_hidden_dim,
        depth=cfg.baseline_depth,
    ).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=cfg.baseline_lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.baseline_epochs + 1):
        optimizer.zero_grad()
        ci_pred = baseline(alpha_ref_t)
        loss = torch.mean((ci_pred - ci_ref_t).pow(2))
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 250 == 0 or epoch == cfg.baseline_epochs:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.item()),
                    "ci_mae": float(torch.mean(torch.abs(ci_pred - ci_ref_t)).item()),
                    "ci_max_abs": float(torch.max(torch.abs(ci_pred - ci_ref_t)).item()),
                }
            )

    return baseline, pd.DataFrame(history)


def evaluate_models(
    *,
    guide: GuideWindowNet,
    main: WindowedSpectralNet,
    baseline: DenseBaselineNet,
    alpha_ref_t: torch.Tensor,
) -> dict[str, np.ndarray]:
    guide.eval()
    main.eval()
    baseline.eval()
    with torch.no_grad():
        mu_ref, width_ref = guide(alpha_ref_t)
        ci_main, latent_ref = main(alpha_ref_t, mu_ref, width_ref)
        ci_baseline = baseline(alpha_ref_t)
    return {
        "guide_mu": mu_ref.cpu().numpy().reshape(-1),
        "guide_width": width_ref.cpu().numpy().reshape(-1),
        "ci_main": ci_main.cpu().numpy().reshape(-1),
        "ci_baseline": ci_baseline.cpu().numpy().reshape(-1),
        "latent": latent_ref.cpu().numpy().reshape(-1),
    }


def summarize_curve(name: str, ci_pred: np.ndarray, ci_ref: np.ndarray) -> dict[str, float | str]:
    abs_err = np.abs(ci_pred - ci_ref)
    return {
        "model": str(name),
        "ci_mae": float(abs_err.mean()),
        "ci_max_abs": float(abs_err.max()),
        "ci_rmse": float(np.sqrt(np.mean((ci_pred - ci_ref) ** 2))),
    }


def save_plots(
    output_dir: Path,
    *,
    alpha_ref: np.ndarray,
    ci_ref: np.ndarray,
    guide_mu: np.ndarray,
    guide_width: np.ndarray,
    ci_main: np.ndarray,
    ci_baseline: np.ndarray,
    anchor_alpha: np.ndarray,
    anchor_ci: np.ndarray,
) -> None:
    low = np.clip(guide_mu - guide_width, a_min=0.0, a_max=None)
    high = guide_mu + guide_width

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(alpha_ref, ci_ref, label="Classique dense", linewidth=2.2, color="black")
    axes[0].plot(alpha_ref, guide_mu, label="Guide mu", linewidth=2.0, color="#1f77b4")
    axes[0].fill_between(alpha_ref, low, high, color="#1f77b4", alpha=0.18, label="Fenetre guide")
    axes[0].plot(alpha_ref, ci_main, "--", label="Reseau borne", linewidth=2.0, color="#d62728")
    axes[0].plot(alpha_ref, ci_baseline, ":", label="Baseline dense", linewidth=2.0, color="#2ca02c")
    axes[0].scatter(anchor_alpha, anchor_ci, label="Ancres clairsemees", color="#ff7f0e", zorder=5)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(alpha_ref, np.abs(ci_main - ci_ref), label="Erreur |borne - classique|", linewidth=2.0, color="#d62728")
    axes[1].plot(alpha_ref, np.abs(ci_baseline - ci_ref), label="Erreur |dense - classique|", linewidth=2.0, color="#2ca02c")
    axes[1].plot(alpha_ref, np.abs(guide_mu - ci_ref), label="Erreur |guide - classique|", linewidth=2.0, color="#1f77b4")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("Erreur absolue")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ci_window_vs_reference.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    cfg = AblationConfig(
        mach=float(args.mach),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        n_reference_alpha=int(args.n_reference_alpha),
        n_anchor_alpha=int(args.n_anchor_alpha),
        guide_hidden_dim=int(args.guide_hidden_dim),
        guide_depth=int(args.guide_depth),
        guide_epochs=int(args.guide_epochs),
        guide_lr=float(args.guide_lr),
        guide_min_width=float(args.guide_min_width),
        guide_sigma_penalty=float(args.guide_sigma_penalty),
        guide_mu_smoothness=float(args.guide_mu_smoothness),
        guide_width_smoothness=float(args.guide_width_smoothness),
        main_hidden_dim=int(args.main_hidden_dim),
        main_depth=int(args.main_depth),
        main_epochs=int(args.main_epochs),
        main_lr=float(args.main_lr),
        main_anchor_weight=float(args.main_anchor_weight),
        main_smoothness_weight=float(args.main_smoothness_weight),
        main_latent_weight=float(args.main_latent_weight),
        baseline_hidden_dim=int(args.baseline_hidden_dim),
        baseline_depth=int(args.baseline_depth),
        baseline_epochs=int(args.baseline_epochs),
        baseline_lr=float(args.baseline_lr),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        device=str(args.device),
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    alpha_ref = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_reference_alpha, dtype=float)
    ci_ref = build_reference_ci_grid(alpha_ref, cfg.mach)
    anchor_indices = build_anchor_indices(len(alpha_ref), cfg.n_anchor_alpha)
    anchor_alpha = alpha_ref[anchor_indices]
    anchor_ci = ci_ref[anchor_indices]

    alpha_ref_t = torch.tensor(alpha_ref, dtype=torch.float32, device=device).view(-1, 1)
    ci_ref_t = torch.tensor(ci_ref, dtype=torch.float32, device=device).view(-1, 1)
    alpha_anchor_t = torch.tensor(anchor_alpha, dtype=torch.float32, device=device).view(-1, 1)
    ci_anchor_t = torch.tensor(anchor_ci, dtype=torch.float32, device=device).view(-1, 1)

    print("Subsonic spectral guide-window ablation")
    print(f"mach={cfg.mach:.3f} alpha-range=[{cfg.alpha_min:.3f}, {cfg.alpha_max:.3f}]")
    print(f"n_reference_alpha={cfg.n_reference_alpha} n_anchor_alpha={cfg.n_anchor_alpha}")
    print(
        f"guide: epochs={cfg.guide_epochs} hidden={cfg.guide_hidden_dim} depth={cfg.guide_depth} min_width={cfg.guide_min_width:.3e}"
    )
    print(
        f"main: epochs={cfg.main_epochs} hidden={cfg.main_hidden_dim} depth={cfg.main_depth} "
        f"anchor_weight={cfg.main_anchor_weight:.2f}"
    )
    print(f"baseline: epochs={cfg.baseline_epochs} hidden={cfg.baseline_hidden_dim} depth={cfg.baseline_depth}")

    guide, guide_history = train_guide(
        cfg,
        alpha_ref_t=alpha_ref_t,
        alpha_anchor_t=alpha_anchor_t,
        ci_anchor_t=ci_anchor_t,
        device=device,
    )
    main_model, main_history = train_windowed_main(
        cfg,
        guide=guide,
        alpha_ref_t=alpha_ref_t,
        alpha_anchor_t=alpha_anchor_t,
        ci_anchor_t=ci_anchor_t,
        device=device,
    )
    baseline, baseline_history = train_dense_baseline(
        cfg,
        alpha_ref_t=alpha_ref_t,
        ci_ref_t=ci_ref_t,
        device=device,
    )

    eval_out = evaluate_models(guide=guide, main=main_model, baseline=baseline, alpha_ref_t=alpha_ref_t)
    guide_mu = eval_out["guide_mu"]
    guide_width = eval_out["guide_width"]
    ci_main = eval_out["ci_main"]
    ci_baseline = eval_out["ci_baseline"]

    comparison_df = pd.DataFrame(
        [
            summarize_curve("guide_mu", guide_mu, ci_ref),
            summarize_curve("windowed_main", ci_main, ci_ref),
            summarize_curve("dense_baseline", ci_baseline, ci_ref),
        ]
    )
    curve_df = pd.DataFrame(
        {
            "alpha": alpha_ref,
            "ci_reference": ci_ref,
            "ci_guide_mu": guide_mu,
            "ci_guide_width": guide_width,
            "ci_window_low": np.clip(guide_mu - guide_width, a_min=0.0, a_max=None),
            "ci_window_high": guide_mu + guide_width,
            "ci_windowed_main": ci_main,
            "ci_dense_baseline": ci_baseline,
            "guide_abs_err": np.abs(guide_mu - ci_ref),
            "windowed_abs_err": np.abs(ci_main - ci_ref),
            "dense_abs_err": np.abs(ci_baseline - ci_ref),
            "is_anchor": np.isin(np.arange(len(alpha_ref)), anchor_indices),
        }
    )
    anchor_df = pd.DataFrame({"alpha": anchor_alpha, "ci_reference": anchor_ci})
    history_df = (
        guide_history.assign(stage="guide")
        .pipe(lambda df: pd.concat([df, main_history.assign(stage="windowed_main"), baseline_history.assign(stage="dense_baseline")], ignore_index=True))
    )

    comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    curve_df.to_csv(output_dir / "ci_window_curve_vs_reference.csv", index=False)
    anchor_df.to_csv(output_dir / "guide_anchor_points.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame([asdict(cfg)]).to_csv(output_dir / "config.csv", index=False)
    torch.save(guide.state_dict(), output_dir / "guide_model.pt", _use_new_zipfile_serialization=False)
    torch.save(main_model.state_dict(), output_dir / "windowed_main_model.pt", _use_new_zipfile_serialization=False)
    torch.save(baseline.state_dict(), output_dir / "dense_baseline_model.pt", _use_new_zipfile_serialization=False)

    save_plots(
        output_dir,
        alpha_ref=alpha_ref,
        ci_ref=ci_ref,
        guide_mu=guide_mu,
        guide_width=guide_width,
        ci_main=ci_main,
        ci_baseline=ci_baseline,
        anchor_alpha=anchor_alpha,
        anchor_ci=anchor_ci,
    )

    print("\nComparison:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(comparison_df.to_string(index=False))
    print(f"Wrote {output_dir / 'comparison_summary.csv'}")
    print(f"Wrote {output_dir / 'ci_window_curve_vs_reference.csv'}")
    print(f"Wrote {output_dir / 'ci_window_vs_reference.png'}")


if __name__ == "__main__":
    main()
