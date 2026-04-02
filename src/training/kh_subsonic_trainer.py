from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.data.kh_subsonic_sampling import (
    SubsonicReferenceCache,
    reference_point,
    sample_alpha_adaptive_batch,
    sample_alpha_batch,
    sample_boundary_points,
    sample_interior_points,
)
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN
from src.physics.kh_subsonic_residual import (
    boundary_decay_loss,
    normalization_loss,
    phase_loss,
    pressure_ode_residual,
)


@dataclass
class KHSubsonicTrainingConfig:
    mach: float = 0.5
    alpha_min: float = 0.05
    alpha_max: float = 0.85
    epochs: int = 5000
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    mode_depth: int = 4
    ci_depth: int = 2
    activation: str = "tanh"
    fourier_features: int = 0
    fourier_scale: float = 2.0
    initial_ci: float = 0.2
    mapping_scale: float = 3.0
    trainable_mapping_scale: bool = False
    n_interior: int = 512
    n_boundary: int = 64
    n_alpha_supervision: int = 128
    n_reference_alpha: int = 81
    n_audit_alpha: int = 21
    audit_every: int = 250
    checkpoint_every: int = 500
    focus_fraction: float = 0.6
    focus_half_width: float = 0.03
    error_threshold: float = 0.01
    max_focus_points: int = 8
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_norm: float = 1.0
    w_phase: float = 1.0
    w_ci_supervision: float = 5.0
    output_dir: str = "model_saved/kh_subsonic_fixed_mach"
    device: str = "cpu"


class KingOfTheHill:
    def __init__(self, model: torch.nn.Module):
        self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.best_metric = float("inf")

    def update(self, model: torch.nn.Module, metric: float) -> bool:
        if metric < self.best_metric:
            self.best_metric = float(metric)
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return True
        return False


def audit_ci_curve(
    model: KHSubsonicFixedMachPINN,
    reference_cache: SubsonicReferenceCache,
    *,
    num_points: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    alphas_np, ci_true_np = reference_cache.audit_grid(num_points=num_points)
    alpha_tensor = torch.tensor(alphas_np, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    ci_abs_err = abs(ci_pred - ci_true_np)
    denom = abs(ci_true_np) + 1e-8
    ci_rel_err = ci_abs_err / denom
    return {
        "audit_ci_mae": float(ci_abs_err.mean()),
        "audit_ci_max_abs": float(ci_abs_err.max()),
        "audit_ci_mean_rel": float(ci_rel_err.mean()),
        "audit_ci_max_rel": float(ci_rel_err.max()),
    }, np.asarray(alphas_np, dtype=float), np.asarray(ci_abs_err, dtype=float)


def train_fixed_mach_subsonic_pinn(cfg: KHSubsonicTrainingConfig) -> tuple[KHSubsonicFixedMachPINN, pd.DataFrame]:
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_cache = SubsonicReferenceCache.build(
        mach=cfg.mach,
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        num_alpha=cfg.n_reference_alpha,
    )

    model = KHSubsonicFixedMachPINN(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.hidden_dim,
        mode_depth=cfg.mode_depth,
        ci_depth=cfg.ci_depth,
        activation=cfg.activation,
        fourier_features=cfg.fourier_features,
        fourier_scale=cfg.fourier_scale,
        initial_ci=cfg.initial_ci,
        mapping_scale=cfg.mapping_scale,
        trainable_mapping_scale=cfg.trainable_mapping_scale,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    king = KingOfTheHill(model)
    focus_alphas: np.ndarray | None = None

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        xi_interior = sample_interior_points(cfg.n_interior, device=device)
        alpha_interior = sample_alpha_adaptive_batch(
            cfg.n_interior,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            device=device,
        )
        xi_left, xi_right = sample_boundary_points(cfg.n_boundary, device=device)
        alpha_boundary = sample_alpha_adaptive_batch(
            cfg.n_boundary,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            device=device,
        )
        alpha_ref = sample_alpha_batch(
            1,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            device=device,
        )
        xi_ref = reference_point(device=device)

        alpha_supervision = sample_alpha_adaptive_batch(
            cfg.n_alpha_supervision,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            device=device,
        )
        ci_target = reference_cache.interpolate(alpha_supervision)
        ci_pred = model.get_ci(alpha_supervision)

        res_r, res_i, _ = pressure_ode_residual(model, xi_interior, alpha_interior, cfg.mach)
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))
        loss_bc = boundary_decay_loss(model, xi_left, xi_right, alpha_boundary)
        loss_norm = normalization_loss(model, xi_ref, alpha_ref)
        loss_phase = phase_loss(model, xi_ref, alpha_ref)
        loss_ci = torch.mean((ci_pred - ci_target).pow(2))

        loss = (
            cfg.w_pde * loss_pde
            + cfg.w_bc * loss_bc
            + cfg.w_norm * loss_norm
            + cfg.w_phase * loss_phase
            + cfg.w_ci_supervision * loss_ci
        )
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_norm": float(loss_norm.item()),
            "loss_phase": float(loss_phase.item()),
            "loss_ci_supervision": float(loss_ci.item()),
            "mapping_scale": float(model.get_mapping_scale().item()),
            "ci_mid": float(model.get_ci(torch.tensor([[0.5 * (cfg.alpha_min + cfg.alpha_max)]], device=device)).item()),
        }

        if epoch == 1 or epoch % cfg.audit_every == 0:
            audit_metrics, alpha_grid, ci_abs_err = audit_ci_curve(
                model,
                reference_cache,
                num_points=cfg.n_audit_alpha,
                device=device,
            )
            record.update(audit_metrics)

            failing_mask = ci_abs_err > cfg.error_threshold
            failing_alphas = alpha_grid[failing_mask]
            if len(failing_alphas) > cfg.max_focus_points:
                worst_idx = np.argsort(ci_abs_err[failing_mask])[-cfg.max_focus_points :]
                failing_alphas = failing_alphas[worst_idx]
            focus_alphas = np.asarray(failing_alphas, dtype=float) if len(failing_alphas) else None

            record["n_focus_alphas"] = 0 if focus_alphas is None else int(len(focus_alphas))
            record["focus_alpha_min"] = np.nan if focus_alphas is None else float(np.min(focus_alphas))
            record["focus_alpha_max"] = np.nan if focus_alphas is None else float(np.max(focus_alphas))

            king.update(model, record["audit_ci_mae"])
            print(
                f"Epoch {epoch:5d} | loss={record['loss']:.3e} | "
                f"ci_mae={record['audit_ci_mae']:.3e} | "
                f"n_focus={record['n_focus_alphas']} | "
                f"L={record['mapping_scale']:.3f}"
            )
        history.append(record)

        if epoch % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

    model.load_state_dict(king.best_state)
    return model, pd.DataFrame(history)


def save_training_artifacts(
    model: KHSubsonicFixedMachPINN,
    history: pd.DataFrame,
    cfg: KHSubsonicTrainingConfig,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_best.pt")
    history.to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame([asdict(cfg)]).to_csv(output_dir / "config.csv", index=False)
