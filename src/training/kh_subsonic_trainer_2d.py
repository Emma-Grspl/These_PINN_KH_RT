from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.data.kh_subsonic_sampling import (
    SubsonicReferenceCache2D,
    reference_point,
    sample_alpha_mach_adaptive_neutral_batch,
    sample_boundary_points,
    sample_interior_points,
)
from src.models.kh_subsonic_pinn import KHSubsonicMultiMachPINN
from src.physics.kh_subsonic_residual import (
    boundary_decay_loss_2d,
    normalization_loss_2d,
    phase_loss_2d,
    pressure_ode_residual_2d,
)


@dataclass
class KHSubsonic2DTrainingConfig:
    alpha_min: float = 0.05
    alpha_max: float = 0.85
    mach_min: float = 0.0
    mach_max: float = 0.5
    epochs: int = 3000
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
    n_supervision: int = 128
    n_reference_alpha: int = 41
    n_reference_mach: int = 11
    n_audit_alpha: int = 17
    n_audit_mach: int = 6
    audit_every: int = 100
    checkpoint_every: int = 500
    focus_fraction: float = 0.6
    neutral_fraction: float = 0.2
    focus_alpha_half_width: float = 0.03
    focus_mach_half_width: float = 0.05
    neutral_band_ratio: float = 0.15
    error_threshold: float = 0.02
    max_focus_points: int = 12
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_norm: float = 1.0
    w_phase: float = 1.0
    w_ci_supervision: float = 5.0
    output_dir: str = "model_saved/kh_subsonic_2d_local"
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


def audit_ci_surface(
    model: KHSubsonicMultiMachPINN,
    reference_cache: SubsonicReferenceCache2D,
    *,
    num_alpha: int,
    num_mach: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    aa, mm, ci_true = reference_cache.audit_grid(num_alpha=num_alpha, num_mach=num_mach)
    alpha_tensor = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_tensor = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor, mach_tensor).cpu().numpy().reshape(ci_true.shape)
    ci_abs_err = np.abs(ci_pred - ci_true)
    ci_rel_err = ci_abs_err / (np.abs(ci_true) + 1e-8)
    focus_pairs = np.column_stack([aa.reshape(-1), mm.reshape(-1)])
    return {
        "audit_ci_mae": float(ci_abs_err.mean()),
        "audit_ci_max_abs": float(ci_abs_err.max()),
        "audit_ci_mean_rel": float(ci_rel_err.mean()),
        "audit_ci_max_rel": float(ci_rel_err.max()),
    }, focus_pairs, ci_abs_err.reshape(-1)


def train_subsonic_2d_pinn(cfg: KHSubsonic2DTrainingConfig) -> tuple[KHSubsonicMultiMachPINN, pd.DataFrame]:
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_cache = SubsonicReferenceCache2D.build(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        mach_min=cfg.mach_min,
        mach_max=cfg.mach_max,
        num_alpha=cfg.n_reference_alpha,
        num_mach=cfg.n_reference_mach,
    )

    model = KHSubsonicMultiMachPINN(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        mach_min=cfg.mach_min,
        mach_max=cfg.mach_max,
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
    focus_points: np.ndarray | None = None

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        xi_interior = sample_interior_points(cfg.n_interior, device=device)
        alpha_interior, mach_interior = sample_alpha_mach_adaptive_neutral_batch(
            cfg.n_interior,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            mach_min=cfg.mach_min,
            mach_max=cfg.mach_max,
            focus_points=focus_points,
            focus_fraction=cfg.focus_fraction,
            neutral_fraction=cfg.neutral_fraction,
            alpha_half_width=cfg.focus_alpha_half_width,
            mach_half_width=cfg.focus_mach_half_width,
            neutral_band_ratio=cfg.neutral_band_ratio,
            device=device,
        )
        xi_left, xi_right = sample_boundary_points(cfg.n_boundary, device=device)
        alpha_boundary, mach_boundary = sample_alpha_mach_adaptive_neutral_batch(
            cfg.n_boundary,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            mach_min=cfg.mach_min,
            mach_max=cfg.mach_max,
            focus_points=focus_points,
            focus_fraction=cfg.focus_fraction,
            neutral_fraction=cfg.neutral_fraction,
            alpha_half_width=cfg.focus_alpha_half_width,
            mach_half_width=cfg.focus_mach_half_width,
            neutral_band_ratio=cfg.neutral_band_ratio,
            device=device,
        )
        xi_ref = reference_point(device=device)
        alpha_ref, mach_ref = sample_alpha_mach_adaptive_neutral_batch(
            1,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            mach_min=cfg.mach_min,
            mach_max=cfg.mach_max,
            focus_points=focus_points,
            focus_fraction=cfg.focus_fraction,
            neutral_fraction=cfg.neutral_fraction,
            alpha_half_width=cfg.focus_alpha_half_width,
            mach_half_width=cfg.focus_mach_half_width,
            neutral_band_ratio=cfg.neutral_band_ratio,
            device=device,
        )
        alpha_supervision, mach_supervision = sample_alpha_mach_adaptive_neutral_batch(
            cfg.n_supervision,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            mach_min=cfg.mach_min,
            mach_max=cfg.mach_max,
            focus_points=focus_points,
            focus_fraction=cfg.focus_fraction,
            neutral_fraction=cfg.neutral_fraction,
            alpha_half_width=cfg.focus_alpha_half_width,
            mach_half_width=cfg.focus_mach_half_width,
            neutral_band_ratio=cfg.neutral_band_ratio,
            device=device,
        )

        ci_target = reference_cache.interpolate(alpha_supervision, mach_supervision)
        ci_pred = model.get_ci(alpha_supervision, mach_supervision)

        res_r, res_i, _ = pressure_ode_residual_2d(model, xi_interior, alpha_interior, mach_interior)
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))
        loss_bc = boundary_decay_loss_2d(model, xi_left, xi_right, alpha_boundary, mach_boundary)
        loss_norm = normalization_loss_2d(model, xi_ref, alpha_ref, mach_ref)
        loss_phase = phase_loss_2d(model, xi_ref, alpha_ref, mach_ref)
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
        }

        if epoch == 1 or epoch % cfg.audit_every == 0:
            audit_metrics, point_grid, ci_abs_err = audit_ci_surface(
                model,
                reference_cache,
                num_alpha=cfg.n_audit_alpha,
                num_mach=cfg.n_audit_mach,
                device=device,
            )
            record.update(audit_metrics)
            failing_mask = ci_abs_err > cfg.error_threshold
            failing_points = point_grid[failing_mask]
            if len(failing_points) > cfg.max_focus_points:
                worst_idx = np.argsort(ci_abs_err[failing_mask])[-cfg.max_focus_points :]
                failing_points = failing_points[worst_idx]
            focus_points = np.asarray(failing_points, dtype=float) if len(failing_points) else None
            record["n_focus_points"] = 0 if focus_points is None else int(len(focus_points))

            king.update(model, record["audit_ci_mae"])
            print(
                f"Epoch {epoch:5d} | loss={record['loss']:.3e} | "
                f"ci_mae={record['audit_ci_mae']:.3e} | "
                f"n_focus={record['n_focus_points']} | "
                f"L={record['mapping_scale']:.3f}"
            )
        history.append(record)

        if epoch % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

    model.load_state_dict(king.best_state)
    return model, pd.DataFrame(history)


def save_training_artifacts(
    model: KHSubsonicMultiMachPINN,
    history: pd.DataFrame,
    cfg: KHSubsonic2DTrainingConfig,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_best.pt")
    history.to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame([asdict(cfg)]).to_csv(output_dir / "config.csv", index=False)
