from __future__ import annotations

import torch

from src.physics.kh_subsonic_residual import (
    base_velocity,
    base_velocity_derivative,
    boundary_decay_loss_2d,
    dy_dxi,
    normalization_loss_2d,
    phase_loss_2d,
    pressure_ode_residual_2d,
    reconstruct_pressure_from_riccati_2d,
    reconstruct_pressure_p_y_from_riccati_2d,
    riccati_boundary_loss_components_2d,
    xi_to_y,
)


def riccati_normalization_loss_2d(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    target_pr: float = 1.0,
    target_pi: float = 0.0,
    anchor_xi: float = 0.0,
) -> torch.Tensor:
    pr, pi, _ = reconstruct_pressure_from_riccati_2d(
        model,
        xi_ref,
        alpha,
        mach,
        anchor_xi=anchor_xi,
    )
    pred = torch.cat([pr, pi], dim=-1)
    target = torch.tensor([target_pr, target_pi], dtype=pred.dtype, device=pred.device).view(1, 2)
    return torch.mean((pred - target) ** 2)


def riccati_phase_loss_2d(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> torch.Tensor:
    pr, pi, _ = reconstruct_pressure_from_riccati_2d(
        model,
        xi_ref,
        alpha,
        mach,
        anchor_xi=anchor_xi,
    )
    imag_penalty = torch.mean(pi.pow(2))
    sign_penalty = torch.mean(torch.relu(-pr).pow(2))
    return imag_penalty + sign_penalty


__all__ = [
    "base_velocity",
    "base_velocity_derivative",
    "boundary_decay_loss_2d",
    "dy_dxi",
    "normalization_loss_2d",
    "phase_loss_2d",
    "pressure_ode_residual_2d",
    "reconstruct_pressure_from_riccati_2d",
    "reconstruct_pressure_p_y_from_riccati_2d",
    "riccati_boundary_loss_components_2d",
    "riccati_normalization_loss_2d",
    "riccati_phase_loss_2d",
    "xi_to_y",
]
