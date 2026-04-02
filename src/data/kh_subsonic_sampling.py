from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver


def sample_interior_points(
    n_points: int,
    *,
    xi_min: float = -0.98,
    xi_max: float = 0.98,
    device: torch.device,
) -> torch.Tensor:
    xi = torch.rand(n_points, 1, device=device)
    xi = xi_min + (xi_max - xi_min) * xi
    xi.requires_grad_(True)
    return xi


def sample_boundary_points(
    n_points_per_side: int,
    *,
    xi_boundary: float = 0.995,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xi_left = -xi_boundary * torch.ones(n_points_per_side, 1, device=device)
    xi_right = xi_boundary * torch.ones(n_points_per_side, 1, device=device)
    xi_left.requires_grad_(True)
    xi_right.requires_grad_(True)
    return xi_left, xi_right


def reference_point(*, device: torch.device) -> torch.Tensor:
    xi_ref = torch.zeros(1, 1, device=device)
    xi_ref.requires_grad_(True)
    return xi_ref


def sample_alpha_batch(
    n_points: int,
    *,
    alpha_min: float,
    alpha_max: float,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(n_points, 1, device=device)
    return alpha_min + (alpha_max - alpha_min) * alpha


def sample_alpha_mixed_batch(
    n_points: int,
    *,
    alpha_min: float,
    alpha_max: float,
    high_alpha_fraction: float,
    high_alpha_start_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    n_high = int(round(high_alpha_fraction * n_points))
    n_high = min(max(n_high, 0), n_points)
    n_uniform = n_points - n_high

    chunks: list[torch.Tensor] = []
    if n_uniform > 0:
        chunks.append(
            sample_alpha_batch(
                n_uniform,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                device=device,
            )
        )
    if n_high > 0:
        alpha_high_min = alpha_min + high_alpha_start_ratio * (alpha_max - alpha_min)
        chunks.append(
            sample_alpha_batch(
                n_high,
                alpha_min=alpha_high_min,
                alpha_max=alpha_max,
                device=device,
            )
        )
    alpha = torch.cat(chunks, dim=0)
    permutation = torch.randperm(alpha.shape[0], device=device)
    return alpha[permutation]


def sample_alpha_adaptive_batch(
    n_points: int,
    *,
    alpha_min: float,
    alpha_max: float,
    focus_alphas: np.ndarray | None,
    focus_fraction: float,
    focus_half_width: float,
    device: torch.device,
) -> torch.Tensor:
    n_focus = int(round(focus_fraction * n_points))
    if focus_alphas is None or len(focus_alphas) == 0:
        n_focus = 0
    n_focus = min(max(n_focus, 0), n_points)
    n_uniform = n_points - n_focus

    chunks: list[torch.Tensor] = []
    if n_uniform > 0:
        chunks.append(
            sample_alpha_batch(
                n_uniform,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                device=device,
            )
        )
    if n_focus > 0:
        focus_idx = torch.randint(0, len(focus_alphas), (n_focus,), device=device)
        focus_centers = torch.tensor(focus_alphas, dtype=torch.float32, device=device).view(-1)[focus_idx].view(-1, 1)
        local = (2.0 * torch.rand(n_focus, 1, device=device) - 1.0) * focus_half_width
        focused = torch.clamp(focus_centers + local, min=alpha_min, max=alpha_max)
        chunks.append(focused)

    alpha = torch.cat(chunks, dim=0)
    permutation = torch.randperm(alpha.shape[0], device=device)
    return alpha[permutation]


@dataclass
class SubsonicReferenceCache:
    mach: float
    alpha_values: np.ndarray
    ci_values: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        mach: float,
        alpha_min: float,
        alpha_max: float,
        num_alpha: int,
    ) -> "SubsonicReferenceCache":
        alpha_values = np.linspace(alpha_min, alpha_max, num_alpha, dtype=float)
        ci_values = np.zeros_like(alpha_values)
        for idx, alpha in enumerate(alpha_values):
            solver = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            result = solver.solve()
            ci_values[idx] = float(result.ci)
        return cls(mach=float(mach), alpha_values=alpha_values, ci_values=ci_values)

    def interpolate(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_np = alpha.detach().cpu().numpy().reshape(-1)
        ci_np = np.interp(alpha_np, self.alpha_values, self.ci_values)
        return torch.tensor(ci_np, dtype=alpha.dtype, device=alpha.device).view(-1, 1)

    def audit_grid(self, *, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        alphas = np.linspace(self.alpha_values.min(), self.alpha_values.max(), num_points, dtype=float)
        cis = np.interp(alphas, self.alpha_values, self.ci_values)
        return alphas, cis
