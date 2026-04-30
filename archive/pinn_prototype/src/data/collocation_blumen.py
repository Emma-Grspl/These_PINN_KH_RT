from __future__ import annotations

import torch


def sample_interior_points(n_points: int, xi_min: float = -0.98, xi_max: float = 0.98, device: str = "cpu") -> torch.Tensor:
    xi = torch.rand(n_points, 1, device=device)
    xi = xi_min + (xi_max - xi_min) * xi
    xi.requires_grad_(True)
    return xi


def sample_boundary_points(n_points_per_side: int, xi_boundary: float = 0.995, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    xi_left = -xi_boundary * torch.ones(n_points_per_side, 1, device=device)
    xi_right = xi_boundary * torch.ones(n_points_per_side, 1, device=device)
    xi_left.requires_grad_(True)
    xi_right.requires_grad_(True)
    return xi_left, xi_right


def reference_point(device: str = "cpu") -> torch.Tensor:
    xi_ref = torch.zeros(1, 1, device=device)
    xi_ref.requires_grad_(True)
    return xi_ref
