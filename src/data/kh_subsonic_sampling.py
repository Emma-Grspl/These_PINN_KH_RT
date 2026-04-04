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


def sample_mach_batch(
    n_points: int,
    *,
    mach_min: float,
    mach_max: float,
    device: torch.device,
) -> torch.Tensor:
    mach = torch.rand(n_points, 1, device=device)
    return mach_min + (mach_max - mach_min) * mach


def sample_alpha_mach_adaptive_batch(
    n_points: int,
    *,
    alpha_min: float,
    alpha_max: float,
    mach_min: float,
    mach_max: float,
    focus_points: np.ndarray | None,
    focus_fraction: float,
    alpha_half_width: float,
    mach_half_width: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_focus = int(round(focus_fraction * n_points))
    if focus_points is None or len(focus_points) == 0:
        n_focus = 0
    n_focus = min(max(n_focus, 0), n_points)
    n_uniform = n_points - n_focus

    alpha_chunks: list[torch.Tensor] = []
    mach_chunks: list[torch.Tensor] = []

    if n_uniform > 0:
        alpha_chunks.append(sample_alpha_batch(n_uniform, alpha_min=alpha_min, alpha_max=alpha_max, device=device))
        mach_chunks.append(sample_mach_batch(n_uniform, mach_min=mach_min, mach_max=mach_max, device=device))

    if n_focus > 0:
        focus_idx = torch.randint(0, len(focus_points), (n_focus,), device=device)
        focus_tensor = torch.tensor(focus_points, dtype=torch.float32, device=device)
        centers = focus_tensor[focus_idx]
        alpha_local = (2.0 * torch.rand(n_focus, 1, device=device) - 1.0) * alpha_half_width
        mach_local = (2.0 * torch.rand(n_focus, 1, device=device) - 1.0) * mach_half_width
        alpha_chunks.append(torch.clamp(centers[:, 0:1] + alpha_local, min=alpha_min, max=alpha_max))
        mach_chunks.append(torch.clamp(centers[:, 1:2] + mach_local, min=mach_min, max=mach_max))

    alpha = torch.cat(alpha_chunks, dim=0)
    mach = torch.cat(mach_chunks, dim=0)
    permutation = torch.randperm(alpha.shape[0], device=device)
    return alpha[permutation], mach[permutation]


def sample_alpha_mach_adaptive_neutral_batch(
    n_points: int,
    *,
    alpha_min: float,
    alpha_max: float,
    mach_min: float,
    mach_max: float,
    focus_points: np.ndarray | None,
    focus_fraction: float,
    neutral_fraction: float,
    low_alpha_fraction: float,
    alpha_half_width: float,
    mach_half_width: float,
    neutral_band_ratio: float,
    low_alpha_band_width: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_focus = int(round(focus_fraction * n_points))
    if focus_points is None or len(focus_points) == 0:
        n_focus = 0
    n_focus = min(max(n_focus, 0), n_points)

    remaining = n_points - n_focus
    n_neutral = int(round(neutral_fraction * n_points))
    n_neutral = min(max(n_neutral, 0), remaining)
    remaining -= n_neutral
    n_low_alpha = int(round(low_alpha_fraction * n_points))
    n_low_alpha = min(max(n_low_alpha, 0), remaining)
    n_uniform = n_points - n_focus - n_neutral - n_low_alpha

    alpha_chunks: list[torch.Tensor] = []
    mach_chunks: list[torch.Tensor] = []

    if n_uniform > 0:
        alpha_chunks.append(sample_alpha_batch(n_uniform, alpha_min=alpha_min, alpha_max=alpha_max, device=device))
        mach_chunks.append(sample_mach_batch(n_uniform, mach_min=mach_min, mach_max=mach_max, device=device))

    if n_focus > 0:
        focus_idx = torch.randint(0, len(focus_points), (n_focus,), device=device)
        focus_tensor = torch.tensor(focus_points, dtype=torch.float32, device=device)
        centers = focus_tensor[focus_idx]
        alpha_local = (2.0 * torch.rand(n_focus, 1, device=device) - 1.0) * alpha_half_width
        mach_local = (2.0 * torch.rand(n_focus, 1, device=device) - 1.0) * mach_half_width
        alpha_chunks.append(torch.clamp(centers[:, 0:1] + alpha_local, min=alpha_min, max=alpha_max))
        mach_chunks.append(torch.clamp(centers[:, 1:2] + mach_local, min=mach_min, max=mach_max))

    if n_neutral > 0:
        mach_neutral = sample_mach_batch(n_neutral, mach_min=mach_min, mach_max=mach_max, device=device)
        alpha_c = torch.sqrt(torch.clamp(1.0 - mach_neutral.pow(2), min=0.0))
        band = neutral_band_ratio * alpha_c
        alpha_low = torch.clamp(alpha_c - band, min=alpha_min, max=alpha_max)
        alpha_high = torch.clamp(alpha_c, min=alpha_min, max=alpha_max)
        alpha_neutral = alpha_low + (alpha_high - alpha_low) * torch.rand(n_neutral, 1, device=device)
        alpha_chunks.append(alpha_neutral)
        mach_chunks.append(mach_neutral)

    if n_low_alpha > 0:
        mach_low = sample_mach_batch(n_low_alpha, mach_min=mach_min, mach_max=mach_max, device=device)
        alpha_high = min(alpha_max, alpha_min + low_alpha_band_width)
        alpha_low_edge = sample_alpha_batch(
            n_low_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_high,
            device=device,
        )
        alpha_chunks.append(alpha_low_edge)
        mach_chunks.append(mach_low)

    alpha = torch.cat(alpha_chunks, dim=0)
    mach = torch.cat(mach_chunks, dim=0)
    permutation = torch.randperm(alpha.shape[0], device=device)
    return alpha[permutation], mach[permutation]


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


@dataclass
class SubsonicReferenceCache2D:
    alpha_values: np.ndarray
    mach_values: np.ndarray
    ci_grid: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        num_alpha: int,
        num_mach: int,
    ) -> "SubsonicReferenceCache2D":
        alpha_values = np.linspace(alpha_min, alpha_max, num_alpha, dtype=float)
        mach_values = np.linspace(mach_min, mach_max, num_mach, dtype=float)
        ci_grid = np.zeros((num_mach, num_alpha), dtype=float)
        for i, mach in enumerate(mach_values):
            for j, alpha in enumerate(alpha_values):
                solver = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
                result = solver.solve()
                ci_grid[i, j] = float(result.ci)
        return cls(alpha_values=alpha_values, mach_values=mach_values, ci_grid=ci_grid)

    def interpolate(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        alpha_np = alpha.detach().cpu().numpy().reshape(-1)
        mach_np = mach.detach().cpu().numpy().reshape(-1)
        out = np.zeros_like(alpha_np)
        for idx, (a, m) in enumerate(zip(alpha_np, mach_np)):
            mach_idx = np.searchsorted(self.mach_values, m, side="left")
            mach_idx = int(np.clip(mach_idx, 1, len(self.mach_values) - 1))
            m0, m1 = self.mach_values[mach_idx - 1], self.mach_values[mach_idx]
            w = 0.0 if np.isclose(m1, m0) else (m - m0) / (m1 - m0)
            ci0 = np.interp(a, self.alpha_values, self.ci_grid[mach_idx - 1])
            ci1 = np.interp(a, self.alpha_values, self.ci_grid[mach_idx])
            out[idx] = (1.0 - w) * ci0 + w * ci1
        return torch.tensor(out, dtype=alpha.dtype, device=alpha.device).view(-1, 1)

    def audit_grid(self, *, num_alpha: int, num_mach: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha_eval = np.linspace(self.alpha_values.min(), self.alpha_values.max(), num_alpha, dtype=float)
        mach_eval = np.linspace(self.mach_values.min(), self.mach_values.max(), num_mach, dtype=float)
        aa, mm = np.meshgrid(alpha_eval, mach_eval)
        ci_eval = np.zeros_like(aa)
        for i in range(mm.shape[0]):
            for j in range(mm.shape[1]):
                mach_idx = np.searchsorted(self.mach_values, mm[i, j], side="left")
                mach_idx = int(np.clip(mach_idx, 1, len(self.mach_values) - 1))
                m0, m1 = self.mach_values[mach_idx - 1], self.mach_values[mach_idx]
                w = 0.0 if np.isclose(m1, m0) else (mm[i, j] - m0) / (m1 - m0)
                ci0 = np.interp(aa[i, j], self.alpha_values, self.ci_grid[mach_idx - 1])
                ci1 = np.interp(aa[i, j], self.alpha_values, self.ci_grid[mach_idx])
                ci_eval[i, j] = (1.0 - w) * ci0 + w * ci1
        return aa, mm, ci_eval
