from __future__ import annotations

import math

import torch

from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative


RICCATI_MATCH_STATE_CLAMP = 1.0e3
RICCATI_MATCH_RHS_CLAMP = 1.0e4
RICCATI_MATCH_EPS = 1.0e-8


def y_to_xi(y: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    """
    Inverse stable de `y = L * xi / (1 - xi^2)`.

    La forme rationalisee
        xi = 2 y / (L + sqrt(L^2 + 4 y^2))
    evite les annulations numeriques et preserve le signe de y.
    """
    L = torch.as_tensor(mapping_scale, dtype=y.dtype, device=y.device)
    root = torch.sqrt(L.pow(2) + 4.0 * y.pow(2))
    denom = L + root + float(RICCATI_MATCH_EPS)
    xi = 2.0 * y / denom
    return torch.clamp(xi, min=-0.999999, max=0.999999)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def _finite_complex(z: torch.Tensor, *, limit: float) -> torch.Tensor:
    real = torch.nan_to_num(z.real, nan=0.0, posinf=float(limit), neginf=-float(limit))
    imag = torch.nan_to_num(z.imag, nan=0.0, posinf=float(limit), neginf=-float(limit))
    return torch.complex(real, imag)


def _clamp_complex_abs(z: torch.Tensor, *, max_abs: float) -> torch.Tensor:
    z = _finite_complex(z, limit=float(max_abs))
    magnitude = torch.abs(z).clamp_min(1.0)
    scale = torch.clamp(float(max_abs) / magnitude, max=1.0)
    return z * scale


def _principal_sqrt_torch(z: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(z)
    flip = (root.real < 0) | ((torch.abs(root.real) < 1e-12) & (root.imag < 0))
    return torch.where(flip, -root, root)


def asymptotic_riccati_gammas_2d(
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ci = torch.clamp(ci, min=1.0e-6)
    one = torch.ones_like(ci)
    zero = torch.zeros_like(ci)
    c = torch.complex(zero, ci)
    mach_sq = mach.pow(2)

    left_r = one - mach_sq * ((-one - c) ** 2)
    right_r = one - mach_sq * ((one - c) ** 2)
    alpha_c = torch.complex(alpha, zero)

    gamma_left = alpha_c * _principal_sqrt_torch(left_r)
    gamma_right = -alpha_c * _principal_sqrt_torch(right_r)
    return gamma_left, gamma_right


def riccati_gamma_rhs_2d(
    y: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci: torch.Tensor,
) -> torch.Tensor:
    gamma = _clamp_complex_abs(gamma, max_abs=RICCATI_MATCH_STATE_CLAMP)
    ci = torch.clamp(ci, min=1.0e-6)
    zero = torch.zeros_like(ci)
    one = torch.ones_like(ci)
    c = torch.complex(zero, ci)
    u = torch.complex(base_velocity(y), zero)
    du = torch.complex(base_velocity_derivative(y), zero)
    alpha_c = torch.complex(alpha, zero)

    u_diff = u - c
    p_term = -2.0 * du / (u_diff + float(RICCATI_MATCH_EPS))
    r_term = one - mach.pow(2) * (u_diff**2)
    rhs = -(gamma**2) - p_term * gamma + (alpha_c**2) * r_term
    return _clamp_complex_abs(rhs, max_abs=RICCATI_MATCH_RHS_CLAMP)


def _riccati_rk4_step(
    y_val: torch.Tensor,
    gamma_val: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci: torch.Tensor,
) -> torch.Tensor:
    k1 = riccati_gamma_rhs_2d(y_val, gamma_val, alpha, mach, ci)
    k2 = riccati_gamma_rhs_2d(y_val + 0.5 * h, gamma_val + 0.5 * h * k1, alpha, mach, ci)
    k3 = riccati_gamma_rhs_2d(y_val + 0.5 * h, gamma_val + 0.5 * h * k2, alpha, mach, ci)
    k4 = riccati_gamma_rhs_2d(y_val + h, gamma_val + h * k3, alpha, mach, ci)
    gamma_next = gamma_val + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _clamp_complex_abs(gamma_next, max_abs=RICCATI_MATCH_STATE_CLAMP)


def _riccati_segment_integrate_batch(
    *,
    y_start: torch.Tensor,
    y_end: torch.Tensor,
    gamma_start: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci: torch.Tensor,
    n_substeps: int,
) -> torch.Tensor:
    if n_substeps <= 0:
        return gamma_start
    gamma_curr = gamma_start
    y_curr = y_start
    h = (y_end - y_start) / float(n_substeps)
    for _ in range(int(n_substeps)):
        gamma_curr = _riccati_rk4_step(y_curr, gamma_curr, h, alpha, mach, ci)
        y_curr = y_curr + h
    return gamma_curr


def _allocate_segment_substeps(y_nodes: torch.Tensor, total_steps: int) -> list[int]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if y_nodes.ndim != 1 or y_nodes.numel() < 2:
        raise ValueError("y_nodes must contain at least two points.")

    deltas = torch.abs(torch.diff(y_nodes)).detach().cpu().numpy().astype(float)
    total = float(np_sum := deltas.sum())
    if not math.isfinite(total) or total <= 0.0:
        return [max(1, int(total_steps // max(1, len(deltas))))] * len(deltas)

    raw = deltas / total * float(total_steps)
    steps = [max(1, int(round(value))) for value in raw]
    while sum(steps) < int(total_steps):
        idx = int(max(range(len(steps)), key=lambda j: raw[j] - steps[j]))
        steps[idx] += 1
    while sum(steps) > int(total_steps):
        candidates = [idx for idx, value in enumerate(steps) if value > 1]
        if not candidates:
            break
        idx = int(max(candidates, key=lambda j: steps[j] - raw[j]))
        steps[idx] -= 1
    return steps


def _integrate_gamma_from_boundary_to_targets(
    *,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci: torch.Tensor,
    y_boundary: float,
    y_targets: torch.Tensor,
    boundary_side: str,
    n_steps: int,
) -> torch.Tensor:
    if y_targets.ndim != 1:
        raise ValueError("y_targets must be a 1D tensor.")
    if y_targets.numel() == 0:
        return torch.empty(
            alpha.shape[0],
            0,
            dtype=_complex_dtype(alpha.dtype),
            device=alpha.device,
        )

    gamma_left_0, gamma_right_0 = asymptotic_riccati_gammas_2d(alpha, mach, ci)
    if boundary_side == "left":
        gamma_curr = gamma_left_0[:, 0]
    elif boundary_side == "right":
        gamma_curr = gamma_right_0[:, 0]
    else:
        raise ValueError(f"Unsupported boundary_side={boundary_side!r}.")

    y_nodes = torch.cat(
        [
            torch.tensor([float(y_boundary)], dtype=alpha.dtype, device=alpha.device),
            y_targets.to(device=alpha.device, dtype=alpha.dtype),
        ]
    )
    segment_steps = _allocate_segment_substeps(y_nodes, int(n_steps))
    batch_shape = alpha[:, 0]

    y_curr = torch.full_like(batch_shape, float(y_boundary))
    targets: list[torch.Tensor] = []
    for idx, n_substeps in enumerate(segment_steps, start=1):
        y_next = torch.full_like(batch_shape, float(y_nodes[idx]))
        gamma_curr = _riccati_segment_integrate_batch(
            y_start=y_curr,
            y_end=y_next,
            gamma_start=gamma_curr,
            alpha=alpha[:, 0],
            mach=mach[:, 0],
            ci=ci[:, 0],
            n_substeps=int(n_substeps),
        )
        targets.append(gamma_curr)
        y_curr = y_next

    gamma_targets = torch.stack(targets, dim=1)
    if not torch.isfinite(gamma_targets.real).all() or not torch.isfinite(gamma_targets.imag).all():
        raise FloatingPointError("Non-finite gamma targets encountered during Riccati matching integration.")
    return gamma_targets


def _complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    loss = diff.real.pow(2) + diff.imag.pow(2)
    return torch.mean(loss)


def _prepare_match_y_values(
    match_y_values: torch.Tensor | tuple[float, ...] | list[float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if match_y_values is None:
        values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=device)
    elif isinstance(match_y_values, torch.Tensor):
        values = match_y_values.to(device=device, dtype=dtype).reshape(-1)
    else:
        values = torch.tensor(list(match_y_values), dtype=dtype, device=device).reshape(-1)
    if values.numel() == 0:
        raise ValueError("match_y_values cannot be empty.")
    return values


def riccati_left_right_matching_loss_2d(
    model,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    match_y_values: torch.Tensor | tuple[float, ...] | list[float] | None = None,
    y_left: float = -40.0,
    y_right: float = 40.0,
    n_steps: int = 512,
    ci_override: torch.Tensor | None = None,
    detach_ci: bool = True,
    weight_net_left: float = 1.0,
    weight_net_right: float = 1.0,
    weight_left_right: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if getattr(model, "mode_representation", "cartesian") != "riccati":
        raise RuntimeError("riccati_left_right_matching_loss_2d requires a Riccati mode representation.")
    if alpha.shape != mach.shape:
        raise ValueError(f"alpha and mach must share the same shape, got {alpha.shape} vs {mach.shape}.")
    if alpha.ndim != 2 or alpha.shape[1] != 1:
        raise ValueError(f"alpha and mach must have shape [N, 1], got {alpha.shape}.")
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive.")

    device = alpha.device
    dtype = alpha.dtype
    match_y = _prepare_match_y_values(match_y_values, device=device, dtype=dtype)
    sorted_y, sort_idx = torch.sort(match_y)
    inverse_sort = torch.empty_like(sort_idx)
    inverse_sort[sort_idx] = torch.arange(sort_idx.numel(), device=device)

    if ci_override is None:
        ci = model.get_ci(alpha, mach)
    else:
        ci = ci_override
    if detach_ci:
        ci = ci.detach()
    ci = torch.clamp(ci, min=1.0e-6)

    mapping_scale = model.get_mapping_scale().detach()
    xi_match_sorted = y_to_xi(sorted_y.view(1, -1).expand(alpha.shape[0], -1), mapping_scale)
    alpha_match = alpha.expand(-1, sorted_y.numel()).reshape(-1, 1)
    mach_match = mach.expand(-1, sorted_y.numel()).reshape(-1, 1)
    xi_match = xi_match_sorted.reshape(-1, 1)
    gamma_pred_sorted = model(xi_match, alpha_match, mach_match)
    gamma_pred_sorted = torch.complex(gamma_pred_sorted[:, 0], gamma_pred_sorted[:, 1]).view(alpha.shape[0], -1)

    if not torch.isfinite(gamma_pred_sorted.real).all() or not torch.isfinite(gamma_pred_sorted.imag).all():
        raise FloatingPointError("Non-finite gamma prediction encountered at matching points.")

    if detach_ci:
        with torch.no_grad():
            gamma_left_sorted = _integrate_gamma_from_boundary_to_targets(
                alpha=alpha,
                mach=mach,
                ci=ci,
                y_boundary=float(y_left),
                y_targets=sorted_y,
                boundary_side="left",
                n_steps=int(n_steps),
            )
            gamma_right_desc = _integrate_gamma_from_boundary_to_targets(
                alpha=alpha,
                mach=mach,
                ci=ci,
                y_boundary=float(y_right),
                y_targets=torch.flip(sorted_y, dims=[0]),
                boundary_side="right",
                n_steps=int(n_steps),
            )
    else:
        gamma_left_sorted = _integrate_gamma_from_boundary_to_targets(
            alpha=alpha,
            mach=mach,
            ci=ci,
            y_boundary=float(y_left),
            y_targets=sorted_y,
            boundary_side="left",
            n_steps=int(n_steps),
        )
        gamma_right_desc = _integrate_gamma_from_boundary_to_targets(
            alpha=alpha,
            mach=mach,
            ci=ci,
            y_boundary=float(y_right),
            y_targets=torch.flip(sorted_y, dims=[0]),
            boundary_side="right",
            n_steps=int(n_steps),
        )

    gamma_right_sorted = torch.flip(gamma_right_desc, dims=[1])
    gamma_pred = gamma_pred_sorted[:, inverse_sort]
    gamma_left = gamma_left_sorted[:, inverse_sort]
    gamma_right = gamma_right_sorted[:, inverse_sort]

    loss_match_net_left = _complex_mse(gamma_pred, gamma_left)
    loss_match_net_right = _complex_mse(gamma_pred, gamma_right)
    loss_match_left_right = _complex_mse(gamma_left, gamma_right)

    abs_lr = torch.abs(gamma_left - gamma_right)
    metrics = {
        "loss_match_net_left": loss_match_net_left,
        "loss_match_net_right": loss_match_net_right,
        "loss_match_left_right": loss_match_left_right,
        "gamma_left_right_abs_mean": torch.mean(abs_lr),
        "gamma_left_right_abs_max": torch.max(abs_lr),
    }
    total_loss = (
        float(weight_net_left) * loss_match_net_left
        + float(weight_net_right) * loss_match_net_right
        + float(weight_left_right) * loss_match_left_right
    )
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite total Riccati matching loss.")
    return total_loss, metrics


__all__ = [
    "RICCATI_MATCH_EPS",
    "RICCATI_MATCH_RHS_CLAMP",
    "RICCATI_MATCH_STATE_CLAMP",
    "asymptotic_riccati_gammas_2d",
    "riccati_gamma_rhs_2d",
    "riccati_left_right_matching_loss_2d",
    "y_to_xi",
]
