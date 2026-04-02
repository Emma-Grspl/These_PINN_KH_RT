from __future__ import annotations

import torch


def xi_to_y(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return mapping_scale * xi / (1.0 - xi.pow(2) + 1e-8)


def dy_dxi(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return mapping_scale * (1.0 + xi.pow(2)) / (1.0 - xi.pow(2) + 1e-8).pow(2)


def d2y_dxi2(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return 2.0 * mapping_scale * xi * (3.0 + xi.pow(2)) / (1.0 - xi.pow(2) + 1e-8).pow(3)


def base_velocity(y: torch.Tensor) -> torch.Tensor:
    return torch.tanh(y)


def base_velocity_derivative(y: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(y).pow(2)


def _differentiate(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        values,
        x,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]


def pressure_ode_residual(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equation modale de pression en subsonique, avec c = i c_i(alpha).
    """
    if not xi.requires_grad:
        xi.requires_grad_(True)

    outputs = model(xi, alpha)
    p_r = outputs[:, 0:1]
    p_i = outputs[:, 1:2]

    p_r_xi = _differentiate(p_r, xi)
    p_i_xi = _differentiate(p_i, xi)
    p_r_xixi = _differentiate(p_r_xi, xi)
    p_i_xixi = _differentiate(p_i_xi, xi)

    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)
    y_xi = dy_dxi(xi, mapping_scale)
    y_xixi = d2y_dxi2(xi, mapping_scale)

    p_r_y = p_r_xi / y_xi
    p_i_y = p_i_xi / y_xi
    p_r_yy = p_r_xixi / y_xi.pow(2) - p_r_xi * y_xixi / y_xi.pow(3)
    p_i_yy = p_i_xixi / y_xi.pow(2) - p_i_xi * y_xixi / y_xi.pow(3)

    u = base_velocity(y)
    du = base_velocity_derivative(y)
    ci = model.get_ci(alpha)

    ur = u
    ui = -ci
    q_r = mach**2 * (ur.pow(2) - ui.pow(2)) - 1.0
    q_i = 2.0 * mach**2 * ur * ui

    res_r = ur * p_r_yy - ui * p_i_yy - 2.0 * du * p_r_y + alpha.pow(2) * (q_r * p_r - q_i * p_i)
    res_i = ur * p_i_yy + ui * p_r_yy - 2.0 * du * p_i_y + alpha.pow(2) * (q_r * p_i + q_i * p_r)
    return res_r, res_i, y


def boundary_decay_loss(model, xi_left: torch.Tensor, xi_right: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    pred_left = model(xi_left, alpha)
    pred_right = model(xi_right, alpha)
    return pred_left.pow(2).mean() + pred_right.pow(2).mean()


def normalization_loss(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
    *,
    target_pr: float = 1.0,
    target_pi: float = 0.0,
) -> torch.Tensor:
    pred = model(xi_ref, alpha)
    target = torch.tensor([target_pr, target_pi], dtype=pred.dtype, device=pred.device).view(1, 2)
    return torch.mean((pred - target) ** 2)


def phase_loss(model, xi_ref: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    pred = model(xi_ref, alpha)
    return torch.mean(pred[:, 1:2].pow(2))


def pressure_ode_residual_2d(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not xi.requires_grad:
        xi.requires_grad_(True)

    outputs = model(xi, alpha, mach)
    p_r = outputs[:, 0:1]
    p_i = outputs[:, 1:2]

    p_r_xi = _differentiate(p_r, xi)
    p_i_xi = _differentiate(p_i, xi)
    p_r_xixi = _differentiate(p_r_xi, xi)
    p_i_xixi = _differentiate(p_i_xi, xi)

    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)
    y_xi = dy_dxi(xi, mapping_scale)
    y_xixi = d2y_dxi2(xi, mapping_scale)

    p_r_y = p_r_xi / y_xi
    p_i_y = p_i_xi / y_xi
    p_r_yy = p_r_xixi / y_xi.pow(2) - p_r_xi * y_xixi / y_xi.pow(3)
    p_i_yy = p_i_xixi / y_xi.pow(2) - p_i_xi * y_xixi / y_xi.pow(3)

    u = base_velocity(y)
    du = base_velocity_derivative(y)
    ci = model.get_ci(alpha, mach)

    ur = u
    ui = -ci
    q_r = mach.pow(2) * (ur.pow(2) - ui.pow(2)) - 1.0
    q_i = 2.0 * mach.pow(2) * ur * ui

    res_r = ur * p_r_yy - ui * p_i_yy - 2.0 * du * p_r_y + alpha.pow(2) * (q_r * p_r - q_i * p_i)
    res_i = ur * p_i_yy + ui * p_r_yy - 2.0 * du * p_i_y + alpha.pow(2) * (q_r * p_i + q_i * p_r)
    return res_r, res_i, y


def boundary_decay_loss_2d(model, xi_left: torch.Tensor, xi_right: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
    pred_left = model(xi_left, alpha, mach)
    pred_right = model(xi_right, alpha, mach)
    return pred_left.pow(2).mean() + pred_right.pow(2).mean()


def normalization_loss_2d(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    target_pr: float = 1.0,
    target_pi: float = 0.0,
) -> torch.Tensor:
    pred = model(xi_ref, alpha, mach)
    target = torch.tensor([target_pr, target_pi], dtype=pred.dtype, device=pred.device).view(1, 2)
    return torch.mean((pred - target) ** 2)


def phase_loss_2d(model, xi_ref: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
    pred = model(xi_ref, alpha, mach)
    return torch.mean(pred[:, 1:2].pow(2))
