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


def _principal_sqrt_torch(z: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(z)
    flip = (root.real < 0) | ((torch.abs(root.real) < 1e-12) & (root.imag < 0))
    return torch.where(flip, -root, root)


def asymptotic_riccati_gammas(alpha: torch.Tensor, mach: float, ci: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    one = torch.ones_like(ci)
    c = torch.complex(torch.zeros_like(ci), ci)
    mach_t = torch.as_tensor(float(mach), dtype=ci.dtype, device=ci.device)

    r_inf_left = one - mach_t**2 * ((-one - c) ** 2)
    r_inf_right = one - mach_t**2 * ((one - c) ** 2)

    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    gamma_left = alpha_c * _principal_sqrt_torch(r_inf_left)
    gamma_right = -alpha_c * _principal_sqrt_torch(r_inf_right)
    return gamma_left, gamma_right


def riccati_gamma_rhs(
    y: torch.Tensor,
    gamma: torch.Tensor,
    alpha: torch.Tensor,
    mach: float,
    ci: torch.Tensor,
) -> torch.Tensor:
    one = torch.ones_like(ci)
    c = torch.complex(torch.zeros_like(ci), ci)
    u = torch.complex(base_velocity(y), torch.zeros_like(y))
    du = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    mach_t = torch.as_tensor(float(mach), dtype=ci.dtype, device=ci.device)

    u_diff = u - c
    p_term = -2.0 * du / u_diff
    r_term = one - mach_t**2 * (u_diff**2)
    return -(gamma**2) - p_term * gamma + (alpha_c**2) * r_term


def reconstruct_pressure_from_riccati(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha)
    kappa = outputs[:, 0:1]
    q = outputs[:, 1:2]

    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)

    order = torch.argsort(y[:, 0])
    y_sorted = y[order, 0]
    kappa_sorted = kappa[order, 0]
    q_sorted = q[order, 0]

    anchor_index = int(torch.argmin(torch.abs(y_sorted - xi_to_y(torch.tensor([[anchor_xi]], dtype=xi.dtype, device=xi.device), mapping_scale)[0, 0])).item())

    ln_amp_sorted = torch.zeros_like(y_sorted)
    phi_sorted = torch.zeros_like(y_sorted)

    for idx in range(anchor_index + 1, y_sorted.shape[0]):
        dy = y_sorted[idx] - y_sorted[idx - 1]
        ln_amp_sorted[idx] = ln_amp_sorted[idx - 1] + 0.5 * (kappa_sorted[idx] + kappa_sorted[idx - 1]) * dy
        phi_sorted[idx] = phi_sorted[idx - 1] + 0.5 * (q_sorted[idx] + q_sorted[idx - 1]) * dy

    for idx in range(anchor_index - 1, -1, -1):
        dy = y_sorted[idx + 1] - y_sorted[idx]
        ln_amp_sorted[idx] = ln_amp_sorted[idx + 1] - 0.5 * (kappa_sorted[idx] + kappa_sorted[idx + 1]) * dy
        phi_sorted[idx] = phi_sorted[idx + 1] - 0.5 * (q_sorted[idx] + q_sorted[idx + 1]) * dy

    amp_sorted = torch.exp(torch.clamp(ln_amp_sorted, min=-20.0, max=20.0))
    pr_sorted = amp_sorted * torch.cos(phi_sorted)
    pi_sorted = amp_sorted * torch.sin(phi_sorted)

    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0], device=order.device)
    pr = pr_sorted[inverse].view(-1, 1)
    pi = pi_sorted[inverse].view(-1, 1)
    return pr, pi, y


def reconstruct_pressure_from_riccati_2d(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha, mach)
    kappa = outputs[:, 0:1]
    q = outputs[:, 1:2]

    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)

    order = torch.argsort(y[:, 0])
    y_sorted = y[order, 0]
    kappa_sorted = kappa[order, 0]
    q_sorted = q[order, 0]

    anchor_y = xi_to_y(torch.tensor([[anchor_xi]], dtype=xi.dtype, device=xi.device), mapping_scale)[0, 0]
    anchor_index = int(torch.argmin(torch.abs(y_sorted - anchor_y)).item())

    ln_amp_sorted = torch.zeros_like(y_sorted)
    phi_sorted = torch.zeros_like(y_sorted)

    for idx in range(anchor_index + 1, y_sorted.shape[0]):
        dy = y_sorted[idx] - y_sorted[idx - 1]
        ln_amp_sorted[idx] = ln_amp_sorted[idx - 1] + 0.5 * (kappa_sorted[idx] + kappa_sorted[idx - 1]) * dy
        phi_sorted[idx] = phi_sorted[idx - 1] + 0.5 * (q_sorted[idx] + q_sorted[idx - 1]) * dy

    for idx in range(anchor_index - 1, -1, -1):
        dy = y_sorted[idx + 1] - y_sorted[idx]
        ln_amp_sorted[idx] = ln_amp_sorted[idx + 1] - 0.5 * (kappa_sorted[idx] + kappa_sorted[idx + 1]) * dy
        phi_sorted[idx] = phi_sorted[idx + 1] - 0.5 * (q_sorted[idx] + q_sorted[idx + 1]) * dy

    amp_sorted = torch.exp(torch.clamp(ln_amp_sorted, min=-20.0, max=20.0))
    pr_sorted = amp_sorted * torch.cos(phi_sorted)
    pi_sorted = amp_sorted * torch.sin(phi_sorted)

    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0], device=order.device)
    pr = pr_sorted[inverse].view(-1, 1)
    pi = pi_sorted[inverse].view(-1, 1)
    return pr, pi, y


def pressure_ode_residual(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: float,
    *,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equation modale de pression en subsonique, avec c = i c_i(alpha).
    """
    if not xi.requires_grad:
        xi.requires_grad_(True)

    if getattr(model, "mode_representation", "cartesian") == "riccati":
        outputs = model(xi, alpha)
        kappa = outputs[:, 0:1]
        q = outputs[:, 1:2]

        kappa_xi = _differentiate(kappa, xi)
        q_xi = _differentiate(q, xi)

        mapping_scale = model.get_mapping_scale()
        y = xi_to_y(xi, mapping_scale)
        y_xi = dy_dxi(xi, mapping_scale)

        kappa_y = kappa_xi / y_xi
        q_y = q_xi / y_xi

        u = base_velocity(y)
        du = base_velocity_derivative(y)
        ci = model.get_ci(alpha) if ci_override is None else ci_override

        gamma = torch.complex(kappa, q)
        c = torch.complex(torch.zeros_like(ci), ci)
        u_complex = torch.complex(u, torch.zeros_like(u))
        du_complex = torch.complex(du, torch.zeros_like(du))
        alpha_complex = torch.complex(alpha, torch.zeros_like(alpha))
        mach_t = torch.as_tensor(float(mach), dtype=xi.dtype, device=xi.device)

        u_diff = u_complex - c
        p_term = -2.0 * du_complex / u_diff
        r_term = 1.0 - mach_t**2 * (u_diff**2)
        gamma_rhs = -(gamma**2) - p_term * gamma + (alpha_complex**2) * r_term
        gamma_y = torch.complex(kappa_y, q_y)
        residual = gamma_y - gamma_rhs
        return residual.real, residual.imag, y

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
    ci = model.get_ci(alpha) if ci_override is None else ci_override

    ur = u
    ui = -ci
    q_r = mach**2 * (ur.pow(2) - ui.pow(2)) - 1.0
    q_i = 2.0 * mach**2 * ur * ui

    res_r = ur * p_r_yy - ui * p_i_yy - 2.0 * du * p_r_y + alpha.pow(2) * (q_r * p_r - q_i * p_i)
    res_i = ur * p_i_yy + ui * p_r_yy - 2.0 * du * p_i_y + alpha.pow(2) * (q_r * p_i + q_i * p_r)
    return res_r, res_i, y


def boundary_decay_loss(model, xi_left: torch.Tensor, xi_right: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if getattr(model, "mode_representation", "cartesian") == "riccati":
        loss_kappa, loss_q = riccati_boundary_loss_components(model, xi_left, xi_right, alpha)
        return loss_kappa + loss_q
    pred_left = model(xi_left, alpha)
    pred_right = model(xi_right, alpha)
    return pred_left.pow(2).mean() + pred_right.pow(2).mean()


def riccati_boundary_loss_components(
    model,
    xi_left: torch.Tensor,
    xi_right: torch.Tensor,
    alpha: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_left = model(xi_left, alpha)
    pred_right = model(xi_right, alpha)
    ci = model.get_ci(alpha) if ci_override is None else ci_override
    gamma_left, gamma_right = asymptotic_riccati_gammas(alpha, float(getattr(model, "mach", 0.5)), ci)

    loss_kappa = (
        (pred_left[:, 0:1] - gamma_left.real).pow(2).mean()
        + (pred_right[:, 0:1] - gamma_right.real).pow(2).mean()
    )
    loss_q = (
        (pred_left[:, 1:2] - gamma_left.imag).pow(2).mean()
        + (pred_right[:, 1:2] - gamma_right.imag).pow(2).mean()
    )
    return loss_kappa, loss_q


def riccati_boundary_band_loss_components(
    model,
    alpha: torch.Tensor,
    *,
    n_points: int,
    xi_start: float,
    xi_end: float,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_points <= 0:
        zero = torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()
        return zero, zero

    xi_left = torch.linspace(-float(xi_end), -float(xi_start), int(n_points), device=alpha.device, dtype=alpha.dtype).view(-1, 1)
    xi_right = torch.linspace(float(xi_start), float(xi_end), int(n_points), device=alpha.device, dtype=alpha.dtype).view(-1, 1)
    xi_left.requires_grad_(True)
    xi_right.requires_grad_(True)

    if alpha.shape[0] == 1:
        alpha_left = alpha.repeat(xi_left.shape[0], 1)
        alpha_right = alpha.repeat(xi_right.shape[0], 1)
    else:
        alpha_left = alpha[: xi_left.shape[0]]
        alpha_right = alpha[: xi_right.shape[0]]

    pred_left = model(xi_left, alpha_left)
    pred_right = model(xi_right, alpha_right)
    ci = model.get_ci(alpha_left) if ci_override is None else ci_override
    gamma_left, gamma_right = asymptotic_riccati_gammas(alpha_left, float(getattr(model, "mach", 0.5)), ci)

    loss_kappa = (
        (pred_left[:, 0:1] - gamma_left.real).pow(2).mean()
        + (pred_right[:, 0:1] - gamma_right.real).pow(2).mean()
    )
    loss_q = (
        (pred_left[:, 1:2] - gamma_left.imag).pow(2).mean()
        + (pred_right[:, 1:2] - gamma_right.imag).pow(2).mean()
    )
    return loss_kappa, loss_q


def riccati_center_constraints(
    model,
    alpha: torch.Tensor,
    *,
    center_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    xi_center = torch.full_like(alpha, float(center_xi), requires_grad=True)
    pred = model(xi_center, alpha)
    kappa = pred[:, 0:1]

    kappa_xi = _differentiate(kappa, xi_center)
    mapping_scale = model.get_mapping_scale()
    y_xi = dy_dxi(xi_center, mapping_scale)
    kappa_y = kappa_xi / y_xi

    loss_center = kappa.pow(2).mean()
    loss_peak = torch.relu(kappa_y).pow(2).mean()
    return loss_center, loss_peak


def riccati_shooting_match_loss(
    model,
    alpha: torch.Tensor,
    mach: float,
    *,
    n_steps: int,
    xi_boundary: float,
) -> torch.Tensor:
    if n_steps <= 0:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()

    mapping_scale = model.get_mapping_scale()
    xi_edge = torch.tensor([[float(xi_boundary)]], dtype=alpha.dtype, device=alpha.device)
    y_max = torch.abs(xi_to_y(xi_edge, mapping_scale))[0, 0]
    if not torch.isfinite(y_max) or float(torch.abs(y_max).detach().cpu()) <= 0.0:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()

    losses: list[torch.Tensor] = []
    alpha_samples = alpha[:, 0]
    ci_samples = model.get_ci(alpha)[:, 0]

    def rk4_step(y_val: torch.Tensor, gamma_val: torch.Tensor, h: torch.Tensor, alpha_val: torch.Tensor, ci_val: torch.Tensor) -> torch.Tensor:
        k1 = riccati_gamma_rhs(y_val, gamma_val, alpha_val, mach, ci_val)
        k2 = riccati_gamma_rhs(y_val + 0.5 * h, gamma_val + 0.5 * h * k1, alpha_val, mach, ci_val)
        k3 = riccati_gamma_rhs(y_val + 0.5 * h, gamma_val + 0.5 * h * k2, alpha_val, mach, ci_val)
        k4 = riccati_gamma_rhs(y_val + h, gamma_val + h * k3, alpha_val, mach, ci_val)
        return gamma_val + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for alpha_val, ci_val in zip(alpha_samples, ci_samples):
        alpha_scalar = alpha_val.view(1)
        ci_scalar = ci_val.view(1)
        gamma_left_0, gamma_right_0 = asymptotic_riccati_gammas(alpha_scalar, mach, ci_scalar)
        gamma_left = gamma_left_0[0]
        gamma_right = gamma_right_0[0]

        y_left = -y_max
        y_right = y_max
        h_left = (torch.zeros_like(y_left) - y_left) / float(n_steps)
        h_right = (torch.zeros_like(y_right) - y_right) / float(n_steps)

        y_curr = y_left
        for _ in range(int(n_steps)):
            gamma_left = rk4_step(y_curr, gamma_left, h_left, alpha_scalar[0], ci_scalar[0])
            y_curr = y_curr + h_left

        y_curr = y_right
        for _ in range(int(n_steps)):
            gamma_right = rk4_step(y_curr, gamma_right, h_right, alpha_scalar[0], ci_scalar[0])
            y_curr = y_curr + h_right

        mismatch = torch.abs(gamma_left - gamma_right) ** 2
        losses.append(mismatch.real)

    if not losses:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()
    return torch.stack(losses).mean()


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


def integral_normalization_loss(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    target_energy: float = 1.0,
) -> torch.Tensor:
    pred = model(xi, alpha)
    energy = torch.mean(pred[:, 0:1].pow(2) + pred[:, 1:2].pow(2))
    target = torch.tensor(float(target_energy), dtype=pred.dtype, device=pred.device)
    return (energy - target).pow(2)


def phase_loss(model, xi_ref: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    pred = model(xi_ref, alpha)
    imag_penalty = torch.mean(pred[:, 1:2].pow(2))
    sign_penalty = torch.mean(torch.relu(-pred[:, 0:1]).pow(2))
    return imag_penalty + sign_penalty


def localization_moment_losses(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model(xi, alpha)
    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)
    weight = pred[:, 0:1].pow(2) + pred[:, 1:2].pow(2) + 1e-12
    weight_sum = torch.mean(weight) + 1e-12
    y_bar = torch.mean(y * weight) / weight_sum
    spread = torch.mean((y - y_bar).pow(2) * weight) / weight_sum
    return y_bar.pow(2), spread


def local_peak_envelope_losses(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not xi_ref.requires_grad:
        xi_ref.requires_grad_(True)
    pred = model(xi_ref, alpha)
    env2 = pred[:, 0:1].pow(2) + pred[:, 1:2].pow(2)
    env2_xi = _differentiate(env2, xi_ref)
    env2_xixi = _differentiate(env2_xi, xi_ref)
    slope_loss = torch.mean(env2_xi.pow(2))
    curvature_loss = torch.mean(torch.relu(env2_xixi).pow(2))
    return slope_loss, curvature_loss


def pressure_ode_residual_2d(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not xi.requires_grad:
        xi.requires_grad_(True)

    if getattr(model, "mode_representation", "cartesian") == "riccati":
        outputs = model(xi, alpha, mach)
        kappa = outputs[:, 0:1]
        q = outputs[:, 1:2]

        kappa_xi = _differentiate(kappa, xi)
        q_xi = _differentiate(q, xi)

        mapping_scale = model.get_mapping_scale()
        y = xi_to_y(xi, mapping_scale)
        y_xi = dy_dxi(xi, mapping_scale)

        kappa_y = kappa_xi / y_xi
        q_y = q_xi / y_xi

        u = base_velocity(y)
        du = base_velocity_derivative(y)
        ci = ci_override if ci_override is not None else model.get_ci(alpha, mach)

        gamma = torch.complex(kappa, q)
        c = torch.complex(torch.zeros_like(ci), ci)
        u_complex = torch.complex(u, torch.zeros_like(u))
        du_complex = torch.complex(du, torch.zeros_like(du))
        alpha_complex = torch.complex(alpha, torch.zeros_like(alpha))
        mach_c = torch.complex(mach, torch.zeros_like(mach))

        u_diff = u_complex - c
        p_term = -2.0 * du_complex / u_diff
        r_term = 1.0 - mach_c.pow(2) * (u_diff**2)
        gamma_rhs = -(gamma**2) - p_term * gamma + (alpha_complex**2) * r_term
        gamma_y = torch.complex(kappa_y, q_y)
        residual = gamma_y - gamma_rhs
        return residual.real, residual.imag, y

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
    ci = ci_override if ci_override is not None else model.get_ci(alpha, mach)

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


def riccati_boundary_loss_components_2d(
    model,
    xi_left: torch.Tensor,
    xi_right: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_left = model(xi_left, alpha, mach)
    pred_right = model(xi_right, alpha, mach)
    ci = ci_override if ci_override is not None else model.get_ci(alpha, mach)

    one = torch.ones_like(ci)
    c = torch.complex(torch.zeros_like(ci), ci)
    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    mach_c = torch.complex(mach, torch.zeros_like(mach))

    r_inf_left = one - mach_c.pow(2) * ((-one - c) ** 2)
    r_inf_right = one - mach_c.pow(2) * ((one - c) ** 2)

    gamma_left = alpha_c * _principal_sqrt_torch(r_inf_left)
    gamma_right = -alpha_c * _principal_sqrt_torch(r_inf_right)

    loss_kappa = (
        (pred_left[:, 0:1] - gamma_left.real).pow(2).mean()
        + (pred_right[:, 0:1] - gamma_right.real).pow(2).mean()
    )
    loss_q = (
        (pred_left[:, 1:2] - gamma_left.imag).pow(2).mean()
        + (pred_right[:, 1:2] - gamma_right.imag).pow(2).mean()
    )
    return loss_kappa, loss_q


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
