from __future__ import annotations

import torch

RICCATI_STATE_CLAMP = 1.0e3
RICCATI_RHS_CLAMP = 1.0e4


def xi_to_y(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return mapping_scale * xi / (1.0 - xi.pow(2) + 1e-8)


def dy_dxi(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return mapping_scale * (1.0 + xi.pow(2)) / (1.0 - xi.pow(2) + 1e-8).pow(2)


def d2y_dxi2(xi: torch.Tensor, mapping_scale: torch.Tensor | float) -> torch.Tensor:
    return 2.0 * mapping_scale * xi * (3.0 + xi.pow(2)) / (1.0 - xi.pow(2) + 1e-8).pow(3)


def base_velocity(y: torch.Tensor) -> torch.Tensor:
    return torch.tanh(y)


def base_velocity_derivative(y: torch.Tensor) -> torch.Tensor:
    # Stable sech^2(y). torch.cosh overflows in float32 for the far-field
    # compactified points used by the shooting losses.
    exp_term = torch.exp(-2.0 * torch.abs(y))
    return 4.0 * exp_term / (1.0 + exp_term).pow(2)


def _finite_complex(z: torch.Tensor, *, limit: float) -> torch.Tensor:
    real = torch.nan_to_num(z.real, nan=0.0, posinf=float(limit), neginf=-float(limit))
    imag = torch.nan_to_num(z.imag, nan=0.0, posinf=float(limit), neginf=-float(limit))
    return torch.complex(real, imag)


def _clamp_complex_abs(z: torch.Tensor, *, max_abs: float) -> torch.Tensor:
    z = _finite_complex(z, limit=float(max_abs))
    magnitude = torch.abs(z).clamp_min(1.0)
    scale = torch.clamp(float(max_abs) / magnitude, max=1.0)
    return z * scale


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
    gamma = _clamp_complex_abs(gamma, max_abs=RICCATI_STATE_CLAMP)
    ci = torch.clamp(ci, min=1e-5)
    one = torch.ones_like(ci)
    c = torch.complex(torch.zeros_like(ci), ci)
    u = torch.complex(base_velocity(y), torch.zeros_like(y))
    du = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    mach_t = torch.as_tensor(float(mach), dtype=ci.dtype, device=ci.device)

    u_diff = u - c
    p_term = -2.0 * du / u_diff
    r_term = one - mach_t**2 * (u_diff**2)
    rhs = -(gamma**2) - p_term * gamma + (alpha_c**2) * r_term
    return _clamp_complex_abs(rhs, max_abs=RICCATI_RHS_CLAMP)


def _reconstruct_riccati_pressure_core(
    outputs: torch.Tensor,
    xi: torch.Tensor,
    mapping_scale: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kappa = outputs[:, 0:1]
    q = outputs[:, 1:2]

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
    gamma = torch.complex(kappa, q)
    p = torch.complex(pr, pi)
    p_y = gamma * p
    return pr, pi, p_y, gamma, y


def reconstruct_pressure_from_riccati(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha)
    pr, pi, _, _, y = _reconstruct_riccati_pressure_core(
        outputs,
        xi,
        model.get_mapping_scale(),
        anchor_xi=anchor_xi,
    )
    return pr, pi, y


def reconstruct_pressure_p_y_from_riccati(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha)
    return _reconstruct_riccati_pressure_core(
        outputs,
        xi,
        model.get_mapping_scale(),
        anchor_xi=anchor_xi,
    )


def extract_pressure_components(outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return outputs[:, 0:1], outputs[:, 1:2]


def reconstruct_pressure_from_riccati_2d(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha, mach)
    pr, pi, _, _, y = _reconstruct_riccati_pressure_core(
        outputs,
        xi,
        model.get_mapping_scale(),
        anchor_xi=anchor_xi,
    )
    return pr, pi, y


def reconstruct_pressure_p_y_from_riccati_2d(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    anchor_xi: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(xi, alpha, mach)
    return _reconstruct_riccati_pressure_core(
        outputs,
        xi,
        model.get_mapping_scale(),
        anchor_xi=anchor_xi,
    )


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

    mode_representation = getattr(model, "mode_representation", "cartesian")
    if mode_representation == "riccati":
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
    if mode_representation == "first_order_real":
        p_r = outputs[:, 0:1]
        p_i = outputs[:, 1:2]
        v_r = outputs[:, 2:3]
        v_i = outputs[:, 3:4]

        p_r_xi = _differentiate(p_r, xi)
        p_i_xi = _differentiate(p_i, xi)
        v_r_xi = _differentiate(v_r, xi)
        v_i_xi = _differentiate(v_i, xi)

        mapping_scale = model.get_mapping_scale()
        y = xi_to_y(xi, mapping_scale)
        y_xi = dy_dxi(xi, mapping_scale)

        u = base_velocity(y)
        du = base_velocity_derivative(y)
        ci = model.get_ci(alpha) if ci_override is None else ci_override

        p = torch.complex(p_r, p_i)
        v = torch.complex(v_r, v_i)
        c = torch.complex(torch.zeros_like(ci), ci)
        u_complex = torch.complex(u, torch.zeros_like(u))
        du_complex = torch.complex(du, torch.zeros_like(du))
        alpha_complex = torch.complex(alpha, torch.zeros_like(alpha))
        mach_t = torch.as_tensor(float(mach), dtype=xi.dtype, device=xi.device)

        u_diff = u_complex - c
        r_term = 1.0 - mach_t**2 * (u_diff**2)
        f = (2.0 * du_complex * v - (alpha_complex**2) * r_term * p) / u_diff

        res_state_r = p_r_xi - y_xi * v_r
        res_state_i = p_i_xi - y_xi * v_i
        res_dyn_r = v_r_xi - y_xi * f.real
        res_dyn_i = v_i_xi - y_xi * f.imag
        return torch.cat([res_state_r, res_dyn_r], dim=0), torch.cat([res_state_i, res_dyn_i], dim=0), y

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
    mode_representation = getattr(model, "mode_representation", "cartesian")
    if mode_representation == "riccati":
        loss_kappa, loss_q = riccati_boundary_loss_components(model, xi_left, xi_right, alpha)
        return loss_kappa + loss_q
    if mode_representation == "first_order_real":
        pred_left = model(xi_left, alpha)
        pred_right = model(xi_right, alpha)
        p_left_r, p_left_i = extract_pressure_components(pred_left)
        p_right_r, p_right_i = extract_pressure_components(pred_right)
        v_left_r = pred_left[:, 2:3]
        v_left_i = pred_left[:, 3:4]
        v_right_r = pred_right[:, 2:3]
        v_right_i = pred_right[:, 3:4]
        ci = model.get_ci(alpha)
        gamma_left, gamma_right = asymptotic_riccati_gammas(alpha, float(getattr(model, "mach", 0.5)), ci)
        target_left = gamma_left * torch.complex(p_left_r, p_left_i)
        target_right = gamma_right * torch.complex(p_right_r, p_right_i)
        loss_left = (v_left_r - target_left.real).pow(2).mean() + (v_left_i - target_left.imag).pow(2).mean()
        loss_right = (v_right_r - target_right.real).pow(2).mean() + (v_right_i - target_right.imag).pow(2).mean()
        return loss_left + loss_right
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

    alpha_samples = alpha[:, 0]
    ci_samples = model.get_ci(alpha)[:, 0]

    return _riccati_shooting_match_loss_from_samples(
        mapping_scale,
        alpha_samples,
        ci_samples,
        mach,
        n_steps=n_steps,
        xi_boundary=xi_boundary,
    ).mean()


def riccati_shooting_path_loss(
    model,
    alpha: torch.Tensor,
    mach: float,
    *,
    n_steps: int,
    xi_boundary: float,
    n_points: int,
) -> torch.Tensor:
    if n_steps <= 0 or n_points < 2:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()

    alpha_samples = alpha[:, 0]
    ci_samples = model.get_ci(alpha)[:, 0]
    mapping_scale = model.get_mapping_scale()

    losses: list[torch.Tensor] = []
    for alpha_val, ci_val in zip(alpha_samples, ci_samples):
        alpha_scalar = alpha_val.view(1)
        ci_scalar = ci_val.view(1)
        sample_loss = _riccati_shooting_path_loss_from_sample(
            model,
            mapping_scale,
            alpha_scalar,
            ci_scalar,
            mach,
            n_steps=n_steps,
            xi_boundary=xi_boundary,
            n_points=n_points,
        )
        losses.append(sample_loss)

    if not losses:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()
    return torch.stack(losses).mean()


def _riccati_rk4_step(
    y_val: torch.Tensor,
    gamma_val: torch.Tensor,
    h: torch.Tensor,
    alpha_val: torch.Tensor,
    mach: float,
    ci_val: torch.Tensor,
) -> torch.Tensor:
    k1 = riccati_gamma_rhs(y_val, gamma_val, alpha_val, mach, ci_val)
    k2 = riccati_gamma_rhs(y_val + 0.5 * h, gamma_val + 0.5 * h * k1, alpha_val, mach, ci_val)
    k3 = riccati_gamma_rhs(y_val + 0.5 * h, gamma_val + 0.5 * h * k2, alpha_val, mach, ci_val)
    k4 = riccati_gamma_rhs(y_val + h, gamma_val + h * k3, alpha_val, mach, ci_val)
    gamma_next = gamma_val + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _clamp_complex_abs(gamma_next, max_abs=RICCATI_STATE_CLAMP)


def _riccati_segment_integrate(
    y_start: torch.Tensor,
    y_end: torch.Tensor,
    gamma_start: torch.Tensor,
    alpha_val: torch.Tensor,
    mach: float,
    ci_val: torch.Tensor,
    *,
    n_substeps: int,
) -> torch.Tensor:
    if n_substeps <= 0:
        return gamma_start
    gamma_curr = gamma_start
    y_curr = y_start
    h = (y_end - y_start) / float(n_substeps)
    for _ in range(int(n_substeps)):
        gamma_curr = _riccati_rk4_step(y_curr, gamma_curr, h, alpha_val, mach, ci_val)
        y_curr = y_curr + h
    return gamma_curr


def _riccati_shooting_path_loss_from_sample(
    model,
    mapping_scale: torch.Tensor,
    alpha_scalar: torch.Tensor,
    ci_scalar: torch.Tensor,
    mach: float,
    *,
    n_steps: int,
    xi_boundary: float,
    n_points: int,
) -> torch.Tensor:
    device = alpha_scalar.device
    dtype = alpha_scalar.dtype

    xi_left = torch.linspace(-float(xi_boundary), 0.0, int(n_points), device=device, dtype=dtype).view(-1, 1)
    xi_right = torch.linspace(0.0, float(xi_boundary), int(n_points), device=device, dtype=dtype).view(-1, 1)

    alpha_left = alpha_scalar.view(1, 1).repeat(xi_left.shape[0], 1)
    alpha_right = alpha_scalar.view(1, 1).repeat(xi_right.shape[0], 1)

    pred_left = model(xi_left, alpha_left)
    pred_right = model(xi_right, alpha_right)
    gamma_left_pred = torch.complex(pred_left[:, 0], pred_left[:, 1])
    gamma_right_pred = torch.complex(pred_right[:, 0], pred_right[:, 1])

    y_left = xi_to_y(xi_left, mapping_scale)[:, 0]
    y_right = xi_to_y(xi_right, mapping_scale)[:, 0]

    gamma_left_0, gamma_right_0 = asymptotic_riccati_gammas(alpha_scalar, mach, ci_scalar)
    gamma_left_curr = gamma_left_0[0]
    gamma_right_curr = gamma_right_0[0]

    left_path = [gamma_left_curr]
    right_desc_path = [gamma_right_curr]

    n_segments = max(int(n_points) - 1, 1)
    n_substeps = max(int(n_steps) // n_segments, 1)

    for idx in range(1, y_left.shape[0]):
        gamma_left_curr = _riccati_segment_integrate(
            y_left[idx - 1],
            y_left[idx],
            gamma_left_curr,
            alpha_scalar[0],
            mach,
            ci_scalar[0],
            n_substeps=n_substeps,
        )
        left_path.append(gamma_left_curr)

    y_right_desc = torch.flip(y_right, dims=[0])
    for idx in range(1, y_right_desc.shape[0]):
        gamma_right_curr = _riccati_segment_integrate(
            y_right_desc[idx - 1],
            y_right_desc[idx],
            gamma_right_curr,
            alpha_scalar[0],
            mach,
            ci_scalar[0],
            n_substeps=n_substeps,
        )
        right_desc_path.append(gamma_right_curr)

    gamma_left_target = torch.stack(left_path)
    gamma_right_target = torch.flip(torch.stack(right_desc_path), dims=[0])

    gamma_left_target = _clamp_complex_abs(gamma_left_target, max_abs=RICCATI_STATE_CLAMP)
    gamma_right_target = _clamp_complex_abs(gamma_right_target, max_abs=RICCATI_STATE_CLAMP)
    gamma_left_pred = _clamp_complex_abs(gamma_left_pred, max_abs=RICCATI_STATE_CLAMP)
    gamma_right_pred = _clamp_complex_abs(gamma_right_pred, max_abs=RICCATI_STATE_CLAMP)

    left_loss = (gamma_left_pred.real - gamma_left_target.real).pow(2) + (gamma_left_pred.imag - gamma_left_target.imag).pow(2)
    right_loss = (gamma_right_pred.real - gamma_right_target.real).pow(2) + (gamma_right_pred.imag - gamma_right_target.imag).pow(2)
    loss = 0.5 * (left_loss.mean() + right_loss.mean())
    return torch.nan_to_num(loss, nan=RICCATI_STATE_CLAMP**2, posinf=RICCATI_STATE_CLAMP**2, neginf=RICCATI_STATE_CLAMP**2)


def _riccati_shooting_match_loss_from_samples(
    mapping_scale: torch.Tensor,
    alpha_samples: torch.Tensor,
    ci_samples: torch.Tensor,
    mach: float,
    *,
    n_steps: int,
    xi_boundary: float,
) -> torch.Tensor:
    if n_steps <= 0:
        return torch.zeros(alpha_samples.shape[0], device=alpha_samples.device, dtype=alpha_samples.dtype)

    xi_edge = torch.tensor([[float(xi_boundary)]], dtype=alpha_samples.dtype, device=alpha_samples.device)
    y_max = torch.abs(xi_to_y(xi_edge, mapping_scale))[0, 0]
    if not torch.isfinite(y_max) or float(torch.abs(y_max).detach().cpu()) <= 0.0:
        return torch.zeros(alpha_samples.shape[0], device=alpha_samples.device, dtype=alpha_samples.dtype)

    def rk4_step(y_val: torch.Tensor, gamma_val: torch.Tensor, h: torch.Tensor, alpha_val: torch.Tensor, ci_val: torch.Tensor) -> torch.Tensor:
        return _riccati_rk4_step(y_val, gamma_val, h, alpha_val, mach, ci_val)

    losses: list[torch.Tensor] = []
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
        losses.append(torch.nan_to_num(mismatch.real, nan=RICCATI_STATE_CLAMP**2, posinf=RICCATI_STATE_CLAMP**2, neginf=RICCATI_STATE_CLAMP**2))

    if not losses:
        return torch.zeros(alpha_samples.shape[0], device=alpha_samples.device, dtype=alpha_samples.dtype)
    return torch.stack(losses)


def riccati_ci_local_min_loss(
    model,
    alpha: torch.Tensor,
    mach: float,
    *,
    n_steps: int,
    xi_boundary: float,
    delta_abs: float,
    delta_rel: float,
    margin: float = 0.0,
) -> torch.Tensor:
    if n_steps <= 0:
        return torch.zeros(1, device=alpha.device, dtype=alpha.dtype).mean()

    ci_center = model.get_ci(alpha)[:, 0]
    delta = torch.clamp(
        float(delta_abs) + float(delta_rel) * torch.abs(ci_center),
        min=max(float(delta_abs), 1e-6),
    )
    ci_minus = torch.clamp(ci_center - delta, min=1e-6)
    ci_plus = ci_center + delta

    mapping_scale = model.get_mapping_scale()
    alpha_samples = alpha[:, 0]
    mismatch_center = _riccati_shooting_match_loss_from_samples(
        mapping_scale,
        alpha_samples,
        ci_center,
        mach,
        n_steps=n_steps,
        xi_boundary=xi_boundary,
    )
    mismatch_minus = _riccati_shooting_match_loss_from_samples(
        mapping_scale,
        alpha_samples,
        ci_minus,
        mach,
        n_steps=n_steps,
        xi_boundary=xi_boundary,
    )
    mismatch_plus = _riccati_shooting_match_loss_from_samples(
        mapping_scale,
        alpha_samples,
        ci_plus,
        mach,
        n_steps=n_steps,
        xi_boundary=xi_boundary,
    )

    margin_t = torch.as_tensor(float(margin), dtype=alpha.dtype, device=alpha.device)
    penalty_minus = torch.relu(mismatch_center - mismatch_minus + margin_t).pow(2)
    penalty_plus = torch.relu(mismatch_center - mismatch_plus + margin_t).pow(2)
    return 0.5 * (penalty_minus.mean() + penalty_plus.mean())


def normalization_loss(
    model,
    xi_ref: torch.Tensor,
    alpha: torch.Tensor,
    *,
    target_pr: float = 1.0,
    target_pi: float = 0.0,
) -> torch.Tensor:
    pred = model(xi_ref, alpha)
    pred_pr, pred_pi = extract_pressure_components(pred)
    pred_mode = torch.cat([pred_pr, pred_pi], dim=-1)
    target = torch.tensor([target_pr, target_pi], dtype=pred.dtype, device=pred.device).view(1, 2)
    return torch.mean((pred_mode - target) ** 2)


def integral_normalization_loss(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    target_energy: float = 1.0,
) -> torch.Tensor:
    pred = model(xi, alpha)
    pred_pr, pred_pi = extract_pressure_components(pred)
    energy = torch.mean(pred_pr.pow(2) + pred_pi.pow(2))
    target = torch.tensor(float(target_energy), dtype=pred.dtype, device=pred.device)
    return (energy - target).pow(2)


def phase_loss(model, xi_ref: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    pred = model(xi_ref, alpha)
    pred_pr, pred_pi = extract_pressure_components(pred)
    imag_penalty = torch.mean(pred_pi.pow(2))
    sign_penalty = torch.mean(torch.relu(-pred_pr).pow(2))
    return imag_penalty + sign_penalty


def localization_moment_losses(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model(xi, alpha)
    pred_pr, pred_pi = extract_pressure_components(pred)
    mapping_scale = model.get_mapping_scale()
    y = xi_to_y(xi, mapping_scale)
    weight = pred_pr.pow(2) + pred_pi.pow(2) + 1e-12
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
    pred_pr, pred_pi = extract_pressure_components(pred)
    env2 = pred_pr.pow(2) + pred_pi.pow(2)
    env2_xi = _differentiate(env2, xi_ref)
    env2_xixi = _differentiate(env2_xi, xi_ref)
    slope_loss = torch.mean(env2_xi.pow(2))
    curvature_loss = torch.mean(torch.relu(env2_xixi).pow(2))
    return slope_loss, curvature_loss


def first_order_stabilization_losses(
    model,
    xi: torch.Tensor,
    alpha: torch.Tensor,
    *,
    amp_cap: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(model, "mode_representation", "cartesian") != "first_order_real":
        zero = torch.zeros(1, device=xi.device, dtype=xi.dtype).mean()
        return zero, zero

    pred = model(xi, alpha)
    pr = pred[:, 0:1]
    pi = pred[:, 1:2]
    vr = pred[:, 2:3]
    vi = pred[:, 3:4]
    amp2 = pr.pow(2) + pi.pow(2)
    v_energy = torch.mean(vr.pow(2) + vi.pow(2))
    amp_cap_loss = torch.mean(torch.relu(amp2 - float(amp_cap) ** 2).pow(2))
    return v_energy, amp_cap_loss


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
