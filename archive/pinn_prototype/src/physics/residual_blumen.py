from __future__ import annotations

import torch


def xi_to_y(xi: torch.Tensor, L: torch.Tensor | float) -> torch.Tensor:
    """
    Mapping algebrique de la these:
        y = L * xi / (1 - xi^2)
    qui envoie xi -> +/-1 vers y -> +/-inf.
    """
    return L * xi / (1.0 - xi.pow(2) + 1e-8)


def dy_dxi(xi: torch.Tensor, L: torch.Tensor | float) -> torch.Tensor:
    numerator = L * (1.0 + xi.pow(2))
    denominator = (1.0 - xi.pow(2) + 1e-8).pow(2)
    return numerator / denominator


def d2y_dxi2(xi: torch.Tensor, L: torch.Tensor | float) -> torch.Tensor:
    numerator = 2.0 * L * xi * (3.0 + xi.pow(2))
    denominator = (1.0 - xi.pow(2) + 1e-8).pow(3)
    return numerator / denominator


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
    alpha: float,
    mach: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcule le residu de l'equation de pression de Blumen en subsonique:

        (U-c) p'' - 2U' p' + alpha^2 (M^2 (U-c)^2 - 1) p = 0

    en travaillant dans la coordonnee bornee xi. On retourne:
    - residu partie reelle
    - residu partie imaginaire
    - y(xi), utile pour le diagnostic ou certaines losses auxiliaires
    """
    if not xi.requires_grad:
        xi.requires_grad_(True)

    outputs = model(xi)
    p_r = outputs[:, 0:1]
    p_i = outputs[:, 1:2]

    p_r_xi = _differentiate(p_r, xi)
    p_i_xi = _differentiate(p_i, xi)
    p_r_xixi = _differentiate(p_r_xi, xi)
    p_i_xixi = _differentiate(p_i_xi, xi)

    L = model.get_L()
    y = xi_to_y(xi, L)
    y_xi = dy_dxi(xi, L)
    y_xixi = d2y_dxi2(xi, L)

    # Regle de chaine:
    # d/dy = (d/dxi) / (dy/dxi)
    # d2/dy2 = d2/dxi2 / (dy/dxi)^2 - (d/dxi) (d2y/dxi2) / (dy/dxi)^3
    p_r_y = p_r_xi / y_xi
    p_i_y = p_i_xi / y_xi
    p_r_yy = p_r_xixi / y_xi.pow(2) - p_r_xi * y_xixi / y_xi.pow(3)
    p_i_yy = p_i_xixi / y_xi.pow(2) - p_i_xi * y_xixi / y_xi.pow(3)

    U = base_velocity(y)
    Uy = base_velocity_derivative(y)
    ci = model.get_ci()

    # En subsonique on impose c = i c_i, donc U - c = U - i c_i.
    # Separation explicite parties reelle / imaginaire pour rester en torch reel.
    ur = U
    ui = -ci.expand_as(U)

    q_r = mach**2 * (ur.pow(2) - ui.pow(2)) - 1.0
    q_i = 2.0 * mach**2 * ur * ui

    res_r = ur * p_r_yy - ui * p_i_yy - 2.0 * Uy * p_r_y + alpha**2 * (q_r * p_r - q_i * p_i)
    res_i = ur * p_i_yy + ui * p_r_yy - 2.0 * Uy * p_i_y + alpha**2 * (q_r * p_i + q_i * p_r)
    return res_r, res_i, y


def boundary_decay_loss(model, xi_left: torch.Tensor, xi_right: torch.Tensor) -> torch.Tensor:
    """
    Penalise une amplitude non nulle aux deux bords du domaine borne en xi.
    """
    pred_left = model(xi_left)
    pred_right = model(xi_right)
    return (pred_left.pow(2).mean() + pred_right.pow(2).mean())


def normalization_loss(model, xi_ref: torch.Tensor, target_pr: float = 1.0, target_pi: float = 0.0) -> torch.Tensor:
    """
    Fixe une amplitude/phase de reference pour eviter la solution triviale p = 0.
    """
    pred = model(xi_ref)
    target = torch.tensor([target_pr, target_pi], dtype=pred.dtype, device=pred.device).view(1, 2)
    return torch.mean((pred - target) ** 2)
