from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kh_subsonic_pinn import build_mlp
from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative


SUPERSONIC_PRESSURE_FIRST_EPS = 1.0e-8


def _differentiate(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def _ensure_finite(name: str, value: torch.Tensor) -> None:
    if torch.is_complex(value):
        finite = torch.isfinite(value.real).all() and torch.isfinite(value.imag).all()
    else:
        finite = torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"Non-finite tensor encountered in {name}.")


def _principal_sqrt_torch(z: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(z)
    flip = (root.real < 0) | ((torch.abs(root.real) < 1e-12) & (root.imag < 0))
    return torch.where(flip, -root, root)


def asymptotic_supersonic_gammas(
    alpha: torch.Tensor,
    mach: torch.Tensor,
    cr: torch.Tensor,
    ci: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ci = torch.clamp(ci, min=1.0e-8)
    c = torch.complex(cr, ci)
    one = torch.ones_like(cr)
    mach_sq = mach.pow(2)

    z_minus = one - mach_sq * ((-one - c) ** 2)
    z_plus = one - mach_sq * ((one - c) ** 2)

    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    gamma_minus = alpha_c * _principal_sqrt_torch(z_minus)
    gamma_plus = -alpha_c * _principal_sqrt_torch(z_plus)

    gamma_minus = torch.where(gamma_minus.real < 0, -gamma_minus, gamma_minus)
    gamma_plus = torch.where(gamma_plus.real > 0, -gamma_plus, gamma_plus)
    return gamma_minus, gamma_plus


class KHSupersonicPressureFirst1D(nn.Module):
    def __init__(
        self,
        *,
        alpha: float,
        mach: float,
        cr: float,
        ci: float,
        hidden_dim: int = 192,
        depth: int = 6,
        activation: str = "tanh",
        ymax: float = 120.0,
        envelope_eps: float = 1.0,
    ):
        super().__init__()
        self.alpha_value = float(alpha)
        self.mach_value = float(mach)
        self.cr_value = float(cr)
        self.ci_value = float(ci)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.activation = str(activation)
        self.ymax = float(ymax)
        self.envelope_eps = float(envelope_eps)
        self.mode_representation = "pressure_first"

        self.pressure_net = build_mlp(
            9,
            2,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=str(activation),
        )

    def export_model_config(self) -> dict[str, object]:
        return {
            "alpha": float(self.alpha_value),
            "mach": float(self.mach_value),
            "cr": float(self.cr_value),
            "ci": float(self.ci_value),
            "hidden_dim": int(self.hidden_dim),
            "depth": int(self.depth),
            "activation": str(self.activation),
            "ymax": float(self.ymax),
            "envelope_eps": float(self.envelope_eps),
        }

    def scalar_parameter_tensors(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        alpha = torch.full_like(y, float(self.alpha_value))
        mach = torch.full_like(y, float(self.mach_value))
        cr = torch.full_like(y, float(self.cr_value))
        ci = torch.full_like(y, float(self.ci_value))
        return {"alpha": alpha, "mach": mach, "cr": cr, "ci": ci}

    def envelope_terms(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        params = self.scalar_parameter_tensors(y)
        gamma_minus, gamma_plus = asymptotic_supersonic_gammas(
            params["alpha"],
            params["mach"],
            params["cr"],
            params["ci"],
        )
        eps = max(float(self.envelope_eps), 1.0e-6)
        mu_minus = gamma_minus.real.clamp_min(1.0e-6)
        mu_plus = (-gamma_plus.real).clamp_min(1.0e-6)

        left_dist = eps * F.softplus(-y / eps)
        right_dist = eps * F.softplus(y / eps)
        center_dist = eps * F.softplus(torch.zeros(1, device=y.device, dtype=y.dtype))
        e_raw = torch.exp(-mu_minus * left_dist - mu_plus * right_dist)
        e0 = torch.exp(-mu_minus * center_dist - mu_plus * center_dist)
        envelope = e_raw / (e0 + float(SUPERSONIC_PRESSURE_FIRST_EPS))
        _ensure_finite("supersonic envelope", envelope)
        return {
            **params,
            "gamma_minus": gamma_minus,
            "gamma_plus": gamma_plus,
            "mu_minus": mu_minus,
            "mu_plus": mu_plus,
            "left_dist": left_dist,
            "right_dist": right_dist,
            "envelope": envelope,
        }

    def encode_inputs(self, y: torch.Tensor, env: dict[str, torch.Tensor]) -> torch.Tensor:
        ymax = max(float(self.ymax), 1.0e-6)
        features = [
            y / ymax,
            torch.tanh(y),
            env["alpha"] * y / 10.0,
            env["mu_minus"] * env["left_dist"] / 10.0,
            env["mu_plus"] * env["right_dist"] / 10.0,
            env["alpha"],
            env["mach"],
            env["cr"],
            env["ci"],
        ]
        encoded = torch.cat(features, dim=-1)
        _ensure_finite("supersonic pressure inputs", encoded)
        return encoded

    def pressure_components(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        env = self.envelope_terms(y)
        raw_h = self.pressure_net(self.encode_inputs(y, env))
        h = torch.complex(raw_h[:, 0:1], raw_h[:, 1:2])
        p = torch.complex(env["envelope"], torch.zeros_like(env["envelope"])) * h
        _ensure_finite("supersonic h", h)
        _ensure_finite("supersonic p", p)
        return {**env, "h": h, "p": p}

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        p = self.pressure_components(y)["p"]
        return torch.cat([p.real, p.imag], dim=-1)


def build_supersonic_pressure_first_model_from_config(config: Mapping[str, object]) -> KHSupersonicPressureFirst1D:
    return KHSupersonicPressureFirst1D(
        alpha=float(config["alpha"]),
        mach=float(config["mach"]),
        cr=float(config["cr"]),
        ci=float(config["ci"]),
        hidden_dim=int(config.get("hidden_dim", 192)),
        depth=int(config.get("depth", 6)),
        activation=str(config.get("activation", "tanh")),
        ymax=float(config.get("ymax", 120.0)),
        envelope_eps=float(config.get("envelope_eps", 1.0)),
    )


def supersonic_pressure_value_and_derivatives(
    model: KHSupersonicPressureFirst1D,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not y.requires_grad:
        y.requires_grad_(True)
    comp = model.pressure_components(y)
    p = comp["p"]
    p_r = p.real
    p_i = p.imag
    p_r_y = _differentiate(p_r, y)
    p_i_y = _differentiate(p_i, y)
    p_r_yy = _differentiate(p_r_y, y)
    p_i_yy = _differentiate(p_i_y, y)
    p_y = torch.complex(p_r_y, p_i_y)
    p_yy = torch.complex(p_r_yy, p_i_yy)
    _ensure_finite("supersonic p_y", p_y)
    _ensure_finite("supersonic p_yy", p_yy)
    return {**comp, "p_y": p_y, "p_yy": p_yy}


def supersonic_pressure_ode_residual(
    model: KHSupersonicPressureFirst1D,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    fields = supersonic_pressure_value_and_derivatives(model, y)
    p = fields["p"]
    p_y = fields["p_y"]
    p_yy = fields["p_yy"]
    c = torch.complex(fields["cr"], fields["ci"])
    u = torch.complex(base_velocity(y), torch.zeros_like(y))
    uy = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    alpha_c = torch.complex(fields["alpha"], torch.zeros_like(fields["alpha"]))
    mach_c = torch.complex(fields["mach"], torch.zeros_like(fields["mach"]))

    u_diff = u - c
    a_coeff = 2.0 * uy / (u_diff + float(SUPERSONIC_PRESSURE_FIRST_EPS))
    b_coeff = alpha_c.pow(2) * (1.0 - mach_c.pow(2) * (u_diff**2))
    residual = p_yy - a_coeff * p_y - b_coeff * p

    scale = 1.0 + torch.abs(p_yy).pow(2) + torch.abs(a_coeff * p_y).pow(2) + torch.abs(b_coeff * p).pow(2)
    residual = residual / scale.detach().clamp_min(1.0)
    _ensure_finite("supersonic residual", residual)
    return residual.real, residual.imag, {
        **fields,
        "a_coeff": a_coeff,
        "b_coeff": b_coeff,
        "residual": residual,
    }


def supersonic_pressure_robin_boundary_loss(
    model: KHSupersonicPressureFirst1D,
    y_left: torch.Tensor,
    y_right: torch.Tensor,
) -> torch.Tensor:
    left_fields = supersonic_pressure_value_and_derivatives(model, y_left)
    right_fields = supersonic_pressure_value_and_derivatives(model, y_right)

    res_left = left_fields["p_y"] - left_fields["gamma_minus"] * left_fields["p"]
    res_right = right_fields["p_y"] - right_fields["gamma_plus"] * right_fields["p"]

    scale_left = 1.0 + torch.abs(left_fields["p_y"]).pow(2) + torch.abs(left_fields["gamma_minus"] * left_fields["p"]).pow(2)
    scale_right = 1.0 + torch.abs(right_fields["p_y"]).pow(2) + torch.abs(right_fields["gamma_plus"] * right_fields["p"]).pow(2)

    loss = torch.mean((res_left.real.pow(2) + res_left.imag.pow(2)) / scale_left.detach().clamp_min(1.0))
    loss = loss + torch.mean((res_right.real.pow(2) + res_right.imag.pow(2)) / scale_right.detach().clamp_min(1.0))
    _ensure_finite("supersonic Robin loss", loss)
    return loss


def supersonic_pressure_gauge_loss(
    model: KHSupersonicPressureFirst1D,
) -> torch.Tensor:
    y0 = torch.zeros(1, 1, dtype=torch.float32, device=next(model.parameters()).device, requires_grad=True)
    fields = model.pressure_components(y0)
    p0 = fields["p"]
    loss = torch.mean((p0.real - 1.0).pow(2) + p0.imag.pow(2))
    _ensure_finite("supersonic gauge loss", loss)
    return loss


__all__ = [
    "KHSupersonicPressureFirst1D",
    "SUPERSONIC_PRESSURE_FIRST_EPS",
    "asymptotic_supersonic_gammas",
    "build_supersonic_pressure_first_model_from_config",
    "supersonic_pressure_gauge_loss",
    "supersonic_pressure_ode_residual",
    "supersonic_pressure_robin_boundary_loss",
    "supersonic_pressure_value_and_derivatives",
]
