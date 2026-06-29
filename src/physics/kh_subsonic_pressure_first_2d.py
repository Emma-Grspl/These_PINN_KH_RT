from __future__ import annotations

from collections.abc import Mapping
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kh_subsonic_pinn import build_mlp
from src.physics.kh_subsonic_residual import base_velocity, base_velocity_derivative


PRESSURE_FIRST_EPS = 1.0e-8


def _differentiate(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def _principal_sqrt_torch(z: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(z)
    flip = (root.real < 0) | ((torch.abs(root.real) < 1e-12) & (root.imag < 0))
    return torch.where(flip, -root, root)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def _ensure_finite(name: str, value: torch.Tensor) -> None:
    if torch.is_complex(value):
        finite = torch.isfinite(value.real).all() and torch.isfinite(value.imag).all()
    else:
        finite = torch.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"Non-finite tensor encountered in {name}.")


def _infer_ci_architecture(ci_net: nn.Module) -> tuple[int, int]:
    linear_layers = [module for module in ci_net.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("Could not infer ci_net architecture from Stage 0.")
    depth = max(len(linear_layers) - 1, 0)
    hidden_dim = int(linear_layers[0].out_features if depth > 0 else linear_layers[0].in_features)
    return hidden_dim, depth


def asymptotic_pressure_gammas(
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

    gamma_minus = alpha_c * _principal_sqrt_torch(left_r)
    gamma_plus = -alpha_c * _principal_sqrt_torch(right_r)
    return gamma_minus, gamma_plus


class KHSubsonicPressureFirst2D(nn.Module):
    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        ci_hidden_dim: int,
        ci_depth: int,
        pressure_hidden_dim: int = 192,
        pressure_depth: int = 4,
        activation: str = "tanh",
        ymax: float = 75.0,
        envelope_eps: float = 1.0,
        mapping_scale: float = 3.0,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)
        self.mode_representation = "pressure_first"
        self.ymax = float(ymax)
        self.envelope_eps = float(envelope_eps)
        self.activation = str(activation)
        self.pressure_hidden_dim = int(pressure_hidden_dim)
        self.pressure_depth = int(pressure_depth)
        self.ci_hidden_dim = int(ci_hidden_dim)
        self.ci_depth = int(ci_depth)

        self.ci_net = build_mlp(
            2,
            1,
            hidden_dim=int(ci_hidden_dim),
            depth=int(ci_depth),
            activation=str(activation),
        )
        self.raw_ci_bias = nn.Parameter(torch.zeros(1))

        initial_raw_L = torch.log(torch.expm1(torch.tensor(float(mapping_scale))))
        self.register_buffer("raw_L", initial_raw_L.view(1))

        self.pressure_net = build_mlp(
            10,
            2,
            hidden_dim=int(pressure_hidden_dim),
            depth=int(pressure_depth),
            activation=str(activation),
        )

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1.0e-8)
        return 2.0 * (alpha - self.alpha_min) / span - 1.0

    def normalize_mach(self, mach: torch.Tensor) -> torch.Tensor:
        span = max(self.mach_max - self.mach_min, 1.0e-8)
        return 2.0 * (mach - self.mach_min) / span - 1.0

    def get_ci(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([self.normalize_alpha(alpha), self.normalize_mach(mach)], dim=-1)
        raw_ci = self.ci_net(inputs) + self.raw_ci_bias
        return F.softplus(raw_ci) + 1.0e-6

    def get_mapping_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_L) + 1.0e-6

    def export_model_config(self) -> dict[str, object]:
        return {
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "mach_min": float(self.mach_min),
            "mach_max": float(self.mach_max),
            "ci_hidden_dim": int(self.ci_hidden_dim),
            "ci_depth": int(self.ci_depth),
            "pressure_hidden_dim": int(self.pressure_hidden_dim),
            "pressure_depth": int(self.pressure_depth),
            "activation": str(self.activation),
            "ymax": float(self.ymax),
            "envelope_eps": float(self.envelope_eps),
            "mapping_scale": float(self.get_mapping_scale().detach().cpu().item()),
        }

    def envelope_terms(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
        ci: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = max(float(self.envelope_eps), 1.0e-6)
        gamma_minus, gamma_plus = asymptotic_pressure_gammas(alpha, mach, ci)
        mu_minus = gamma_minus.real.clamp_min(1.0e-6)
        mu_plus = (-gamma_plus.real).clamp_min(1.0e-6)

        left_dist = eps * F.softplus(-y / eps)
        right_dist = eps * F.softplus(y / eps)
        center_dist = eps * F.softplus(torch.zeros(1, device=y.device, dtype=y.dtype))
        e_raw = torch.exp(-mu_minus * left_dist - mu_plus * right_dist)
        e0 = torch.exp(-mu_minus * center_dist - mu_plus * center_dist)
        envelope = e_raw / (e0 + float(PRESSURE_FIRST_EPS))

        _ensure_finite("pressure envelope", envelope)
        return {
            "gamma_minus": gamma_minus,
            "gamma_plus": gamma_plus,
            "mu_minus": mu_minus,
            "mu_plus": mu_plus,
            "left_dist": left_dist,
            "right_dist": right_dist,
            "envelope": envelope,
        }

    def encode_pressure_inputs(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
        ci: torch.Tensor,
        envelope_terms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ymax = max(float(self.ymax), 1.0e-6)
        features = [
            y / ymax,
            torch.tanh(y),
            alpha,
            mach,
            ci,
            alpha * y / 10.0,
            envelope_terms["mu_minus"] * envelope_terms["left_dist"] / 10.0,
            envelope_terms["mu_plus"] * envelope_terms["right_dist"] / 10.0,
            self.normalize_alpha(alpha),
            self.normalize_mach(mach),
        ]
        encoded = torch.cat(features, dim=-1)
        _ensure_finite("pressure-first inputs", encoded)
        return encoded

    def pressure_components(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
        *,
        ci_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if y.shape != alpha.shape or y.shape != mach.shape:
            raise ValueError(f"Expected matching shapes for y/alpha/mach, got {y.shape}, {alpha.shape}, {mach.shape}.")
        ci = self.get_ci(alpha, mach) if ci_override is None else ci_override
        ci = torch.clamp(ci, min=1.0e-6)
        env = self.envelope_terms(y, alpha, mach, ci)
        features = self.encode_pressure_inputs(y, alpha, mach, ci, env)
        raw_h = self.pressure_net(features)
        h = torch.complex(raw_h[:, 0:1], raw_h[:, 1:2])
        p = torch.complex(env["envelope"], torch.zeros_like(env["envelope"])) * h
        _ensure_finite("pressure correction h", h)
        _ensure_finite("pressure field p", p)
        return {
            **env,
            "ci": ci,
            "features": features,
            "h": h,
            "p": p,
        }

    def forward(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        mach: torch.Tensor,
        *,
        ci_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self.pressure_components(y, alpha, mach, ci_override=ci_override)["p"]
        return torch.cat([p.real, p.imag], dim=-1)


def initialize_pressure_first_from_stage0(
    model: KHSubsonicPressureFirst2D,
    stage0_model: nn.Module,
) -> None:
    if getattr(stage0_model, "ci_net", None) is None:
        raise RuntimeError("Stage 0 ci_net is missing.")
    model.ci_net.load_state_dict(copy.deepcopy(stage0_model.ci_net.state_dict()))
    with torch.no_grad():
        model.raw_ci_bias.copy_(stage0_model.raw_ci_bias.detach())
        model.raw_L.copy_(stage0_model.raw_L.detach())


def build_pressure_first_model_from_stage0(
    stage0_model: nn.Module,
    *,
    pressure_hidden_dim: int = 192,
    pressure_depth: int = 4,
    activation: str = "tanh",
    ymax: float = 75.0,
    envelope_eps: float = 1.0,
) -> KHSubsonicPressureFirst2D:
    if getattr(stage0_model, "ci_net", None) is None:
        raise RuntimeError("Stage 0 ci_net is missing.")
    ci_hidden_dim, ci_depth = _infer_ci_architecture(stage0_model.ci_net)
    model = KHSubsonicPressureFirst2D(
        alpha_min=float(stage0_model.alpha_min),
        alpha_max=float(stage0_model.alpha_max),
        mach_min=float(stage0_model.mach_min),
        mach_max=float(stage0_model.mach_max),
        ci_hidden_dim=int(ci_hidden_dim),
        ci_depth=int(ci_depth),
        pressure_hidden_dim=int(pressure_hidden_dim),
        pressure_depth=int(pressure_depth),
        activation=str(activation),
        ymax=float(ymax),
        envelope_eps=float(envelope_eps),
        mapping_scale=float(stage0_model.get_mapping_scale().detach().cpu().item()),
    )
    initialize_pressure_first_from_stage0(model, stage0_model)
    return model


def build_pressure_first_model_from_config(config: Mapping[str, object]) -> KHSubsonicPressureFirst2D:
    return KHSubsonicPressureFirst2D(
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        mach_min=float(config["mach_min"]),
        mach_max=float(config["mach_max"]),
        ci_hidden_dim=int(config["ci_hidden_dim"]),
        ci_depth=int(config["ci_depth"]),
        pressure_hidden_dim=int(config["pressure_hidden_dim"]),
        pressure_depth=int(config["pressure_depth"]),
        activation=str(config.get("activation", "tanh")),
        ymax=float(config.get("ymax", 75.0)),
        envelope_eps=float(config.get("envelope_eps", 1.0)),
        mapping_scale=float(config.get("mapping_scale", 3.0)),
    )


def pressure_first_value_and_derivatives(
    model: KHSubsonicPressureFirst2D,
    y: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not y.requires_grad:
        y.requires_grad_(True)
    comp = model.pressure_components(y, alpha, mach, ci_override=ci_override)
    p = comp["p"]
    p_r = p.real
    p_i = p.imag
    p_r_y = _differentiate(p_r, y)
    p_i_y = _differentiate(p_i, y)
    p_r_yy = _differentiate(p_r_y, y)
    p_i_yy = _differentiate(p_i_y, y)
    p_y = torch.complex(p_r_y, p_i_y)
    p_yy = torch.complex(p_r_yy, p_i_yy)
    _ensure_finite("pressure p_y", p_y)
    _ensure_finite("pressure p_yy", p_yy)
    return {
        **comp,
        "p_y": p_y,
        "p_yy": p_yy,
    }


def pressure_first_ode_residual(
    model: KHSubsonicPressureFirst2D,
    y: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    fields = pressure_first_value_and_derivatives(model, y, alpha, mach, ci_override=ci_override)
    p = fields["p"]
    p_y = fields["p_y"]
    p_yy = fields["p_yy"]
    ci = fields["ci"]

    zero = torch.zeros_like(ci)
    c = torch.complex(zero, ci)
    u = torch.complex(base_velocity(y), torch.zeros_like(y))
    uy = torch.complex(base_velocity_derivative(y), torch.zeros_like(y))
    alpha_c = torch.complex(alpha, torch.zeros_like(alpha))
    mach_c = torch.complex(mach, torch.zeros_like(mach))

    u_diff = u - c
    a_coeff = 2.0 * uy / (u_diff + float(PRESSURE_FIRST_EPS))
    b_coeff = alpha_c.pow(2) * (1.0 - mach_c.pow(2) * (u_diff**2))
    residual = p_yy - a_coeff * p_y - b_coeff * p

    scale = 1.0 + torch.abs(p_yy).pow(2) + torch.abs(a_coeff * p_y).pow(2) + torch.abs(b_coeff * p).pow(2)
    residual = residual / torch.sqrt(scale.detach().clamp_min(1.0))
    _ensure_finite("pressure residual", residual)
    return residual.real, residual.imag, {
        **fields,
        "a_coeff": a_coeff,
        "b_coeff": b_coeff,
        "residual": residual,
    }


def pressure_first_robin_boundary_loss(
    model: KHSubsonicPressureFirst2D,
    y_left: torch.Tensor,
    y_right: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> torch.Tensor:
    left_fields = pressure_first_value_and_derivatives(model, y_left, alpha, mach, ci_override=ci_override)
    right_fields = pressure_first_value_and_derivatives(model, y_right, alpha, mach, ci_override=ci_override)

    res_left = left_fields["p_y"] - left_fields["gamma_minus"] * left_fields["p"]
    res_right = right_fields["p_y"] - right_fields["gamma_plus"] * right_fields["p"]
    loss = torch.mean(res_left.real.pow(2) + res_left.imag.pow(2)) + torch.mean(
        res_right.real.pow(2) + res_right.imag.pow(2)
    )
    _ensure_finite("pressure Robin loss", loss)
    return loss


def pressure_first_gauge_loss(
    model: KHSubsonicPressureFirst2D,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> torch.Tensor:
    y0 = torch.zeros_like(alpha)
    fields = model.pressure_components(y0, alpha, mach, ci_override=ci_override)
    p0 = fields["p"]
    loss = torch.mean((p0.real - 1.0).pow(2) + p0.imag.pow(2))
    _ensure_finite("pressure gauge loss", loss)
    return loss



def pressure_first_symmetry_loss(
    model: KHSubsonicPressureFirst2D,
    y_abs_or_y: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Parity regularization for the centered pressure mode.

    For the centered subsonic KH setup with c = i ci and the gauge
    Re p(0)=1, Im p(0)=0, the pressure is expected to satisfy:
    Re p(y) even, Im p(y) odd.
    """
    y_abs = torch.abs(y_abs_or_y)
    y_plus = y_abs
    y_minus = -y_abs

    p_plus = model.pressure_components(y_plus, alpha, mach, ci_override=ci_override)["p"]
    p_minus = model.pressure_components(y_minus, alpha, mach, ci_override=ci_override)["p"]

    loss = (p_plus.real - p_minus.real).pow(2) + (p_plus.imag + p_minus.imag).pow(2)
    return torch.mean(loss)


def pressure_first_center_derivative_loss(
    model: KHSubsonicPressureFirst2D,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    *,
    ci_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center derivative parity loss.

    Re p is even, so Re p_y(0)=0. Do not penalize Im p_y(0), because
    Im p is odd and its derivative at zero is generally nonzero.
    """
    y0 = torch.zeros_like(alpha)
    y0.requires_grad_(True)
    fields = pressure_first_value_and_derivatives(model, y0, alpha, mach, ci_override=ci_override)
    return torch.mean(fields["p_y"].real.pow(2))

__all__ = [
    "KHSubsonicPressureFirst2D",
    "PRESSURE_FIRST_EPS",
    "asymptotic_pressure_gammas",
    "build_pressure_first_model_from_config",
    "build_pressure_first_model_from_stage0",
    "initialize_pressure_first_from_stage0",
    "pressure_first_gauge_loss",
    "pressure_first_ode_residual",
    "pressure_first_robin_boundary_loss",
    "pressure_first_symmetry_loss",
    "pressure_first_center_derivative_loss",
    "pressure_first_value_and_derivatives",
]
