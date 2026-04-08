from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoding(nn.Module):
    def __init__(self, in_features: int, num_frequencies: int, scale: float = 1.0):
        super().__init__()
        if num_frequencies <= 0:
            self.register_buffer("B", torch.empty(in_features, 0))
        else:
            self.register_buffer("B", torch.randn(in_features, num_frequencies) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.B.numel() == 0:
            return x
        projection = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


def build_mlp(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    depth: int,
    activation: str,
) -> nn.Sequential:
    if activation == "tanh":
        act: nn.Module = nn.Tanh()
    elif activation == "silu":
        act = nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation={activation!r}.")

    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(act)
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    net = nn.Sequential(*layers)

    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return net


class KHSubsonicFixedMachPINN(nn.Module):
    """
    Prototype PINN subsonique a Mach fixe.

    - entrees du reseau de mode : (xi, alpha_normalise)
    - sortie : (p_r, p_i)
    - tete spectrale separee : c_i(alpha)
    - mapping du projet : y = L * xi / (1 - xi^2)
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        hidden_dim: int = 128,
        mode_depth: int = 4,
        ci_depth: int = 2,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        initial_ci: float = 0.3,
        mapping_scale: float = 3.0,
        trainable_mapping_scale: bool = False,
        enforce_mode_symmetry: bool = False,
        mode_representation: str = "cartesian",
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.enforce_mode_symmetry = bool(enforce_mode_symmetry)
        if mode_representation not in {"cartesian", "amplitude_phase"}:
            raise ValueError(f"Unsupported mode_representation={mode_representation!r}.")
        self.mode_representation = str(mode_representation)

        self.mode_fourier = FourierEncoding(2, fourier_features, fourier_scale) if fourier_features > 0 else None
        mode_input_dim = 4 * fourier_features if fourier_features > 0 else 2
        self.mode_net = build_mlp(
            mode_input_dim,
            2,
            hidden_dim=hidden_dim,
            depth=mode_depth,
            activation=activation,
        )
        self.ci_net = build_mlp(
            1,
            1,
            hidden_dim=hidden_dim // 2,
            depth=ci_depth,
            activation=activation,
        )

        initial_raw_ci_bias = torch.log(torch.expm1(torch.tensor(float(initial_ci))))
        self.raw_ci_bias = nn.Parameter(initial_raw_ci_bias.view(1))

        initial_raw_L = torch.log(torch.expm1(torch.tensor(float(mapping_scale))))
        if trainable_mapping_scale:
            self.raw_L = nn.Parameter(initial_raw_L.view(1))
        else:
            self.register_buffer("raw_L", initial_raw_L.view(1))

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1e-8)
        alpha_scaled = 2.0 * (alpha - self.alpha_min) / span - 1.0
        return alpha_scaled

    def encode_mode_inputs(self, xi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha_n = self.normalize_alpha(alpha)
        inputs = torch.cat([xi, alpha_n], dim=-1)
        if self.mode_fourier is None:
            return inputs
        return self.mode_fourier(inputs)

    def decode_mode_outputs(self, xi: torch.Tensor, raw_outputs: torch.Tensor) -> torch.Tensor:
        if self.mode_representation == "cartesian":
            if not self.enforce_mode_symmetry:
                return raw_outputs
            pr = raw_outputs[:, 0:1]
            pi = raw_outputs[:, 1:2] * torch.sign(xi)
            return torch.cat([pr, pi], dim=-1)

        amp = F.softplus(raw_outputs[:, 0:1]) + 1e-6
        phase = raw_outputs[:, 1:2]
        if self.enforce_mode_symmetry:
            phase = phase * torch.sign(xi)
        pr = amp * torch.cos(phase)
        pi = amp * torch.sin(phase)
        return torch.cat([pr, pi], dim=-1)

    def forward(self, xi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        xi_in = torch.abs(xi) if self.enforce_mode_symmetry else xi
        features = self.encode_mode_inputs(xi_in, alpha)
        raw_outputs = self.mode_net(features)
        return self.decode_mode_outputs(xi, raw_outputs)

    def get_ci(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_n = self.normalize_alpha(alpha)
        raw_ci = self.ci_net(alpha_n) + self.raw_ci_bias
        return F.softplus(raw_ci) + 1e-6

    def get_mapping_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_L) + 1e-6


class KHSubsonicMultiMachPINN(nn.Module):
    """
    Prototype PINN subsonique 2D sur (alpha, Mach).

    - reseau de mode : (xi, alpha_normalise, mach_normalise) -> (p_r, p_i)
    - tete spectrale : (alpha_normalise, mach_normalise) -> c_i(alpha, Mach)
    """

    def __init__(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        hidden_dim: int = 128,
        mode_depth: int = 4,
        ci_depth: int = 2,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        initial_ci: float = 0.3,
        mapping_scale: float = 3.0,
        trainable_mapping_scale: bool = False,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)

        self.mode_fourier = FourierEncoding(3, fourier_features, fourier_scale) if fourier_features > 0 else None
        mode_input_dim = 6 * fourier_features if fourier_features > 0 else 3
        self.mode_net = build_mlp(
            mode_input_dim,
            2,
            hidden_dim=hidden_dim,
            depth=mode_depth,
            activation=activation,
        )
        self.ci_net = build_mlp(
            2,
            1,
            hidden_dim=hidden_dim // 2,
            depth=ci_depth,
            activation=activation,
        )

        initial_raw_ci_bias = torch.log(torch.expm1(torch.tensor(float(initial_ci))))
        self.raw_ci_bias = nn.Parameter(initial_raw_ci_bias.view(1))

        initial_raw_L = torch.log(torch.expm1(torch.tensor(float(mapping_scale))))
        if trainable_mapping_scale:
            self.raw_L = nn.Parameter(initial_raw_L.view(1))
        else:
            self.register_buffer("raw_L", initial_raw_L.view(1))

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        span = max(self.alpha_max - self.alpha_min, 1e-8)
        return 2.0 * (alpha - self.alpha_min) / span - 1.0

    def normalize_mach(self, mach: torch.Tensor) -> torch.Tensor:
        span = max(self.mach_max - self.mach_min, 1e-8)
        return 2.0 * (mach - self.mach_min) / span - 1.0

    def encode_mode_inputs(self, xi: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([xi, self.normalize_alpha(alpha), self.normalize_mach(mach)], dim=-1)
        if self.mode_fourier is None:
            return inputs
        return self.mode_fourier(inputs)

    def forward(self, xi: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        return self.mode_net(self.encode_mode_inputs(xi, alpha, mach))

    def get_ci(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([self.normalize_alpha(alpha), self.normalize_mach(mach)], dim=-1)
        raw_ci = self.ci_net(inputs) + self.raw_ci_bias
        return F.softplus(raw_ci) + 1e-6

    def get_mapping_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_L) + 1e-6
