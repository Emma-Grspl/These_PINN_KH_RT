from __future__ import annotations

from collections.abc import Mapping
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
        mode_hidden_dim: int | None = None,
        ci_hidden_dim: int | None = None,
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
        mode_experts: int = 1,
        alpha_split_threshold: float | None = None,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.enforce_mode_symmetry = bool(enforce_mode_symmetry)
        if mode_representation not in {"cartesian", "amplitude_phase", "log_amplitude_phase", "riccati"}:
            raise ValueError(f"Unsupported mode_representation={mode_representation!r}.")
        self.mode_representation = str(mode_representation)
        self.mode_experts = int(mode_experts)
        if self.mode_experts not in {1, 2}:
            raise ValueError(f"Unsupported mode_experts={mode_experts!r}.")
        if self.mode_experts == 2 and alpha_split_threshold is None:
            alpha_split_threshold = 0.5 * (self.alpha_min + self.alpha_max)
        self.alpha_split_threshold = None if alpha_split_threshold is None else float(alpha_split_threshold)
        mode_hidden_dim = int(mode_hidden_dim if mode_hidden_dim is not None else hidden_dim)
        ci_hidden_dim = int(ci_hidden_dim if ci_hidden_dim is not None else max(hidden_dim // 2, 1))

        self.mode_fourier = FourierEncoding(2, fourier_features, fourier_scale) if fourier_features > 0 else None
        mode_input_dim = 4 * fourier_features if fourier_features > 0 else 2
        self.mode_nets = nn.ModuleList(
            [
                build_mlp(
                    mode_input_dim,
                    2,
                    hidden_dim=mode_hidden_dim,
                    depth=mode_depth,
                    activation=activation,
                )
                for _ in range(self.mode_experts)
            ]
        )
        self.mode_net = self.mode_nets[0]
        self.ci_net = build_mlp(
            1,
            1,
            hidden_dim=ci_hidden_dim,
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

    def _forward_mode_raw(self, features: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.mode_experts == 1:
            return self.mode_net(features)

        raw_outputs = torch.empty(features.shape[0], 2, dtype=features.dtype, device=features.device)
        alpha_values = alpha[:, 0]
        left_mask = alpha_values <= float(self.alpha_split_threshold)
        right_mask = ~left_mask
        if torch.any(left_mask):
            raw_outputs[left_mask] = self.mode_nets[0](features[left_mask])
        if torch.any(right_mask):
            raw_outputs[right_mask] = self.mode_nets[1](features[right_mask])
        return raw_outputs

    def decode_mode_outputs(self, xi: torch.Tensor, raw_outputs: torch.Tensor) -> torch.Tensor:
        if self.mode_representation == "cartesian":
            if not self.enforce_mode_symmetry:
                return raw_outputs
            pr = raw_outputs[:, 0:1]
            pi = raw_outputs[:, 1:2] * torch.sign(xi)
            return torch.cat([pr, pi], dim=-1)

        if self.mode_representation == "riccati":
            if not self.enforce_mode_symmetry:
                return raw_outputs
            kappa = raw_outputs[:, 0:1]
            q = raw_outputs[:, 1:2] * torch.sign(xi)
            return torch.cat([kappa, q], dim=-1)

        if self.mode_representation == "amplitude_phase":
            amp = F.softplus(raw_outputs[:, 0:1]) + 1e-6
        else:
            amp = torch.exp(torch.clamp(raw_outputs[:, 0:1], min=-20.0, max=20.0))
        phase = raw_outputs[:, 1:2]
        if self.enforce_mode_symmetry:
            phase = phase * torch.sign(xi)
        pr = amp * torch.cos(phase)
        pi = amp * torch.sin(phase)
        return torch.cat([pr, pi], dim=-1)

    def forward(self, xi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        xi_in = torch.abs(xi) if self.enforce_mode_symmetry else xi
        features = self.encode_mode_inputs(xi_in, alpha)
        raw_outputs = self._forward_mode_raw(features, alpha)
        return self.decode_mode_outputs(xi, raw_outputs)

    def get_ci(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_n = self.normalize_alpha(alpha)
        raw_ci = self.ci_net(alpha_n) + self.raw_ci_bias
        return F.softplus(raw_ci) + 1e-6

    def get_mapping_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_L) + 1e-6


def _config_get(config: Mapping[str, object] | object, key: str, default: object = None) -> object:
    if isinstance(config, Mapping):
        value = config.get(key, default)
    elif hasattr(config, "index"):
        value = config[key] if key in config.index else default
    else:
        value = getattr(config, key, default)
    if value is None:
        return default
    try:
        if bool(torch.isnan(torch.tensor(value, dtype=torch.float32)).item()):
            return default
    except Exception:
        pass
    return value


def build_fixed_mach_model_from_config(config: Mapping[str, object] | object) -> KHSubsonicFixedMachPINN:
    mode_hidden_dim = _config_get(config, "mode_hidden_dim")
    ci_hidden_dim = _config_get(config, "ci_hidden_dim")
    return KHSubsonicFixedMachPINN(
        alpha_min=float(_config_get(config, "alpha_min")),
        alpha_max=float(_config_get(config, "alpha_max")),
        hidden_dim=int(_config_get(config, "hidden_dim")),
        mode_hidden_dim=None if mode_hidden_dim is None else int(mode_hidden_dim),
        ci_hidden_dim=None if ci_hidden_dim is None else int(ci_hidden_dim),
        mode_depth=int(_config_get(config, "mode_depth")),
        ci_depth=int(_config_get(config, "ci_depth")),
        activation=str(_config_get(config, "activation")),
        fourier_features=int(_config_get(config, "fourier_features", 0)),
        fourier_scale=float(_config_get(config, "fourier_scale", 2.0)),
        initial_ci=float(_config_get(config, "initial_ci", 0.2)),
        mapping_scale=float(_config_get(config, "mapping_scale", 3.0)),
        trainable_mapping_scale=bool(_config_get(config, "trainable_mapping_scale", False)),
        enforce_mode_symmetry=bool(_config_get(config, "enforce_mode_symmetry", False)),
        mode_representation=str(_config_get(config, "mode_representation", "cartesian")),
        mode_experts=int(_config_get(config, "mode_experts", 1)),
        alpha_split_threshold=_config_get(config, "alpha_split_threshold"),
    )


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
        mode_hidden_dim: int | None = None,
        ci_hidden_dim: int | None = None,
        mode_depth: int = 4,
        ci_depth: int = 2,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        initial_ci: float = 0.3,
        mapping_scale: float = 3.0,
        trainable_mapping_scale: bool = False,
        mode_representation: str = "cartesian",
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)
        if mode_representation not in {"cartesian", "riccati"}:
            raise ValueError(f"Unsupported mode_representation={mode_representation!r}.")
        self.mode_representation = str(mode_representation)
        mode_hidden_dim = int(mode_hidden_dim if mode_hidden_dim is not None else hidden_dim)
        ci_hidden_dim = int(ci_hidden_dim if ci_hidden_dim is not None else max(hidden_dim // 2, 1))

        self.mode_fourier = FourierEncoding(3, fourier_features, fourier_scale) if fourier_features > 0 else None
        mode_input_dim = 6 * fourier_features if fourier_features > 0 else 3
        self.mode_net = build_mlp(
            mode_input_dim,
            2,
            hidden_dim=mode_hidden_dim,
            depth=mode_depth,
            activation=activation,
        )
        self.ci_net = build_mlp(
            2,
            1,
            hidden_dim=ci_hidden_dim,
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
        raw_outputs = self.mode_net(self.encode_mode_inputs(xi, alpha, mach))
        if self.mode_representation == "riccati":
            kappa = raw_outputs[:, 0:1]
            q = raw_outputs[:, 1:2]
            return torch.cat([kappa, q], dim=-1)
        return raw_outputs

    def get_ci(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([self.normalize_alpha(alpha), self.normalize_mach(mach)], dim=-1)
        raw_ci = self.ci_net(inputs) + self.raw_ci_bias
        return F.softplus(raw_ci) + 1e-6

    def get_mapping_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_L) + 1e-6
