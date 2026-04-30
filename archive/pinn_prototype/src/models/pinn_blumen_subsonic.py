from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    """
    Encodage Fourier optionnel pour aider le reseau a representer des structures
    oscillantes ou fortement localisees.
    """

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


class BlumenSubsonicPINN(nn.Module):
    """
    PINN minimal pour le probleme de Blumen subsonique a (alpha, M) fixes.

    Entree:
    - xi : coordonnee bornee dans (-1, 1)

    Sortie:
    - p_r(xi), p_i(xi) : parties reelle et imaginaire de la perturbation de pression

    Parametres physiques appris:
    - c_i > 0 : partie imaginaire de la vitesse de phase, parametree via softplus
    - L > 0   : parametre de mapping xi -> y, optionnellement appris
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        fourier_features: int = 0,
        fourier_scale: float = 2.0,
        initial_ci: float = 0.3,
        initial_L: float = 10.0,
        trainable_L: bool = True,
    ):
        super().__init__()

        self.fourier = FourierEncoding(1, fourier_features, fourier_scale) if fourier_features > 0 else None
        input_dim = 2 * fourier_features if fourier_features > 0 else 1

        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        # c_i = softplus(raw_ci) assure c_i > 0 pendant tout l'entrainement.
        initial_raw_ci = torch.log(torch.expm1(torch.tensor(float(initial_ci))))
        self.raw_ci = nn.Parameter(initial_raw_ci.view(1))

        initial_raw_L = torch.log(torch.expm1(torch.tensor(float(initial_L))))
        if trainable_L:
            self.raw_L = nn.Parameter(initial_raw_L.view(1))
        else:
            self.register_buffer("raw_L", initial_raw_L.view(1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, xi: torch.Tensor) -> torch.Tensor:
        if self.fourier is None:
            return xi
        return self.fourier(xi)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        features = self.encode(xi)
        return self.net(features)

    def get_ci(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_ci) + 1e-6

    def get_L(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_L) + 1e-6
