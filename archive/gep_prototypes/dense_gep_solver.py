from __future__ import annotations

"""
Solveur GEP dense minimal pour la couche de melange compressible.

Objectif :
- construire un premier solveur single-case robuste avec le mapping du PINN ;
- diagonaliser le probleme dense A X = c B X avec scipy.linalg.eig ;
- extraire un mode instable dominant pour comparaison avec les solveurs de tir.

Cette version reste volontairement simple :
- grille uniforme en xi sur un sous-intervalle [-xi_max, xi_max] de (-1, 1) ;
- mapping du PINN : y = L * xi / (1 - xi^2) ;
- systeme lineaireise [u, v, p] identique au prototype classique archive ;
- conditions aux bords homogenes imposees en supprimant les inconnues de bord.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig


def first_derivative_matrix_uniform(x: np.ndarray) -> np.ndarray:
    """
    Matrice de derivee premiere sur une grille uniforme 1D.
    Les lignes de bord utilisent des differences decalees du second ordre.
    """
    n = len(x)
    h = float(x[1] - x[0])
    d = np.zeros((n, n), dtype=float)

    for i in range(1, n - 1):
        d[i, i - 1] = -0.5 / h
        d[i, i + 1] = 0.5 / h

    d[0, 0] = -1.5 / h
    d[0, 1] = 2.0 / h
    d[0, 2] = -0.5 / h

    d[-1, -3] = 0.5 / h
    d[-1, -2] = -2.0 / h
    d[-1, -1] = 1.5 / h
    return d


@dataclass
class DenseGEPResult:
    alpha: float
    Mach: float
    c: complex
    cr: float
    ci: float
    omega_i: float
    n_modes_positive: int
    selection_source: str
    success: bool


class DenseGEPCompressibleSolver:
    def __init__(
        self,
        *,
        alpha: float,
        Mach: float,
        n_points: int = 181,
        xi_max: float = 0.98,
        mapping_scale: float = 5.0,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.n_points = int(n_points)
        self.xi_max = float(xi_max)
        self.mapping_scale = float(mapping_scale)

        self.xi = np.linspace(-self.xi_max, self.xi_max, self.n_points)
        self.y = self.xi_to_y(self.xi)
        self.jac = self.dy_dxi(self.xi)
        self.u = np.tanh(self.y)
        self.uy = 1.0 / np.cosh(self.y) ** 2

        self.dxi = first_derivative_matrix_uniform(self.xi)
        self.dy = np.diag(1.0 / self.jac) @ self.dxi

    def xi_to_y(self, xi: np.ndarray) -> np.ndarray:
        return self.mapping_scale * xi / (1.0 - xi**2)

    def dy_dxi(self, xi: np.ndarray) -> np.ndarray:
        return self.mapping_scale * (1.0 + xi**2) / (1.0 - xi**2) ** 2

    def assemble_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        n_int = self.n_points - 2
        alpha = self.alpha
        mach = self.Mach

        dy_int = self.dy[1:-1, 1:-1]
        u_int = self.u[1:-1]
        uy_int = self.uy[1:-1]

        eye = np.eye(n_int, dtype=complex)
        zero = np.zeros((n_int, n_int), dtype=complex)
        u_mat = np.diag(u_int.astype(complex))
        uy_mat = np.diag(uy_int.astype(complex))

        a11 = alpha * eye
        a12 = -1j * dy_int.astype(complex)
        a13 = alpha * (mach**2) * u_mat

        a21 = alpha * u_mat
        a22 = -1j * uy_mat
        a23 = alpha * eye

        a31 = zero
        a32 = alpha * u_mat
        a33 = -1j * dy_int.astype(complex)

        a = np.block(
            [
                [a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33],
            ]
        )

        b13 = alpha * (mach**2) * eye
        b21 = alpha * eye
        b32 = alpha * eye
        b = np.block(
            [
                [zero, zero, b13],
                [b21, zero, zero],
                [zero, b32, zero],
            ]
        )
        return a, b

    def solve_all(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.assemble_matrices()
        vals, vecs = eig(a, b)
        return vals, vecs

    def candidate_modes(self) -> list[dict]:
        vals, vecs = self.solve_all()
        modes = []
        for idx, val in enumerate(vals):
            if not np.isfinite(val):
                continue
            if abs(val) > 5.0:
                continue
            ci = float(np.imag(val))
            if ci <= 0.0:
                continue
            cr = float(np.real(val))
            omega_i = float(self.alpha * ci)
            if omega_i > 1.0:
                continue
            modes.append(
                {
                    "c": complex(val),
                    "cr": cr,
                    "ci": ci,
                    "omega_i": omega_i,
                    "abs_cr": abs(cr),
                    "vector": vecs[:, idx],
                }
            )

        unique_modes = []
        for mode in sorted(modes, key=lambda item: item["omega_i"], reverse=True):
            if any(abs(mode["c"] - other["c"]) < 1e-8 for other in unique_modes):
                continue
            unique_modes.append(mode)
        return unique_modes

    @staticmethod
    def _distance_to_target(mode: dict, target_guess: tuple[float, float], cr_scale: float, ci_scale: float) -> float:
        return float(
            np.hypot(
                (mode["cr"] - target_guess[0]) / max(cr_scale, 1e-8),
                (mode["ci"] - target_guess[1]) / max(ci_scale, 1e-8),
            )
        )

    def select_mode(
        self,
        modes: list[dict],
        *,
        target_guess: tuple[float, float] | None = None,
        prefer_positive_cr: bool = True,
    ) -> tuple[dict, str]:
        if prefer_positive_cr:
            positive = [mode for mode in modes if mode["cr"] >= -1e-10]
            if positive:
                modes = positive

        if target_guess is not None:
            cr_scale = max(0.25, max(abs(mode["cr"]) for mode in modes))
            ci_scale = max(0.10, max(mode["ci"] for mode in modes))
            chosen = min(
                modes,
                key=lambda mode: self._distance_to_target(mode, target_guess, cr_scale, ci_scale),
            )
            return chosen, "target_guided"

        if self.Mach < 1.0:
            chosen = max(modes, key=lambda item: item["omega_i"])
            return chosen, "max_growth"

        moderate = [mode for mode in modes if mode["abs_cr"] <= 0.35]
        if moderate:
            chosen = max(moderate, key=lambda item: item["omega_i"])
            return chosen, "moderate_cr"

        chosen = min(modes, key=lambda item: item["abs_cr"])
        return chosen, "min_abs_cr"

    def solve_dominant_mode(
        self,
        *,
        target_guess: tuple[float, float] | None = None,
        prefer_positive_cr: bool = True,
    ) -> DenseGEPResult:
        if self.Mach < 1.0 and self.alpha**2 + self.Mach**2 >= 1.0:
            return DenseGEPResult(
                alpha=self.alpha,
                Mach=self.Mach,
                c=0.0j,
                cr=0.0,
                ci=0.0,
                omega_i=0.0,
                n_modes_positive=0,
                selection_source="neutral_boundary",
                success=True,
            )

        modes = self.candidate_modes()
        if not modes:
            return DenseGEPResult(
                alpha=self.alpha,
                Mach=self.Mach,
                c=0.0j,
                cr=0.0,
                ci=0.0,
                omega_i=0.0,
                n_modes_positive=0,
                selection_source="no_mode",
                success=False,
            )

        chosen, selection_source = self.select_mode(
            modes,
            target_guess=target_guess,
            prefer_positive_cr=prefer_positive_cr,
        )
        c = chosen["c"]
        return DenseGEPResult(
            alpha=self.alpha,
            Mach=self.Mach,
            c=c,
            cr=float(np.real(c)),
            ci=float(np.imag(c)),
            omega_i=float(self.alpha * np.imag(c)),
            n_modes_positive=len(modes),
            selection_source=selection_source,
            success=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solveur GEP dense single-case avec mapping du PINN.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--n-points", type=int, default=181)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--target-cr", type=float, default=None)
    parser.add_argument("--target-ci", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    solver = DenseGEPCompressibleSolver(
        alpha=args.alpha,
        Mach=args.mach,
        n_points=args.n_points,
        xi_max=args.xi_max,
        mapping_scale=args.mapping_scale,
    )
    target_guess = None
    if args.target_cr is not None and args.target_ci is not None:
        target_guess = (float(args.target_cr), float(args.target_ci))
    result = solver.solve_dominant_mode(target_guess=target_guess)
    print(result)


if __name__ == "__main__":
    main()
