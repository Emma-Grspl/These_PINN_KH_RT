from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def _principal_sqrt(z: complex) -> complex:
    root = np.sqrt(z)
    if root.real < 0 or (np.isclose(root.real, 0.0, atol=1e-12) and root.imag < 0):
        root = -root
    return root


@dataclass
class ShootingResult:
    alpha: float
    Mach: float
    ci: float
    omega_i: float
    mismatch: float
    domain_size: float
    success: bool


class SubsonicShootingSolver:
    """
    Solveur de tir subsonique pour l'équation de pression de Blumen.

    Hypothèses:
    - régime subsonique M < 1
    - branche principale c_r = 0
    - recherche de c = i c_i avec c_i >= 0
    """

    def __init__(
        self,
        alpha: float,
        Mach: float,
        *,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        min_domain_size: float = 8.0,
        max_domain_size: float = 80.0,
        match_point: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.rtol = rtol
        self.atol = atol
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.match_point = match_point
        self.ci_floor = 1e-5

    @staticmethod
    def base_velocity(y: float | np.ndarray) -> float | np.ndarray:
        return np.tanh(y)

    @staticmethod
    def base_velocity_derivative(y: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / np.cosh(y) ** 2

    def phase_speed(self, ci: float) -> complex:
        return 1j * float(max(ci, self.ci_floor))

    def stability_limit(self) -> float:
        return np.sqrt(max(0.0, 1.0 - self.Mach**2))

    def is_stable_region(self) -> bool:
        return self.alpha >= self.stability_limit() - 1e-12

    def asymptotic_decay_rates(self, ci: float) -> tuple[complex, complex]:
        c = self.phase_speed(ci)
        z_minus = 1.0 - self.Mach**2 * (c + 1.0) ** 2
        z_plus = 1.0 - self.Mach**2 * (c - 1.0) ** 2

        root_minus = _principal_sqrt(z_minus)
        root_plus = _principal_sqrt(z_plus)

        gamma_left = self.alpha * root_minus
        gamma_right = -self.alpha * root_plus
        return gamma_left, gamma_right

    def estimate_domain_size(self, ci: float) -> float:
        gamma_left, gamma_right = self.asymptotic_decay_rates(ci)
        decay_left = max(gamma_left.real, 1e-6)
        decay_right = max(-gamma_right.real, 1e-6)
        estimate = 4.0 * (1.0 / decay_left + 1.0 / decay_right)
        return float(np.clip(estimate, self.min_domain_size, self.max_domain_size))

    def riccati_rhs(self, y: float, gamma: np.ndarray, ci: float) -> np.ndarray:
        c = self.phase_speed(ci)
        u = self.base_velocity(y)
        up = self.base_velocity_derivative(y)
        denominator = u - c
        p_term = -2.0 * up / denominator
        r_term = 1.0 - self.Mach**2 * denominator**2
        return np.array([-gamma[0] ** 2 - p_term * gamma[0] + self.alpha**2 * r_term], dtype=complex)

    def integrate_gamma(self, ci: float, domain_size: float | None = None) -> tuple[complex, complex, float, bool]:
        domain_size = self.estimate_domain_size(ci) if domain_size is None else float(domain_size)
        gamma_left_0, gamma_right_0 = self.asymptotic_decay_rates(ci)

        left_solution = solve_ivp(
            lambda y, state: self.riccati_rhs(y, state, ci),
            (-domain_size, self.match_point),
            y0=np.array([gamma_left_0], dtype=complex),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.array([self.match_point]),
        )
        right_solution = solve_ivp(
            lambda y, state: self.riccati_rhs(y, state, ci),
            (domain_size, self.match_point),
            y0=np.array([gamma_right_0], dtype=complex),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.array([self.match_point]),
        )

        success = bool(left_solution.success and right_solution.success)
        if not success:
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan, domain_size, False

        return left_solution.y[0, -1], right_solution.y[0, -1], domain_size, True

    def mismatch(self, ci: float) -> float:
        if ci < 0.0:
            return 1e6
        gamma_left, gamma_right, _, success = self.integrate_gamma(max(ci, self.ci_floor))
        if not success:
            return 1e6
        return float(abs(gamma_left - gamma_right))

    def solve_ci(
        self,
        *,
        ci_max: float = 1.0,
        n_scan: int = 81,
        previous_ci: float | None = None,
    ) -> ShootingResult:
        if self.is_stable_region():
            return ShootingResult(
                alpha=self.alpha,
                Mach=self.Mach,
                ci=0.0,
                omega_i=0.0,
                mismatch=0.0,
                domain_size=self.estimate_domain_size(1e-6),
                success=True,
            )

        scan_min = self.ci_floor
        scan_max = ci_max
        if previous_ci is not None and previous_ci > 0.0:
            scan_min = max(self.ci_floor, previous_ci - 0.15)
            scan_max = min(ci_max, previous_ci + 0.15)

        ci_values = np.linspace(scan_min, scan_max, n_scan)
        mismatches = np.array([self.mismatch(ci) for ci in ci_values])

        local_minima = []
        for idx in range(1, len(ci_values) - 1):
            if mismatches[idx] <= mismatches[idx - 1] and mismatches[idx] <= mismatches[idx + 1]:
                if ci_values[idx] > max(0.03, 3.0 * self.ci_floor):
                    local_minima.append((float(ci_values[idx]), float(mismatches[idx])))

        if local_minima:
            if previous_ci is not None and previous_ci > 0.0:
                coarse_ci = min(local_minima, key=lambda item: abs(item[0] - previous_ci))[0]
            else:
                coarse_ci = min(local_minima, key=lambda item: item[1])[0]
        else:
            admissible_idx = np.where(ci_values > max(0.03, 3.0 * self.ci_floor))[0]
            if len(admissible_idx) == 0:
                coarse_ci = float(ci_values[int(np.argmin(mismatches))])
            else:
                best_local = admissible_idx[int(np.argmin(mismatches[admissible_idx]))]
                coarse_ci = float(ci_values[best_local])

        delta_ci = ci_values[1] - ci_values[0]
        bracket_left = max(self.ci_floor, coarse_ci - delta_ci)
        bracket_right = min(ci_max, coarse_ci + delta_ci)

        optimum = minimize_scalar(
            self.mismatch,
            bounds=(bracket_left, bracket_right),
            method="bounded",
            options={"xatol": 1e-4},
        )

        ci_star = float(optimum.x)
        mismatch_star = float(optimum.fun)
        domain_size = self.estimate_domain_size(max(ci_star, 1e-6))
        success = bool(optimum.success and mismatch_star < 5e-2)

        if ci_star < 1e-4:
            ci_star = 0.0

        return ShootingResult(
            alpha=self.alpha,
            Mach=self.Mach,
            ci=ci_star,
            omega_i=self.alpha * ci_star,
            mismatch=mismatch_star,
            domain_size=domain_size,
            success=success,
        )


def sample_subsonic_growth_map(
    alphas,
    machs,
    *,
    ci_max: float = 1.0,
    n_scan: int = 81,
) -> pd.DataFrame:
    rows = []
    alphas = np.asarray(alphas, dtype=float)
    machs = np.asarray(machs, dtype=float)

    for mach in machs:
        alpha_cut = np.sqrt(max(0.0, 1.0 - float(mach) ** 2))
        unstable_alphas = np.sort(alphas[alphas < alpha_cut - 1e-12])[::-1]
        stable_alphas = np.sort(alphas[alphas >= alpha_cut - 1e-12])
        previous_ci = None

        for alpha in unstable_alphas:
            solver = SubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            result = solver.solve_ci(ci_max=ci_max, n_scan=n_scan, previous_ci=previous_ci)
            if result.success:
                previous_ci = result.ci
            rows.append(
                {
                    "alpha": result.alpha,
                    "Mach": result.Mach,
                    "ci": result.ci,
                    "omega_i": result.omega_i,
                    "mismatch": result.mismatch,
                    "domain_size": result.domain_size,
                    "success": result.success,
                }
            )

        for alpha in stable_alphas:
            solver = SubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            result = ShootingResult(
                alpha=float(alpha),
                Mach=float(mach),
                ci=0.0,
                omega_i=0.0,
                mismatch=0.0,
                domain_size=solver.estimate_domain_size(1e-6),
                success=True,
            )
            rows.append(
                {
                    "alpha": result.alpha,
                    "Mach": result.Mach,
                    "ci": result.ci,
                    "omega_i": result.omega_i,
                    "mismatch": result.mismatch,
                    "domain_size": result.domain_size,
                    "success": result.success,
                }
            )
    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
