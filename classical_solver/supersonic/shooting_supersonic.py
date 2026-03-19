from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def _principal_sqrt(z: complex) -> complex:
    root = np.sqrt(z)
    if root.real < 0 or (np.isclose(root.real, 0.0, atol=1e-12) and root.imag < 0):
        root = -root
    return root


@dataclass
class SupersonicShootingResult:
    alpha: float
    Mach: float
    cr: float
    ci: float
    omega_i: float
    mismatch: float
    domain_size: float
    success: bool


class SupersonicShootingSolver:
    """
    Solveur de tir supersonique base sur l'equation de Riccati pour la pression.

    On recherche directement l'eigenvaleur complexe c = c_r + i c_i.
    La paire spectrale +/- c_r partage le meme c_i ; on contraint ici c_r >= 0
    pour suivre une seule branche representative.
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

    def phase_speed(self, cr: float, ci: float) -> complex:
        return float(max(cr, 0.0)) + 1j * float(max(ci, self.ci_floor))

    def asymptotic_decay_rates(self, cr: float, ci: float) -> tuple[complex, complex]:
        c = self.phase_speed(cr, ci)
        z_minus = 1.0 - self.Mach**2 * (c + 1.0) ** 2
        z_plus = 1.0 - self.Mach**2 * (c - 1.0) ** 2
        root_minus = _principal_sqrt(z_minus)
        root_plus = _principal_sqrt(z_plus)
        gamma_left = self.alpha * root_minus
        gamma_right = -self.alpha * root_plus
        return gamma_left, gamma_right

    def estimate_domain_size(self, cr: float, ci: float) -> float:
        gamma_left, gamma_right = self.asymptotic_decay_rates(cr, ci)
        decay_left = max(gamma_left.real, 1e-6)
        decay_right = max(-gamma_right.real, 1e-6)
        estimate = 4.0 * (1.0 / decay_left + 1.0 / decay_right)
        return float(np.clip(estimate, self.min_domain_size, self.max_domain_size))

    def riccati_rhs(self, y: float, gamma: np.ndarray, cr: float, ci: float) -> np.ndarray:
        c = self.phase_speed(cr, ci)
        u = self.base_velocity(y)
        up = self.base_velocity_derivative(y)
        denominator = u - c
        p_term = -2.0 * up / denominator
        r_term = 1.0 - self.Mach**2 * denominator**2
        return np.array([-gamma[0] ** 2 - p_term * gamma[0] + self.alpha**2 * r_term], dtype=complex)

    def integrate_gamma(
        self,
        cr: float,
        ci: float,
        domain_size: float | None = None,
    ) -> tuple[complex, complex, float, bool]:
        domain_size = self.estimate_domain_size(cr, ci) if domain_size is None else float(domain_size)
        gamma_left_0, gamma_right_0 = self.asymptotic_decay_rates(cr, ci)

        left_solution = solve_ivp(
            lambda y, state: self.riccati_rhs(y, state, cr, ci),
            (-domain_size, self.match_point),
            y0=np.array([gamma_left_0], dtype=complex),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.array([self.match_point]),
        )
        right_solution = solve_ivp(
            lambda y, state: self.riccati_rhs(y, state, cr, ci),
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

    def mismatch(self, params: np.ndarray) -> float:
        cr = float(params[0])
        ci = float(params[1])
        if cr < 0.0 or ci <= 0.0:
            return 1e6
        gamma_left, gamma_right, _, success = self.integrate_gamma(cr, ci)
        if not success:
            return 1e6
        mismatch_value = abs(gamma_left - gamma_right)
        return float(mismatch_value)

    @staticmethod
    def _tracking_distance(
        params: tuple[float, float],
        target_guess: tuple[float, float],
        cr_max: float,
        ci_max: float,
    ) -> float:
        cr_scale = max(cr_max, 1e-6)
        ci_scale = max(ci_max, 1e-6)
        return float(
            np.hypot(
                (params[0] - target_guess[0]) / cr_scale,
                (params[1] - target_guess[1]) / ci_scale,
            )
        )

    def solve_mode(
        self,
        *,
        previous_guess: tuple[float, float] | None = None,
        previous_guesses: list[tuple[float, float]] | None = None,
        anchor_guess: tuple[float, float] | None = None,
        ci_max: float = 0.12,
        cr_max: float = 0.20,
        ci_min_search: float = 0.005,
        tracking_weight: float = 2e-2,
    ) -> SupersonicShootingResult:
        seeds = []
        if previous_guesses is not None:
            for guess in previous_guesses:
                if guess is None:
                    continue
                seeds.append((max(guess[0], 0.0), max(guess[1], ci_min_search)))
        target_guess = None
        continuation_pool = [guess for guess in (previous_guesses or []) if guess is not None]
        if previous_guess is not None:
            continuation_pool.append(previous_guess)
        if continuation_pool:
            target_guess = max(continuation_pool, key=lambda guess: guess[1])
        if anchor_guess is not None:
            seeds.append((max(anchor_guess[0], 0.0), max(anchor_guess[1], ci_min_search)))
        if previous_guess is not None:
            seeds.append((max(previous_guess[0], 0.0), max(previous_guess[1], ci_min_search)))
        seeds.extend(
            [
                (0.00, max(ci_min_search, 0.02)),
                (0.02, 0.02),
                (0.04, 0.03),
                (0.06, 0.05),
                (0.08, 0.08),
            ]
        )

        seed_neighbors = []
        if previous_guesses is not None:
            seed_neighbors.extend(previous_guesses)
        if previous_guess is not None:
            seed_neighbors.append(previous_guess)
        for neighbor in seed_neighbors:
            seed_cr, seed_ci = neighbor
            for delta_cr in (-0.03, 0.0, 0.03):
                for delta_ci in (-0.02, 0.0, 0.02):
                    seeds.append((max(0.0, seed_cr + delta_cr), max(ci_min_search, seed_ci + delta_ci)))
        if anchor_guess is not None:
            seed_cr, seed_ci = anchor_guess
            for delta_cr in (-0.03, 0.0, 0.03):
                for delta_ci in (-0.02, 0.0, 0.02):
                    seeds.append((max(0.0, seed_cr + delta_cr), max(ci_min_search, seed_ci + delta_ci)))
        else:
            coarse_cr = np.linspace(0.0, cr_max, 5)
            coarse_ci = np.linspace(max(ci_min_search, self.ci_floor), ci_max, 7)
            coarse_evals = []
            for cr in coarse_cr:
                for ci in coarse_ci:
                    mismatch_value = self.mismatch(np.array([cr, ci], dtype=float))
                    coarse_evals.append((mismatch_value, cr, ci))
            coarse_evals.sort(key=lambda item: item[0])
            nontrivial_coarse = [item for item in coarse_evals if item[2] >= ci_min_search]
            seeds.extend((item[1], item[2]) for item in nontrivial_coarse[:6])

        best_result = None
        bounds = [(0.0, cr_max), (ci_min_search, ci_max)]
        seen = set()
        for seed in seeds:
            rounded_seed = (round(seed[0], 4), round(seed[1], 4))
            if rounded_seed in seen:
                continue
            seen.add(rounded_seed)
            if target_guess is None:
                objective = self.mismatch
            else:
                objective = lambda params, target_guess=target_guess: (
                    self.mismatch(params)
                    + tracking_weight
                    * self._tracking_distance(
                        (float(params[0]), float(params[1])),
                        target_guess,
                        cr_max,
                        ci_max,
                    )
                )
            optimum = minimize(
                objective,
                x0=np.array(seed, dtype=float),
                method="Powell",
                bounds=bounds,
                options={"xtol": 2e-3, "ftol": 5e-4, "maxiter": 20},
            )
            candidate_cr = float(np.clip(optimum.x[0], 0.0, cr_max))
            candidate_ci = float(np.clip(optimum.x[1], self.ci_floor, ci_max))
            candidate_mismatch = float(self.mismatch(np.array([candidate_cr, candidate_ci], dtype=float)))
            candidate_score = candidate_mismatch
            if target_guess is not None:
                candidate_score += tracking_weight * self._tracking_distance(
                    (candidate_cr, candidate_ci),
                    target_guess,
                    cr_max,
                    ci_max,
                )
            if best_result is None or candidate_score < best_result[0]:
                best_result = (
                    candidate_score,
                    candidate_mismatch,
                    candidate_cr,
                    candidate_ci,
                    bool(optimum.success),
                )

        _, mismatch_value, cr_star, ci_star, opt_success = best_result
        domain_size = self.estimate_domain_size(cr_star, ci_star)
        success = bool(opt_success and mismatch_value < 5e-2)
        return SupersonicShootingResult(
            alpha=self.alpha,
            Mach=self.Mach,
            cr=cr_star,
            ci=ci_star,
            omega_i=self.alpha * ci_star,
            mismatch=mismatch_value,
            domain_size=domain_size,
            success=success,
        )


def sample_supersonic_growth_map(
    alphas,
    machs,
    *,
    ci_max: float = 0.12,
    cr_max: float = 0.20,
    anchor_points: list[dict] | None = None,
    tracking_weight: float = 2e-2,
) -> pd.DataFrame:
    rows = []
    alphas = np.asarray(alphas, dtype=float)
    machs = np.asarray(machs, dtype=float)
    alpha_values = np.sort(alphas)[::-1]

    def nearest_anchor(alpha: float, mach: float):
        if not anchor_points:
            return None
        best_item = None
        best_distance = None
        for item in anchor_points:
            distance = np.hypot((mach - item["Mach"]) / 1.0, (alpha - item["alpha"]) / 0.6)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_item = item
        if best_item is None:
            return None
        return float(best_item["cr_seed"]), float(best_item["ci_seed"])

    previous_mach_guesses: dict[int, tuple[float, float] | None] = {}
    for mach in np.sort(machs):
        previous_guess = None
        current_mach_guesses: dict[int, tuple[float, float] | None] = {}
        for alpha_index, alpha in enumerate(alpha_values):
            solver = SupersonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            anchor_guess = nearest_anchor(float(alpha), float(mach))
            continuation_guesses = []
            if previous_guess is not None:
                continuation_guesses.append(previous_guess)
            if alpha_index in previous_mach_guesses and previous_mach_guesses[alpha_index] is not None:
                continuation_guesses.append(previous_mach_guesses[alpha_index])
            if alpha_index > 0 and current_mach_guesses.get(alpha_index - 1) is not None:
                continuation_guesses.append(current_mach_guesses[alpha_index - 1])
            result = solver.solve_mode(
                previous_guess=previous_guess,
                previous_guesses=continuation_guesses or None,
                anchor_guess=anchor_guess,
                ci_max=ci_max,
                cr_max=cr_max,
                tracking_weight=tracking_weight,
            )
            if result.success:
                previous_guess = (result.cr, result.ci)
                current_mach_guesses[alpha_index] = previous_guess
            else:
                current_mach_guesses[alpha_index] = None
            rows.append(
                {
                    "alpha": result.alpha,
                    "Mach": result.Mach,
                    "cr": result.cr,
                    "ci": result.ci,
                    "omega_i": result.omega_i,
                    "mismatch": result.mismatch,
                    "domain_size": result.domain_size,
                    "success": result.success,
                }
            )
        previous_mach_guesses = current_mach_guesses

    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
