from __future__ import annotations

"""
Adaptation subsonique de mstab17.py en solveur reutilisable.

L'idee reste celle du script d'origine :
1. on integre le systeme de Riccati reel [kappa, q, ln|p|, phi] depuis la gauche
   et depuis la droite ;
2. on determine c_i en minimisant l'ecart entre les variables de Riccati
   au point de raccord ;
3. une fois c_i determine, on ajuste l'amplitude initiale de la branche droite
   pour raccorder |p| au centre.

Dans cette version subsonique on impose :
    c = i c_i
donc c_r = 0.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def _principal_sqrt(z: complex) -> complex:
    root = np.sqrt(z + 0j)
    if root.real < 0 or (np.isclose(root.real, 0.0, atol=1e-12) and root.imag < 0):
        root = -root
    return root


@dataclass
class Mstab17SubsonicResult:
    alpha: float
    Mach: float
    ci: float
    omega_i: float
    stage1_mismatch: float
    stage2_mismatch: float
    y_limit: float
    ln_p_start_right: float
    success: bool


class Mstab17SubsonicSolver:
    def __init__(
        self,
        alpha: float,
        Mach: float,
        *,
        match_y: float = 1.0,
        ln_p_start_left: float = -5.0,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        min_y_limit: float = 10.0,
        max_y_limit: float = 80.0,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.match_y = float(match_y)
        self.ln_p_start_left = float(ln_p_start_left)
        self.rtol = rtol
        self.atol = atol
        self.min_y_limit = min_y_limit
        self.max_y_limit = max_y_limit
        self.ci_floor = 1e-6

    @staticmethod
    def base_velocity(y: np.ndarray | float) -> np.ndarray | float:
        return np.tanh(y)

    @staticmethod
    def base_velocity_derivative(y: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / np.cosh(y) ** 2

    def phase_speed(self, ci: float) -> complex:
        return 1j * max(float(ci), self.ci_floor)

    def asymptotic_gammas(self, ci: float) -> tuple[complex, complex]:
        c = self.phase_speed(ci)
        r_inf_left = 1.0 - self.Mach**2 * ((-1.0 - c) ** 2)
        r_inf_right = 1.0 - self.Mach**2 * ((1.0 - c) ** 2)

        gamma_left = self.alpha * _principal_sqrt(r_inf_left)
        gamma_right = -self.alpha * _principal_sqrt(r_inf_right)
        return gamma_left, gamma_right

    def estimate_y_limit(self, ci: float) -> float:
        gamma_left, gamma_right = self.asymptotic_gammas(ci)
        decay_left = max(gamma_left.real, 1e-8)
        decay_right = max(-gamma_right.real, 1e-8)
        estimate = 4.0 * max(1.0 / decay_left, 1.0 / decay_right)
        return float(np.clip(estimate, self.min_y_limit, self.max_y_limit))

    def riccati_system_real_split(self, y: float, state: np.ndarray, ci: float) -> list[float]:
        """
        Etat = [kappa, q, ln|p|, phi]
        avec gamma = p'/p = kappa + i q.
        """
        kappa, q, ln_p, phi = state
        gamma = kappa + 1j * q
        c = self.phase_speed(ci)

        u = self.base_velocity(y)
        up = self.base_velocity_derivative(y)
        u_diff = u - c

        p_term = -2.0 * up / u_diff
        r_term = 1.0 - self.Mach**2 * (u_diff**2)
        rhs_complex = -(gamma**2) - p_term * gamma + (self.alpha**2) * r_term

        return [rhs_complex.real, rhs_complex.imag, kappa, q]

    def get_trajectories(self, ci: float, ln_p_start_right: float = -5.0) -> tuple[solve_ivp, solve_ivp, float]:
        """
        Integre une branche gauche jusqu'a y = 0 et une branche droite jusqu'a y = 0.
        Le point de raccord pour l'eigenvaleur peut etre distinct de 0.
        """
        gamma_left_inf, gamma_right_inf = self.asymptotic_gammas(ci)
        y_limit = self.estimate_y_limit(ci)

        init_left = [gamma_left_inf.real, gamma_left_inf.imag, self.ln_p_start_left, 0.0]
        init_right = [gamma_right_inf.real, gamma_right_inf.imag, ln_p_start_right, 0.0]

        target_left = max(self.match_y, 0.0)
        y_eval_left = np.linspace(-y_limit, target_left, 2500)
        y_eval_right = np.linspace(y_limit, 0.0, 2500)

        sol_left = solve_ivp(
            self.riccati_system_real_split,
            (-y_limit, target_left),
            init_left,
            t_eval=y_eval_left,
            args=(ci,),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
        )
        sol_right = solve_ivp(
            self.riccati_system_real_split,
            (y_limit, 0.0),
            init_right,
            t_eval=y_eval_right,
            args=(ci,),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
        )
        return sol_left, sol_right, y_limit

    @staticmethod
    def _interp_component(y_target: float, solution: solve_ivp, component_index: int) -> float:
        t = np.asarray(solution.t)
        values = np.asarray(solution.y[component_index])
        if t[0] > t[-1]:
            t = t[::-1]
            values = values[::-1]
        return float(np.interp(y_target, t, values))

    def stage1_mismatch(self, ci: float) -> float:
        if ci <= 0.0:
            return 1e6
        sol_left, sol_right, _ = self.get_trajectories(ci)
        if not sol_left.success or not sol_right.success:
            return 1e6

        kappa_left = self._interp_component(self.match_y, sol_left, 0)
        q_left = self._interp_component(self.match_y, sol_left, 1)
        kappa_right = self._interp_component(self.match_y, sol_right, 0)
        q_right = self._interp_component(self.match_y, sol_right, 1)

        return float(np.hypot(kappa_left - kappa_right, q_left - q_right))

    def solve_ci(
        self,
        *,
        ci_min: float = 1e-3,
        ci_max: float = 1.0,
        n_scan: int = 61,
    ) -> tuple[float, float]:
        ci_values = np.linspace(max(ci_min, self.ci_floor), ci_max, n_scan)
        mismatches = np.array([self.stage1_mismatch(ci) for ci in ci_values])

        best_index = int(np.argmin(mismatches))
        coarse_ci = float(ci_values[best_index])
        delta_ci = ci_values[1] - ci_values[0]
        bracket_left = max(ci_min, coarse_ci - delta_ci)
        bracket_right = min(ci_max, coarse_ci + delta_ci)

        optimum = minimize_scalar(
            self.stage1_mismatch,
            bounds=(bracket_left, bracket_right),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return float(optimum.x), float(optimum.fun)

    def stage2_objective(self, ln_p_start_right: float, ci: float) -> float:
        sol_left, sol_right, _ = self.get_trajectories(ci, ln_p_start_right=ln_p_start_right)
        if not sol_left.success or not sol_right.success:
            return 1e6
        ln_p_left_0 = self._interp_component(0.0, sol_left, 2)
        ln_p_right_0 = self._interp_component(0.0, sol_right, 2)
        return float((ln_p_left_0 - ln_p_right_0) ** 2)

    def solve(self, *, ci_min: float = 1e-3, ci_max: float = 1.0, n_scan: int = 61) -> Mstab17SubsonicResult:
        ci_star, stage1_err = self.solve_ci(ci_min=ci_min, ci_max=ci_max, n_scan=n_scan)
        amp_opt = minimize_scalar(
            lambda ln_p_right: self.stage2_objective(ln_p_right, ci_star),
            bounds=(-15.0, 5.0),
            method="bounded",
        )
        stage2_err = float(amp_opt.fun)
        _, _, y_limit = self.get_trajectories(ci_star, ln_p_start_right=float(amp_opt.x))
        success = bool(stage1_err < 5e-2 and stage2_err < 1e-2)
        return Mstab17SubsonicResult(
            alpha=self.alpha,
            Mach=self.Mach,
            ci=ci_star,
            omega_i=self.alpha * ci_star,
            stage1_mismatch=stage1_err,
            stage2_mismatch=stage2_err,
            y_limit=y_limit,
            ln_p_start_right=float(amp_opt.x),
            success=success,
        )

    def plot_mode(self, result: Mstab17SubsonicResult, output_path: Path | None = None) -> None:
        sol_left, sol_right, _ = self.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

        y_left = np.asarray(sol_left.t)
        y_right = np.asarray(sol_right.t)
        k_left, q_left, ln_p_left, phi_left = sol_left.y
        k_right, q_right, ln_p_right, phi_right = sol_right.y

        abs_p_left = np.exp(ln_p_left)
        abs_p_right = np.exp(ln_p_right)

        phi_left_0 = self._interp_component(0.0, sol_left, 3)
        phi_right_0 = self._interp_component(0.0, sol_right, 3)
        phase_shift = phi_left_0 - phi_right_0

        mode_left = abs_p_left * np.cos(phi_left)
        mode_right = abs_p_right * np.cos(phi_right + phase_shift)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].plot(y_left, abs_p_left, label="Left")
        axes[0, 0].plot(y_right, abs_p_right, "--", label="Right")
        axes[0, 0].set_title(r"Amplitude $|p|$")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(y_left, k_left, label=r"$\kappa$ left")
        axes[0, 1].plot(y_left, q_left, ":", label=r"$q$ left")
        axes[0, 1].plot(y_right, k_right, "--", label=r"$\kappa$ right")
        axes[0, 1].plot(y_right, q_right, "-.", label=r"$q$ right")
        axes[0, 1].set_title("Variables de Riccati")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(y_left, mode_left, label="Left")
        axes[1, 0].plot(y_right, mode_right, "--", label="Right")
        axes[1, 0].set_title(r"Mode physique $\Re(p)$")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        yy = np.linspace(-result.y_limit, result.y_limit, 400)
        axes[1, 1].plot(yy, np.tanh(yy), label=r"$U(y)=\tanh(y)$")
        axes[1, 1].axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        axes[1, 1].set_title("Profil moyen")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle(
            f"mstab17 subsonic | alpha={self.alpha:.3f}, M={self.Mach:.3f}, "
            f"ci={result.ci:.5f}, omega_i={result.omega_i:.5f}"
        )
        fig.tight_layout()
        if output_path is None:
            plt.show()
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solveur subsonique inspire de mstab17.py.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--ci-min", type=float, default=1e-3)
    parser.add_argument("--ci-max", type=float, default=1.0)
    parser.add_argument("--n-scan", type=int, default=61)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("assets/blumen_shooting/mstab17_subsonic_mode.png"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    solver = Mstab17SubsonicSolver(alpha=args.alpha, Mach=args.mach, match_y=args.match_y)
    result = solver.solve(ci_min=args.ci_min, ci_max=args.ci_max, n_scan=args.n_scan)
    print(result)
    if args.plot:
        solver.plot_mode(result, output_path=args.output)
        print(f"Figure enregistree dans {args.output}")


if __name__ == "__main__":
    main()
