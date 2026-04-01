from __future__ import annotations

"""
Adaptation reutilisable de mstab17.py pour le regime supersonique.

La logique reste volontairement proche du script d'origine :
1. on integre le systeme reel [kappa, q, ln|p|, phi] depuis la gauche
   et depuis la droite ;
2. on cherche l'eigenvaleur complexe c = c_r + i c_i en minimisant
   l'ecart entre les variables de Riccati au point de raccord y = match_y ;
3. une fois c connu, on ajuste l'amplitude initiale de la branche droite
   pour raccorder |p| au centre y = 0.

Cette version sert d'abord de solveur "single case" robuste pour tester
des couples (alpha, Mach) supersoniques avant tout balayage global.
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
class Mstab17SupersonicResult:
    alpha: float
    Mach: float
    cr: float
    ci: float
    omega_i: float
    stage1_mismatch: float
    stage2_mismatch: float
    y_limit: float
    ln_p_start_right: float
    spectral_success: bool
    mode_success: bool
    success: bool
    use_mapping: bool
    mapping_scale: float


class Mstab17SupersonicSolver:
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
        amplitude_match_y: float | None = None,
        use_mapping: bool = False,
        mapping_scale: float = 5.0,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.match_y = float(match_y)
        self.ln_p_start_left = float(ln_p_start_left)
        self.rtol = rtol
        self.atol = atol
        self.min_y_limit = min_y_limit
        self.max_y_limit = max_y_limit
        self.amplitude_match_y = self.match_y if amplitude_match_y is None else float(amplitude_match_y)
        self.use_mapping = bool(use_mapping)
        self.mapping_scale = float(mapping_scale)
        self.ci_floor = 1e-6

    @staticmethod
    def base_velocity(y: np.ndarray | float) -> np.ndarray | float:
        return np.tanh(y)

    @staticmethod
    def base_velocity_derivative(y: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / np.cosh(y) ** 2

    def phase_speed(self, cr: float, ci: float) -> complex:
        return max(float(cr), 0.0) + 1j * max(float(ci), self.ci_floor)

    def asymptotic_gammas(self, cr: float, ci: float) -> tuple[complex, complex]:
        """
        Branches decroissantes dans les far fields U -> -1 et U -> +1.
        """
        c = self.phase_speed(cr, ci)
        r_inf_left = 1.0 - self.Mach**2 * ((-1.0 - c) ** 2)
        r_inf_right = 1.0 - self.Mach**2 * ((1.0 - c) ** 2)

        gamma_left = self.alpha * _principal_sqrt(r_inf_left)
        if gamma_left.real < 0:
            gamma_left = -gamma_left

        gamma_right = -self.alpha * _principal_sqrt(r_inf_right)
        if gamma_right.real > 0:
            gamma_right = -gamma_right

        return gamma_left, gamma_right

    def estimate_y_limit(self, cr: float, ci: float) -> float:
        gamma_left, gamma_right = self.asymptotic_gammas(cr, ci)
        decay_left = max(gamma_left.real, 1e-8)
        decay_right = max(-gamma_right.real, 1e-8)
        estimate = 4.0 * max(1.0 / decay_left, 1.0 / decay_right)
        return float(np.clip(estimate, self.min_y_limit, self.max_y_limit))

    def xi_to_y(self, xi: np.ndarray | float) -> np.ndarray | float:
        return self.mapping_scale * np.asarray(xi) / (1.0 - np.asarray(xi) ** 2)

    def dy_dxi(self, xi: np.ndarray | float) -> np.ndarray | float:
        xi = np.asarray(xi)
        return self.mapping_scale * (1.0 + xi**2) / (1.0 - xi**2) ** 2

    def y_to_xi(self, y: float) -> float:
        if np.isclose(y, 0.0):
            return 0.0
        return float((2.0 * y) / (self.mapping_scale + np.sqrt(self.mapping_scale**2 + 4.0 * y**2)))

    def riccati_system_real_split(
        self,
        y: float,
        state: np.ndarray,
        cr: float,
        ci: float,
    ) -> list[float]:
        """
        Etat = [kappa, q, ln|p|, phi] avec gamma = p'/p = kappa + i q.
        """
        kappa, q, ln_p, phi = state
        gamma = kappa + 1j * q
        c = self.phase_speed(cr, ci)

        u = self.base_velocity(y)
        up = self.base_velocity_derivative(y)
        u_diff = u - c

        p_term = -2.0 * up / u_diff
        r_term = 1.0 - self.Mach**2 * (u_diff**2)
        rhs_complex = -(gamma**2) - p_term * gamma + (self.alpha**2) * r_term

        return [rhs_complex.real, rhs_complex.imag, kappa, q]

    def riccati_system_real_split_xi(
        self,
        xi: float,
        state: np.ndarray,
        cr: float,
        ci: float,
    ) -> list[float]:
        y = float(self.xi_to_y(xi))
        jac = float(self.dy_dxi(xi))
        rhs_y = self.riccati_system_real_split(y, state, cr, ci)
        return [jac * value for value in rhs_y]

    def get_trajectories(
        self,
        cr: float,
        ci: float,
        ln_p_start_right: float = -5.0,
    ) -> tuple[solve_ivp, solve_ivp, solve_ivp, float]:
        """
        Retourne :
        - branche gauche jusqu'a match_y
        - branche droite evaluee a match_y
        - branche droite complete jusqu'a 0
        - taille de domaine utilisee
        """
        gamma_left_inf, gamma_right_inf = self.asymptotic_gammas(cr, ci)
        y_limit = self.estimate_y_limit(cr, ci)

        init_left = [gamma_left_inf.real, gamma_left_inf.imag, self.ln_p_start_left, 0.0]
        init_right = [gamma_right_inf.real, gamma_right_inf.imag, ln_p_start_right, 0.0]

        if self.use_mapping:
            xi_left = self.y_to_xi(-y_limit)
            xi_right = self.y_to_xi(y_limit)
            xi_match = self.y_to_xi(self.match_y)
            xi_center = self.y_to_xi(0.0)
            xi_eval_left = np.linspace(xi_left, xi_match, 2500)
            xi_eval_right = np.linspace(xi_right, xi_center, 2500)

            sol_left = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_left, xi_match),
                init_left,
                t_eval=xi_eval_left,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            sol_right_match = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_right, xi_match),
                init_right,
                t_eval=np.array([xi_match]),
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            sol_right_full = solve_ivp(
                self.riccati_system_real_split_xi,
                (xi_right, xi_center),
                init_right,
                t_eval=xi_eval_right,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            sol_left.t = self.xi_to_y(sol_left.t)
            sol_right_match.t = self.xi_to_y(sol_right_match.t)
            sol_right_full.t = self.xi_to_y(sol_right_full.t)
        else:
            y_eval_left = np.linspace(-y_limit, self.match_y, 2500)
            y_eval_right = np.linspace(y_limit, 0.0, 2500)

            sol_left = solve_ivp(
                self.riccati_system_real_split,
                (-y_limit, self.match_y),
                init_left,
                t_eval=y_eval_left,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            sol_right_match = solve_ivp(
                self.riccati_system_real_split,
                (y_limit, self.match_y),
                init_right,
                t_eval=np.array([self.match_y]),
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            sol_right_full = solve_ivp(
                self.riccati_system_real_split,
                (y_limit, 0.0),
                init_right,
                t_eval=y_eval_right,
                args=(cr, ci),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
        return sol_left, sol_right_match, sol_right_full, y_limit

    @staticmethod
    def _interp_component(y_target: float, solution: solve_ivp, component_index: int) -> float:
        t = np.asarray(solution.t)
        values = np.asarray(solution.y[component_index])
        if t[0] > t[-1]:
            t = t[::-1]
            values = values[::-1]
        return float(np.interp(y_target, t, values))

    def stage1_mismatch(self, cr: float, ci: float) -> float:
        if cr < 0.0 or ci <= 0.0:
            return 1e6
        sol_left, sol_right_match, _, _ = self.get_trajectories(cr, ci)
        if not (sol_left.success and sol_right_match.success):
            return 1e6

        kappa_left = self._interp_component(self.match_y, sol_left, 0)
        q_left = self._interp_component(self.match_y, sol_left, 1)
        kappa_right = float(sol_right_match.y[0, 0])
        q_right = float(sol_right_match.y[1, 0])
        return float(np.hypot(kappa_left - kappa_right, q_left - q_right))

    def solve_eigenvalue(
        self,
        *,
        cr_min: float = 0.03,
        cr_max: float = 0.35,
        ci_min: float = 0.01,
        ci_max: float = 0.12,
        max_iter: int = 12,
        tol: float = 1e-7,
        grid_size: int = 4,
    ) -> tuple[float, float, float]:
        """
        Raffinement par boites successives, comme dans mstab17.py.
        """
        best_cr = max(cr_min, 0.0)
        best_ci = max(ci_min, self.ci_floor)
        best_err = np.inf

        current_cr_min = float(cr_min)
        current_cr_max = float(cr_max)
        current_ci_min = float(max(ci_min, self.ci_floor))
        current_ci_max = float(ci_max)

        for _ in range(max_iter):
            cr_vals = np.linspace(current_cr_min, current_cr_max, grid_size)
            ci_vals = np.linspace(current_ci_min, current_ci_max, grid_size)
            local_errors = np.empty((grid_size, grid_size), dtype=float)

            for i, ci in enumerate(ci_vals):
                for j, cr in enumerate(cr_vals):
                    local_errors[i, j] = self.stage1_mismatch(float(cr), float(ci))

            idx = np.unravel_index(int(np.argmin(local_errors)), local_errors.shape)
            best_ci = float(ci_vals[idx[0]])
            best_cr = float(cr_vals[idx[1]])
            best_err = float(local_errors[idx])
            if best_err < tol:
                break

            dcr = (current_cr_max - current_cr_min) / grid_size
            dci = (current_ci_max - current_ci_min) / grid_size
            current_cr_min = max(0.0, best_cr - dcr)
            current_cr_max = best_cr + dcr
            current_ci_min = max(self.ci_floor, best_ci - dci)
            current_ci_max = best_ci + dci

        return best_cr, best_ci, best_err

    def stage2_objective(self, ln_p_start_right: float, cr: float, ci: float) -> float:
        sol_left, _, sol_right_full, _ = self.get_trajectories(cr, ci, ln_p_start_right=ln_p_start_right)
        if not (sol_left.success and sol_right_full.success):
            return 1e6
        target_y = self.amplitude_match_y
        ln_p_left_target = self._interp_component(target_y, sol_left, 2)
        ln_p_right_target = self._interp_component(target_y, sol_right_full, 2)
        return float((ln_p_left_target - ln_p_right_target) ** 2)

    def solve(
        self,
        *,
        cr_min: float = 0.03,
        cr_max: float = 0.35,
        ci_min: float = 0.01,
        ci_max: float = 0.12,
        max_iter: int = 12,
        tol: float = 1e-7,
        grid_size: int = 4,
    ) -> Mstab17SupersonicResult:
        cr_star, ci_star, stage1_err = self.solve_eigenvalue(
            cr_min=cr_min,
            cr_max=cr_max,
            ci_min=ci_min,
            ci_max=ci_max,
            max_iter=max_iter,
            tol=tol,
            grid_size=grid_size,
        )
        amp_opt = minimize_scalar(
            lambda ln_p_right: self.stage2_objective(ln_p_right, cr_star, ci_star),
            bounds=(-15.0, 5.0),
            method="bounded",
        )
        stage2_err = float(amp_opt.fun)
        _, _, _, y_limit = self.get_trajectories(cr_star, ci_star, ln_p_start_right=float(amp_opt.x))
        spectral_success = bool(stage1_err < 5e-2)
        mode_success = bool(stage2_err < 1e-2)
        success = bool(spectral_success and mode_success)
        return Mstab17SupersonicResult(
            alpha=self.alpha,
            Mach=self.Mach,
            cr=cr_star,
            ci=ci_star,
            omega_i=self.alpha * ci_star,
            stage1_mismatch=stage1_err,
            stage2_mismatch=stage2_err,
            y_limit=y_limit,
            ln_p_start_right=float(amp_opt.x),
            spectral_success=spectral_success,
            mode_success=mode_success,
            success=success,
            use_mapping=self.use_mapping,
            mapping_scale=self.mapping_scale,
        )

    def plot_mode(self, result: Mstab17SupersonicResult, output_path: Path | None = None) -> None:
        sol_left, _, sol_right_full, _ = self.get_trajectories(
            result.cr,
            result.ci,
            ln_p_start_right=result.ln_p_start_right,
        )

        y_left = np.asarray(sol_left.t)
        y_right = np.asarray(sol_right_full.t)
        k_left, q_left, ln_p_left, phi_left = sol_left.y
        k_right, q_right, ln_p_right, phi_right = sol_right_full.y

        abs_p_left = np.exp(ln_p_left)
        abs_p_right = np.exp(ln_p_right)

        phi_left_0 = self._interp_component(0.0, sol_left, 3)
        phi_right_0 = self._interp_component(0.0, sol_right_full, 3)
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
        axes[1, 1].axhline(result.cr, color="orange", linestyle="--", label=r"$c_r$")
        axes[1, 1].set_title("Profil moyen")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle(
            f"mstab17 supersonic | alpha={self.alpha:.3f}, M={self.Mach:.3f}, "
            f"cr={result.cr:.5f}, ci={result.ci:.5f}, omega_i={result.omega_i:.5f}, "
            f"mapping={'on' if result.use_mapping else 'off'}"
        )
        fig.tight_layout()
        if output_path is None:
            plt.show()
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solveur supersonique inspire de mstab17.py.")
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--mach", type=float, default=1.1)
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--amplitude-match-y", type=float, default=None)
    parser.add_argument("--use-mapping", action="store_true")
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--cr-min", type=float, default=0.03)
    parser.add_argument("--cr-max", type=float, default=0.35)
    parser.add_argument("--ci-min", type=float, default=0.01)
    parser.add_argument("--ci-max", type=float, default=0.12)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/blumen_shooting_supersonic/mstab17_supersonic_mode.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    solver = Mstab17SupersonicSolver(
        alpha=args.alpha,
        Mach=args.mach,
        match_y=args.match_y,
        amplitude_match_y=args.amplitude_match_y,
        use_mapping=args.use_mapping,
        mapping_scale=args.mapping_scale,
    )
    result = solver.solve(
        cr_min=args.cr_min,
        cr_max=args.cr_max,
        ci_min=args.ci_min,
        ci_max=args.ci_max,
        max_iter=args.max_iter,
        grid_size=args.grid_size,
    )
    print(result)
    if args.plot:
        solver.plot_mode(result, output_path=args.output)
        print(f"Figure enregistree dans {args.output}")


if __name__ == "__main__":
    main()
