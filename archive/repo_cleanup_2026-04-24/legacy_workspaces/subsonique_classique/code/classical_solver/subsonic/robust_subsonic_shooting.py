from __future__ import annotations

"""
Solveur subsonique robuste en methode du tir.

Strategie :
- solveur principal : SubsonicShootingSolver
- solveur de secours / validation : Mstab17SubsonicSolver

Le solveur principal est plus leger et bien adapte aux balayages denses.
Le solveur mstab17 est plus complet sur la reconstruction du mode et semble
plus robuste pres de la neutralite. On l'utilise donc :
- en fallback si le solveur principal echoue ;
- automatiquement pres de la frontiere neutre ;
- ou en validation si les deux solutions divergent trop.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from classical_solver.subsonic.shooting_subsonic import SubsonicShootingSolver


@dataclass
class RobustSubsonicResult:
    alpha: float
    Mach: float
    ci: float
    omega_i: float
    success: bool
    source: str
    primary_ci: float
    primary_omega_i: float
    primary_success: bool
    primary_mismatch: float
    secondary_ci: float | None
    secondary_omega_i: float | None
    secondary_success: bool | None
    secondary_stage1_mismatch: float | None
    secondary_stage2_mismatch: float | None
    ci_abs_diff: float | None
    omega_abs_diff: float | None


class RobustSubsonicShootingSolver:
    def __init__(
        self,
        alpha: float,
        Mach: float,
        *,
        neutral_ratio_threshold: float = 0.85,
        omega_diff_tolerance: float = 1e-2,
        ci_diff_tolerance: float = 5e-2,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.neutral_ratio_threshold = float(neutral_ratio_threshold)
        self.omega_diff_tolerance = float(omega_diff_tolerance)
        self.ci_diff_tolerance = float(ci_diff_tolerance)

    def stability_limit(self) -> float:
        return max(0.0, (1.0 - self.Mach**2)) ** 0.5

    def is_near_neutral(self) -> bool:
        alpha_cut = self.stability_limit()
        if alpha_cut <= 1e-12:
            return True
        return self.alpha >= self.neutral_ratio_threshold * alpha_cut

    def solve(
        self,
        *,
        primary_ci_max: float = 1.0,
        primary_n_scan: int = 81,
        secondary_ci_min: float = 1e-3,
        secondary_ci_max: float = 1.0,
        secondary_n_scan: int = 41,
        force_cross_check: bool = False,
    ) -> RobustSubsonicResult:
        primary_solver = SubsonicShootingSolver(alpha=self.alpha, Mach=self.Mach)
        primary = primary_solver.solve_ci(ci_max=primary_ci_max, n_scan=primary_n_scan)

        secondary = None
        need_secondary = force_cross_check or self.is_near_neutral() or (not primary.success)

        if need_secondary:
            secondary_solver = Mstab17SubsonicSolver(alpha=self.alpha, Mach=self.Mach)
            secondary = secondary_solver.solve(
                ci_min=secondary_ci_min,
                ci_max=secondary_ci_max,
                n_scan=secondary_n_scan,
            )

        chosen_ci = primary.ci
        chosen_omega = primary.omega_i
        chosen_success = primary.success
        source = "primary"
        ci_abs_diff = None
        omega_abs_diff = None

        if secondary is not None:
            ci_abs_diff = abs(primary.ci - secondary.ci)
            omega_abs_diff = abs(primary.omega_i - secondary.omega_i)

            if not primary.success and secondary.success:
                chosen_ci = secondary.ci
                chosen_omega = secondary.omega_i
                chosen_success = True
                source = "secondary_fallback"
            elif self.is_near_neutral() and secondary.success:
                # Pres de la neutralite on privilegie mstab17, qui suit mieux cette zone.
                chosen_ci = secondary.ci
                chosen_omega = secondary.omega_i
                chosen_success = True
                source = "secondary_neutral"
            elif secondary.success and (
                ci_abs_diff > self.ci_diff_tolerance or omega_abs_diff > self.omega_diff_tolerance
            ):
                # Si les solveurs divergent significativement, on choisit le plus prudent :
                # mstab17 s'il est en succes et proche de la neutralite, sinon le primaire.
                if self.is_near_neutral():
                    chosen_ci = secondary.ci
                    chosen_omega = secondary.omega_i
                    chosen_success = True
                    source = "secondary_disagreement"
                else:
                    source = "primary_disagreement"

        return RobustSubsonicResult(
            alpha=self.alpha,
            Mach=self.Mach,
            ci=chosen_ci,
            omega_i=chosen_omega,
            success=chosen_success,
            source=source,
            primary_ci=primary.ci,
            primary_omega_i=primary.omega_i,
            primary_success=primary.success,
            primary_mismatch=primary.mismatch,
            secondary_ci=None if secondary is None else secondary.ci,
            secondary_omega_i=None if secondary is None else secondary.omega_i,
            secondary_success=None if secondary is None else secondary.success,
            secondary_stage1_mismatch=None if secondary is None else secondary.stage1_mismatch,
            secondary_stage2_mismatch=None if secondary is None else secondary.stage2_mismatch,
            ci_abs_diff=ci_abs_diff,
            omega_abs_diff=omega_abs_diff,
        )


def sample_robust_subsonic_growth_map(
    alphas,
    machs,
    *,
    force_cross_check: bool = False,
) -> pd.DataFrame:
    rows = []
    for mach in machs:
        for alpha in alphas:
            solver = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach))
            result = solver.solve(force_cross_check=force_cross_check)
            rows.append(result.__dict__)
    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solveur subsonique robuste combinant deux methodes du tir.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--force-cross-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    solver = RobustSubsonicShootingSolver(alpha=args.alpha, Mach=args.mach)
    result = solver.solve(force_cross_check=args.force_cross_check)
    print(result)


if __name__ == "__main__":
    main()
