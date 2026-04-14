from __future__ import annotations

"""
Version "style notebook" du solveur GEP dense.

Cette implementation suit au plus pres la structure des notebooks de la
stagiaire :
- matrice de differences finies sur une grille auxiliaire uniforme ;
- passage a l'operateur D_y par regle de chaine ;
- construction des blocs A, B ;
- elimination des DOF contraints v=0 aux bords ;
- diagonalisation dense via scipy.linalg.eig, alias dense_eig.

Seule difference volontaire :
- on utilise le mapping de reference du projet
      y = L * xi / (1 - xi^2)
  pour rester coherent avec le futur PINN.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig as dense_eig
from scipy.sparse import bmat, csr_matrix, diags


def finite_diff_matrix(n: int, h: float) -> np.ndarray:
    d = np.zeros((n, n), dtype=float)
    inv2h = 0.5 / h
    for i in range(1, n - 1):
        d[i, i - 1] = -inv2h
        d[i, i + 1] = inv2h
    d[0, 0:3] = [-1.5 / h, 2.0 / h, -0.5 / h]
    d[-1, -3:] = [0.5 / h, -2.0 / h, 1.5 / h]
    return d


def eliminate_dofs(a: csr_matrix, b: csr_matrix, constrained: list[int]) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    keep = np.ones(a.shape[0], dtype=bool)
    keep[constrained] = False
    return a[keep][:, keep], b[keep][:, keep], keep


def pin_mapping(xi: np.ndarray, scale: float) -> np.ndarray:
    return scale * xi / (1.0 - xi**2)


def dzeta_dy(xi: np.ndarray, scale: float) -> np.ndarray:
    return (1.0 - xi**2) ** 2 / (scale * (1.0 + xi**2))


def cubic_mapping(xi: np.ndarray, scale: float, delta: float) -> np.ndarray:
    return delta * xi + (scale - delta) * xi**3


def cubic_dzeta_dy(xi: np.ndarray, scale: float, delta: float) -> np.ndarray:
    return 1.0 / (delta + 3.0 * (scale - delta) * xi**2)


@dataclass
class NotebookStyleGEPResult:
    alpha: float
    Mach: float
    c: complex
    cr: float
    ci: float
    omega_i: float
    n_finite_modes: int
    selection_source: str
    success: bool


class NotebookStyleDenseGEPSolver:
    def __init__(
        self,
        *,
        alpha: float,
        Mach: float,
        n_points: int = 301,
        mapping_kind: str = "pin",
        mapping_scale: float = 5.0,
        cubic_delta: float = 0.2,
        xi_max: float = 0.98,
        rho_bar: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.Mach = float(Mach)
        self.n_points = int(n_points)
        self.mapping_kind = str(mapping_kind)
        self.mapping_scale = float(mapping_scale)
        self.cubic_delta = float(cubic_delta)
        self.xi_max = float(xi_max)
        self.rho_bar = float(rho_bar)

        self.a_squared = (1.0 / self.Mach) ** 2
        self.i_alpha = 1j * self.alpha

        self.xi, self.dxi = np.linspace(-self.xi_max, self.xi_max, self.n_points, retstep=True)
        if self.mapping_kind == "pin":
            self.y = pin_mapping(self.xi, self.mapping_scale)
            metric = dzeta_dy(self.xi, self.mapping_scale)
        elif self.mapping_kind == "cubic":
            self.y = cubic_mapping(self.xi, self.mapping_scale, self.cubic_delta)
            metric = cubic_dzeta_dy(self.xi, self.mapping_scale, self.cubic_delta)
        else:
            raise ValueError(f"Unknown mapping_kind={self.mapping_kind!r}. Expected 'pin' or 'cubic'.")

        d_zeta = finite_diff_matrix(self.n_points, self.dxi)
        self.d_y = metric[:, None] * d_zeta

    def construct_matrices(self) -> tuple[csr_matrix, csr_matrix]:
        u = np.tanh(self.y)
        du = 1.0 - u**2
        n = self.n_points

        eye = diags(np.ones(n), format="csr")
        u_d = diags(u, format="csr")
        du_d = diags(du, format="csr")
        d = csr_matrix(self.d_y)
        zero = csr_matrix((n, n), dtype=complex)

        a_blocks = [
            [self.rho_bar * self.i_alpha * eye, self.rho_bar * d, (self.i_alpha / self.a_squared) * u_d],
            [self.rho_bar * self.i_alpha * u_d, self.rho_bar * du_d, self.i_alpha * eye],
            [zero, self.rho_bar * self.i_alpha * u_d, d],
        ]
        b_blocks = [
            [zero, zero, (self.i_alpha / self.a_squared) * eye],
            [self.rho_bar * self.i_alpha * eye, zero, zero],
            [zero, self.rho_bar * self.i_alpha * eye, zero],
        ]
        return bmat(a_blocks, format="csr"), bmat(b_blocks, format="csr")

    def solve_all(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.construct_matrices()
        v_bc = [self.n_points, 2 * self.n_points - 1]
        a_r, b_r, keep = eliminate_dofs(a, b, v_bc)
        vals, vecs = dense_eig(a_r.toarray(), b_r.toarray())
        full_vecs = np.zeros((3 * self.n_points, vecs.shape[1]), dtype=complex)
        full_vecs[keep, :] = vecs
        return vals, full_vecs

    def _mode_signature(self, vector: np.ndarray, *, n_signature: int = 256) -> np.ndarray:
        # Use the pressure component as a robust modal fingerprint that can be
        # compared across different spatial resolutions N.
        p = np.asarray(vector[2 * self.n_points : 3 * self.n_points], dtype=complex)
        idx = int(np.argmax(np.abs(p)))
        if np.abs(p[idx]) > 0.0:
            p = p * np.exp(-1j * np.angle(p[idx]))

        y_sig = np.linspace(self.y[0], self.y[-1], n_signature)
        p_real = np.interp(y_sig, self.y, np.real(p))
        p_imag = np.interp(y_sig, self.y, np.imag(p))
        signature = np.concatenate([p_real, p_imag]).astype(float)
        norm = np.linalg.norm(signature)
        if norm > 0.0:
            signature /= norm
        return signature

    def finite_modes(self) -> list[dict]:
        vals, vecs = self.solve_all()
        finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
        vals = vals[finite]
        vecs = vecs[:, finite]

        mask = np.abs(vals) < 10.0
        vals = vals[mask]
        vecs = vecs[:, mask]

        modes = []
        for idx, val in enumerate(vals):
            ci = float(np.imag(val))
            if ci <= 0.0:
                continue
            modes.append(
                {
                    "c": complex(val),
                    "cr": float(np.real(val)),
                    "ci": ci,
                    "omega_i": float(self.alpha * ci),
                    "abs_cr": abs(float(np.real(val))),
                    "vector": vecs[:, idx],
                    "signature": self._mode_signature(vecs[:, idx]),
                }
            )
        return sorted(modes, key=lambda item: item["omega_i"], reverse=True)

    @staticmethod
    def _distance_to_target(mode: dict, target_guess: tuple[float, float], cr_scale: float, ci_scale: float) -> float:
        return float(
            np.hypot(
                (mode["cr"] - target_guess[0]) / max(cr_scale, 1e-8),
                (mode["ci"] - target_guess[1]) / max(ci_scale, 1e-8),
            )
        )

    @staticmethod
    def spectral_distance(mode: dict, target_guess: tuple[float, float], *, ci_weight: float = 2.0) -> float:
        return float(
            np.sqrt(
                (mode["cr"] - target_guess[0]) ** 2
                + (ci_weight * (mode["ci"] - target_guess[1])) ** 2
            )
        )

    @staticmethod
    def signature_overlap(mode: dict, previous_signature: np.ndarray) -> float:
        signature = mode.get("signature")
        if signature is None or previous_signature is None:
            return 0.0
        return float(abs(np.vdot(signature, previous_signature)))

    def select_mode(
        self,
        modes: list[dict],
        *,
        target_guess: tuple[float, float] | None = None,
        prefer_positive_cr: bool = True,
        cr_window: float = 0.4,
    ) -> tuple[dict, str]:
        if prefer_positive_cr:
            positive = [mode for mode in modes if mode["cr"] >= -1e-10]
            if positive:
                modes = positive

        if target_guess is not None:
            guided = [mode for mode in modes if abs(mode["cr"] - target_guess[0]) <= cr_window]
            pool = guided if guided else modes
            cr_scale = max(0.25, max(abs(mode["cr"]) for mode in pool))
            ci_scale = max(0.10, max(mode["ci"] for mode in pool))
            chosen = min(pool, key=lambda mode: self._distance_to_target(mode, target_guess, cr_scale, ci_scale))
            return chosen, "target_guided"

        if self.Mach < 1.0:
            return max(modes, key=lambda item: item["omega_i"]), "max_growth"

        moderate = [mode for mode in modes if mode["abs_cr"] <= 0.35]
        if moderate:
            return max(moderate, key=lambda item: item["omega_i"]), "moderate_cr"

        return min(modes, key=lambda item: item["abs_cr"]), "min_abs_cr"

    def get_selected_mode(
        self,
        *,
        target_guess: tuple[float, float] | None = None,
        prefer_positive_cr: bool = True,
        cr_window: float = 0.4,
    ) -> tuple[dict | None, str, int]:
        modes = self.finite_modes()
        if not modes:
            return None, "no_mode", 0
        chosen, selection_source = self.select_mode(
            modes,
            target_guess=target_guess,
            prefer_positive_cr=prefer_positive_cr,
            cr_window=cr_window,
        )
        return chosen, selection_source, len(modes)

    def get_nearest_mode_to_target(
        self,
        *,
        target_guess: tuple[float, float],
        prefer_positive_cr: bool = True,
        ci_weight: float = 2.0,
    ) -> tuple[dict | None, str, int]:
        modes = self.finite_modes()
        if prefer_positive_cr:
            positive = [mode for mode in modes if mode["cr"] >= -1e-10]
            if positive:
                modes = positive
        if not modes:
            return None, "no_mode", 0
        chosen = min(modes, key=lambda mode: self.spectral_distance(mode, target_guess, ci_weight=ci_weight))
        return chosen, "nearest_to_target", len(modes)

    def get_branch_mode(
        self,
        *,
        target_guess: tuple[float, float],
        previous_guess: tuple[float, float] | None = None,
        previous_signature: np.ndarray | None = None,
        prefer_positive_cr: bool = True,
        ci_weight: float = 2.0,
        spectral_window_factor: float = 1.5,
        spectral_window_floor: float = 0.01,
        overlap_top_k: int = 5,
        overlap_weight: float = 0.35,
        jump_cr_weight: float = 0.45,
        jump_ci_weight: float = 0.20,
    ) -> tuple[dict | None, str, int]:
        modes = self.finite_modes()
        if prefer_positive_cr:
            positive = [mode for mode in modes if mode["cr"] >= -1e-10]
            if positive:
                modes = positive
        if not modes:
            return None, "no_mode", 0

        distances = np.array([self.spectral_distance(mode, target_guess, ci_weight=ci_weight) for mode in modes])
        if previous_signature is None:
            chosen = modes[int(np.argmin(distances))]
            return chosen, "nearest_to_target", len(modes)

        best_distance = float(np.min(distances))
        window = max(best_distance * spectral_window_factor, best_distance + spectral_window_floor)
        candidate_indices = [idx for idx, dist in enumerate(distances) if dist <= window]
        candidates = [modes[idx] for idx in candidate_indices]
        if not candidates:
            chosen = modes[int(np.argmin(distances))]
            return chosen, "nearest_to_target", len(modes)

        # Keep shooting/target proximity as the primary criterion and use modal
        # overlap only to disambiguate the nearest candidates.
        candidates = sorted(
            candidates,
            key=lambda mode: self.spectral_distance(mode, target_guess, ci_weight=ci_weight),
        )[: max(1, overlap_top_k)]

        if previous_guess is None:
            chosen = max(
                candidates,
                key=lambda mode: (
                    self.signature_overlap(mode, previous_signature),
                    -self.spectral_distance(mode, target_guess, ci_weight=ci_weight),
                ),
            )
            return chosen, "target_then_overlap", len(modes)

        distance_scale = max(
            max(self.spectral_distance(mode, target_guess, ci_weight=ci_weight) for mode in candidates),
            1e-8,
        )
        jump_cr_scale = max(max(abs(mode["cr"] - previous_guess[0]) for mode in candidates), 1e-8)
        jump_ci_scale = max(max(abs(mode["ci"] - previous_guess[1]) for mode in candidates), 1e-8)

        def composite_score(mode: dict) -> tuple[float, float]:
            distance_term = self.spectral_distance(mode, target_guess, ci_weight=ci_weight) / distance_scale
            overlap_term = 1.0 - self.signature_overlap(mode, previous_signature)
            jump_cr_term = abs(mode["cr"] - previous_guess[0]) / jump_cr_scale
            jump_ci_term = abs(mode["ci"] - previous_guess[1]) / jump_ci_scale
            score = (
                distance_term
                + overlap_weight * overlap_term
                + jump_cr_weight * jump_cr_term
                + jump_ci_weight * jump_ci_term
            )
            return score, distance_term

        chosen = min(candidates, key=composite_score)
        return chosen, "composite_branch_score", len(modes)

    def solve_most_unstable(
        self,
        *,
        target_guess: tuple[float, float] | None = None,
        prefer_positive_cr: bool = True,
        cr_window: float = 0.4,
    ) -> NotebookStyleGEPResult:
        if self.Mach < 1.0 and self.alpha**2 + self.Mach**2 >= 1.0:
            return NotebookStyleGEPResult(
                alpha=self.alpha,
                Mach=self.Mach,
                c=0.0j,
                cr=0.0,
                ci=0.0,
                omega_i=0.0,
                n_finite_modes=0,
                selection_source="neutral_boundary",
                success=True,
            )

        chosen, selection_source, n_modes = self.get_selected_mode(
            target_guess=target_guess,
            prefer_positive_cr=prefer_positive_cr,
            cr_window=cr_window,
        )
        if chosen is None:
            return NotebookStyleGEPResult(
                alpha=self.alpha,
                Mach=self.Mach,
                c=0.0j,
                cr=0.0,
                ci=0.0,
                omega_i=0.0,
                n_finite_modes=0,
                selection_source=selection_source,
                success=False,
            )

        c = chosen["c"]
        return NotebookStyleGEPResult(
            alpha=self.alpha,
            Mach=self.Mach,
            c=c,
            cr=float(np.real(c)),
            ci=float(np.imag(c)),
            omega_i=float(self.alpha * np.imag(c)),
            n_finite_modes=n_modes,
            selection_source=selection_source,
            success=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solveur GEP dense, style notebook stagiaire, avec mapping PINN.")
    parser.add_argument("--alpha", type=float, default=0.181)
    parser.add_argument("--mach", type=float, default=1.33)
    parser.add_argument("--n-points", type=int, default=301)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=0.98)
    parser.add_argument("--target-cr", type=float, default=None)
    parser.add_argument("--target-ci", type=float, default=None)
    parser.add_argument("--cr-window", type=float, default=0.4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    solver = NotebookStyleDenseGEPSolver(
        alpha=args.alpha,
        Mach=args.mach,
        n_points=args.n_points,
        mapping_scale=args.mapping_scale,
        xi_max=args.xi_max,
    )
    target_guess = None
    if args.target_cr is not None and args.target_ci is not None:
        target_guess = (float(args.target_cr), float(args.target_ci))
    result = solver.solve_most_unstable(target_guess=target_guess, cr_window=args.cr_window)
    print(result)


if __name__ == "__main__":
    main()
