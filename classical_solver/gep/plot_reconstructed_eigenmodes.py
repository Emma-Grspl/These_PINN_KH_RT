from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def normalize_gep_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = vector[0:n_points]
    v = vector[n_points : 2 * n_points]
    p = vector[2 * n_points : 3 * n_points]
    rho = p * mach**2

    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def extract_shooting_mode(alpha: float, mach: float, cr: float, ci: float, mapping_scale: float) -> dict | None:
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        use_mapping=True,
        mapping_scale=mapping_scale,
    )
    result = solver.solve(
        cr_min=max(0.0, cr - 0.08),
        cr_max=cr + 0.08,
        ci_min=max(1e-3, ci - 0.03),
        ci_max=ci + 0.03,
        max_iter=10,
        grid_size=4,
    )
    if not result.spectral_success:
        return None

    sol_left, _, sol_right_full, _ = solver.get_trajectories(
        result.cr,
        result.ci,
        ln_p_start_right=result.ln_p_start_right,
    )
    if not (sol_left.success and sol_right_full.success):
        return None

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right_full.t)
    ln_p_left, phi_left = sol_left.y[2], sol_left.y[3]
    ln_p_right, phi_right = sol_right_full.y[2], sol_right_full.y[3]

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right_full, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))

    y = np.concatenate([y_left[y_left < 0.0], y_right[::-1]])
    p = np.concatenate([p_left[y_left < 0.0], p_right[::-1]])
    scale = max(np.max(np.abs(np.real(p))), np.max(np.abs(np.imag(p))), 1e-12)
    return {
        "y": y,
        "p": p / scale,
        "cr": result.cr,
        "ci": result.ci,
        "spectral_success": result.spectral_success,
        "mode_success": result.mode_success,
    }


def solve_gep_mode(
    alpha: float,
    mach: float,
    *,
    n_values: list[int],
    mapping_kind: str,
    mapping_scale: float,
    cubic_delta: float,
    xi_max: float,
    ci_weight: float,
) -> tuple[dict, NotebookStyleDenseGEPSolver, dict]:
    shooting = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        use_mapping=True,
        mapping_scale=mapping_scale,
    ).solve(
        cr_min=0.0,
        cr_max=min(0.7, max(0.3, 0.5 * mach)),
        ci_min=0.001,
        ci_max=0.12,
        max_iter=12,
        grid_size=4,
    )
    target = (shooting.cr, shooting.ci)

    attempts: list[dict] = []
    best_payload: tuple[dict, NotebookStyleDenseGEPSolver] | None = None
    best_distance = np.inf

    for n_points in n_values:
        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=n_points,
            mapping_kind=mapping_kind,
            mapping_scale=mapping_scale,
            cubic_delta=cubic_delta,
            xi_max=xi_max,
        )
        mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
            target_guess=target,
            ci_weight=ci_weight,
        )
        if mode is None:
            attempts.append(
                {
                    "alpha": alpha,
                    "Mach": mach,
                    "N": n_points,
                    "success": False,
                    "selection_source": selection_source,
                    "n_finite_modes": n_modes,
                }
            )
            continue

        distance = solver.spectral_distance(mode, target, ci_weight=ci_weight)
        payload = {
            "alpha": alpha,
            "Mach": mach,
            "N": n_points,
            "shooting_cr": shooting.cr,
            "shooting_ci": shooting.ci,
            "gep_cr": mode["cr"],
            "gep_ci": mode["ci"],
            "gep_omega_i": mode["omega_i"],
            "distance_to_shooting": distance,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "mode": mode,
        }
        attempts.append({key: value for key, value in payload.items() if key != "mode"})
        if distance < best_distance:
            best_distance = distance
            best_payload = (payload, solver)

    if best_payload is None:
        raise RuntimeError(f"Aucun mode GEP trouve pour alpha={alpha}, Mach={mach}.")
    payload, solver = best_payload
    return payload, solver, {"shooting": shooting, "attempts": attempts}


def plot_mode_page(
    pdf: PdfPages,
    row: dict,
    solver: NotebookStyleDenseGEPSolver,
    *,
    shooting_mode: dict | None,
) -> None:
    mode = row["mode"]
    u, v, p, rho = normalize_gep_mode(mode["vector"], solver.n_points, solver.Mach)
    fields = [
        (rho, r"Densite $\hat{\rho}$"),
        (u, r"Vitesse longitudinale $\hat{u}$"),
        (v, r"Vitesse transverse $\hat{v}$"),
        (p, r"Pression $\hat{p}$"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=False)
    axes = axes.ravel()

    for ax, (field, title) in zip(axes[:4], fields):
        ax.plot(solver.y, np.real(field), label="Re")
        ax.plot(solver.y, np.imag(field), "--", label="Im")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[4].plot(solver.y, np.real(p), label="GEP Re(p)")
    axes[4].plot(solver.y, np.imag(p), "--", label="GEP Im(p)")
    if shooting_mode is not None:
        axes[4].plot(shooting_mode["y"], np.real(shooting_mode["p"]), color="black", alpha=0.8, label="Tir Re(p)")
        axes[4].plot(shooting_mode["y"], np.imag(shooting_mode["p"]), color="gray", linestyle=":", alpha=0.8, label="Tir Im(p)")
    axes[4].set_title(r"Comparaison pression mode $\hat{p}$")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(solver.y, np.tanh(solver.y), label=r"$U(y)=\tanh(y)$")
    axes[5].axhline(row["gep_cr"], color="tab:red", linestyle="--", label=r"$c_r$ GEP")
    axes[5].axhline(row["shooting_cr"], color="black", linestyle=":", label=r"$c_r$ tir")
    axes[5].set_title("Profil moyen et vitesse de phase")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    title = (
        f"Modes reconstruits | alpha={row['alpha']:.3f}, M={row['Mach']:.3f}, N={row['N']}\n"
        f"GEP: c={row['gep_cr']:.5f}+i{row['gep_ci']:.5f} | "
        f"tir: c={row['shooting_cr']:.5f}+i{row['shooting_ci']:.5f} | "
        f"distance={row['distance_to_shooting']:.4e}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def parse_points(raw_points: list[str]) -> list[tuple[float, float]]:
    points = []
    for item in raw_points:
        alpha_str, mach_str = item.split(",")
        points.append((float(alpha_str), float(mach_str)))
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace des modes propres reconstruits pour des points GEP supersoniques.")
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Point sous la forme alpha,mach. Peut etre repete.",
    )
    parser.add_argument("--points-csv", type=Path, default=None, help="CSV contenant des colonnes alpha et Mach.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[401, 481, 561, 641])
    parser.add_argument("--mapping-kind", type=str, default="pin")
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.99)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--compare-shooting", action="store_true")
    parser.add_argument("--output-stem", type=str, default="supersonic_reconstructed_eigenmodes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    points = parse_points(args.point)
    if args.points_csv is not None:
        df_points = pd.read_csv(args.points_csv)
        if not {"alpha", "Mach"}.issubset(df_points.columns):
            raise ValueError("Le CSV doit contenir les colonnes alpha et Mach.")
        points.extend((float(row["alpha"]), float(row["Mach"])) for _, row in df_points.iterrows())

    if not points:
        points = [(0.18, 1.10), (0.24, 1.20), (0.18, 1.70)]

    pdf_path = OUTPUT_DIR / f"{args.output_stem}.pdf"
    csv_path = OUTPUT_DIR / f"{args.output_stem}.csv"

    rows: list[dict] = []
    with PdfPages(pdf_path) as pdf:
        for alpha, mach in points:
            row, solver, extras = solve_gep_mode(
                alpha,
                mach,
                n_values=args.n_values,
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
                ci_weight=args.ci_weight,
            )
            shooting_mode = None
            if args.compare_shooting:
                shooting_mode = extract_shooting_mode(alpha, mach, row["shooting_cr"], row["shooting_ci"], args.mapping_scale)
            plot_mode_page(pdf, row, solver, shooting_mode=shooting_mode)
            rows.append({key: value for key, value in row.items() if key != "mode"})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nModes PDF: {pdf_path}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
