from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


OUTPUT_DIR = ROOT_DIR / "assets" / "blumen_gep"


def normalize_mode(vector: np.ndarray, n_points: int, mach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def interpolate_complex(field: np.ndarray, y_native: np.ndarray, y_export: np.ndarray) -> np.ndarray:
    real = np.interp(y_export, y_native, np.real(field))
    imag = np.interp(y_export, y_native, np.imag(field))
    return real + 1j * imag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit une base de donnees de modes GEP supersoniques.")
    parser.add_argument("--mach-min", type=float, required=True)
    parser.add_argument("--mach-max", type=float, required=True)
    parser.add_argument("--num-mach", type=int, default=21)
    parser.add_argument("--alpha-min", type=float, required=True)
    parser.add_argument("--alpha-max", type=float, required=True)
    parser.add_argument("--num-alpha", type=int, default=21)
    parser.add_argument("--n-values", type=int, nargs="+", default=[561])
    parser.add_argument("--mapping-kind", type=str, choices=["pin", "cubic"], default="pin")
    parser.add_argument("--mapping-scale", type=float, default=1.5)
    parser.add_argument("--cubic-delta", type=float, default=0.2)
    parser.add_argument("--xi-max", type=float, default=0.90)
    parser.add_argument("--distance-tol", type=float, default=0.02)
    parser.add_argument("--ci-weight", type=float, default=2.0)
    parser.add_argument("--previous-weight", type=float, default=0.6)
    parser.add_argument("--y-points", type=int, default=561)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-stem", type=str, default="supersonic_mode_database")
    return parser


def solve_point(
    *,
    alpha: float,
    mach: float,
    n_values: list[int],
    mapping_kind: str,
    mapping_scale: float,
    cubic_delta: float,
    xi_max: float,
    ci_weight: float,
    distance_tol: float,
    target_guess: tuple[float, float],
    shooting_guess: tuple[float, float],
    previous_signature: np.ndarray | None,
) -> tuple[dict, NotebookStyleDenseGEPSolver | None, dict | None]:
    best_row: dict | None = None
    best_solver: NotebookStyleDenseGEPSolver | None = None
    best_mode: dict | None = None
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
        mode, selection_source, n_modes = solver.get_branch_mode(
            target_guess=target_guess,
            previous_signature=previous_signature,
            prefer_positive_cr=True,
            ci_weight=ci_weight,
        )
        if mode is None:
            continue

        dist_target = solver.spectral_distance(mode, target_guess, ci_weight=ci_weight)
        dist_shooting = solver.spectral_distance(mode, shooting_guess, ci_weight=ci_weight)
        overlap = np.nan if previous_signature is None else solver.signature_overlap(mode, previous_signature)
        row = {
            "alpha": alpha,
            "Mach": mach,
            "N": n_points,
            "gep_cr": mode["cr"],
            "gep_ci": mode["ci"],
            "gep_omega_i": mode["omega_i"],
            "distance_to_target": dist_target,
            "distance_to_shooting": dist_shooting,
            "overlap_to_previous": overlap,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": True,
            "accepted": dist_shooting <= distance_tol,
        }
        if dist_shooting < best_distance:
            best_distance = dist_shooting
            best_row = row
            best_solver = solver
            best_mode = mode

        if row["accepted"]:
            return row, solver, mode

    if best_row is None:
        return {
            "alpha": alpha,
            "Mach": mach,
            "N": np.nan,
            "gep_cr": np.nan,
            "gep_ci": np.nan,
            "gep_omega_i": np.nan,
            "distance_to_target": np.nan,
            "distance_to_shooting": np.nan,
            "overlap_to_previous": np.nan,
            "selection_source": "no_mode",
            "n_finite_modes": 0,
            "success": False,
            "accepted": False,
        }, None, None

    return best_row, best_solver, best_mode


def plot_isolines(df: pd.DataFrame, png_path: Path) -> None:
    valid = df[df["success"]].copy()
    if valid.empty:
        return

    pivot_ci = valid.pivot(index="Mach", columns="alpha", values="gep_ci").sort_index()
    pivot_cr = valid.pivot(index="Mach", columns="alpha", values="gep_cr").sort_index()
    pivot_acc = valid.pivot(index="Mach", columns="alpha", values="accepted").sort_index()

    alpha_grid = pivot_ci.columns.to_numpy(dtype=float)
    mach_grid = pivot_ci.index.to_numpy(dtype=float)
    A, M = np.meshgrid(alpha_grid, mach_grid)
    ci_grid = pivot_ci.to_numpy(dtype=float)
    cr_grid = pivot_cr.to_numpy(dtype=float)
    accepted = pivot_acc.to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    levels_ci = np.linspace(np.nanmin(ci_grid), np.nanmax(ci_grid), 12)
    levels_cr = np.linspace(np.nanmin(cr_grid), np.nanmax(cr_grid), 12)

    cf0 = axes[0].contourf(A, M, ci_grid, levels=levels_ci, cmap="viridis")
    cs0 = axes[0].contour(A, M, ci_grid, levels=levels_ci[::2], colors="white", linewidths=0.8)
    axes[0].clabel(cs0, inline=True, fontsize=8, fmt="%.3f")
    axes[0].contour(A, M, accepted, levels=[0.5], colors="red", linewidths=1.2)
    axes[0].set_title(r"Isolignes de $c_i$")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$M$")
    fig.colorbar(cf0, ax=axes[0], label=r"$c_i$")

    cf1 = axes[1].contourf(A, M, cr_grid, levels=levels_cr, cmap="magma")
    cs1 = axes[1].contour(A, M, cr_grid, levels=levels_cr[::2], colors="white", linewidths=0.8)
    axes[1].clabel(cs1, inline=True, fontsize=8, fmt="%.3f")
    axes[1].contour(A, M, accepted, levels=[0.5], colors="cyan", linewidths=1.2)
    axes[1].set_title(r"Isolignes de $c_r$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$M$")
    fig.colorbar(cf1, ax=axes[1], label=r"$c_r$")

    fig.suptitle("Base GEP supersonique figee")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_checkpoint_payload(
    *,
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    summary_df: pd.DataFrame,
    modes_df: pd.DataFrame,
    mode_fields: dict[str, list[np.ndarray]],
    y_export: np.ndarray | None,
) -> None:
    summary_df.to_csv(checkpoint_meta, index=False)
    payload: dict[str, np.ndarray] = {
        "summary_alpha": summary_df["alpha"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_mach": summary_df["Mach"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_gep_cr": summary_df["gep_cr"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_gep_ci": summary_df["gep_ci"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_gep_omega_i": summary_df["gep_omega_i"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_distance_to_shooting": summary_df["distance_to_shooting"].to_numpy(dtype=np.float64) if not summary_df.empty else np.asarray([], dtype=np.float64),
        "summary_success": summary_df["success"].to_numpy(dtype=bool) if not summary_df.empty else np.asarray([], dtype=bool),
        "summary_accepted": summary_df["accepted"].to_numpy(dtype=bool) if not summary_df.empty else np.asarray([], dtype=bool),
        "mode_alpha": modes_df["alpha"].to_numpy(dtype=np.float64) if not modes_df.empty else np.asarray([], dtype=np.float64),
        "mode_mach": modes_df["Mach"].to_numpy(dtype=np.float64) if not modes_df.empty else np.asarray([], dtype=np.float64),
        "mode_accepted": modes_df["accepted"].to_numpy(dtype=bool) if not modes_df.empty else np.asarray([], dtype=bool),
        "mode_gep_cr": modes_df["gep_cr"].to_numpy(dtype=np.float64) if not modes_df.empty else np.asarray([], dtype=np.float64),
        "mode_gep_ci": modes_df["gep_ci"].to_numpy(dtype=np.float64) if not modes_df.empty else np.asarray([], dtype=np.float64),
        "mode_distance_to_shooting": modes_df["distance_to_shooting"].to_numpy(dtype=np.float64) if not modes_df.empty else np.asarray([], dtype=np.float64),
    }
    if y_export is not None:
        payload["y"] = y_export.astype(np.float64)
    if mode_fields["u"]:
        payload["u"] = np.asarray(mode_fields["u"], dtype=np.complex128)
        payload["v"] = np.asarray(mode_fields["v"], dtype=np.complex128)
        payload["p"] = np.asarray(mode_fields["p"], dtype=np.complex128)
        payload["rho"] = np.asarray(mode_fields["rho"], dtype=np.complex128)
    np.savez_compressed(checkpoint_npz, **payload)


def load_checkpoint_payload(
    *,
    checkpoint_npz: Path,
    checkpoint_meta: Path,
) -> tuple[list[dict], list[dict], dict[str, list[np.ndarray]], np.ndarray | None]:
    summary_rows: list[dict] = []
    mode_rows: list[dict] = []
    mode_fields: dict[str, list[np.ndarray]] = {"u": [], "v": [], "p": [], "rho": []}
    y_export: np.ndarray | None = None

    if checkpoint_meta.exists():
        summary_df = pd.read_csv(checkpoint_meta)
        summary_rows = summary_df.to_dict(orient="records")

    if checkpoint_npz.exists():
        data = np.load(checkpoint_npz, allow_pickle=False)
        if "y" in data:
            y_export = data["y"]
        if "mode_alpha" in data and data["mode_alpha"].size:
            mode_rows = [
                {
                    "mode_index": idx,
                    "alpha": float(data["mode_alpha"][idx]),
                    "Mach": float(data["mode_mach"][idx]),
                    "accepted": bool(data["mode_accepted"][idx]),
                    "gep_cr": float(data["mode_gep_cr"][idx]),
                    "gep_ci": float(data["mode_gep_ci"][idx]),
                    "distance_to_shooting": float(data["mode_distance_to_shooting"][idx]),
                }
                for idx in range(data["mode_alpha"].shape[0])
            ]
            for key in ("u", "v", "p", "rho"):
                if key in data:
                    mode_fields[key] = [arr for arr in data[key]]
    return summary_rows, mode_rows, mode_fields, y_export


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)
    machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)

    summary_csv = OUTPUT_DIR / f"{args.output_stem}_surface.csv"
    modes_csv = OUTPUT_DIR / f"{args.output_stem}_modes.csv"
    modes_npz = OUTPUT_DIR / f"{args.output_stem}_modes.npz"
    png_path = OUTPUT_DIR / f"{args.output_stem}_isolines.png"
    checkpoint_meta = OUTPUT_DIR / f"{args.output_stem}_checkpoint_summary.csv"
    checkpoint_npz = OUTPUT_DIR / f"{args.output_stem}_checkpoint_modes.npz"

    if args.resume:
        summary_rows, mode_rows, mode_fields, y_export = load_checkpoint_payload(
            checkpoint_npz=checkpoint_npz,
            checkpoint_meta=checkpoint_meta,
        )
    else:
        summary_rows = []
        mode_rows = []
        mode_fields = {"u": [], "v": [], "p": [], "rho": []}
        y_export = None

    completed_pairs = {(round(float(row["alpha"]), 12), round(float(row["Mach"]), 12)) for row in summary_rows}

    for mach in machs:
        previous_gep: tuple[float, float] | None = None
        previous_signature: np.ndarray | None = None

        for idx, alpha in enumerate(alphas):
            key = (round(float(alpha), 12), round(float(mach), 12))
            if key in completed_pairs:
                existing = [row for row in summary_rows if round(float(row["alpha"]), 12) == key[0] and round(float(row["Mach"]), 12) == key[1]]
                if existing:
                    row0 = existing[-1]
                    if bool(row0.get("success", False)):
                        previous_gep = (float(row0["gep_cr"]), float(row0["gep_ci"]))
                continue

            shooting = Mstab17SupersonicSolver(alpha=float(alpha), Mach=float(mach)).solve(
                cr_min=0.03,
                cr_max=min(0.7, max(0.35, 0.5 * mach)),
                ci_min=0.001,
                ci_max=0.12,
                max_iter=10,
            )
            shooting_guess = (shooting.cr, shooting.ci)

            if idx == 0 or previous_gep is None:
                target_guess = shooting_guess
                target_source = "shooting_anchor"
            else:
                target_guess = (
                    args.previous_weight * previous_gep[0] + (1.0 - args.previous_weight) * shooting_guess[0],
                    args.previous_weight * previous_gep[1] + (1.0 - args.previous_weight) * shooting_guess[1],
                )
                target_source = "blended_continuation"

            chosen, solver, mode = solve_point(
                alpha=float(alpha),
                mach=float(mach),
                n_values=list(args.n_values),
                mapping_kind=args.mapping_kind,
                mapping_scale=args.mapping_scale,
                cubic_delta=args.cubic_delta,
                xi_max=args.xi_max,
                ci_weight=args.ci_weight,
                distance_tol=args.distance_tol,
                target_guess=target_guess,
                shooting_guess=shooting_guess,
                previous_signature=previous_signature,
            )

            summary_row = dict(chosen)
            summary_row["target_cr"] = target_guess[0]
            summary_row["target_ci"] = target_guess[1]
            summary_row["target_source"] = target_source
            summary_row["shooting_cr"] = shooting.cr
            summary_row["shooting_ci"] = shooting.ci
            summary_row["shooting_omega_i"] = shooting.omega_i
            summary_row["shooting_spectral_success"] = shooting.spectral_success
            summary_rows.append(summary_row)
            completed_pairs.add(key)

            if not chosen["success"] or solver is None or mode is None:
                write_checkpoint_payload(
                    checkpoint_npz=checkpoint_npz,
                    checkpoint_meta=checkpoint_meta,
                    summary_df=pd.DataFrame(summary_rows),
                    modes_df=pd.DataFrame(mode_rows),
                    mode_fields=mode_fields,
                    y_export=y_export,
                )
                continue
            if args.accepted_only and not chosen["accepted"]:
                write_checkpoint_payload(
                    checkpoint_npz=checkpoint_npz,
                    checkpoint_meta=checkpoint_meta,
                    summary_df=pd.DataFrame(summary_rows),
                    modes_df=pd.DataFrame(mode_rows),
                    mode_fields=mode_fields,
                    y_export=y_export,
                )
                continue

            if y_export is None:
                y_export = np.linspace(float(np.min(solver.y)), float(np.max(solver.y)), args.y_points)

            u, v, p, rho = normalize_mode(mode["vector"], solver.n_points, mach)
            mode_fields["u"].append(interpolate_complex(u, solver.y, y_export))
            mode_fields["v"].append(interpolate_complex(v, solver.y, y_export))
            mode_fields["p"].append(interpolate_complex(p, solver.y, y_export))
            mode_fields["rho"].append(interpolate_complex(rho, solver.y, y_export))
            mode_rows.append(
                {
                    "mode_index": len(mode_rows),
                    "alpha": alpha,
                    "Mach": mach,
                    "accepted": chosen["accepted"],
                    "gep_cr": chosen["gep_cr"],
                    "gep_ci": chosen["gep_ci"],
                    "gep_omega_i": chosen["gep_omega_i"],
                    "distance_to_shooting": chosen["distance_to_shooting"],
                    "selection_source": chosen["selection_source"],
                    "N": chosen["N"],
                }
            )

            previous_gep = (chosen["gep_cr"], chosen["gep_ci"])
            previous_signature = mode.get("signature")
            write_checkpoint_payload(
                checkpoint_npz=checkpoint_npz,
                checkpoint_meta=checkpoint_meta,
                summary_df=pd.DataFrame(summary_rows),
                modes_df=pd.DataFrame(mode_rows),
                mode_fields=mode_fields,
                y_export=y_export,
            )

    summary_df = pd.DataFrame(summary_rows)
    modes_df = pd.DataFrame(mode_rows)

    summary_df.to_csv(summary_csv, index=False)
    modes_df.to_csv(modes_csv, index=False)
    plot_isolines(summary_df, png_path)

    if y_export is not None and mode_rows:
        payload = {
            "y": y_export.astype(np.float64),
            "alpha": modes_df["alpha"].to_numpy(dtype=np.float64),
            "Mach": modes_df["Mach"].to_numpy(dtype=np.float64),
            "accepted": modes_df["accepted"].to_numpy(dtype=bool),
            "gep_cr": modes_df["gep_cr"].to_numpy(dtype=np.float64),
            "gep_ci": modes_df["gep_ci"].to_numpy(dtype=np.float64),
            "distance_to_shooting": modes_df["distance_to_shooting"].to_numpy(dtype=np.float64),
            "u": np.asarray(mode_fields["u"], dtype=np.complex128),
            "v": np.asarray(mode_fields["v"], dtype=np.complex128),
            "p": np.asarray(mode_fields["p"], dtype=np.complex128),
            "rho": np.asarray(mode_fields["rho"], dtype=np.complex128),
        }
        np.savez_compressed(modes_npz, **payload)

    progress = {
        "completed_points": int(len(summary_rows)),
        "stored_modes": int(len(mode_rows)),
        "total_points": int(len(alphas) * len(machs)),
        "output_stem": args.output_stem,
    }
    (OUTPUT_DIR / f"{args.output_stem}_progress.json").write_text(json.dumps(progress, indent=2))

    print(summary_df.to_string(index=False))
    print(f"\nSurface CSV: {summary_csv}")
    print(f"Modes CSV: {modes_csv}")
    if y_export is not None and mode_rows:
        print(f"Modes NPZ: {modes_npz}")
    print(f"Isoline figure: {png_path}")


if __name__ == "__main__":
    main()
