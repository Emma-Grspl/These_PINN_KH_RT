from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import DEFAULT_OUTPUT_DIR  # noqa: E402


DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "supersonic_shooting_ci_map_a020_m120_130_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation visuelle supersonique du shooting : eigenvaleurs vs Blumen, mode principal, partenaire et reconstruction."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--threshold-ratio", type=float, default=0.02)
    parser.add_argument("--min-half-width", type=float, default=8.0)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def infer_summary_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    triplets = [
        ("best_shooting_cr", "best_shooting_ci", "best_ln_p_start_right"),
        ("shooting_cr", "shooting_ci", "ln_p_start_right"),
        ("cr", "ci", "ln_p_start_right"),
    ]
    for cr_col, ci_col, ln_col in triplets:
        if {cr_col, ci_col, ln_col}.issubset(df.columns):
            return cr_col, ci_col, ln_col
    raise ValueError("Impossible d'identifier les colonnes cr/ci/ln_p_start_right dans le summary CSV.")


def parse_bool_like(value: object, *, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def compute_visible_xlim(
    y: np.ndarray,
    fields: list[np.ndarray],
    *,
    threshold_ratio: float,
    min_half_width: float,
) -> tuple[float, float]:
    envelope = np.zeros_like(y, dtype=float)
    for field in fields:
        envelope = np.maximum(envelope, np.abs(np.real(field)))
        envelope = np.maximum(envelope, np.abs(np.imag(field)))

    peak = float(np.max(envelope))
    if peak <= 0.0:
        return float(y[0]), float(y[-1])

    mask = envelope >= threshold_ratio * peak
    if not np.any(mask):
        return float(y[0]), float(y[-1])

    y_vis = y[mask]
    half_width = max(float(np.max(np.abs(y_vis))), float(min_half_width))
    return -half_width, half_width


def normalize_fields(
    *,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u, v, p, rho = u * phase, v * phase, p * phase, rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u, v, p, rho = -u, -v, -p, -rho

    scale = max(np.max(np.abs(np.real(rho))), np.max(np.abs(np.imag(rho))), 1e-12)
    return u / scale, v / scale, p / scale, rho / scale


def reconstruct_shooting_fields(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    match_y: float,
    use_mapping: bool,
    mapping_scale: float,
) -> dict[str, np.ndarray]:
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=match_y,
        use_mapping=use_mapping,
        mapping_scale=mapping_scale,
    )
    sol_left, _, sol_right_full, _ = solver.get_trajectories(cr, ci, ln_p_start_right=ln_p_start_right)
    if not (sol_left.success and sol_right_full.success):
        raise RuntimeError(
            f"Echec de reconstruction du mode shooting pour alpha={alpha:.3f}, M={mach:.3f}."
        )

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right_full.t)

    k_left, q_left, ln_p_left, phi_left = sol_left.y
    k_right, q_right, ln_p_right, phi_right = sol_right_full.y

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right_full, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    left_mask = y_left < 0.0
    y = np.concatenate([y_left[left_mask], y_right[::-1]])
    p = np.concatenate([p_left[left_mask], p_right[::-1]])
    gamma = np.concatenate([gamma_left[left_mask], gamma_right[::-1]])

    u_bar = solver.base_velocity(y)
    du_bar = solver.base_velocity_derivative(y)
    c = complex(cr, ci)
    i_alpha = 1j * float(alpha)

    p_y = gamma * p
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = (float(mach) ** 2) * p

    u, v, p, rho = normalize_fields(y=y, u=u, v=v, p=p, rho=rho)
    return {"y": y, "rho": rho, "u": u, "v": v, "p": p}


def build_partner_and_reconstruction(fields: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    partner = {
        "y": fields["y"].copy(),
        "rho": np.conj(fields["rho"]),
        "u": np.conj(fields["u"]),
        "v": np.conj(fields["v"]),
        "p": np.conj(fields["p"]),
    }

    reconstructed = {"y": fields["y"].copy()}
    for name in ("rho", "u", "v", "p"):
        reconstructed[name] = fields[name] + partner[name]

    scale = max(np.max(np.abs(np.real(reconstructed["rho"]))), 1e-12)
    for name in ("rho", "u", "v", "p"):
        reconstructed[name] = reconstructed[name] / scale

    return partner, reconstructed


def plot_eigenvalue_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    mach = summary_df["Mach"].to_numpy(dtype=float)

    axes[0, 0].plot(mach, summary_df["blumen_cr"], marker="o", label="Blumen")
    axes[0, 0].plot(mach, summary_df["shooting_cr"], marker="s", label="shooting")
    axes[0, 0].set_title(r"$c_r$")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(mach, summary_df["blumen_ci"], marker="o", label="Blumen")
    axes[0, 1].plot(mach, summary_df["shooting_ci"], marker="s", label="shooting")
    axes[0, 1].set_title(r"$c_i$")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(mach, np.abs(summary_df["shooting_cr"] - summary_df["blumen_cr"]), marker="o")
    axes[1, 0].set_title(r"$|c_r^{shoot} - c_r^{Blumen}|$")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(mach, np.abs(summary_df["shooting_ci"] - summary_df["blumen_ci"]), marker="o")
    axes[1, 1].set_title(r"$|c_i^{shoot} - c_i^{Blumen}|$")
    axes[1, 1].grid(True, alpha=0.25)

    for ax in axes.ravel():
        ax.set_xlabel("Mach")

    alpha_values = sorted(float(value) for value in summary_df["alpha"].unique())
    alpha_label = ", ".join(f"{value:.3f}" for value in alpha_values)
    fig.suptitle(f"Shooting supersonique vs Blumen | alpha={alpha_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_visual_validation_pdf(
    *,
    summary_df: pd.DataFrame,
    field_pages: list[dict[str, object]],
    output_path: Path,
    threshold_ratio: float,
    min_half_width: float,
) -> None:
    field_names = ["rho", "u", "v", "p"]
    field_titles = [
        r"Density Perturbation $\hat{\rho}$",
        r"Streamwise Velocity $\hat{u}$",
        r"Vertical Velocity $\hat{v}$",
        r"Pressure Perturbation $\hat{p}$",
    ]

    with PdfPages(output_path) as pdf:
        for page in field_pages:
            row = page["summary_row"]
            principal = page["principal"]
            partner = page["partner"]
            reconstructed = page["reconstructed"]
            y = principal["y"]

            x_limits = compute_visible_xlim(
                y,
                [principal[name] for name in field_names],
                threshold_ratio=threshold_ratio,
                min_half_width=min_half_width,
            )

            fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
            rows = [
                ("Principal mode", principal, True),
                ("Conjugate partner", partner, True),
                ("Reconstructed physical mode", reconstructed, False),
            ]
            for j, title in enumerate(field_titles):
                axes[0, j].set_title(title)

            for i, (row_label, fields, with_imag) in enumerate(rows):
                for j, field_name in enumerate(field_names):
                    ax = axes[i, j]
                    field = fields[field_name]
                    ax.plot(y, np.real(field), color="black", linewidth=1.8, label="Real")
                    if with_imag:
                        ax.plot(y, np.imag(field), color="#D97706", linestyle="--", linewidth=1.3, label="Imag")
                    else:
                        ax.plot(y, np.imag(field), color="#D97706", linestyle="--", linewidth=1.0, alpha=0.45, label="Imag")
                    ax.axvline(0.0, color="#9CA3AF", linewidth=1.0, alpha=0.6)
                    ax.set_xlim(*x_limits)
                    ax.grid(True, alpha=0.25)
                    if i == 0 and j == 0:
                        ax.legend(frameon=False, fontsize=8)
                    if j == 0:
                        ax.set_ylabel(row_label)
                    if i == 2:
                        ax.set_xlabel("y")

            fig.suptitle(
                f"alpha={float(row['alpha']):.3f}, M={float(row['Mach']):.3f}\n"
                f"Blumen=(c_r={float(row['blumen_cr']):.5f}, c_i={float(row['blumen_ci']):.5f}) | "
                f"shooting=(c_r={float(row['shooting_cr']):.5f}, c_i={float(row['shooting_ci']):.5f})\n"
                "Rows: principal mode, conjugate partner representation, reconstructed physical mode at t=0"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    summary_df = pd.read_csv(args.summary_csv).copy()
    cr_col, ci_col, ln_col = infer_summary_columns(summary_df)

    required = {"alpha", "Mach", "blumen_cr", "blumen_ci", cr_col, ci_col, ln_col}
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {missing}")

    summary_out = summary_df.rename(
        columns={
            cr_col: "shooting_cr",
            ci_col: "shooting_ci",
            ln_col: "ln_p_start_right",
        }
    ).copy()

    field_rows: list[dict[str, float | str]] = []
    visual_summary_rows: list[dict[str, float]] = []
    field_pages: list[dict[str, object]] = []

    for _, row in summary_out.sort_values(["alpha", "Mach"]).iterrows():
        principal = reconstruct_shooting_fields(
            alpha=float(row["alpha"]),
            mach=float(row["Mach"]),
            cr=float(row["shooting_cr"]),
            ci=float(row["shooting_ci"]),
            ln_p_start_right=float(row["ln_p_start_right"]),
            match_y=float(row["match_y"]) if "match_y" in row.index and pd.notna(row["match_y"]) else 1.0,
            use_mapping=parse_bool_like(row["use_mapping"], default=True) if "use_mapping" in row.index else True,
            mapping_scale=float(row["mapping_scale"]) if "mapping_scale" in row.index and pd.notna(row["mapping_scale"]) else 5.0,
        )
        partner, reconstructed = build_partner_and_reconstruction(principal)

        field_pages.append(
            {
                "summary_row": row,
                "principal": principal,
                "partner": partner,
                "reconstructed": reconstructed,
            }
        )

        x_left, x_right = compute_visible_xlim(
            principal["y"],
            [principal["rho"], principal["u"], principal["v"], principal["p"]],
            threshold_ratio=float(args.threshold_ratio),
            min_half_width=float(args.min_half_width),
        )
        visual_summary_rows.append(
            {
                "alpha": float(row["alpha"]),
                "Mach": float(row["Mach"]),
                "blumen_cr": float(row["blumen_cr"]),
                "blumen_ci": float(row["blumen_ci"]),
                "shooting_cr": float(row["shooting_cr"]),
                "shooting_ci": float(row["shooting_ci"]),
                "err_cr_abs": abs(float(row["shooting_cr"]) - float(row["blumen_cr"])),
                "err_ci_abs": abs(float(row["shooting_ci"]) - float(row["blumen_ci"])),
                "visible_xmin": float(x_left),
                "visible_xmax": float(x_right),
                "principal_peak_y_p": float(principal["y"][int(np.argmax(np.abs(principal["p"])))]),
                "principal_peak_y_rho": float(principal["y"][int(np.argmax(np.abs(principal["rho"])))]),
                "reconstructed_peak_y_rho": float(reconstructed["y"][int(np.argmax(np.abs(np.real(reconstructed["rho"]))))]),
            }
        )

        for mode_name, fields in (
            ("principal", principal),
            ("partner", partner),
            ("reconstructed", reconstructed),
        ):
            for y_value, rho_value, u_value, v_value, p_value in zip(
                fields["y"], fields["rho"], fields["u"], fields["v"], fields["p"]
            ):
                field_rows.append(
                    {
                        "alpha": float(row["alpha"]),
                        "Mach": float(row["Mach"]),
                        "mode_family": mode_name,
                        "y": float(y_value),
                        "rho_real": float(np.real(rho_value)),
                        "rho_imag": float(np.imag(rho_value)),
                        "u_real": float(np.real(u_value)),
                        "u_imag": float(np.imag(u_value)),
                        "v_real": float(np.real(v_value)),
                        "v_imag": float(np.imag(v_value)),
                        "p_real": float(np.real(p_value)),
                        "p_imag": float(np.imag(p_value)),
                    }
                )

    visual_summary_df = pd.DataFrame(visual_summary_rows)
    fields_df = pd.DataFrame(field_rows)

    summary_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_summary.csv"
    fields_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_fields.csv"
    eig_png_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_eigenvalues_vs_blumen.png"
    pdf_path = DEFAULT_OUTPUT_DIR / f"{args.output_stem}_visual_validation.pdf"

    visual_summary_df.to_csv(summary_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    plot_eigenvalue_comparison(visual_summary_df, eig_png_path)
    plot_visual_validation_pdf(
        summary_df=visual_summary_df,
        field_pages=field_pages,
        output_path=pdf_path,
        threshold_ratio=float(args.threshold_ratio),
        min_half_width=float(args.min_half_width),
    )

    print("Supersonic shooting visual validation summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(visual_summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {fields_path}")
    print(f"Wrote {eig_png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
