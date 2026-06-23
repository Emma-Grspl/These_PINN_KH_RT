from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kh_stage1bis_article_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager

font_manager._get_macos_fonts = lambda: []
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver  # noqa: E402
from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver  # noqa: E402
from src.models.kh_subsonic_pinn import build_fixed_mach_model_from_config, load_fixed_mach_state_dict_compat  # noqa: E402
from src.physics.kh_subsonic_residual import (  # noqa: E402
    base_velocity,
    base_velocity_derivative,
    reconstruct_pressure_p_y_from_riccati,
    xi_to_y,
)


EPS = 1.0e-8
DEFAULT_NUM_ALPHA = 71
DEFAULT_N_Y = 1201
FIELD_ORDER = ["p", "rho", "v", "u"]
FIELD_TITLES = {
    "p": r"Pressure $\hat{p}$",
    "rho": r"Density $\hat{\rho}$",
    "v": r"Transverse velocity $\hat{v}$",
    "u": r"Streamwise velocity $\hat{u}$",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    short_label: str
    color: str
    required: bool
    candidate_dirs: tuple[Path, ...]


RUN_SPECS = (
    RunSpec(
        key="pure_physics",
        label="Pure physics",
        short_label="pure physics",
        color="#c84c09",
        required=True,
        candidate_dirs=(),
    ),
    RunSpec(
        key="hybrid_ci4",
        label="Hybrid ci4",
        short_label="ci4",
        color="#1f77b4",
        required=True,
        candidate_dirs=(Path("model_saved/kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined/hybrid_ci4"),),
    ),
    RunSpec(
        key="hybrid_ci8",
        label="Hybrid ci8",
        short_label="ci8",
        color="#0b6e4f",
        required=False,
        candidate_dirs=(Path("model_saved/kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined/hybrid_ci8"),),
    ),
    RunSpec(
        key="hybrid_ci16",
        label="Hybrid ci16",
        short_label="ci16",
        color="#8e44ad",
        required=False,
        candidate_dirs=(Path("model_saved/kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined/hybrid_ci16"),),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render final article assets for subsonic Stage 1bis M=0.5 alpha sweep."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.80)
    parser.add_argument("--num-alpha", type=int, default=DEFAULT_NUM_ALPHA)
    parser.add_argument("--modal-alphas", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/pinn_subsonic/stage1bis_article"),
    )
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--n-y", type=int, default=DEFAULT_N_Y)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )


def write_dual_figure(fig: plt.Figure, png_path: Path, *, save_pdf: bool) -> list[Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    outputs = [png_path]
    if save_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs.append(pdf_path)
    plt.close(fig)
    return outputs


def validate_run_dir(path: Path) -> bool:
    return (path / "config.csv").is_file() and (path / "history.csv").is_file() and (path / "model_best.pt").is_file()


def find_pure_physics_run() -> tuple[Path | None, str]:
    direct_matches = sorted(
        {
            path.parent
            for path in ROOT_DIR.glob("model_saved/**/*")
            if path.is_file() and path.name == "model_best.pt" and "pure_physics_reference" in str(path.parent)
        }
    )
    for candidate in direct_matches:
        if validate_run_dir(candidate):
            return candidate, "model_saved"

    archived_matches = sorted(
        {
            path.parent
            for path in ROOT_DIR.glob("assets/pinn_subsonic/experiment_*/model_saved/**/*")
            if path.is_file() and path.name == "model_best.pt" and "pure_physics_reference" in str(path.parent)
        }
    )
    for candidate in archived_matches:
        if validate_run_dir(candidate):
            return candidate, "archived_experiment"

    frozen = ROOT_DIR / "assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current"
    if validate_run_dir(frozen):
        return frozen, "frozen_fallback"
    return None, "missing"


def resolve_run_dir(spec: RunSpec) -> tuple[Path | None, str]:
    if spec.key == "pure_physics":
        return find_pure_physics_run()
    for rel_path in spec.candidate_dirs:
        candidate = ROOT_DIR / rel_path
        if validate_run_dir(candidate):
            return candidate, "direct"
    return None, "missing"


def load_model(run_dir: Path, device: torch.device):
    config = pd.read_csv(run_dir / "config.csv").iloc[0]
    history = pd.read_csv(run_dir / "history.csv")
    model = build_fixed_mach_model_from_config(config)
    state_dict = torch.load(run_dir / "model_best.pt", map_location=device)
    load_fixed_mach_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()
    return model, config, history


def solve_reference_curve(mach: float, alpha_values: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        result = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(mach)).solve()
        rows.append({"alpha": float(alpha), "ci_classic": float(result.ci)})
    return pd.DataFrame(rows)


def predict_ci_curve(model, alpha_values: np.ndarray, device: torch.device) -> np.ndarray:
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_values = model.get_ci(alpha_tensor).detach().cpu().numpy().reshape(-1)
    return ci_values


def normalize_full_mode(y: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray, rho: np.ndarray) -> dict[str, np.ndarray]:
    idx = int(np.argmax(np.abs(rho)))
    if np.abs(rho[idx]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[idx]))
        u = u * phase
        v = v * phase
        p = p * phase
        rho = rho * phase
    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u = -u
        v = -v
        p = -p
        rho = -rho
    scale = max(
        float(np.max(np.abs(np.real(rho)))),
        float(np.max(np.abs(np.imag(rho)))),
        1e-12,
    )
    return {
        "y": np.asarray(y, dtype=float),
        "u": np.asarray(u / scale, dtype=np.complex128),
        "v": np.asarray(v / scale, dtype=np.complex128),
        "p": np.asarray(p / scale, dtype=np.complex128),
        "rho": np.asarray(rho / scale, dtype=np.complex128),
    }


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def load_classic_full_mode(alpha: float, mach: float) -> tuple[dict[str, np.ndarray], float]:
    solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
    result = solver.solve()
    sol_left, sol_right, _ = solver.get_trajectories(result.ci, ln_p_start_right=result.ln_p_start_right)

    y_left = np.asarray(sol_left.t)
    y_right = np.asarray(sol_right.t)
    k_left = np.asarray(sol_left.y[0])
    q_left = np.asarray(sol_left.y[1])
    ln_p_left = np.asarray(sol_left.y[2])
    phi_left = np.asarray(sol_left.y[3])
    k_right = np.asarray(sol_right.y[0])
    q_right = np.asarray(sol_right.y[1])
    ln_p_right = np.asarray(sol_right.y[2])
    phi_right = np.asarray(sol_right.y[3])

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)
    phi_left_0 = solver._interp_component(0.0, sol_left, 3)
    phi_right_0 = solver._interp_component(0.0, sol_right, 3)
    phase_shift = phi_left_0 - phi_right_0

    p_left = abs_p_left * np.exp(1j * phi_left)
    p_right = abs_p_right * np.exp(1j * (phi_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    mask_left = y_left < 0.0
    y = np.concatenate([y_left[mask_left], y_right[::-1]])
    p = np.concatenate([p_left[mask_left], p_right[::-1]])
    gamma = np.concatenate([gamma_left[mask_left], gamma_right[::-1]])

    p_y = gamma * p
    c = 1j * float(result.ci)
    u_bar = np.tanh(y)
    du_bar = 1.0 / np.cosh(y) ** 2
    i_alpha = 1j * float(alpha)
    v = -p_y / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p) / (i_alpha * (u_bar - c))
    rho = p * (float(mach) ** 2)
    return normalize_full_mode(y, u, v, p, rho), float(result.ci)


def load_pinn_full_mode(run_dir: Path, *, alpha: float, n_y: int, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    model, config, _history = load_model(run_dir, device)
    xi = torch.linspace(-0.98, 0.98, int(n_y), device=device).view(-1, 1)
    xi.requires_grad_(True)
    alpha_tensor = torch.full_like(xi, float(alpha))

    pr, pi, p_y, _gamma, y_t = reconstruct_pressure_p_y_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
    p = torch.complex(pr, pi)

    ci = float(model.get_ci(torch.tensor([[alpha]], dtype=torch.float32, device=device)).item())
    mach = float(config["mach"])
    c = 1j * ci
    y = y_t[:, 0]
    u_bar = base_velocity(y)
    du_bar = base_velocity_derivative(y)
    i_alpha = 1j * float(alpha)
    v = -p_y[:, 0] / (i_alpha * (u_bar - c))
    u = -(du_bar * v + i_alpha * p[:, 0]) / (i_alpha * (u_bar - c))
    rho = p[:, 0] * (mach**2)

    fields = normalize_full_mode(
        y.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        p[:, 0].detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    )
    return fields, ci


def common_grid(y_a: np.ndarray, y_b: np.ndarray, n_common: int = 1200) -> np.ndarray:
    y_min = max(float(np.min(y_a)), float(np.min(y_b)))
    y_max = min(float(np.max(y_a)), float(np.max(y_b)))
    return np.linspace(y_min, y_max, int(n_common), dtype=float)


def align_mode_to_reference(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], *, anchor_field: str = "rho") -> dict[str, np.ndarray]:
    y_common = common_grid(reference["y"], candidate["y"], n_common=1200)
    ref_field = interp_complex(reference["y"], reference[anchor_field], y_common)
    cand_field = interp_complex(candidate["y"], candidate[anchor_field], y_common)
    inner = np.vdot(cand_field, ref_field)
    phase = 1.0 + 0.0j if abs(inner) < 1e-14 else np.exp(1j * np.angle(inner))

    aligned = {"y": np.asarray(candidate["y"], dtype=float)}
    for field in FIELD_ORDER:
        aligned[field] = np.asarray(candidate[field] * phase, dtype=np.complex128)
    return aligned


def compute_mode_metrics(classic: dict[str, np.ndarray], pinn: dict[str, np.ndarray], *, n_common: int = 1200) -> dict[str, float]:
    y_common = common_grid(classic["y"], pinn["y"], n_common=n_common)

    def rel(field_name: str) -> float:
        ref = interp_complex(classic["y"], classic[field_name], y_common)
        pred = interp_complex(pinn["y"], pinn[field_name], y_common)
        return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12))

    return {
        "p_rel": rel("p"),
        "rho_rel": rel("rho"),
        "v_rel": rel("v"),
        "u_rel": rel("u"),
    }


def compute_visible_xlim(modes: list[dict[str, np.ndarray]], *, threshold_ratio: float = 0.02, min_half_width: float = 8.0) -> tuple[float, float]:
    xmin = np.inf
    xmax = -np.inf
    for fields in modes:
        y = fields["y"]
        envelope = np.zeros_like(y, dtype=float)
        for field_name in FIELD_ORDER:
            field = fields[field_name]
            envelope = np.maximum(envelope, np.abs(np.real(field)))
            envelope = np.maximum(envelope, np.abs(np.imag(field)))
        peak = float(np.max(envelope))
        if peak <= 0.0:
            continue
        mask = envelope >= threshold_ratio * peak
        if not np.any(mask):
            continue
        y_vis = y[mask]
        half_width = max(float(np.max(np.abs(y_vis))), float(min_half_width))
        xmin = min(xmin, -half_width)
        xmax = max(xmax, half_width)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return -10.0, 10.0
    return xmin, xmax


def save_curve_comparison_figure(
    output_path: Path,
    alpha_values: np.ndarray,
    ci_classic: np.ndarray,
    ci_curves: dict[str, np.ndarray],
    run_specs: list[RunSpec],
    *,
    title: str,
    save_pdf: bool,
) -> list[Path]:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2]},
        constrained_layout=True,
    )

    ax_top.plot(alpha_values, ci_classic, color="black", linewidth=2.4, label="classic")
    for spec in run_specs:
        if spec.key not in ci_curves:
            continue
        ax_top.plot(alpha_values, ci_curves[spec.key], linewidth=2.0, color=spec.color, label=spec.label)
    ax_top.set_ylabel(r"$c_i$")
    ax_top.set_title(title)
    ax_top.legend(ncol=3)

    heat_keys = [spec.key for spec in run_specs if spec.key in ci_curves]
    heat_labels = [spec.short_label for spec in run_specs if spec.key in ci_curves]
    heat_rows = []
    for key in heat_keys:
        rel_err = np.abs(ci_curves[key] - ci_classic) / np.maximum(np.abs(ci_classic), EPS)
        heat_rows.append(rel_err)
    heat = np.asarray(heat_rows, dtype=float)
    if heat.ndim == 1:
        heat = heat[None, :]

    vmax = float(np.nanmax(heat)) if heat.size else 0.10
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.05, vmax=vmax) if vmax > 0.05 else None
    image = ax_bottom.imshow(
        heat,
        aspect="auto",
        origin="lower",
        extent=[alpha_values[0], alpha_values[-1], -0.5, len(heat_labels) - 0.5],
        cmap="magma",
        norm=norm,
    )
    ax_bottom.set_yticks(np.arange(len(heat_labels)))
    ax_bottom.set_yticklabels(heat_labels)
    ax_bottom.set_xlabel(r"$\alpha$")
    ax_bottom.set_ylabel("config")
    ax_bottom.set_title(r"Relative error $\left|c_i^{PINN}-c_i^{classic}\right|/\max(\left|c_i^{classic}\right|,\varepsilon)$")
    cbar = fig.colorbar(image, ax=ax_bottom, pad=0.02)
    cbar.set_label("relative error")
    cbar.ax.axhline(0.05, color="white", linestyle="--", linewidth=1.4)
    cbar.ax.text(1.8, 0.05, "5%", color="white", fontsize=9, va="center")

    return write_dual_figure(fig, output_path, save_pdf=save_pdf)


def save_mode_grid_figure(
    output_path: Path,
    alpha: float,
    mach: float,
    classic_fields: dict[str, np.ndarray],
    mode_entries: list[tuple[RunSpec, dict[str, np.ndarray]]],
    *,
    title: str,
    save_pdf: bool,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=False, constrained_layout=True)
    axes = axes.flatten()

    x_limits = compute_visible_xlim([classic_fields] + [fields for _spec, fields in mode_entries])

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.4, linestyle="-", label="classic Re"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="classic Im"),
    ]
    for spec, _fields in mode_entries:
        legend_handles.append(Line2D([0], [0], color=spec.color, linewidth=2.0, linestyle="-", label=f"{spec.short_label} Re"))
        legend_handles.append(Line2D([0], [0], color=spec.color, linewidth=2.0, linestyle="--", label=f"{spec.short_label} Im"))

    for ax, field_name in zip(axes, FIELD_ORDER):
        ax.plot(classic_fields["y"], np.real(classic_fields[field_name]), color="black", linewidth=2.4, linestyle="-")
        ax.plot(classic_fields["y"], np.imag(classic_fields[field_name]), color="black", linewidth=2.0, linestyle="--")
        for spec, fields in mode_entries:
            ax.plot(fields["y"], np.real(fields[field_name]), color=spec.color, linewidth=2.0, linestyle="-")
            ax.plot(fields["y"], np.imag(fields[field_name]), color=spec.color, linewidth=2.0, linestyle="--")
        ax.set_title(FIELD_TITLES[field_name])
        ax.set_xlim(*x_limits)
        ax.set_xlabel("y")

    fig.suptitle(f"{title} | alpha={alpha:.2f}, M={mach:.2f}")
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.02))
    return write_dual_figure(fig, output_path, save_pdf=save_pdf)


def save_error_bar_figure(
    output_path: Path,
    summary_df: pd.DataFrame,
    *,
    save_pdf: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
    x = np.arange(len(summary_df))
    colors = [row["color"] for _, row in summary_df.iterrows()]

    axes[0].bar(x, summary_df["ci_l2_rel"], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df["config"], rotation=0)
    axes[0].set_ylabel("relative L2")
    axes[0].set_title(r"Relative L2 error on $c_i(\alpha)$")

    axes[1].bar(x, summary_df["mode_l2_rel_mean"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["config"], rotation=0)
    axes[1].set_ylabel("relative L2")
    axes[1].set_title("Mean modal relative L2 error")

    return write_dual_figure(fig, output_path, save_pdf=save_pdf)


def write_article_readme(output_dir: Path, found_specs: list[RunSpec]) -> Path:
    lines = [
        "# Stage1bis Article Assets",
        "",
        "This directory contains final article-oriented figures for the subsonic PINN study at M=0.5 over alpha in [0.10, 0.80].",
        "",
        "Compared configurations:",
    ]
    for spec in found_specs:
        lines.append(f"- `{spec.key}`: {spec.label}")
    lines.extend(
        [
            "",
            "Baseline interpretation:",
            "- `hybrid_ci4` is the minimal spectral supervision baseline.",
            "- `pure_physics` is the non-supervised PINN baseline.",
            "- `hybrid_ci8` and `hybrid_ci16` are comparison runs.",
            "",
            "Important reminder:",
            "- Classical supervision targets `c_i` only.",
            "- There is no direct modal supervision in the loss.",
            "- Modes are reconstructed and evaluated only in post-processing.",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def read_history_tail(run_dir: Path) -> list[dict[str, str]]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return []
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-3:]


def main() -> None:
    args = build_parser().parse_args()
    setup_matplotlib()

    output_dir = (ROOT_DIR / args.output_dir).resolve()
    baseline_dir = output_dir / "baseline_ci4"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    mach = float(args.mach)
    alpha_values = np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.num_alpha), dtype=float)
    modal_alphas = [float(value) for value in args.modal_alphas]
    device = torch.device(args.device)

    found_specs: list[RunSpec] = []
    missing_specs: list[str] = []
    run_dirs: dict[str, Path] = {}
    run_sources: dict[str, str] = {}
    generated_paths: list[Path] = []
    warnings_list: list[str] = []

    for spec in RUN_SPECS:
        run_dir, source = resolve_run_dir(spec)
        if run_dir is None:
            missing_specs.append(spec.key)
            message = f"Missing run for {spec.key}."
            warnings.warn(message)
            warnings_list.append(message)
            continue
        found_specs.append(spec)
        run_dirs[spec.key] = run_dir
        run_sources[spec.key] = source

    if not found_specs:
        raise RuntimeError("No available runs found. Nothing to render.")

    classic_df = solve_reference_curve(mach, alpha_values)
    ci_classic = classic_df["ci_classic"].to_numpy(dtype=float)

    ci_curves: dict[str, np.ndarray] = {}
    mode_metric_rows: list[dict[str, float | str]] = []
    models_cache: dict[str, tuple[object, pd.Series, pd.DataFrame]] = {}

    for spec in found_specs:
        model, config, history = load_model(run_dirs[spec.key], device)
        models_cache[spec.key] = (model, config, history)
        ci_curves[spec.key] = predict_ci_curve(model, alpha_values, device)

    generated_paths.extend(
        save_curve_comparison_figure(
            output_dir / "01_global_ci_vs_alpha_with_error_heatmap.png",
            alpha_values,
            ci_classic,
            ci_curves,
            found_specs,
            title=r"Global spectral comparison at $M=0.5$",
            save_pdf=args.save_pdf,
        )
    )

    figure_name_by_alpha = {
        0.30: "02_global_modes_alpha030_2x2.png",
        0.50: "03_global_modes_alpha050_2x2.png",
        0.70: "04_global_modes_alpha070_2x2.png",
    }
    baseline_name_by_alpha = {
        0.30: "02_baseline_ci4_modes_alpha030_2x2.png",
        0.50: "03_baseline_ci4_modes_alpha050_2x2.png",
        0.70: "04_baseline_ci4_modes_alpha070_2x2.png",
    }

    for alpha in modal_alphas:
        classic_fields, ci_mode_classic = load_classic_full_mode(alpha, mach)
        all_mode_entries: list[tuple[RunSpec, dict[str, np.ndarray]]] = []
        baseline_entries: list[tuple[RunSpec, dict[str, np.ndarray]]] = []

        for spec in found_specs:
            pinn_fields_raw, ci_mode_pinn = load_pinn_full_mode(run_dirs[spec.key], alpha=alpha, n_y=int(args.n_y), device=device)
            pinn_fields = align_mode_to_reference(classic_fields, pinn_fields_raw)
            metrics = compute_mode_metrics(classic_fields, pinn_fields)
            mode_metric_rows.append(
                {
                    "config": spec.key,
                    "alpha": float(alpha),
                    "ci_classic": float(ci_mode_classic),
                    "ci_pinn": float(ci_mode_pinn),
                    "ci_rel": abs(float(ci_mode_pinn) - float(ci_mode_classic)) / max(abs(float(ci_mode_classic)), EPS),
                    **metrics,
                }
            )
            all_mode_entries.append((spec, pinn_fields))
            if spec.key in {"pure_physics", "hybrid_ci4"}:
                baseline_entries.append((spec, pinn_fields))

        target_name = figure_name_by_alpha.get(round(alpha, 2), f"modes_alpha_{alpha:.2f}.png".replace(".", "p"))
        generated_paths.extend(
            save_mode_grid_figure(
                output_dir / target_name,
                alpha,
                mach,
                classic_fields,
                all_mode_entries,
                title="Global modal comparison",
                save_pdf=args.save_pdf,
            )
        )

        baseline_target = baseline_name_by_alpha.get(round(alpha, 2), f"baseline_modes_alpha_{alpha:.2f}.png".replace(".", "p"))
        generated_paths.extend(
            save_mode_grid_figure(
                baseline_dir / baseline_target,
                alpha,
                mach,
                classic_fields,
                baseline_entries,
                title="Baseline ci4 modal comparison",
                save_pdf=args.save_pdf,
            )
        )

    mode_metrics_df = pd.DataFrame(mode_metric_rows)
    summary_rows: list[dict[str, float | str]] = []
    for spec in found_specs:
        ci_pred = ci_curves[spec.key]
        ci_rel = np.abs(ci_pred - ci_classic) / np.maximum(np.abs(ci_classic), EPS)
        mode_sub = mode_metrics_df[mode_metrics_df["config"] == spec.key]
        if mode_sub.empty:
            mode_l2_rel_mean = float("nan")
            p_mean = float("nan")
            rho_mean = float("nan")
            v_mean = float("nan")
            u_mean = float("nan")
        else:
            p_mean = float(mode_sub["p_rel"].mean())
            rho_mean = float(mode_sub["rho_rel"].mean())
            v_mean = float(mode_sub["v_rel"].mean())
            u_mean = float(mode_sub["u_rel"].mean())
            mode_l2_rel_mean = float(np.mean([p_mean, rho_mean, v_mean, u_mean]))
        summary_rows.append(
            {
                "config": spec.key,
                "label": spec.label,
                "color": spec.color,
                "ci_l2_rel": float(np.linalg.norm(ci_pred - ci_classic) / max(np.linalg.norm(ci_classic), 1e-12)),
                "ci_max_rel": float(np.max(ci_rel)),
                "mode_l2_rel_mean": mode_l2_rel_mean,
                "p_l2_rel_mean": p_mean,
                "rho_l2_rel_mean": rho_mean,
                "v_l2_rel_mean": v_mean,
                "u_l2_rel_mean": u_mean,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "stage1bis_article_error_summary.csv"
    summary_df.drop(columns=["label", "color"]).to_csv(summary_csv, index=False)
    generated_paths.append(summary_csv)

    generated_paths.extend(
        save_error_bar_figure(
            output_dir / "05_global_error_bars_ci_and_modes.png",
            summary_df,
            save_pdf=args.save_pdf,
        )
    )

    baseline_specs = [spec for spec in found_specs if spec.key in {"pure_physics", "hybrid_ci4"}]
    baseline_curves = {spec.key: ci_curves[spec.key] for spec in baseline_specs}
    generated_paths.extend(
        save_curve_comparison_figure(
            baseline_dir / "01_baseline_ci4_ci_vs_alpha_with_error_heatmap.png",
            alpha_values,
            ci_classic,
            baseline_curves,
            baseline_specs,
            title=r"hybrid\_ci4 minimal spectral supervision baseline",
            save_pdf=args.save_pdf,
        )
    )

    readme_path = write_article_readme(output_dir, found_specs)
    generated_paths.append(readme_path)

    print("\nGenerated figures and files:")
    for path in generated_paths:
        print(f"- {path}")

    print("\nConfigs found:")
    for spec in found_specs:
        print(f"- {spec.key}: {run_dirs[spec.key]} ({run_sources[spec.key]})")

    print("\nConfigs missing:")
    if missing_specs:
        for key in missing_specs:
            print(f"- {key}")
    else:
        print("- none")

    if warnings_list:
        print("\nWarnings:")
        for message in warnings_list:
            print(f"- {message}")

    print("\nMain metrics:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.drop(columns=["color"]).to_string(index=False))

    print("\nRecent history tail:")
    for spec in found_specs:
        print(f"- {spec.key}")
        for row in read_history_tail(run_dirs[spec.key]):
            epoch = row.get("epoch", "?")
            loss = row.get("loss", "")
            audit_ci = row.get("audit_ci_mae", "")
            print(f"  epoch={epoch} loss={loss} audit_ci_mae={audit_ci}")


if __name__ == "__main__":
    main()
