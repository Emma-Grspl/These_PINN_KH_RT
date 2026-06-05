from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_kh_subsonic_pinn_2d_ci_modes import evaluate_run  # noqa: E402
from src.training.kh_subsonic_trainer_2d import (  # noqa: E402
    KHSubsonic2DTrainingConfig,
    save_training_artifacts,
    train_subsonic_2d_pinn,
)


DEFAULT_MODE_POINTS = [
    "0.10:0.50",
    "0.25:0.50",
    "0.65:0.50",
    "0.10:0.55",
    "0.25:0.55",
    "0.65:0.55",
    "0.10:0.60",
    "0.25:0.60",
    "0.65:0.60",
]


def build_pilot_parser(
    *,
    description: str,
    default_warmstart_run_dir: Path,
    default_output_dir: Path,
    default_mode_points: list[str],
    default_alpha_min: float,
    default_alpha_max: float,
    default_mach_min: float,
    default_mach_max: float,
    default_n_reference_mach: int = 7,
    default_n_audit_mach: int = 5,
    default_upper_mach_min: float = 0.58,
    default_lower_mach_max: float = 0.52,
    default_focus_mach_half_width: float = 0.02,
    default_ci_plot_mach: int = 7,
    default_mode_audit_mach: int = 3,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--warmstart-run-dir",
        type=Path,
        default=default_warmstart_run_dir,
    )
    parser.add_argument("--warmstart-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--warmstart-kind",
        type=str,
        choices=["auto", "fixed_mach", "multimach"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--alpha-min", type=float, default=default_alpha_min)
    parser.add_argument("--alpha-max", type=float, default=default_alpha_max)
    parser.add_argument("--mach-min", type=float, default=default_mach_min)
    parser.add_argument("--mach-max", type=float, default=default_mach_max)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--ci-branch-lr", type=float, default=2.5e-4)
    parser.add_argument("--mode-branch-lr", type=float, default=5e-4)
    parser.add_argument("--n-interior", type=int, default=384)
    parser.add_argument("--n-boundary", type=int, default=64)
    parser.add_argument("--n-supervision", type=int, default=96)
    parser.add_argument("--n-reference-alpha", type=int, default=31)
    parser.add_argument("--n-reference-mach", type=int, default=default_n_reference_mach)
    parser.add_argument("--n-audit-alpha", type=int, default=13)
    parser.add_argument("--n-audit-mach", type=int, default=default_n_audit_mach)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--focus-fraction", type=float, default=0.55)
    parser.add_argument("--neutral-fraction", type=float, default=0.20)
    parser.add_argument("--low-alpha-fraction", type=float, default=0.15)
    parser.add_argument("--upper-corner-fraction", type=float, default=0.10)
    parser.add_argument("--lower-corner-fraction", type=float, default=0.10)
    parser.add_argument("--focus-alpha-half-width", type=float, default=0.03)
    parser.add_argument("--focus-mach-half-width", type=float, default=default_focus_mach_half_width)
    parser.add_argument("--neutral-band-ratio", type=float, default=0.15)
    parser.add_argument("--low-alpha-band-width", type=float, default=0.06)
    parser.add_argument("--upper-alpha-min", type=float, default=0.65)
    parser.add_argument("--upper-mach-min", type=float, default=default_upper_mach_min)
    parser.add_argument("--lower-alpha-max", type=float, default=0.20)
    parser.add_argument("--lower-mach-max", type=float, default=default_lower_mach_max)
    parser.add_argument("--error-threshold", type=float, default=0.01)
    parser.add_argument("--max-focus-points", type=int, default=12)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc-kappa", type=float, default=10.0)
    parser.add_argument("--w-bc-q", type=float, default=10.0)
    parser.add_argument("--w-norm", type=float, default=1.0)
    parser.add_argument("--w-phase", type=float, default=1.0)
    parser.add_argument("--w-ci-supervision", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ci-plot-alpha", type=int, default=31)
    parser.add_argument("--ci-plot-mach", type=int, default=default_ci_plot_mach)
    parser.add_argument("--ci-n-levels", type=int, default=8)
    parser.add_argument("--mode-audit-alpha", type=int, default=5)
    parser.add_argument("--mode-audit-mach", type=int, default=default_mode_audit_mach)
    parser.add_argument("--mode-points", type=str, nargs="*", default=list(default_mode_points))
    parser.add_argument("--mode-n-y", type=int, default=1001)
    parser.add_argument("--mode-n-common", type=int, default=1200)
    parser.add_argument("--phase-threshold", type=float, default=1e-2)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return build_pilot_parser(
        description=(
            "Pilote PINN subsonique 2D sur la bande Mach [0.5, 0.6], "
            "bootstrape depuis une reference 1D a Mach fixe."
        ),
        default_warmstart_run_dir=Path("assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current"),
        default_output_dir=Path("model_saved/kh_subsonic_2d_pilot_M05_M06"),
        default_mode_points=DEFAULT_MODE_POINTS,
        default_alpha_min=0.05,
        default_alpha_max=0.75,
        default_mach_min=0.50,
        default_mach_max=0.60,
    )


def load_warmstart_config(run_dir: Path) -> pd.Series:
    config_path = run_dir / "config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Warmstart config not found: {config_path}")
    config_df = pd.read_csv(config_path)
    if config_df.empty:
        raise RuntimeError(f"Warmstart config is empty: {config_path}")
    return config_df.iloc[0]


def resolve_warmstart_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = args.warmstart_checkpoint
    if checkpoint is None:
        checkpoint = args.warmstart_run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Warmstart checkpoint not found: {checkpoint}")
    return checkpoint


def infer_warmstart_kind(args: argparse.Namespace, warm_config: pd.Series) -> str:
    requested = str(getattr(args, "warmstart_kind", "auto"))
    if requested != "auto":
        return requested
    if "mach_min" in warm_config.index and "mach_max" in warm_config.index:
        if not pd.isna(warm_config["mach_min"]) and not pd.isna(warm_config["mach_max"]):
            return "multimach"
    return "fixed_mach"


def build_training_config(
    args: argparse.Namespace,
    warm_config: pd.Series,
    warm_checkpoint: Path,
    warmstart_kind: str,
) -> KHSubsonic2DTrainingConfig:
    return KHSubsonic2DTrainingConfig(
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        mach_min=float(args.mach_min),
        mach_max=float(args.mach_max),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(warm_config["hidden_dim"]),
        mode_hidden_dim=None if pd.isna(warm_config.get("mode_hidden_dim")) else int(warm_config["mode_hidden_dim"]),
        ci_hidden_dim=None if pd.isna(warm_config.get("ci_hidden_dim")) else int(warm_config["ci_hidden_dim"]),
        mode_depth=int(warm_config["mode_depth"]),
        ci_depth=int(warm_config["ci_depth"]),
        activation=str(warm_config["activation"]),
        fourier_features=int(warm_config.get("fourier_features", 0)),
        fourier_scale=float(warm_config.get("fourier_scale", 2.0)),
        initial_ci=float(warm_config.get("initial_ci", 0.2)),
        mapping_scale=float(warm_config.get("mapping_scale", 3.0)),
        trainable_mapping_scale=False,
        n_interior=int(args.n_interior),
        n_boundary=int(args.n_boundary),
        n_supervision=int(args.n_supervision),
        n_reference_alpha=int(args.n_reference_alpha),
        n_reference_mach=int(args.n_reference_mach),
        n_audit_alpha=int(args.n_audit_alpha),
        n_audit_mach=int(args.n_audit_mach),
        audit_every=int(args.audit_every),
        checkpoint_every=int(args.checkpoint_every),
        separate_branch_optimizers=True,
        detach_ci_in_mode_branch=True,
        ci_branch_lr=float(args.ci_branch_lr),
        mode_branch_lr=float(args.mode_branch_lr),
        focus_fraction=float(args.focus_fraction),
        neutral_fraction=float(args.neutral_fraction),
        low_alpha_fraction=float(args.low_alpha_fraction),
        upper_corner_fraction=float(args.upper_corner_fraction),
        lower_corner_fraction=float(args.lower_corner_fraction),
        focus_alpha_half_width=float(args.focus_alpha_half_width),
        focus_mach_half_width=float(args.focus_mach_half_width),
        neutral_band_ratio=float(args.neutral_band_ratio),
        low_alpha_band_width=float(args.low_alpha_band_width),
        upper_alpha_min=float(args.upper_alpha_min),
        upper_mach_min=float(args.upper_mach_min),
        lower_alpha_max=float(args.lower_alpha_max),
        lower_mach_max=float(args.lower_mach_max),
        error_threshold=float(args.error_threshold),
        max_focus_points=int(args.max_focus_points),
        w_pde=float(args.w_pde),
        w_bc=10.0,
        w_bc_kappa=float(args.w_bc_kappa),
        w_bc_q=float(args.w_bc_q),
        w_norm=float(args.w_norm),
        w_phase=float(args.w_phase),
        w_ci_supervision=float(args.w_ci_supervision),
        mode_representation=str(warm_config.get("mode_representation", "riccati")),
        initial_model_path=str(warm_checkpoint),
        initial_model_kind=warmstart_kind,
        initial_model_config_path=str(args.warmstart_run_dir / "config.csv") if warmstart_kind == "fixed_mach" else None,
        initial_model_strict=True,
        output_dir=str(args.output_dir),
        device=str(args.device),
    )


def run_pilot(args: argparse.Namespace, *, pilot_name: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warm_config = load_warmstart_config(args.warmstart_run_dir)
    warm_checkpoint = resolve_warmstart_checkpoint(args)
    warmstart_kind = infer_warmstart_kind(args, warm_config)

    cfg = build_training_config(args, warm_config, warm_checkpoint, warmstart_kind)
    pd.DataFrame(
        [
            {
                "pilot_name": pilot_name,
                "warmstart_run_dir": str(args.warmstart_run_dir),
                "warmstart_checkpoint": str(warm_checkpoint),
                "warmstart_kind": warmstart_kind,
                "alpha_min": float(args.alpha_min),
                "alpha_max": float(args.alpha_max),
                "mach_min": float(args.mach_min),
                "mach_max": float(args.mach_max),
                "mode_points": " ".join(args.mode_points),
            }
        ]
    ).to_csv(args.output_dir / "pilot_manifest.csv", index=False)

    print("Subsonic 2D PINN pilot")
    print(f"warmstart_run_dir={args.warmstart_run_dir}")
    print(f"warmstart_checkpoint={warm_checkpoint}")
    print(f"warmstart_kind={warmstart_kind}")
    print(f"alpha-range=[{args.alpha_min:.3f}, {args.alpha_max:.3f}]")
    print(f"mach-range=[{args.mach_min:.3f}, {args.mach_max:.3f}]")
    print(f"epochs={args.epochs} device={args.device}")
    print(f"mode_points={' '.join(args.mode_points)}")

    if args.skip_training:
        print(f"Skip training enabled; reusing existing model at {args.output_dir / 'model_best.pt'}")
    else:
        model, history = train_subsonic_2d_pinn(cfg)
        save_training_artifacts(model, history, cfg)

    outputs = evaluate_run(
        args.output_dir,
        device=args.device,
        ci_num_alpha=args.ci_plot_alpha,
        ci_num_mach=args.ci_plot_mach,
        ci_n_levels=args.ci_n_levels,
        mode_num_alpha=args.mode_audit_alpha,
        mode_num_mach=args.mode_audit_mach,
        mode_points=args.mode_points,
        mode_n_y=args.mode_n_y,
        mode_n_common=args.mode_n_common,
        phase_threshold=args.phase_threshold,
    )
    print(f"Artifacts saved in {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


def main() -> None:
    args = build_parser().parse_args()
    run_pilot(args, pilot_name="subsonic_2d_M05_M06_band")


if __name__ == "__main__":
    main()
