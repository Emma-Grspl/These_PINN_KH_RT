from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.kh_subsonic_trainer_2d import (
    KHSubsonic2DTrainingConfig,
    save_training_artifacts,
    train_subsonic_2d_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype PINN subsonique 2D local sur (alpha, Mach).")
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--mach-min", type=float, default=0.0)
    parser.add_argument("--mach-max", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mode-hidden-dim", type=int, default=None)
    parser.add_argument("--ci-hidden-dim", type=int, default=None)
    parser.add_argument("--mode-depth", type=int, default=4)
    parser.add_argument("--ci-depth", type=int, default=2)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--initial-ci", type=float, default=0.2)
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--trainable-mapping-scale", action="store_true")
    parser.add_argument("--n-interior", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=64)
    parser.add_argument("--n-supervision", type=int, default=128)
    parser.add_argument("--n-reference-alpha", type=int, default=41)
    parser.add_argument("--n-reference-mach", type=int, default=11)
    parser.add_argument("--n-audit-alpha", type=int, default=17)
    parser.add_argument("--n-audit-mach", type=int, default=6)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--separate-branch-optimizers", action="store_true")
    parser.add_argument("--detach-ci-in-mode-branch", action="store_true")
    parser.add_argument("--ci-branch-lr", type=float, default=None)
    parser.add_argument("--mode-branch-lr", type=float, default=None)
    parser.add_argument("--focus-fraction", type=float, default=0.6)
    parser.add_argument("--neutral-fraction", type=float, default=0.2)
    parser.add_argument("--low-alpha-fraction", type=float, default=0.15)
    parser.add_argument("--focus-alpha-half-width", type=float, default=0.03)
    parser.add_argument("--focus-mach-half-width", type=float, default=0.05)
    parser.add_argument("--neutral-band-ratio", type=float, default=0.15)
    parser.add_argument("--low-alpha-band-width", type=float, default=0.06)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--max-focus-points", type=int, default=12)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=10.0)
    parser.add_argument("--w-norm", type=float, default=1.0)
    parser.add_argument("--w-phase", type=float, default=1.0)
    parser.add_argument("--w-ci-supervision", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/kh_subsonic_2d_local"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KHSubsonic2DTrainingConfig(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        mach_min=args.mach_min,
        mach_max=args.mach_max,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        mode_hidden_dim=args.mode_hidden_dim,
        ci_hidden_dim=args.ci_hidden_dim,
        mode_depth=args.mode_depth,
        ci_depth=args.ci_depth,
        activation=args.activation,
        fourier_features=args.fourier_features,
        fourier_scale=args.fourier_scale,
        initial_ci=args.initial_ci,
        mapping_scale=args.mapping_scale,
        trainable_mapping_scale=args.trainable_mapping_scale,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_supervision=args.n_supervision,
        n_reference_alpha=args.n_reference_alpha,
        n_reference_mach=args.n_reference_mach,
        n_audit_alpha=args.n_audit_alpha,
        n_audit_mach=args.n_audit_mach,
        audit_every=args.audit_every,
        checkpoint_every=args.checkpoint_every,
        separate_branch_optimizers=args.separate_branch_optimizers,
        detach_ci_in_mode_branch=args.detach_ci_in_mode_branch,
        ci_branch_lr=args.ci_branch_lr,
        mode_branch_lr=args.mode_branch_lr,
        focus_fraction=args.focus_fraction,
        neutral_fraction=args.neutral_fraction,
        low_alpha_fraction=args.low_alpha_fraction,
        focus_alpha_half_width=args.focus_alpha_half_width,
        focus_mach_half_width=args.focus_mach_half_width,
        neutral_band_ratio=args.neutral_band_ratio,
        low_alpha_band_width=args.low_alpha_band_width,
        error_threshold=args.error_threshold,
        max_focus_points=args.max_focus_points,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_norm=args.w_norm,
        w_phase=args.w_phase,
        w_ci_supervision=args.w_ci_supervision,
        output_dir=str(args.output_dir),
        device=args.device,
    )
    model, history = train_subsonic_2d_pinn(cfg)
    save_training_artifacts(model, history, cfg)
    print(f"Modele et historique enregistres dans {args.output_dir}")



if __name__ == "__main__":
    main()
