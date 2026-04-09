from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.kh_subsonic_trainer import (
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype PINN subsonique a Mach fixe pour Kelvin-Helmholtz compressible.")
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    parser.add_argument("--n-alpha-supervision", type=int, default=128)
    parser.add_argument("--n-anchor-alpha", type=int, default=32)
    parser.add_argument("--n-norm-interior", type=int, default=256)
    parser.add_argument("--n-reference-alpha", type=int, default=81)
    parser.add_argument("--n-audit-alpha", type=int, default=21)
    parser.add_argument("--n-mode-audit-alpha", type=int, default=7)
    parser.add_argument("--n-mode-audit-y", type=int, default=801)
    parser.add_argument("--audit-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--focus-fraction", type=float, default=0.6)
    parser.add_argument("--focus-half-width", type=float, default=0.03)
    parser.add_argument("--error-threshold", type=float, default=0.01)
    parser.add_argument("--mode-error-threshold", type=float, default=0.12)
    parser.add_argument("--max-focus-points", type=int, default=8)
    parser.add_argument("--anchor-strategy", type=str, default="band", choices=["point", "band", "max", "point_max"])
    parser.add_argument("--anchor-half-width", type=float, default=0.12)
    parser.add_argument("--anchor-max-candidates", type=int, default=257)
    parser.add_argument("--mode-center-fraction", type=float, default=0.5)
    parser.add_argument("--mode-center-half-width", type=float, default=0.3)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=10.0)
    parser.add_argument("--w-norm", type=float, default=1.0)
    parser.add_argument("--w-integral-norm", type=float, default=1.0)
    parser.add_argument("--w-phase", type=float, default=1.0)
    parser.add_argument("--w-ci-supervision", type=float, default=5.0)
    parser.add_argument("--audit-ci-weight", type=float, default=10.0)
    parser.add_argument("--audit-mode-weight", type=float, default=1.0)
    parser.add_argument("--classic-n-points", type=int, default=561)
    parser.add_argument("--classic-mapping-scale", type=float, default=3.0)
    parser.add_argument("--classic-xi-max", type=float, default=0.99)
    parser.add_argument("--enforce-mode-symmetry", action="store_true")
    parser.add_argument("--mode-representation", type=str, default="cartesian", choices=["cartesian", "amplitude_phase", "log_amplitude_phase"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/kh_subsonic_fixed_mach"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KHSubsonicTrainingConfig(
        mach=args.mach,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
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
        n_alpha_supervision=args.n_alpha_supervision,
        n_anchor_alpha=args.n_anchor_alpha,
        n_norm_interior=args.n_norm_interior,
        n_reference_alpha=args.n_reference_alpha,
        n_audit_alpha=args.n_audit_alpha,
        n_mode_audit_alpha=args.n_mode_audit_alpha,
        n_mode_audit_y=args.n_mode_audit_y,
        audit_every=args.audit_every,
        checkpoint_every=args.checkpoint_every,
        focus_fraction=args.focus_fraction,
        focus_half_width=args.focus_half_width,
        error_threshold=args.error_threshold,
        mode_error_threshold=args.mode_error_threshold,
        max_focus_points=args.max_focus_points,
        anchor_strategy=args.anchor_strategy,
        anchor_half_width=args.anchor_half_width,
        anchor_max_candidates=args.anchor_max_candidates,
        mode_center_fraction=args.mode_center_fraction,
        mode_center_half_width=args.mode_center_half_width,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_norm=args.w_norm,
        w_integral_norm=args.w_integral_norm,
        w_phase=args.w_phase,
        w_ci_supervision=args.w_ci_supervision,
        audit_ci_weight=args.audit_ci_weight,
        audit_mode_weight=args.audit_mode_weight,
        classic_n_points=args.classic_n_points,
        classic_mapping_scale=args.classic_mapping_scale,
        classic_xi_max=args.classic_xi_max,
        enforce_mode_symmetry=args.enforce_mode_symmetry,
        mode_representation=args.mode_representation,
        output_dir=str(args.output_dir),
        device=args.device,
    )
    model, history = train_fixed_mach_subsonic_pinn(cfg)
    save_training_artifacts(model, history, cfg)
    print(f"Modele et historique enregistres dans {args.output_dir}")


if __name__ == "__main__":
    main()
