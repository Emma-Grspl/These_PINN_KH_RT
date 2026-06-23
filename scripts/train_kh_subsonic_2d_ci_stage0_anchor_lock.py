from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.kh_subsonic_sampling_2d import DEFAULT_ANCHOR_ALPHAS, normalize_float_list
from src.training.kh_subsonic_trainer_2d_hybrid4ci import (
    KHSubsonic2DHybrid4CIStage0Config,
    train_kh_subsonic_2d_stage0_anchor_lock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 0 subsonic 2D hybrid4ci: train only the spectral head "
            "c_i(alpha, Mach) from sparse classical anchors, with no modal supervision."
        )
    )
    parser.add_argument("--mach-values", type=float, nargs="+", required=True)
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.80)
    parser.add_argument("--anchor-alphas", type=float, nargs="+", default=list(DEFAULT_ANCHOR_ALPHAS))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--mode-hidden-dim", type=int, default=None)
    parser.add_argument("--ci-hidden-dim", type=int, default=None)
    parser.add_argument("--mode-depth", type=int, default=4)
    parser.add_argument("--ci-depth", type=int, default=2)
    parser.add_argument("--activation", choices=["tanh", "silu"], default="tanh")
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--initial-ci", type=float, default=0.2)
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--w-anchor", type=float, default=100.0)
    parser.add_argument("--w-monotone-alpha", type=float, default=1.0)
    parser.add_argument("--w-smooth-alpha", type=float, default=0.0)
    parser.add_argument("--w-smooth-mach", type=float, default=0.0)
    parser.add_argument("--n-shape-alpha", type=int, default=65)
    parser.add_argument("--n-shape-mach", type=int, default=25)
    parser.add_argument("--audit-every", type=int, default=50)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument("--target-max-abs", type=float, default=1e-3)
    parser.add_argument("--target-max-rel", type=float, default=5e-2)
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock"))
    parser.add_argument("--fail-on-target", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = KHSubsonic2DHybrid4CIStage0Config(
        mach_values=normalize_float_list(args.mach_values),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        anchor_alphas=normalize_float_list(args.anchor_alphas),
        output_dir=str(args.output_dir),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        mode_hidden_dim=args.mode_hidden_dim,
        ci_hidden_dim=args.ci_hidden_dim,
        mode_depth=int(args.mode_depth),
        ci_depth=int(args.ci_depth),
        activation=str(args.activation),
        fourier_features=int(args.fourier_features),
        fourier_scale=float(args.fourier_scale),
        initial_ci=float(args.initial_ci),
        mapping_scale=float(args.mapping_scale),
        device=str(args.device),
        seed=int(args.seed),
        reference_cache=args.reference_cache,
        w_anchor=float(args.w_anchor),
        w_monotone_alpha=float(args.w_monotone_alpha),
        w_smooth_alpha=float(args.w_smooth_alpha),
        w_smooth_mach=float(args.w_smooth_mach),
        n_shape_alpha=int(args.n_shape_alpha),
        n_shape_mach=int(args.n_shape_mach),
        audit_every=max(1, int(args.audit_every)),
        lbfgs_steps=max(0, int(args.lbfgs_steps)),
        lbfgs_lr=float(args.lbfgs_lr),
        target_max_abs=float(args.target_max_abs),
        target_max_rel=float(args.target_max_rel),
        fail_on_target=bool(args.fail_on_target),
    )
    raise SystemExit(train_kh_subsonic_2d_stage0_anchor_lock(cfg))


if __name__ == "__main__":
    main()
