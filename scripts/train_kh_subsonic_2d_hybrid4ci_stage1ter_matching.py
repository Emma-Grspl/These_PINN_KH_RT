from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.kh_subsonic_sampling_2d import normalize_float_list
from src.training.kh_subsonic_trainer_2d_stage1ter_matching import (
    KHSubsonic2DStage1terMatchingConfig,
    train_kh_subsonic_2d_stage1ter_matching,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2D subsonic Stage 1ter — Riccati left/right matching without direct modal supervision."
    )
    parser.add_argument(
        "--stage0-checkpoint",
        type=str,
        default="model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt",
    )
    parser.add_argument("--mach-values", type=float, nargs="+", default=None)
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.80)
    parser.add_argument("--anchor-alphas", type=float, nargs="+", default=[0.10, 0.30, 0.55, 0.80])
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_2d_hybrid4ci_stage1ter_riccati_matching"),
    )
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--freeze-ci", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach-ci-in-mode-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-interior", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=96)
    parser.add_argument("--n-center", type=int, default=256)
    parser.add_argument("--center-width", type=float, default=2.0)
    parser.add_argument("--center-fraction", type=float, default=0.5)
    parser.add_argument("--n-alpha-samples", type=int, default=12)
    parser.add_argument("--n-mach-samples", type=int, default=None)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc-kappa", type=float, default=20.0)
    parser.add_argument("--w-bc-q", type=float, default=60.0)
    parser.add_argument("--w-match", type=float, default=1.0)
    parser.add_argument("--w-center-pde", type=float, default=1.0)
    parser.add_argument("--w-norm", type=float, default=0.0)
    parser.add_argument("--w-phase", type=float, default=0.0)
    parser.add_argument("--w-ci-anchor", type=float, default=1.0)
    parser.add_argument("--match-y-values", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    parser.add_argument("--shoot-ymax", type=float, default=40.0)
    parser.add_argument("--shoot-steps", type=int, default=512)
    parser.add_argument("--match-warmup-epochs", type=int, default=1000)
    parser.add_argument("--best-metric", type=str, default="loss_total")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mach_values = () if args.mach_values is None else normalize_float_list(args.mach_values)
    cfg = KHSubsonic2DStage1terMatchingConfig(
        stage0_checkpoint=str(args.stage0_checkpoint),
        mach_values=mach_values,
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        anchor_alphas=normalize_float_list(args.anchor_alphas),
        output_dir=str(args.output_dir),
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        grad_clip_norm=float(args.grad_clip_norm),
        n_interior=int(args.n_interior),
        n_boundary=int(args.n_boundary),
        n_center=int(args.n_center),
        center_width=float(args.center_width),
        center_fraction=float(args.center_fraction),
        n_alpha_samples=int(args.n_alpha_samples),
        n_mach_samples=args.n_mach_samples,
        audit_every=max(1, int(args.audit_every)),
        checkpoint_every=max(1, int(args.checkpoint_every)),
        device=str(args.device),
        seed=int(args.seed),
        freeze_ci=bool(args.freeze_ci),
        detach_ci_in_mode_branch=bool(args.detach_ci_in_mode_branch),
        reference_cache=args.reference_cache,
        w_pde=float(args.w_pde),
        w_bc_kappa=float(args.w_bc_kappa),
        w_bc_q=float(args.w_bc_q),
        w_match=float(args.w_match),
        w_center_pde=float(args.w_center_pde),
        w_norm=float(args.w_norm),
        w_phase=float(args.w_phase),
        w_ci_anchor=float(args.w_ci_anchor),
        match_y_values=normalize_float_list(args.match_y_values),
        shoot_ymax=float(args.shoot_ymax),
        shoot_steps=int(args.shoot_steps),
        match_warmup_epochs=int(args.match_warmup_epochs),
        best_metric=str(args.best_metric),
    )
    raise SystemExit(train_kh_subsonic_2d_stage1ter_matching(cfg))


if __name__ == "__main__":
    main()
