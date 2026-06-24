from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.kh_subsonic_trainer_2d_stage1quater_pressure import (
    KHSubsonic2DStage1quaterPressureConfig,
    train_kh_subsonic_2d_stage1quater_pressure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the subsonic 2D Stage 1quater pressure-first PINN.")
    parser.add_argument("--stage0-checkpoint", type=str, default=KHSubsonic2DStage1quaterPressureConfig.stage0_checkpoint)
    parser.add_argument("--output-dir", type=str, default=KHSubsonic2DStage1quaterPressureConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=KHSubsonic2DStage1quaterPressureConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=KHSubsonic2DStage1quaterPressureConfig.learning_rate)
    parser.add_argument("--mach-values", type=float, nargs="*", default=list(KHSubsonic2DStage1quaterPressureConfig.mach_values))
    parser.add_argument("--alpha-min", type=float, default=KHSubsonic2DStage1quaterPressureConfig.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=KHSubsonic2DStage1quaterPressureConfig.alpha_max)
    parser.add_argument("--anchor-alphas", type=float, nargs="*", default=list(KHSubsonic2DStage1quaterPressureConfig.anchor_alphas))
    parser.add_argument("--n-interior", type=int, default=KHSubsonic2DStage1quaterPressureConfig.n_interior)
    parser.add_argument("--n-boundary", type=int, default=KHSubsonic2DStage1quaterPressureConfig.n_boundary)
    parser.add_argument("--n-center", type=int, default=KHSubsonic2DStage1quaterPressureConfig.n_center)
    parser.add_argument("--n-alpha-samples", type=int, default=KHSubsonic2DStage1quaterPressureConfig.n_alpha_samples)
    parser.add_argument("--n-mach-samples", type=int, default=KHSubsonic2DStage1quaterPressureConfig.n_mach_samples)
    parser.add_argument("--hidden-dim", type=int, default=KHSubsonic2DStage1quaterPressureConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=KHSubsonic2DStage1quaterPressureConfig.depth)
    parser.add_argument("--activation", type=str, default=KHSubsonic2DStage1quaterPressureConfig.activation)
    parser.add_argument("--freeze-ci", dest="freeze_ci", action="store_true")
    parser.add_argument("--no-freeze-ci", dest="freeze_ci", action="store_false")
    parser.set_defaults(freeze_ci=KHSubsonic2DStage1quaterPressureConfig.freeze_ci)
    parser.add_argument("--detach-ci-in-mode-branch", dest="detach_ci_in_mode_branch", action="store_true")
    parser.add_argument("--no-detach-ci-in-mode-branch", dest="detach_ci_in_mode_branch", action="store_false")
    parser.set_defaults(detach_ci_in_mode_branch=KHSubsonic2DStage1quaterPressureConfig.detach_ci_in_mode_branch)
    parser.add_argument("--w-pde", type=float, default=KHSubsonic2DStage1quaterPressureConfig.w_pde)
    parser.add_argument("--w-bc", type=float, default=KHSubsonic2DStage1quaterPressureConfig.w_bc)
    parser.add_argument("--w-gauge", type=float, default=KHSubsonic2DStage1quaterPressureConfig.w_gauge)
    parser.add_argument("--w-center-pde", type=float, default=KHSubsonic2DStage1quaterPressureConfig.w_center_pde)
    parser.add_argument("--w-ci-anchor", type=float, default=KHSubsonic2DStage1quaterPressureConfig.w_ci_anchor)
    parser.add_argument("--ymax", type=float, default=KHSubsonic2DStage1quaterPressureConfig.ymax)
    parser.add_argument("--envelope-eps", type=float, default=KHSubsonic2DStage1quaterPressureConfig.envelope_eps)
    parser.add_argument("--center-width", type=float, default=KHSubsonic2DStage1quaterPressureConfig.center_width)
    parser.add_argument("--center-fraction", type=float, default=KHSubsonic2DStage1quaterPressureConfig.center_fraction)
    parser.add_argument("--grad-clip-norm", type=float, default=KHSubsonic2DStage1quaterPressureConfig.grad_clip_norm)
    parser.add_argument("--audit-every", type=int, default=KHSubsonic2DStage1quaterPressureConfig.audit_every)
    parser.add_argument("--checkpoint-every", type=int, default=KHSubsonic2DStage1quaterPressureConfig.checkpoint_every)
    parser.add_argument("--best-metric", type=str, default=KHSubsonic2DStage1quaterPressureConfig.best_metric)
    parser.add_argument("--reference-cache", type=str, default=KHSubsonic2DStage1quaterPressureConfig.reference_cache)
    parser.add_argument("--device", type=str, default=KHSubsonic2DStage1quaterPressureConfig.device)
    parser.add_argument("--seed", type=int, default=KHSubsonic2DStage1quaterPressureConfig.seed)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = KHSubsonic2DStage1quaterPressureConfig(
        stage0_checkpoint=str(args.stage0_checkpoint),
        mach_values=tuple(float(value) for value in args.mach_values),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        anchor_alphas=tuple(float(value) for value in args.anchor_alphas),
        output_dir=str(args.output_dir),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        grad_clip_norm=float(args.grad_clip_norm),
        n_interior=int(args.n_interior),
        n_boundary=int(args.n_boundary),
        n_center=int(args.n_center),
        center_width=float(args.center_width),
        center_fraction=float(args.center_fraction),
        n_alpha_samples=int(args.n_alpha_samples),
        n_mach_samples=None if args.n_mach_samples is None else int(args.n_mach_samples),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        activation=str(args.activation),
        audit_every=int(args.audit_every),
        checkpoint_every=int(args.checkpoint_every),
        device=str(args.device),
        seed=int(args.seed),
        freeze_ci=bool(args.freeze_ci),
        detach_ci_in_mode_branch=bool(args.detach_ci_in_mode_branch),
        reference_cache=None if args.reference_cache in {None, "", "None"} else str(args.reference_cache),
        w_pde=float(args.w_pde),
        w_bc=float(args.w_bc),
        w_gauge=float(args.w_gauge),
        w_center_pde=float(args.w_center_pde),
        w_ci_anchor=float(args.w_ci_anchor),
        ymax=float(args.ymax),
        envelope_eps=float(args.envelope_eps),
        best_metric=str(args.best_metric),
    )
    return train_kh_subsonic_2d_stage1quater_pressure(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
