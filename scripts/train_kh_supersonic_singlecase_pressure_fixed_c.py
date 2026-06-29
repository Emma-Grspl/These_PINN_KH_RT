from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.kh_supersonic_trainer_singlecase_pressure import (
    KHSupersonicSingleCasePressureConfig,
    train_kh_supersonic_singlecase_pressure,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = KHSupersonicSingleCasePressureConfig()
    parser = argparse.ArgumentParser(description="Train a supersonic single-case pressure-first PINN at fixed complex c.")
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--mach", type=float, default=defaults.mach)
    parser.add_argument("--cr", type=float, default=defaults.cr)
    parser.add_argument("--ci", type=float, default=defaults.ci)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-interior", type=int, default=defaults.n_interior)
    parser.add_argument("--n-boundary", type=int, default=defaults.n_boundary)
    parser.add_argument("--n-center", type=int, default=defaults.n_center)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--activation", type=str, default=defaults.activation)
    parser.add_argument("--w-pde", type=float, default=defaults.w_pde)
    parser.add_argument("--w-bc", type=float, default=defaults.w_bc)
    parser.add_argument("--w-gauge", type=float, default=defaults.w_gauge)
    parser.add_argument("--w-center-pde", type=float, default=defaults.w_center_pde)
    parser.add_argument("--ymax", type=float, default=defaults.ymax)
    parser.add_argument("--envelope-eps", type=float, default=defaults.envelope_eps)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--audit-every", type=int, default=defaults.audit_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = KHSupersonicSingleCasePressureConfig(
        alpha=float(args.alpha),
        mach=float(args.mach),
        cr=float(args.cr),
        ci=float(args.ci),
        output_dir=str(args.output_dir),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        n_interior=int(args.n_interior),
        n_boundary=int(args.n_boundary),
        n_center=int(args.n_center),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        activation=str(args.activation),
        w_pde=float(args.w_pde),
        w_bc=float(args.w_bc),
        w_gauge=float(args.w_gauge),
        w_center_pde=float(args.w_center_pde),
        ymax=float(args.ymax),
        envelope_eps=float(args.envelope_eps),
        device=str(args.device),
        seed=int(args.seed),
        grad_clip_norm=float(args.grad_clip_norm),
        audit_every=int(args.audit_every),
        checkpoint_every=int(args.checkpoint_every),
    )
    return train_kh_supersonic_singlecase_pressure(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
