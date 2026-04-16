from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.kh_subsonic_trainer import (  # noqa: E402
    KHSubsonicTrainingConfig,
    save_training_artifacts,
    train_fixed_mach_subsonic_pinn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recherche externe sur c_i pour un single-case KH subsonique.")
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--ci-min", type=float, default=0.02)
    parser.add_argument("--ci-max", type=float, default=0.22)
    parser.add_argument("--ci-count", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/kh_subsonic_singlecase_external_ci_search"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_singlecase_config(args: argparse.Namespace, ci_value: float, output_dir: Path) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=args.mach,
        alpha_min=args.alpha,
        alpha_max=args.alpha,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=160,
        mode_depth=4,
        ci_depth=2,
        fixed_scalar_ci=True,
        freeze_ci=True,
        initial_ci=ci_value,
        mapping_scale=3.0,
        n_interior=512,
        n_boundary=64,
        n_alpha_supervision=64,
        n_anchor_alpha=16,
        n_norm_interior=256,
        n_reference_alpha=1,
        n_audit_alpha=1,
        n_mode_audit_alpha=1,
        n_mode_audit_y=801,
        audit_every=100,
        checkpoint_every=500,
        enable_classic_ci_supervision=False,
        enable_classic_mode_audit=True,
        focus_fraction=0.0,
        max_focus_points=1,
        anchor_strategy="band",
        anchor_half_width=0.12,
        mode_center_fraction=1.0,
        mode_center_half_width=0.3,
        w_pde=1.0,
        w_bc_kappa=10.0,
        w_bc_q=20.0,
        w_ci_supervision=0.0,
        audit_ci_weight=10.0,
        audit_env_weight=1.0,
        audit_phase_weight=0.5,
        audit_peak_weight=0.25,
        mode_representation="riccati",
        mode_experts=1,
        w_riccati_center_kappa=5.0,
        w_riccati_center_peak=2.0,
        w_riccati_boundary_band_kappa=2.0,
        w_riccati_boundary_band_q=8.0,
        riccati_boundary_band_points=32,
        riccati_boundary_band_start=0.94,
        riccati_boundary_band_end=0.995,
        output_dir=str(output_dir),
        device=args.device,
    )


def select_score(history: pd.DataFrame) -> tuple[int, float]:
    if history.empty:
        return -1, float("inf")
    score = history["loss"].to_numpy(dtype=float)
    best_idx = int(np.argmin(score))
    return best_idx, float(score[best_idx])


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_values = np.linspace(float(args.ci_min), float(args.ci_max), max(int(args.ci_count), 2))
    rows: list[dict] = []
    best_score = float("inf")
    best_dir: Path | None = None

    for idx, ci_value in enumerate(ci_values, start=1):
        run_dir = output_dir / f"candidate_{idx:02d}_ci_{ci_value:.6f}"
        cfg = build_singlecase_config(args, float(ci_value), run_dir)
        model, history = train_fixed_mach_subsonic_pinn(cfg)
        save_training_artifacts(model, history, cfg)

        best_idx, score = select_score(history)
        best_row = history.iloc[best_idx] if best_idx >= 0 else pd.Series(dtype=float)
        last_row = history.iloc[-1] if not history.empty else pd.Series(dtype=float)

        row = {
            "ci_candidate": float(ci_value),
            "run_dir": str(run_dir),
            "best_loss": score,
            "best_epoch": int(best_row.get("epoch", -1)),
            "best_ci_mid": float(best_row.get("ci_mid", np.nan)),
            "best_loss_pde": float(best_row.get("loss_pde", np.nan)),
            "best_loss_bc_kappa": float(best_row.get("loss_bc_kappa", np.nan)),
            "best_loss_bc_q": float(best_row.get("loss_bc_q", np.nan)),
            "best_loss_riccati_center_kappa": float(best_row.get("loss_riccati_center_kappa", np.nan)),
            "best_loss_riccati_center_peak": float(best_row.get("loss_riccati_center_peak", np.nan)),
            "best_loss_riccati_boundary_band_kappa": float(best_row.get("loss_riccati_boundary_band_kappa", np.nan)),
            "best_loss_riccati_boundary_band_q": float(best_row.get("loss_riccati_boundary_band_q", np.nan)),
            "last_epoch": int(last_row.get("epoch", -1)),
            "last_loss": float(last_row.get("loss", np.nan)),
            "last_ci_mae": float(last_row.get("audit_ci_mae", np.nan)),
            "last_p_rel": float(last_row.get("audit_p_rel_l2_mean", np.nan)),
            "last_env": float(last_row.get("audit_env_rel_mean", np.nan)),
            "last_phase": float(last_row.get("audit_phase_rel_mean", np.nan)),
            "last_peak": float(last_row.get("audit_peak_shift_mean", np.nan)),
        }
        rows.append(row)

        if score < best_score:
            best_score = score
            best_dir = run_dir

        print(
            f"[{idx}/{len(ci_values)}] ci={ci_value:.6f} "
            f"best_loss={score:.3e} last_ci_mae={row['last_ci_mae']:.3e} "
            f"last_p_rel={row['last_p_rel']:.3e}"
        )

    summary = pd.DataFrame(rows).sort_values("best_loss").reset_index(drop=True)
    summary_path = output_dir / "external_ci_search_summary.csv"
    summary.to_csv(summary_path, index=False)

    if not summary.empty and best_dir is not None:
        best_txt = output_dir / "external_ci_search_best.txt"
        best = summary.iloc[0]
        best_txt.write_text(
            "\n".join(
                [
                    f"best_run_dir={best['run_dir']}",
                    f"best_ci_candidate={best['ci_candidate']:.12f}",
                    f"best_loss={best['best_loss']:.12e}",
                    f"best_epoch={int(best['best_epoch'])}",
                    f"last_ci_mae={best['last_ci_mae']:.12e}",
                    f"last_p_rel={best['last_p_rel']:.12e}",
                    f"last_env={best['last_env']:.12e}",
                    f"last_phase={best['last_phase']:.12e}",
                ]
            )
            + "\n"
        )

        best_alias = output_dir / "best_candidate"
        if best_alias.exists() or best_alias.is_symlink():
            best_alias.unlink()
        best_alias.symlink_to(best_dir.name)

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
