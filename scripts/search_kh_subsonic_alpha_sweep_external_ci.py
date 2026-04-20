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
    parser = argparse.ArgumentParser(
        description="Sweep en alpha avec recherche externe sur c_i point par point."
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--alpha-count", type=int, default=5)
    parser.add_argument("--ci-min", type=float, default=0.2)
    parser.add_argument("--ci-max", type=float, default=0.34)
    parser.add_argument("--ci-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_saved/kh_subsonic_fixed_mach_M05_external_ci_alpha_sweep"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_singlecase_config(
    args: argparse.Namespace,
    alpha_value: float,
    ci_value: float,
    output_dir: Path,
) -> KHSubsonicTrainingConfig:
    return KHSubsonicTrainingConfig(
        mach=args.mach,
        alpha_min=alpha_value,
        alpha_max=alpha_value,
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


def select_min_idx(frame: pd.DataFrame, column: str) -> int:
    series = pd.to_numeric(frame[column], errors="coerce")
    if series.isna().all():
        return -1
    return int(series.idxmin())


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = np.linspace(
        float(args.alpha_min), float(args.alpha_max), max(int(args.alpha_count), 2)
    )
    ci_values = np.linspace(
        float(args.ci_min), float(args.ci_max), max(int(args.ci_count), 2)
    )

    rows: list[dict] = []
    total_runs = len(alpha_values) * len(ci_values)
    run_index = 0

    for alpha_value in alpha_values:
        alpha_tag = f"a_{alpha_value:.6f}"
        alpha_dir = output_dir / alpha_tag
        alpha_dir.mkdir(parents=True, exist_ok=True)
        for ci_value in ci_values:
            run_index += 1
            run_dir = alpha_dir / f"ci_{ci_value:.6f}"
            cfg = build_singlecase_config(args, float(alpha_value), float(ci_value), run_dir)
            model, history = train_fixed_mach_subsonic_pinn(cfg)
            save_training_artifacts(model, history, cfg)

            best_idx, score = select_score(history)
            best_row = history.iloc[best_idx] if best_idx >= 0 else pd.Series(dtype=float)
            last_row = history.iloc[-1] if not history.empty else pd.Series(dtype=float)

            row = {
                "alpha": float(alpha_value),
                "ci_candidate": float(ci_value),
                "run_dir": str(run_dir),
                "best_loss": score,
                "best_epoch": int(best_row.get("epoch", -1)),
                "last_epoch": int(last_row.get("epoch", -1)),
                "last_loss": float(last_row.get("loss", np.nan)),
                "last_ci_mae": float(last_row.get("audit_ci_mae", np.nan)),
                "last_p_rel": float(last_row.get("audit_p_rel_l2_mean", np.nan)),
                "last_env": float(last_row.get("audit_env_rel_mean", np.nan)),
                "last_phase": float(last_row.get("audit_phase_rel_mean", np.nan)),
                "last_peak": float(last_row.get("audit_peak_shift_mean", np.nan)),
            }
            rows.append(row)

            print(
                f"[{run_index}/{total_runs}] alpha={alpha_value:.6f} ci={ci_value:.6f} "
                f"best_loss={score:.3e} last_ci_mae={row['last_ci_mae']:.3e} "
                f"last_p_rel={row['last_p_rel']:.3e}"
            )

    candidates = pd.DataFrame(rows).sort_values(["alpha", "ci_candidate"]).reset_index(drop=True)
    candidates_path = output_dir / "external_ci_alpha_sweep_candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    best_by_loss_rows: list[dict] = []
    best_by_ci_rows: list[dict] = []
    overview_rows: list[dict] = []

    for alpha_value, group in candidates.groupby("alpha", sort=True):
        loss_idx = select_min_idx(group, "best_loss")
        ci_idx = select_min_idx(group, "last_ci_mae")
        best_loss_row = group.loc[loss_idx] if loss_idx >= 0 else pd.Series(dtype=float)
        best_ci_row = group.loc[ci_idx] if ci_idx >= 0 else pd.Series(dtype=float)

        if not best_loss_row.empty:
            best_by_loss_rows.append(best_loss_row.to_dict())
        if not best_ci_row.empty:
            best_by_ci_rows.append(best_ci_row.to_dict())

        overview_rows.append(
            {
                "alpha": float(alpha_value),
                "best_loss_ci_candidate": float(best_loss_row.get("ci_candidate", np.nan)),
                "best_loss_value": float(best_loss_row.get("best_loss", np.nan)),
                "best_loss_last_ci_mae": float(best_loss_row.get("last_ci_mae", np.nan)),
                "best_loss_last_p_rel": float(best_loss_row.get("last_p_rel", np.nan)),
                "best_audit_ci_candidate": float(best_ci_row.get("ci_candidate", np.nan)),
                "best_audit_ci_mae": float(best_ci_row.get("last_ci_mae", np.nan)),
                "best_audit_last_p_rel": float(best_ci_row.get("last_p_rel", np.nan)),
                "loss_vs_audit_ci_gap": float(
                    best_loss_row.get("ci_candidate", np.nan) - best_ci_row.get("ci_candidate", np.nan)
                ),
            }
        )

    best_by_loss = pd.DataFrame(best_by_loss_rows).sort_values("alpha").reset_index(drop=True)
    best_by_ci = pd.DataFrame(best_by_ci_rows).sort_values("alpha").reset_index(drop=True)
    overview = pd.DataFrame(overview_rows).sort_values("alpha").reset_index(drop=True)

    best_by_loss_path = output_dir / "external_ci_alpha_sweep_best_by_loss.csv"
    best_by_ci_path = output_dir / "external_ci_alpha_sweep_best_by_ci_mae.csv"
    overview_path = output_dir / "external_ci_alpha_sweep_overview.csv"

    best_by_loss.to_csv(best_by_loss_path, index=False)
    best_by_ci.to_csv(best_by_ci_path, index=False)
    overview.to_csv(overview_path, index=False)

    print(f"Candidates written to {candidates_path}")
    print(f"Best-by-loss summary written to {best_by_loss_path}")
    print(f"Best-by-audit-ci summary written to {best_by_ci_path}")
    print(f"Overview written to {overview_path}")


if __name__ == "__main__":
    main()
