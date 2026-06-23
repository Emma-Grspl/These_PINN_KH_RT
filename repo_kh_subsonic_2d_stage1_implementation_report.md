# KH Subsonic 2D Stage 1 Implementation Report

## Scope

This implementation adds the first `Stage 1` pipeline for:

`2D hybrid4ci Stage 1 — physics-only modal reconstruction from sparse spectral supervision`

## Files Created

- `src/physics/kh_subsonic_residual_2d.py`
- `src/training/kh_subsonic_trainer_2d_stage1.py`
- `scripts/train_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.py`
- `scripts/run_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.sh`
- `launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1.slurm`
- `scripts/render_kh_subsonic_2d_stage1_diagnostics.py`

## Files Modified

- none

## Architecture Justification

- Stage 1 is implemented in a separate trainer to avoid breaking:
  - the 1D validated pipelines;
  - the 2D dense-supervision prototype;
  - the 2D Stage 0 spectral lock baseline.
- The 2D Stage 1 model is loaded directly from the Stage 0 checkpoint and keeps the same architecture.
- By default, the spectral head `c_i(alpha, Mach)` is frozen.
- Only the modal branch is trained through physical constraints.

## Stage 0 Source

The default Stage 0 checkpoint used by the new Stage 1 CLI is:

`model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt`

## Spectral Head

- `c_i` is frozen by default.
- The CLI exposes `--freeze-ci` / `--no-freeze-ci`.
- When `freeze_ci=false`, there is still no modal supervision; only a light spectral anchor-preservation loss is allowed.

## Direct Modal Supervision

No direct modal supervision has been added.

## Implemented Physical Losses

- pressure PDE residual
- Riccati far-field boundary loss
- modal normalization
- phase constraint
- shooting term logged as a placeholder and disabled by default in this first implementation
- spectral anchor preservation only when `freeze_ci=false`

## Local Commands

### Stage 1 training

```bash
python scripts/train_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.py \
  --stage0-checkpoint model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt \
  --mach-values 0.1 0.3 0.5 0.7 \
  --alpha-min 0.10 \
  --alpha-max 0.80 \
  --epochs 5000 \
  --lr 1e-4 \
  --device cuda \
  --freeze-ci \
  --output-dir model_saved/kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0
```

### Wrapper

```bash
bash scripts/run_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.sh
```

### Diagnostics

```bash
python scripts/render_kh_subsonic_2d_stage1_diagnostics.py \
  --stage1-dir model_saved/kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0 \
  --output-dir assets/pinn_subsonic/baseline_2D_Stage1
```

## SLURM Command

```bash
sbatch launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1.slurm
```

## Tests Executed

- Python compilation:
  - `python -m py_compile src/physics/kh_subsonic_residual_2d.py src/training/kh_subsonic_trainer_2d_stage1.py scripts/train_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.py scripts/render_kh_subsonic_2d_stage1_diagnostics.py`
  - result: pass
- Shell syntax:
  - `bash -n scripts/run_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.sh`
  - `bash -n launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1.slurm`
  - result: pass
- Smoke training:
  - `python scripts/train_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.py --stage0-checkpoint model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt --mach-values 0.3 0.5 --alpha-min 0.10 --alpha-max 0.80 --epochs 5 --n-interior 128 --n-boundary 32 --device cpu --freeze-ci --output-dir model_saved/_smoke_kh_subsonic_2d_hybrid4ci_stage1`
  - result: pass
  - observed spectral lock during smoke:
    - `ci_anchor_max_abs ≈ 2.92e-4`
    - `ci_anchor_max_rel_unstable ≈ 8.24e-4`
- Smoke diagnostics:
  - `python scripts/render_kh_subsonic_2d_stage1_diagnostics.py --stage1-dir model_saved/_smoke_kh_subsonic_2d_hybrid4ci_stage1 --output-dir assets/pinn_subsonic/_smoke_2D_Stage1`
  - result: pass
  - produced:
    - `01_stage1_ci_vs_alpha_by_mach.png`
    - `02_stage1_ci_error_heatmap.png`
    - `03_stage1_loss_history.png`
    - `04_stage1_modes_M05_alpha030_2x2.png`
    - `05_stage1_modes_M05_alpha050_2x2.png`
    - `06_stage1_modes_M05_alpha070_2x2.png`

## Remaining Limits

- The shooting contribution is not yet active in this first Stage 1 version.
- The present Stage 1 uses physical losses already available in the 2D residual stack, without introducing the full single-case Riccati shooting path machinery.
- The modal diagnostic figures are best-effort and do not block the spectral diagnostics.
- In Riccati mode, the center-anchor reconstruction fixes the local gauge. As a result, the current normalization and phase losses behave mostly as gauge checks and may stay near zero during smoke tests.

## Recommended Next Step

- Run the short smoke test first.
- Then launch a real Stage 1 run with `freeze_ci=true`.
- Inspect:
  - `ci_anchor_max_abs`
  - `ci_anchor_max_rel_unstable`
  - `loss_pde`
  - `loss_bc`
  - `loss_norm`
  - `loss_phase`
- If the modal fields remain poorly constrained, the next clean extension is to add a true 2D Riccati shooting consistency term without introducing any modal labels.
