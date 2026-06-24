# Stage 1ter 2D Riccati Matching Report

## Objective

Add a new subsonic 2D PINN workflow that keeps the Stage 0 spectral head locked and improves modal reconstruction through a purely physical left-right Riccati matching constraint, without any direct modal supervision.

## Files Created

- `src/physics/kh_subsonic_riccati_matching_2d.py`
- `src/training/kh_subsonic_trainer_2d_stage1ter_matching.py`
- `scripts/train_kh_subsonic_2d_hybrid4ci_stage1ter_matching.py`
- `scripts/run_kh_subsonic_2d_hybrid4ci_stage1ter_matching.sh`
- `launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1ter_matching.slurm`
- `scripts/render_kh_subsonic_2d_stage1ter_matching_diagnostics.py`
- `repo_stage1ter_2d_riccati_matching_report.md`

## Existing Workflows Left Intact

The current Stage 1 and Stage 1bis workflows were not removed or rewritten. The new Stage 1ter path is additive:

- Stage 0 still provides the locked `c_i(alpha, M)` head.
- Stage 1 and Stage 1bis remain available as previous modal-reconstruction attempts.
- Stage 1ter introduces a separate trainer and separate launch path.

## Scientific Change

### Training-side physics

The new Stage 1ter loss keeps the existing subsonic ingredients:

- pressure/Riccati PDE residual,
- Riccati far-field boundary constraints,
- Stage 0 spectral anchor lock,
- no direct modal labels.

It adds a non-tautological physical constraint:

- integrate the Riccati ODE from the left far field to a set of matching points,
- integrate the Riccati ODE from the right far field to the same matching points,
- compare the network prediction `gamma = p_y / p` against both integrated solutions,
- optionally compare left-integrated and right-integrated solutions directly.

This gives a new `loss_match` term driven only by the Riccati ODE and far-field asymptotics.

### No modal supervision

No classical mode fields are used inside the training loss.

Classical modes appear only in the diagnostics renderer, after training, to assess:

- `p`,
- `rho`,
- `u`,
- `v`,
- `gamma`,
- `p_y`.

## New Stage 1ter Configuration

Main defaults in the new trainer:

- `w_pde = 1.0`
- `w_bc_kappa = 20.0`
- `w_bc_q = 60.0`
- `w_match = 1.0`
- `w_center_pde = 1.0`
- `w_norm = 0.0`
- `w_phase = 0.0`
- `w_ci_anchor = 1.0`
- `n_interior = 512`
- `n_boundary = 96`
- `n_center = 256`
- `center_width = 2.0`
- `center_fraction = 0.5`
- `match_y_values = (-1.0, -0.5, 0.0, 0.5, 1.0)`
- `shoot_ymax = 40.0`
- `shoot_steps = 512`
- `match_warmup_epochs = 1000`

Important runtime defaults:

- `freeze_ci = True`
- `detach_ci_in_mode_branch = True`
- best checkpoint metric defaults to `loss_total`

## Training Outputs

The new trainer writes:

- `model_best.pt`
- `model_final.pt`
- `history.csv`
- `anchor_predictions_stage1ter.csv`
- `stage1ter_report.txt`
- periodic checkpoints when requested

The history now includes:

- `loss_total`
- `loss_pde`
- `loss_bc`
- `loss_bc_kappa`
- `loss_bc_q`
- `loss_match`
- `loss_match_net_left`
- `loss_match_net_right`
- `loss_match_left_right`
- `loss_center_pde`
- `loss_norm`
- `loss_phase`
- `loss_ci_anchor`
- `w_match_effective`
- `gamma_left_right_abs_mean`
- `gamma_left_right_abs_max`
- `ci_anchor_max_abs`
- `ci_anchor_max_rel_unstable`
- `ci_neutral_max_abs`

## Diagnostics Outputs

The new renderer writes:

- `01_stage1ter_ci_vs_alpha_by_mach.png`
- `02_stage1ter_ci_error_heatmap.png`
- `03_stage1ter_loss_history.png`
- modal panels at `M=0.5`, `alpha = 0.30, 0.50, 0.70`
- `gamma` / `p_y` panels at the same points
- `diagnostics_summary.csv`
- `stage1ter_ci_surface_table.csv`
- `README.md`

The renderer uses classical solutions only for post-training comparisons.

## Commands Executed

### Compile checks

```bash
python3 -m py_compile \
  src/physics/kh_subsonic_riccati_matching_2d.py \
  src/training/kh_subsonic_trainer_2d_stage1ter_matching.py \
  scripts/train_kh_subsonic_2d_hybrid4ci_stage1ter_matching.py \
  scripts/render_kh_subsonic_2d_stage1ter_matching_diagnostics.py
```

### Shell checks

```bash
bash -n scripts/run_kh_subsonic_2d_hybrid4ci_stage1ter_matching.sh
bash -n launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1ter_matching.slurm
```

### Smoke training

```bash
MPLCONFIGDIR=/tmp/kh_stage1ter_smoke python3 scripts/train_kh_subsonic_2d_hybrid4ci_stage1ter_matching.py \
  --stage0-checkpoint model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt \
  --mach-values 0.3 0.5 \
  --alpha-min 0.10 \
  --alpha-max 0.80 \
  --epochs 10 \
  --n-interior 64 \
  --n-boundary 16 \
  --n-center 32 \
  --n-alpha-samples 3 \
  --n-mach-samples 2 \
  --shoot-steps 64 \
  --match-y-values -0.5 0.0 0.5 \
  --w-match 1.0 \
  --match-warmup-epochs 5 \
  --device cuda \
  --freeze-ci \
  --detach-ci-in-mode-branch \
  --output-dir model_saved/_smoke_kh_subsonic_2d_stage1ter_matching
```

### Smoke diagnostics

```bash
MPLCONFIGDIR=/tmp/kh_stage1ter_smoke_render python3 scripts/render_kh_subsonic_2d_stage1ter_matching_diagnostics.py \
  --stage1ter-dir model_saved/_smoke_kh_subsonic_2d_stage1ter_matching \
  --output-dir assets/pinn_subsonic/_smoke_2D_Stage1ter_matching \
  --device cpu
```

## Test Results

### Compile and shell checks

All requested compile and shell checks passed.

### Smoke training

The short smoke run completed successfully and produced all expected training artifacts.

Observed smoke indicators:

- `loss_match` is nonzero from epoch 1,
- `w_match_effective` ramps from `0.0` to `1.0`,
- `ci_anchor_max_abs` stayed at about `2.92e-4`,
- `ci_anchor_max_rel_unstable` stayed at about `8.24e-4`.

Representative rows from `history.csv`:

- epoch 1: `loss_total = 1.79e+01`, `loss_match = 9.44e-02`, `w_match_effective = 0.00`
- epoch 10: `loss_total = 9.82e+00`, `loss_match = 3.73e-01`, `w_match_effective = 1.00`

There was one large transient spike during the 10-epoch smoke run, which is acceptable for a smoke test and does not invalidate the workflow. This run was only meant to validate execution, logging, and artifact generation.

### Smoke diagnostics

The diagnostics renderer completed successfully and wrote:

- CI plots,
- CI error heatmap,
- loss history plot,
- modal comparison plots,
- `gamma` / `p_y` plots,
- `diagnostics_summary.csv`.

## Recommended Jean Zay Command

```bash
N_INTERIOR=512 \
N_BOUNDARY=96 \
N_CENTER=256 \
N_ALPHA_SAMPLES=12 \
N_MACH_SAMPLES=4 \
EPOCHS=8000 \
LEARNING_RATE=5e-5 \
W_BC_KAPPA=20.0 \
W_BC_Q=60.0 \
W_MATCH=1.0 \
W_CENTER_PDE=1.0 \
MATCH_WARMUP_EPOCHS=1000 \
SHOOT_YMAX=40.0 \
SHOOT_STEPS=512 \
MATCH_Y_VALUES="-1.0 -0.5 0.0 0.5 1.0" \
FREEZE_CI=1 \
DETACH_CI_IN_MODE_BRANCH=1 \
OUTPUT_DIR=model_saved/kh_subsonic_2d_hybrid4ci_stage1ter_riccati_matching \
sbatch launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage1ter_matching.slurm
```

## Remaining Limitations

- The smoke test validates execution, not final modal quality.
- The current matching term constrains `gamma` physically, but full improvement on `u` and `v` still needs long runs to be evaluated.
- Center-focused sampling helps the shear core, but may still need tuning through `n_center`, `center_width`, and `w_center_pde`.
- The best metric is still `loss_total` by default; depending on long-run behavior, a different checkpoint criterion may become preferable.
