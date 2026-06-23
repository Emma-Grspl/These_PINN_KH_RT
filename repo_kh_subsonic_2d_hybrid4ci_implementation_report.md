# KH Subsonic 2D Hybrid4CI Implementation Report

## Scope

This first implementation step adds only:

- a parametric subsonic 2D `Stage 0` spectral lock on sparse `c_i` anchors;
- light `c_i` diagnostics;
- no physics-mode training yet;
- no direct modal supervision.

## Files Created

- `src/models/kh_subsonic_pinn_2d.py`
- `src/data/kh_subsonic_sampling_2d.py`
- `src/training/kh_subsonic_trainer_2d_hybrid4ci.py`
- `scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py`
- `scripts/render_kh_subsonic_2d_hybrid4ci_diagnostics.py`
- `scripts/run_kh_subsonic_2d_hybrid4ci_stage0_anchor_lock.sh`
- `launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage0.slurm`

## Files Modified

- none

## 2D PINN Architecture

- The new 2D entry point is `KHSubsonicPINN2D`.
- It reuses the validated existing multi-Mach architecture already present in `src/models/kh_subsonic_pinn.py`.
- Inputs: `xi`, `alpha`, `Mach`.
- Spectral head: predicts `c_i(alpha, Mach)`.
- Mode branch remains present in the model definition for future Stage 1 use, but Stage 0 freezes it entirely.

## Hybrid4CI Anchor Definition

- For each Mach value `M_j`, Stage 0 supervises only 4 spectral anchors.
- Default anchor alphas:
  - `0.10`
  - `0.30`
  - `0.55`
  - `0.80`
- Total number of anchors:
  - `4 * len(mach_values)`

## Classical Reference Logic

- First choice: use `assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv` if available.
- If the cache is absent or insufficient, fallback to `classical_solver/subsonic/mstab17_subsonic_solver.py`.
- All anchors actually used are saved in `anchors_used.csv`.
- Stable or neutral reference values with `c_i = 0` are preserved as-is when they exist in the cache.

## Stage 0 Training

- Script:
  - `scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py`
- Output directory default:
  - `model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock`
- Loss terms:
  - anchor fit on sparse `c_i`
  - monotonicity penalty in `alpha`
  - optional smoothness penalty in `alpha`
  - optional smoothness penalty in `Mach`
- No PDE loss.
- No Riccati loss.
- No boundary-condition loss.
- No direct modal supervision.

## Diagnostics

- Script:
  - `scripts/render_kh_subsonic_2d_hybrid4ci_diagnostics.py`
- Output directory default:
  - `assets/pinn_subsonic/2d_hybrid4ci_diagnostics`
- Generated figures:
  - `01_stage0_ci_anchor_fit.png`
  - `02_ci_surface_classic_vs_pinn.png`
  - `03_ci_relative_error_map.png`
- Summary CSV:
  - `diagnostics_summary.csv`

## Commands

### Stage 0 local

```bash
python scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py \
  --mach-values 0.1 0.3 0.5 0.7 \
  --alpha-min 0.10 \
  --alpha-max 0.80 \
  --anchor-alphas 0.10 0.30 0.55 0.80 \
  --epochs 2000 \
  --output-dir model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock
```

### Stage 0 wrapper

```bash
bash scripts/run_kh_subsonic_2d_hybrid4ci_stage0_anchor_lock.sh
```

### Stage 0 Jean Zay

```bash
sbatch launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage0.slurm
```

### Diagnostics

```bash
python scripts/render_kh_subsonic_2d_hybrid4ci_diagnostics.py \
  --stage0-dir model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock
```

## Tests Executed

- Python compilation:
  - `python -m py_compile src/models/kh_subsonic_pinn_2d.py src/data/kh_subsonic_sampling_2d.py src/training/kh_subsonic_trainer_2d_hybrid4ci.py scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py scripts/render_kh_subsonic_2d_hybrid4ci_diagnostics.py`
  - result: pass
- Shell syntax:
  - `bash -n scripts/run_kh_subsonic_2d_hybrid4ci_stage0_anchor_lock.sh`
  - `bash -n launch/jz_submit_kh_subsonic_2d_hybrid4ci_stage0.slurm`
  - result: pass
- Stage 0 smoke test:
  - `python scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py --mach-values 0.3 0.5 --anchor-alphas 0.10 0.30 0.55 0.80 --epochs 5 --audit-every 1 --output-dir model_saved/_smoke_kh_subsonic_2d_hybrid4ci_stage0`
  - result: pass
  - expected scientific quality: fail, because 5 epochs are only a structural smoke test
- Diagnostics smoke test:
  - `python scripts/render_kh_subsonic_2d_hybrid4ci_diagnostics.py --stage0-dir model_saved/_smoke_kh_subsonic_2d_hybrid4ci_stage0 --output-dir assets/pinn_subsonic/2d_hybrid4ci_diagnostics/_smoke_stage0 --grid-alpha 21 --grid-mach 11`
  - result: pass

## Remaining Limits

- No Stage 1 physics-mode training in this first step.
- No Stage 1bis refinement in this first step.
- No modal post-processing figures yet.
- The current diagnostics validate only the spectral `c_i(alpha, Mach)` behavior.

## Explicit Confirmation

This implementation adds **no direct modal supervision**.
