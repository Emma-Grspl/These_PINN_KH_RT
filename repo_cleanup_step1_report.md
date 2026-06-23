# Repo Cleanup Step 1 Report

Date: 2026-06-23

Scope executed: safe organization step only.

No scientific refactor was done. No equations, losses, hyperparameters, results, CSV, checkpoints, or assets were modified intentionally.

## Directories Created

- `scripts/reference_classique/`
- `scripts/reference_classique/subsonique/`
- `scripts/reference_classique/supersonique/`
- `_quarantine/`
- `_quarantine/modal_supervision_experiments/`

## Wrappers Created

Subsonic wrappers:

- `scripts/reference_classique/subsonique/run_hybrid_subsonic_scan.py`
  - Redirects to `classical_solver/subsonic/hybrid_subsonic_scan.py`.
  - Preserves CLI arguments via `runpy.run_path(..., run_name="__main__")`.

- `scripts/reference_classique/subsonique/reconstruct_blumen_subsonic_robust.py`
  - Redirects to `classical_solver/subsonic/reconstruct_blumen_subsonic_robust.py`.
  - Preserves CLI arguments via `runpy.run_path(..., run_name="__main__")`.

Supersonic wrappers:

- `scripts/reference_classique/supersonique/audit_shooting_point_batch.py`
  - Redirects to `scripts/audit_supersonic_shooting_point_batch.py`.

- `scripts/reference_classique/supersonique/audit_shooting_point_batch_branch_guided.py`
  - Redirects to `scripts/audit_supersonic_shooting_point_batch_branch_guided.py`.

- `scripts/reference_classique/supersonique/build_shooting_reference_package.py`
  - Redirects to `scripts/build_supersonic_shooting_reference_package.py`.

- `scripts/reference_classique/supersonique/build_validated_modal_package.py`
  - Redirects to `scripts/build_supersonic_validated_modal_package.py`.

Documentation wrapper folder:

- `scripts/reference_classique/README.md`
  - Explains that this folder contains human-facing entry points.
  - States that real modules remain in `classical_solver/` or `scripts/` to avoid breaking imports.
  - Clarifies that the supersonic reference mainly uses shooting.
  - Clarifies that GEP remains diagnostic.

## Files Moved to Quarantine

Moved with `git mv` into `_quarantine/modal_supervision_experiments/`:

- `scripts/run_kh_subsonic_highalpha_classic_mode_supervision_test.py`
- `scripts/run_kh_subsonic_highalpha_classic_full_mode_supervision_test.py`
- `scripts/run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py`
- `scripts/run_kh_subsonic_highalpha_classic_two_stage_repair_test.py`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_mode_supervision.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_full_mode_supervision.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_balanced_full_mode_supervision.slurm`

Not moved:

- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_two_stage_repair.slurm`

Reason: the instruction said to move associated launchers containing both `highalpha_classic` and `mode_supervision`. This file contains `highalpha_classic` but not `mode_supervision`, so it was left unchanged.

## Checks Run

Wrapper compilation:

```bash
python3 -m py_compile \
  scripts/reference_classique/subsonique/run_hybrid_subsonic_scan.py \
  scripts/reference_classique/subsonique/reconstruct_blumen_subsonic_robust.py \
  scripts/reference_classique/supersonique/audit_shooting_point_batch.py \
  scripts/reference_classique/supersonique/audit_shooting_point_batch_branch_guided.py \
  scripts/reference_classique/supersonique/build_shooting_reference_package.py \
  scripts/reference_classique/supersonique/build_validated_modal_package.py
```

Result: PASS.

PINN core CLI compilation:

```bash
python3 -m py_compile scripts/train_kh_subsonic_pinn.py
```

Result: PASS.

Stage 0 CI anchor lock compilation:

```bash
python3 -m py_compile scripts/train_kh_subsonic_ci_stage0_anchor_lock.py
```

Result: PASS.

Active shell launcher syntax checks:

```bash
bash -n scripts/run_kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock.sh
bash -n scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh
bash -n scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined.sh
```

Result: PASS for all three.

## Check Artifacts

`python3 -m py_compile` generated local `__pycache__/` files under `scripts/reference_classique/`.

They were not removed because this step was constrained to avoid deletion. They should not be committed.

## Points Left Voluntarily Unchanged

- `assets/` was not touched.
- `src/` modules were not moved.
- `classical_solver/` modules were not moved.
- Original scripts wrapped by `scripts/reference_classique/` were not moved.
- Imports were not changed.
- Slurm files for active Stage 0 / Stage 1 / Stage 1bis were not changed.
- GEP scripts were not moved; they remain diagnostic.
- Other `REVIEW` or historical scripts from `repo_cleanup_plan.md` were left unchanged.
- Existing unrelated local modifications/untracked files in the worktree were left unchanged.

## Notes

During quarantine, one parallel `git mv` temporarily hit `.git/index.lock`. The lock disappeared after the other `git mv` operations completed, and the remaining move was redone sequentially.

No destructive cleanup was performed.

