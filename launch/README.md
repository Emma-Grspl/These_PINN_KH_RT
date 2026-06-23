# Launch Jobs

This directory is intended to expose the currently active Slurm entry points.

Active PINN subsonic launchers:

- `jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci4.slurm`
- `jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci8.slurm`
- `jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci16.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci4.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci8.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci16.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci4.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci8.slurm`
- `jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci16.slurm`
- `jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm`

Active supersonic shooting launchers:

- `jz_submit_supersonic_shooting_point_batch.slurm`
- `jz_submit_supersonic_shooting_point_batch_M140.slurm`
- `jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm`
- `jz_submit_supersonic_shooting_blumen_locked.slurm`
- `jz_submit_supersonic_shooting_reference_package.slurm`
- `jz_submit_supersonic_shooting_intermediate_mach_spectral.slurm`
- `jz_submit_supersonic_shooting_modal_front_spectral.slurm`
- `jz_submit_supersonic_shooting_modal_front_spectral_M140_M150_branch_guided.slurm`

Active general classical launcher:

- `jz_submit_subsonic.slurm`

Files not listed above but still present in this directory are intentionally left in `REVIEW`.
They were not moved because their role is ambiguous or diagnostic, and this cleanup step avoids
moving ambiguous files.

Historical launchers are under `_quarantine/launch_historical/`.
GEP diagnostic launchers are under `_quarantine/launch_gep_diagnostics/`.
