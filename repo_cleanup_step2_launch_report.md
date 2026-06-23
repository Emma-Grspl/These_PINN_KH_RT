# Repo Cleanup Step 2 Launch Report

Date: 2026-06-23

Scope executed: `launch/` cleanup only.

No Python file was modified. No asset was touched. No Slurm file content was modified. No file was
deleted.

## Directories Created

- `_quarantine/launch_historical/`
- `_quarantine/launch_gep_diagnostics/`

## Documentation Created

- `launch/README.md`
- `_quarantine/launch_historical/README.md`
- `_quarantine/launch_gep_diagnostics/README.md`

## Launchers Kept as Active

PINN subsonic active launchers:

- `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci4.slurm`
- `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci8.slurm`
- `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci16.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci4.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci8.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci16.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci4.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci8.slurm`
- `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci16.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm`

Supersonic shooting active launchers:

- `launch/jz_submit_supersonic_shooting_point_batch.slurm`
- `launch/jz_submit_supersonic_shooting_point_batch_M140.slurm`
- `launch/jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm`
- `launch/jz_submit_supersonic_shooting_blumen_locked.slurm`
- `launch/jz_submit_supersonic_shooting_reference_package.slurm`
- `launch/jz_submit_supersonic_shooting_intermediate_mach_spectral.slurm`
- `launch/jz_submit_supersonic_shooting_modal_front_spectral.slurm`
- `launch/jz_submit_supersonic_shooting_modal_front_spectral_M140_M150_branch_guided.slurm`

General classical launcher:

- `launch/jz_submit_subsonic.slurm`

## Launchers Moved to Historical Quarantine

Moved with `git mv` into `_quarantine/launch_historical/`:

- `launch/jz_submit.slurm`
- `launch/jz_submit_cpu.slurm`
- `launch/jz_submit_kh_subsonic_mode_error_8pt_vs_first_order_real.slurm`
- `launch/jz_submit_kh_subsonic_mode_error_8pt_vs_modefocus.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_band_M05_M07.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_ci_mode_campaign.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_m0007_neutral.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_m0008_neutral.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_m0008_neutral_h160.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_m0009_neutral_h160.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_multibranch.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_multibranch_riccati.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_multibranch_targeted.slurm`
- `launch/jz_submit_kh_subsonic_pinn_2d_pilot_M05_M06.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M00.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M03.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M07.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_ampphase_multibranch.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_ampphase.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_ampphase_locloss.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_ampphase_modeaudit.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_ampphase_peakloss.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_h192_d6.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_anchor_max_logampphase.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_ci_supervision_budget.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_edge_repair_sweep.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_first_order_real.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_first_order_real_stabilized.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_continuation.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_mode_repair.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_stepwise.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_targeted.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_modefocus.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_modefocus_lowalpha.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_modefocus_sym.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_modefocus_v2.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_core1d.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_core1d_windowed_sparse.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_exp1_q_lowalpha.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_experts.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_experts_anchor.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_experts_light_anchor.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_lowalpha_gamma_repair.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_lowalpha_repair.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_mode_repair_edges.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_modeaudit.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_multibranch.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_multibranch_refined.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_neutral_bcq.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_pure_physics.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_pure_physics_ci_selection.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_spectral_alpha_sweep.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_twostage.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_riccati_twostage_soft.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_ampphase.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_anchor_band.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_anchor_max.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_anchor_point.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_anchor_point_max.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_external_ci_search.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_external_ci_search_v2.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_riccati_pure.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_riccati_pure_centered.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_riccati_pure_scalar_ci.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_a050_m050_riccati_pure_scalar_ci_matching.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_riccati_pure_scalar_ci_matching_param.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_riccati_pure_scalar_ci_matching_phaseA_bis_param.slurm`
- `launch/jz_submit_kh_subsonic_pinn_singlecase_riccati_pure_scalar_ci_matching_phaseA_ter_param.slurm`

Justification: these names match explicit historical categories: `anchor_max`, `ampphase`,
`modefocus`, `highalpha`, `singlecase`, `first_order_real`, `edge_repair`, old `riccati_*`,
`M00`, `M03`, `M07`, old `2d`, broken `ci_supervision_budget`, or old generic launchers.

## Launchers Moved to GEP Diagnostic Quarantine

Moved with `git mv` into `_quarantine/launch_gep_diagnostics/`:

- `launch/jz_merge_gep_supersonic_mode_database_global.slurm`
- `launch/jz_merge_gep_supersonic_mode_database_global_composite.slurm`
- `launch/jz_submit_gep_param_audit.slurm`
- `launch/jz_submit_gep_param_audit_a021_m170.slurm`
- `launch/jz_submit_gep_param_audit_a021_m180.slurm`
- `launch/jz_submit_gep_supersonic_1d_alpha_bidirectional.slurm`
- `launch/jz_submit_gep_supersonic_1d_alpha_sweep.slurm`
- `launch/jz_submit_gep_supersonic_1d_mach_bidirectional.slurm`
- `launch/jz_submit_gep_supersonic_1d_mach_sweep.slurm`
- `launch/jz_submit_gep_supersonic_asymptotic_box.slurm`
- `launch/jz_submit_gep_supersonic_blumen_audit.slurm`
- `launch/jz_submit_gep_supersonic_blumen_guided_sweep.slurm`
- `launch/jz_submit_gep_supersonic_branch_beam_search.slurm`
- `launch/jz_submit_gep_supersonic_branch_continuation_blumen.slurm`
- `launch/jz_submit_gep_supersonic_candidate_modes_near_blumen.slurm`
- `launch/jz_submit_gep_supersonic_cluster_family_audit.slurm`
- `launch/jz_submit_gep_supersonic_compare_candidates_vs_shooting.slurm`
- `launch/jz_submit_gep_supersonic_family_clustering.slurm`
- `launch/jz_submit_gep_supersonic_local_highmach_recipes.slurm`
- `launch/jz_submit_gep_supersonic_local_mode_families.slurm`
- `launch/jz_submit_gep_supersonic_local_spectrum.slurm`
- `launch/jz_submit_gep_supersonic_mach_branch_tracking.slurm`
- `launch/jz_submit_gep_supersonic_modal_branch_selector.slurm`
- `launch/jz_submit_gep_supersonic_modal_branch_selector_calibration.slurm`
- `launch/jz_submit_gep_supersonic_mode_database.slurm`
- `launch/jz_submit_gep_supersonic_mode_database_global_array.slurm`
- `launch/jz_submit_gep_supersonic_mode_database_global_composite_array.slurm`
- `launch/jz_submit_gep_supersonic_raw_spectrum_vs_blumen.slurm`
- `launch/jz_submit_gep_supersonic_shooting_vs_families.slurm`
- `launch/jz_submit_gep_validity_frontier.slurm`
- `launch/jz_validate_gep_supersonic_branch_selector.slurm`

Justification: GEP remains useful as a diagnostic, but the main supersonic reference protocol is
shooting-based.

## Launchers Left in REVIEW

These files remain in `launch/` because they are not explicitly active, but their role is ambiguous
or diagnostic enough that this safe cleanup step should not move them automatically:

- `launch/jz_submit_kh_subsonic_fixed_mach_modal_candidates_compare.slurm`
- `launch/jz_submit_kh_subsonic_mode_error_vs_alpha.slurm`
- `launch/jz_submit_kh_subsonic_pinn.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci4_reference.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci8_reference.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci16_reference.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_ci_supervision_vs_physics.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_ci_window_ablation.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_external_ci_alpha_sweep.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_hybrid_alpha_sweep.slurm`
- `launch/jz_submit_kh_subsonic_pinn_M05_lowalpha_targeted.slurm`
- `launch/jz_submit_supersonic_blumen_ci_visual.slurm`
- `launch/jz_submit_supersonic_blumen_cr_uncertainty.slurm`
- `launch/jz_submit_supersonic_blumen_cr_visual.slurm`
- `launch/jz_submit_supersonic_blumen_eigenconditions.slurm`
- `launch/jz_submit_supersonic_blumen_local_reference.slurm`
- `launch/jz_submit_supersonic_shooting_ci_alpha_continuation.slurm`
- `launch/jz_submit_supersonic_shooting_ci_alpha_lines.slurm`
- `launch/jz_submit_supersonic_shooting_ci_alpha_spectral_continuation.slurm`
- `launch/jz_submit_supersonic_shooting_ci_map.slurm`
- `launch/jz_submit_supersonic_shooting_mach_continuation.slurm`
- `launch/jz_submit_supersonic_shooting_modal_surface.slurm`
- `launch/jz_submit_supersonic_shooting_modal_surface_M140_branch_guided.slurm`
- `launch/jz_submit_supersonic_shooting_mode_structure.slurm`
- `launch/jz_submit_supersonic_shooting_multistart.slurm`
- `launch/jz_submit_supersonic_shooting_visual_validation.slurm`

## Checks

Required check:

```bash
bash -n launch/*.slurm
```

Result: see final command output from this cleanup step.

Observed result: PASS. All remaining `launch/*.slurm` files passed `bash -n`.

## Points Left Unchanged

- No `.py` file was modified.
- No `assets/` file was touched.
- No Slurm file kept in `launch/` was edited.
- No file was deleted.
- Ambiguous launchers were left in place and listed in `REVIEW`.
