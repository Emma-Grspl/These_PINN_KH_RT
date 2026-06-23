# Repo Cleanup Plan

Date: 2026-06-23

Scope: audit only. No move, no delete, no import rewrite, no asset change.

Current branch observed during audit: `cleanup-repo`.

Important working-tree note: the repository already contains unrelated local modifications and untracked outputs. This plan does not rely on moving or deleting any of them.

## Cleanup Policy

Categories used in this report:

- `KEEP`: required by current pipelines, imported by current code, or clear active entry point.
- `MOVE_TO_REFERENCE_CLASSIQUE`: executable reference-classic entry point that could be moved later to `scripts/reference_classique/...`, with wrapper or import update.
- `DIAGNOSTIC`: useful analysis, plotting, comparison, validation, or one-off scientific audit. Keep accessible, but not part of the core pipeline.
- `REVIEW`: ambiguous, old experiment, risky dependency, or unclear status. Do not move/delete without checking logs and imports.
- `QUARANTINE`: probably obsolete, duplicated, or incompatible with the current stated protocol. Candidate for later `_quarantine/`, not deletion.

Non-goals for this first step:

- Do not touch `assets/`.
- Do not touch produced CSV/PDF/PNG/checkpoints.
- Do not move modules under `src/` or `classical_solver/` unless a later refactor updates imports carefully.
- Do not move anything under `archive/`; it is already archived.

## Target Organization

Desired high-level entry points later:

- `scripts/reference_classique/subsonique/`
- `scripts/reference_classique/supersonique/`

Recommendation: only move executable wrappers and plotting/report builders there. Keep imported modules in `classical_solver/`, `src/`, and shared helper scripts unless wrappers are added.

## Current Main Pipelines

### Reference Classique Subsonique

Status: already frozen and usable.

Core modules:

| File | Category | Role | Justification |
|---|---|---|---|
| `classical_solver/subsonic/mstab17_subsonic_solver.py` | KEEP | Mstab17-style subsonic reference solver | Imported by trainer, plotting, and subsonic reference builders. Do not move. |
| `classical_solver/subsonic/robust_subsonic_shooting.py` | KEEP | Robust subsonic shooting wrapper/reference cache | Imported by `src/data/kh_subsonic_sampling.py`, Stage 0, and comparison scripts. Do not move. |
| `classical_solver/subsonic/shooting_subsonic.py` | KEEP | Legacy/direct subsonic shooting implementation | Still used by robust wrapper and comparison scripts. |
| `classical_solver/subsonic/hybrid_subsonic_scan.py` | MOVE_TO_REFERENCE_CLASSIQUE | Executable dense subsonic scan entry point | Launched by `launch/jz_submit_subsonic.slurm`; good candidate for `scripts/reference_classique/subsonique/`. |
| `classical_solver/subsonic/reconstruct_blumen_subsonic_robust.py` | MOVE_TO_REFERENCE_CLASSIQUE | Reconstruct subsonic Blumen maps using robust shooting | Executable reference builder; candidate for target folder. |
| `classical_solver/subsonic/reconstruct_blumen_subsonic_shooting.py` | REVIEW | Older subsonic reconstruction via direct shooting | Probably superseded by robust version, but may be useful for comparison. |
| `classical_solver/subsonic/compare_subsonic_shooting_solvers.py` | DIAGNOSTIC | Compare subsonic solvers | Useful diagnostic, not core pipeline. |
| `classical_solver/subsonic/plot_subsonic_ci_map.py` | DIAGNOSTIC | Plot subsonic `ci` map | Figure/post-processing utility. |
| `classical_solver/subsonic/plot_subsonic_error_map.py` | DIAGNOSTIC | Plot subsonic error map | Figure/post-processing utility. |
| `classical_solver/subsonic/README.md` | KEEP | Subsonic solver notes | Documentation. |
| `classical_solver/subsonic/__init__.py` | KEEP | Package marker | Required for imports. |

Proposed later move to `scripts/reference_classique/subsonique/`:

- `classical_solver/subsonic/hybrid_subsonic_scan.py`
- `classical_solver/subsonic/reconstruct_blumen_subsonic_robust.py`

Move caveat: if moved directly, current Slurm paths and imports may break. Prefer adding thin wrappers in the target folder first, then update Slurm after tests.

### Reference Classique Supersonique

Status: active reference is shooting. GEP remains diagnostic.

Core shooting modules and entry points:

| File | Category | Role | Justification |
|---|---|---|---|
| `classical_solver/supersonic/mstab17_supersonic_solver.py` | KEEP | Supersonic ODE/shooting backend | Imported by shooting diagnostics and GEP comparisons. Do not move. |
| `classical_solver/supersonic/shooting_supersonic.py` | KEEP | Supersonic shooting utilities | Used by Blumen reconstruction scripts. |
| `classical_solver/supersonic/blumen_reference.py` | KEEP | Digitized Blumen reference utilities | Imported widely by supersonic plotting and audits. |
| `scripts/track_supersonic_shooting_multistart.py` | KEEP | Multistart shooting backend | Imported by point batch, branch guided, continuation, and package builders. Do not move without import refactor. |
| `scripts/audit_supersonic_shooting_point_batch.py` | MOVE_TO_REFERENCE_CLASSIQUE | Current pointwise supersonic shooting audit | Main active pointwise shooting pipeline with box robustness. Candidate for `scripts/reference_classique/supersonique/`. |
| `scripts/audit_supersonic_shooting_point_batch_branch_guided.py` | MOVE_TO_REFERENCE_CLASSIQUE | Branch-guided pointwise shooting audit | Active reference expansion path around intermediate Mach. Candidate for target folder. |
| `scripts/audit_supersonic_shooting_blumen_locked.py` | MOVE_TO_REFERENCE_CLASSIQUE | Blumen-locked shooting audit | Recent active diagnostic/reference variant; candidate for target folder if kept. |
| `scripts/audit_supersonic_shooting_visual_validation.py` | KEEP | Reconstruct fields and visual validation helpers | Imported by current shooting scripts. Keep in place or move only after import rewrite. |
| `scripts/audit_supersonic_shooting_ci_map.py` | KEEP | CI scoring, seeds, profile diagnostics | Imported by point batch and continuation scripts. Keep in place unless module path refactored. |
| `scripts/audit_supersonic_families_against_blumen.py` | KEEP | Blumen targets and family comparison helpers | Imported by many supersonic scripts. Keep as shared helper. |
| `scripts/audit_supersonic_shooting_ci_alpha_continuation.py` | DIAGNOSTIC | CI-alpha continuation audit | Useful for branch following; not the primary pointwise reference now. |
| `scripts/audit_supersonic_shooting_ci_alpha_lines.py` | DIAGNOSTIC | CI-alpha line audit | Older/specialized diagnostic. |
| `scripts/audit_supersonic_shooting_mode_structure.py` | DIAGNOSTIC | Mode structure audit | Useful to understand numerical/physical modes. |
| `scripts/audit_supersonic_shooting_vs_gep_families.py` | DIAGNOSTIC | Compare shooting and GEP families | Diagnostic only. |
| `scripts/track_supersonic_shooting_mach_continuation.py` | DIAGNOSTIC | Mach continuation shooting | Useful branch diagnostic, not current reference base. |
| `scripts/track_supersonic_shooting_modal_surface.py` | DIAGNOSTIC | Modal surface shooting tracker | Diagnostic/reference expansion helper. |
| `scripts/densify_supersonic_intermediate_mach_spectral.py` | DIAGNOSTIC | Densify intermediate Mach spectral points | Useful expansion tool. |
| `scripts/densify_supersonic_modal_front_spectral.py` | DIAGNOSTIC | Densify modal front/spectral points | Useful expansion tool. |
| `scripts/build_supersonic_shooting_reference_package.py` | MOVE_TO_REFERENCE_CLASSIQUE | Build packaged shooting reference | Final package builder; candidate for target folder. |
| `scripts/build_supersonic_validated_modal_package.py` | MOVE_TO_REFERENCE_CLASSIQUE | Build validated modal/spectral point package | Important packaging step for sparse supersonic reference. |
| `scripts/build_supersonic_reference_tables.py` | DIAGNOSTIC | Build/reference table summaries | Useful post-processing. |
| `classical_solver/supersonic/reconstruct_blumen_supersonic_shooting.py` | MOVE_TO_REFERENCE_CLASSIQUE | Supersonic Blumen reconstruction by shooting | Executable reference-classic entry point. |
| `classical_solver/supersonic/scan_mstab17_supersonic_local.py` | DIAGNOSTIC | Local supersonic scan | Useful diagnostic around points. |
| `classical_solver/supersonic/plot_mstab17_supersonic_point_checks.py` | DIAGNOSTIC | Plot point checks | Diagnostic plot. |
| `classical_solver/supersonic/__init__.py` | KEEP | Package marker | Required for imports. |

Proposed later move to `scripts/reference_classique/supersonique/`:

- `scripts/audit_supersonic_shooting_point_batch.py`
- `scripts/audit_supersonic_shooting_point_batch_branch_guided.py`
- `scripts/audit_supersonic_shooting_blumen_locked.py`
- `scripts/build_supersonic_shooting_reference_package.py`
- `scripts/build_supersonic_validated_modal_package.py`
- `classical_solver/supersonic/reconstruct_blumen_supersonic_shooting.py`

Move caveat: `audit_supersonic_shooting_point_batch.py` is imported by many other scripts. Do not move it directly unless all imports are updated or a compatibility wrapper remains at the old path.

### GEP Supersonique

Status: diagnostic, not primary reference.

| File family | Category | Role | Justification |
|---|---|---|---|
| `classical_solver/gep/dense_gep_notebook_style.py` | DIAGNOSTIC | Core GEP solver | Needed for GEP diagnostics, but not primary reference. |
| `classical_solver/gep/build_supersonic_mode_database.py` | DIAGNOSTIC | Build GEP mode database | Launched by several GEP Slurm jobs; keep as diagnostic. |
| `classical_solver/gep/run_supersonic_mode_database_chunk.py` | DIAGNOSTIC | Chunked GEP DB run | Diagnostic HPC helper. |
| `classical_solver/gep/merge_supersonic_mode_database_chunks.py` | DIAGNOSTIC | Merge GEP chunks | Diagnostic HPC helper. |
| `classical_solver/gep/audit_gep_parameter_grid.py` | DIAGNOSTIC | GEP parameter audit | Useful to explain why GEP is not primary reference. |
| `classical_solver/gep/audit_local_supersonic_spectrum.py` | DIAGNOSTIC | Local spectrum audit | Diagnostic. |
| `classical_solver/gep/audit_supersonic_blumen_points.py` | DIAGNOSTIC | GEP vs Blumen audit | Diagnostic. |
| `classical_solver/gep/audit_supersonic_gep_asymptotic_box.py` | DIAGNOSTIC | GEP box/asymptotic audit | Diagnostic. |
| `classical_solver/gep/adaptive_continuation_sweep_gep.py` | DIAGNOSTIC | GEP continuation sweep | Diagnostic. |
| `classical_solver/gep/adaptive_gep_guided_by_shooting.py` | REVIEW | Hybrid GEP guided by shooting | Could be useful but not current reference. |
| `classical_solver/gep/scan_gep_supersonic_local.py` | DIAGNOSTIC | Local GEP scan | Diagnostic. |
| `classical_solver/gep/scan_gep_validity_frontier.py` | DIAGNOSTIC | GEP validity frontier | Diagnostic. |
| `classical_solver/gep/compare_gep_vs_shooting_supersonic.py` | DIAGNOSTIC | Compare GEP to shooting | Useful for appendix/validation. |
| `classical_solver/gep/compare_sweep_gep_vs_shooting.py` | DIAGNOSTIC | Sweep comparison | Useful diagnostic. |
| `classical_solver/gep/plot_*.py` | DIAGNOSTIC | GEP plotting utilities | Figures only. |
| `classical_solver/gep/sweep_alpha_gep_notebook_style.py` | REVIEW | Older alpha sweep GEP | Likely legacy, but harmless diagnostic. |
| `classical_solver/gep/__init__.py` | KEEP | Package marker | Required for imports. |

Recommendation: keep GEP under `classical_solver/gep/` as diagnostic backend. Do not mix it into `scripts/reference_classique/supersonique/` except possibly wrapper scripts named explicitly as diagnostics.

### PINN Subsonique

Current stated protocol:

- Reconstruct `c_i(alpha)`.
- Reconstruct modes in post-processing.
- Classical supervision only on `c_i`.
- No modal supervision in the loss.
- Mode constrained by PDE, Riccati, BC, normalization, phase, and physical shooting/matching when present.

Core model/training modules:

| File | Category | Role | Justification |
|---|---|---|---|
| `src/models/kh_subsonic_pinn.py` | KEEP | PINN model definitions | Central model. Do not move. |
| `src/training/kh_subsonic_trainer.py` | KEEP | Main fixed-Mach subsonic trainer | Central active trainer. |
| `src/training/kh_subsonic_trainer_2d.py` | REVIEW | 2D alpha/Mach trainer | Relevant long-term, but current immediate work is fixed Mach. |
| `src/data/kh_subsonic_sampling.py` | KEEP | Sampling/reference caches | Imported by trainer and many scripts. |
| `src/physics/kh_subsonic_residual.py` | KEEP | PDE/Riccati residuals and reconstruction | Central physics. |
| `src/**/__init__.py` | KEEP | Package markers | Required for imports. |
| `scripts/train_kh_subsonic_pinn.py` | KEEP | Main PINN CLI | Used by most current subsonic Slurm jobs. |
| `scripts/train_kh_subsonic_ci_stage0_anchor_lock.py` | KEEP | Stage 0 CI anchor lock | Active `ci4/ci8/ci16` sparse spectral pipeline. |
| `scripts/run_kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock.sh` | KEEP | Stage 0 launcher wrapper | Active sparse spectral pipeline. |
| `scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh` | KEEP | Stage 1 mode-from-CI wrapper | Active no-modal-supervision pipeline. |
| `scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined.sh` | KEEP | Stage 1bis physics-refined wrapper | Active no-modal-supervision refinement. |
| `scripts/run_kh_subsonic_M05_alpha010_080_pure_physics_reference.sh` | KEEP | Pure-physics alpha sweep baseline | Current baseline comparison. |
| `scripts/run_kh_subsonic_M05_alpha010_080_ci_sparse_reference.sh` | KEEP | Older sparse CI reference wrapper | Keep until superseded by Stage 0/1/1bis reports. |
| `scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching_phaseA_ter.sh` | KEEP | Recent pure-physics single-case baseline | Used for frozen diagnostic baseline. |
| `scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching_phaseA_bis.sh` | REVIEW | Phase A-bis predecessor | Keep until final baseline provenance is clarified. |
| `scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching.sh` | REVIEW | Older scalar CI matching | Likely predecessor. |
| `scripts/run_kh_subsonic_singlecase_pure_physics*.sh` | REVIEW | Older single-case variants | Historical experiments; do not delete until baseline provenance checked. |
| `scripts/train_kh_subsonic_pinn_2d.py` | REVIEW | 2D PINN CLI | Long-term target, but current fixed-Mach protocol differs. |
| `scripts/run_kh_subsonic_pinn_2d_*.py` | REVIEW | 2D campaigns | Keep for future 2D work, but not current fixed-Mach cleanup priority. |

Active launch files for current PINN protocol:

| File | Category | Role | Justification |
|---|---|---|---|
| `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci4.slurm` | KEEP | Stage 0 sparse CI 4 anchors | Active. |
| `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci8.slurm` | KEEP | Stage 0 sparse CI 8 anchors | Active. |
| `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci16.slurm` | KEEP | Stage 0 sparse CI 16 anchors | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci4.slurm` | KEEP | Stage 1 mode training from Stage 0 `ci4` | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci8.slurm` | KEEP | Stage 1 mode training from Stage 0 `ci8` | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci16.slurm` | KEEP | Stage 1 mode training from Stage 0 `ci16` | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci4.slurm` | KEEP | Stage 1bis physics-refined `ci4` | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci8.slurm` | KEEP | Stage 1bis physics-refined `ci8` | Active. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci16.slurm` | KEEP | Stage 1bis physics-refined `ci16` | Active. |
| `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm` | KEEP | Pure-physics fixed-Mach alpha sweep | Active baseline. |
| `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci4_reference.slurm` | REVIEW | Older hybrid sparse reference `ci4` | Superseded by Stage 0/1. |
| `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci8_reference.slurm` | REVIEW | Older hybrid sparse reference `ci8` | Superseded by Stage 0/1. |
| `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci16_reference.slurm` | REVIEW | Older hybrid sparse reference `ci16` | Superseded by Stage 0/1. |

PINN scripts probably incompatible with current "no modal supervision" rule:

| File | Category | Role | Justification |
|---|---|---|---|
| `scripts/run_kh_subsonic_highalpha_classic_mode_supervision_test.py` | QUARANTINE | Classical modal supervision test | Violates current no-modal-supervision protocol. Keep only as historical negative test. |
| `scripts/run_kh_subsonic_highalpha_classic_full_mode_supervision_test.py` | QUARANTINE | Full modal supervision test | Violates current no-modal-supervision protocol. |
| `scripts/run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py` | QUARANTINE | Balanced full modal supervision test | Violates current no-modal-supervision protocol. |
| `scripts/run_kh_subsonic_highalpha_classic_two_stage_repair_test.py` | QUARANTINE | Two-stage repair with classical mode info | Likely incompatible with stated PINN protocol. |
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_mode_supervision.slurm` | QUARANTINE | Slurm for modal supervision test | Quarantine with its script. |
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_full_mode_supervision.slurm` | QUARANTINE | Slurm for full modal supervision test | Quarantine with its script. |
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_balanced_full_mode_supervision.slurm` | QUARANTINE | Slurm for balanced full modal supervision test | Quarantine with its script. |
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_two_stage_repair.slurm` | QUARANTINE | Slurm for two-stage modal repair | Quarantine with its script. |

PINN diagnostic/ablation scripts:

| File family | Category | Role | Justification |
|---|---|---|---|
| `scripts/run_kh_subsonic_riccati_*.py` and `scripts/run_kh_subsonic_riccati_*.sh` | REVIEW | Riccati ablations/repairs | Some ideas may remain useful; many are older than Stage 0/1/1bis. |
| `scripts/run_kh_subsonic_first_order_real*.py` | REVIEW | Representation alternative | Not current pipeline but useful ablation. |
| `scripts/run_kh_subsonic_modefocus_lowalpha.py` | REVIEW | Low-alpha mode focus | Older attempt; keep until compared to Stage 1bis. |
| `scripts/run_kh_subsonic_edge_repair_sweep.py` | REVIEW | Edge repair sweep | Diagnostic, maybe historical. |
| `scripts/run_kh_subsonic_ci_window_ablation.py` | DIAGNOSTIC | CI window ablation | Useful for methodology. |
| `scripts/run_kh_subsonic_ci_supervision_vs_physics.py` | DIAGNOSTIC | CI supervision vs physics comparison | Useful final comparison. |
| `scripts/run_kh_subsonic_ci_supervision_budget.sh` | REVIEW | CI budget wrapper | References deleted `scripts/ablate_kh_subsonic_ci_supervision_budget.py`; likely broken until restored or quarantined. |
| `scripts/search_kh_subsonic_*external_ci*.py/.sh` | REVIEW | External CI search | Older supervised-search experiments. |
| `scripts/run_kh_subsonic_alpha_sweep_external_ci.sh` | REVIEW | External CI alpha sweep | Older protocol, not current sparse Stage 0. |
| `scripts/run_kh_subsonic_alpha_sweep_spectral.sh` | REVIEW | Earlier spectral alpha sweep | Possibly superseded by Stage 0/1. |
| `scripts/run_kh_subsonic_hybrid_alpha_sweep.py` | REVIEW | Older hybrid alpha sweep | Superseded by Stage 0/1/1bis. |
| `scripts/run_kh_subsonic_normalization_ablation_local.py` | DIAGNOSTIC | Local normalization ablation | Useful scientific diagnostic. |
| `scripts/run_kh_subsonic_representation_ablation_local.py` | DIAGNOSTIC | Representation ablation | Useful diagnostic. |
| `scripts/run_kh_subsonic_targeted_regime_test.py` | REVIEW | Targeted regime tests | Experimental. |
| `scripts/run_kh_subsonic_highalpha_*` excluding modal-supervision files | REVIEW | High-alpha repair experiments | Historical attempts; classify after Stage 1bis results. |
| `scripts/submit_kh_subsonic_singlecase_phaseA*.sh` | REVIEW | Convenience submit wrappers | Historical, not core. |

## Diagnostics and Post-Processing

### Subsonic PINN Figures and Comparisons

| File | Category | Role | Justification |
|---|---|---|---|
| `scripts/render_kh_subsonic_exact_10figs.py` | KEEP | Current requested 10-figure comparison renderer | Recent final-figure script. |
| `scripts/render_kh_subsonic_reference_comparison_pngs.py` | KEEP | PNG renderer for reference comparisons | Recent plotting helper. |
| `scripts/build_kh_subsonic_M05_alpha010_080_reference_comparison.py` | KEEP | Build M05 alpha-band comparison | Recent comparison generator. |
| `scripts/build_kh_subsonic_M05_alpha010_080_reference_comparison_plotly.py` | DIAGNOSTIC | Plotly comparison variant | Useful interactive diagnostic; not necessary for pipeline. |
| `scripts/plot_kh_subsonic_ci_supervision_vs_physics.py` | KEEP | Compare CI sparse/PINN physics | Useful final result. |
| `scripts/plot_kh_subsonic_ci_error_heatmap.py` | DIAGNOSTIC | CI error heatmap | Useful figure. |
| `scripts/plot_kh_subsonic_ci_supervision_budget_summary.py` | REVIEW | CI budget plot | Depends on budget ablation status. |
| `scripts/plot_kh_subsonic_fixed_mach_classic_vs_pinn_modes.py` | DIAGNOSTIC | Classic/PINN modes comparison | Useful post-processing. |
| `scripts/plot_kh_subsonic_fixed_mach_mode_field_error_heatmaps.py` | DIAGNOSTIC | Mode field error heatmaps | Useful post-processing. |
| `scripts/plot_kh_subsonic_mode_error_vs_alpha.py` | DIAGNOSTIC | Mode error vs alpha | Useful post-processing. |
| `scripts/plot_kh_subsonic_mode_error_heatmaps_2d.py` | REVIEW | 2D mode heatmaps | Future 2D work. |
| `scripts/plot_kh_subsonic_pinn_results.py` | DIAGNOSTIC | Generic PINN result plotter | Useful but broad/older. |
| `scripts/plot_kh_subsonic_pinn_results_2d.py` | REVIEW | 2D result plotter | Future 2D work. |
| `scripts/plot_kh_subsonic_pinn_single_mode_like_thesis.py` | DIAGNOSTIC | Thesis-like single-mode plot | Useful final illustration. |
| `scripts/compare_kh_subsonic_*.py` | DIAGNOSTIC | Subsonic mode/CI comparisons | Keep as analysis utilities. |
| `scripts/build_subsonic_1d_presentation.py` | DIAGNOSTIC | Presentation figure builder | Useful but presentation-specific. |
| `scripts/build_presentation_plots.py` | DIAGNOSTIC | Generic presentation plots | Useful but presentation-specific. |

### Supersonic Figures and Diagnostics

| File | Category | Role | Justification |
|---|---|---|---|
| `scripts/plot_supersonic_presentation_summary.py` | DIAGNOSTIC | Presentation summary figures | Useful for slides. |
| `scripts/plot_supersonic_shooting_reference_vs_blumen.py` | DIAGNOSTIC | Shooting reference vs Blumen | Useful validation figure. |
| `scripts/plot_supersonic_blumen_digitized_diagnostic.py` | DIAGNOSTIC | Digitized Blumen diagnostic | Useful for checking reference interpolation. |
| `scripts/plot_supersonic_cr_digitized_points.py` | DIAGNOSTIC | CR digitized points plot | Useful digitization check. |
| `scripts/plot_supersonic_database_presentation.py` | DIAGNOSTIC | Supersonic database presentation | Useful slides. |
| `scripts/plot_supersonic_gep_vs_blumen_overlay.py` | DIAGNOSTIC | GEP vs Blumen overlay | Diagnostic only. |
| `scripts/plot_blumen_supersonic_paper_figure.py` | DIAGNOSTIC | Blumen paper-style figure | Useful for article/slides. |
| `scripts/plot_blumen_supersonic_reference.py` | DIAGNOSTIC | Blumen reference plot | Useful. |
| `scripts/plot_blumen_isocontours.py` | DIAGNOSTIC | Blumen isocontours | Useful. |
| `scripts/reconstruct_supersonic_blumen_ci_visual.py` | DIAGNOSTIC | Visual CI reconstruction | Diagnostic. |
| `scripts/reconstruct_supersonic_blumen_cr_visual.py` | DIAGNOSTIC | Visual CR reconstruction | Diagnostic. |
| `scripts/audit_supersonic_blumen_local_reference.py` | DIAGNOSTIC | Local Blumen interpolation audit | Important to avoid wrong CI targets. |
| `scripts/audit_supersonic_blumen_cr_uncertainty.py` | DIAGNOSTIC | CR uncertainty audit | Useful. |
| `scripts/audit_supersonic_blumen_eigenconditions.py` | DIAGNOSTIC | Blumen eigencondition audit | Useful but not core. |
| `scripts/check_blumen_supersonic_digitization.py` | DIAGNOSTIC | Digitization check | Useful. |

## Launch Files

The `launch/` folder mixes active production jobs, diagnostics, and many historical experiments. Do not delete now.

### KEEP Launch Files

| File family | Category | Justification |
|---|---|---|
| `launch/jz_submit_kh_subsonic_ci_stage0_M05_alpha010_080_hybrid_ci{4,8,16}.slurm` | KEEP | Active sparse CI Stage 0. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1_M05_alpha010_080_ci{4,8,16}.slurm` | KEEP | Active Stage 1. |
| `launch/jz_submit_kh_subsonic_hybrid_stage1bis_M05_alpha010_080_ci{4,8,16}.slurm` | KEEP | Active Stage 1bis. |
| `launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm` | KEEP | Active pure-physics baseline. |
| `launch/jz_submit_supersonic_shooting_point_batch.slurm` | KEEP | Active pointwise supersonic shooting. |
| `launch/jz_submit_supersonic_shooting_point_batch_M140.slurm` | KEEP | Active fixed-Mach point batch variant. |
| `launch/jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm` | KEEP | Active branch-guided point batch. |
| `launch/jz_submit_supersonic_shooting_blumen_locked.slurm` | KEEP | Recent Blumen-locked shooting variant. |
| `launch/jz_submit_supersonic_shooting_reference_package.slurm` | KEEP | Builds/package shooting reference. |
| `launch/jz_submit_supersonic_shooting_intermediate_mach_spectral.slurm` | KEEP | Expands sparse supersonic reference. |
| `launch/jz_submit_supersonic_shooting_modal_front_spectral*.slurm` | KEEP | Expands modal/spectral front. |

### DIAGNOSTIC Launch Files

| File family | Category | Justification |
|---|---|---|
| `launch/jz_submit_gep_*.slurm`, `launch/jz_merge_gep_*.slurm`, `launch/jz_validate_gep_*.slurm` | DIAGNOSTIC | GEP diagnostics, not primary reference. |
| `launch/jz_submit_supersonic_blumen_*.slurm` | DIAGNOSTIC | Blumen digitization/eigencondition checks. |
| `launch/jz_submit_supersonic_shooting_ci_*.slurm` | DIAGNOSTIC | CI maps/continuation/lines diagnostics. |
| `launch/jz_submit_supersonic_shooting_mach_continuation.slurm` | DIAGNOSTIC | Branch continuation diagnostic. |
| `launch/jz_submit_supersonic_shooting_modal_surface*.slurm` | DIAGNOSTIC | Modal surface diagnostics/reference expansion. |
| `launch/jz_submit_supersonic_shooting_mode_structure.slurm` | DIAGNOSTIC | Mode structure diagnostic. |
| `launch/jz_submit_supersonic_shooting_visual_validation.slurm` | DIAGNOSTIC | Visual validation diagnostic. |
| `launch/jz_submit_kh_subsonic_mode_error*.slurm` | DIAGNOSTIC | Mode error post-processing. |

### REVIEW Launch Files

| File family | Category | Justification |
|---|---|---|
| `launch/jz_submit_kh_subsonic_pinn_M05_riccati_*.slurm` | REVIEW | Many historical Riccati variants; keep until mapped to results. |
| `launch/jz_submit_kh_subsonic_pinn_M05_anchor_*.slurm` | REVIEW | Anchor strategy ablations; likely historical. |
| `launch/jz_submit_kh_subsonic_pinn_M05_modefocus*.slurm` | REVIEW | Mode-focus experiments; maybe superseded by Stage 1bis. |
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_*.slurm` | REVIEW | High-alpha repair experiments; several are incompatible with current no-modal-supervision rule. |
| `launch/jz_submit_kh_subsonic_pinn_singlecase_*.slurm` | REVIEW | Single-case prototypes and Phase A/B/Ter history. |
| `launch/jz_submit_kh_subsonic_pinn_2d*.slurm` | REVIEW | Future 2D work, not current fixed-Mach pipeline. |
| `launch/jz_submit_kh_subsonic_pinn_M00/M03/M07.slurm` | REVIEW | Older fixed-Mach experiments. |
| `launch/jz_submit.slurm`, `launch/jz_submit_cpu.slurm` | REVIEW | Generic/old launchers; target unclear. |

### QUARANTINE Launch Files

| File family | Category | Justification |
|---|---|---|
| `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_*mode_supervision*.slurm` | QUARANTINE | Classical modal supervision conflicts with current PINN loss rule. |
| `launch/jz_submit_kh_subsonic_pinn_M05_ci_supervision_budget.slurm` | REVIEW | Currently references deleted script; decide restore or quarantine. |

## Markdown and Protocol Files

| File | Category | Role | Justification |
|---|---|---|---|
| `protocole_experimental_classique.md` | KEEP | Classical reference protocol | Active documentation. |
| `protocole_experimental_pinn.md` | KEEP | PINN protocol | Active documentation. |
| `protocole_pinn_subsonic_alpha_mach_sweep.md` | KEEP | PINN alpha/Mach sweep protocol | Active/future protocol. |
| `protocole_shooting_supersonique_pointwise.md` | KEEP | Supersonic pointwise shooting protocol | Active reference protocol. |
| `protocole_pinn_subsonic_M05_alpha010_080_pure_physics.md` | KEEP | Pure-physics fixed-Mach protocol | Active baseline doc. |
| `protocole_pinn_subsonic_M05_alpha010_080_hybrid_ci_sparse.md` | KEEP | Sparse CI hybrid protocol | Active Stage 0/1 protocol context. |
| `pinn_caracteristiques_actuelles.md` | KEEP | Current PINN characteristics | Useful technical reference. |
| `roadmap_6_semaines_references_et_pinn.md` | KEEP | Roadmap | Active planning. |
| `presentation_plan_kh_pinn_avancement.md` | DIAGNOSTIC | Presentation plan | Useful, but not pipeline. |
| `article.md` | REVIEW | Article draft/plan | Keep, but role not pipeline. |
| `pinn_subsonic.md` | REVIEW | Older PINN notes | May overlap with newer protocol docs. |
| `problèmes_supersoniques.md` | KEEP | Supersonic issues notes | Important context. |
| `protocole_experimental.md` | REVIEW | Generic older protocol | Potential duplicate of split classical/PINN protocol docs. |
| `assets/**/README.md` | KEEP | Asset-local documentation | Do not move/touch under `assets/`. |
| `archive/**/README.md` | KEEP | Archive documentation | Already archived; leave unchanged. |

## YAML/YML

No active `.yaml` or `.yml` files were found in the current non-archive inventory from `rg --files`. If config files are later added under `config/`, classify them as `KEEP` only if loaded by an active launcher.

## Files Proposed for `scripts/reference_classique/`

### `scripts/reference_classique/subsonique/`

Recommended first wave, as wrappers or moved entry points after import-safe refactor:

- `classical_solver/subsonic/hybrid_subsonic_scan.py`
- `classical_solver/subsonic/reconstruct_blumen_subsonic_robust.py`
- possibly `classical_solver/subsonic/plot_subsonic_ci_map.py`
- possibly `classical_solver/subsonic/plot_subsonic_error_map.py`

Keep in place as modules:

- `classical_solver/subsonic/mstab17_subsonic_solver.py`
- `classical_solver/subsonic/robust_subsonic_shooting.py`
- `classical_solver/subsonic/shooting_subsonic.py`

### `scripts/reference_classique/supersonique/`

Recommended first wave, as wrappers or moved entry points after import-safe refactor:

- `scripts/audit_supersonic_shooting_point_batch.py`
- `scripts/audit_supersonic_shooting_point_batch_branch_guided.py`
- `scripts/audit_supersonic_shooting_blumen_locked.py`
- `scripts/build_supersonic_shooting_reference_package.py`
- `scripts/build_supersonic_validated_modal_package.py`
- `classical_solver/supersonic/reconstruct_blumen_supersonic_shooting.py`
- `scripts/plot_supersonic_shooting_reference_vs_blumen.py`

Keep in place as modules/helpers unless compatibility wrappers are added:

- `scripts/track_supersonic_shooting_multistart.py`
- `scripts/audit_supersonic_shooting_visual_validation.py`
- `scripts/audit_supersonic_shooting_ci_map.py`
- `scripts/audit_supersonic_families_against_blumen.py`
- `classical_solver/supersonic/mstab17_supersonic_solver.py`
- `classical_solver/supersonic/blumen_reference.py`

## Files Proposed for `_quarantine/`

No file should be moved now. Later quarantine candidates:

| File/family | Reason |
|---|---|
| `scripts/run_kh_subsonic_highalpha_classic_mode_supervision_test.py` | Uses classical modal supervision, contrary to current PINN protocol. |
| `scripts/run_kh_subsonic_highalpha_classic_full_mode_supervision_test.py` | Uses full modal supervision. |
| `scripts/run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py` | Uses balanced full modal supervision. |
| `scripts/run_kh_subsonic_highalpha_classic_two_stage_repair_test.py` | Likely uses classical mode repair logic. |
| Matching `launch/jz_submit_kh_subsonic_pinn_M05_highalpha_classic_*mode_supervision*.slurm` | Launchers for the above. |
| `scripts/run_kh_subsonic_ci_supervision_budget.sh` | References deleted `scripts/ablate_kh_subsonic_ci_supervision_budget.py`; likely broken unless restored. |
| `launch/jz_submit_kh_subsonic_pinn_M05_ci_supervision_budget.slurm` | Launcher for likely broken budget script. |
| Old `anchor_max`, `ampphase`, `modefocus`, `first_order_real`, `edge_repair`, `highalpha` launch/script variants | Historical ablations likely superseded by Stage 0/1/1bis, but require provenance check before quarantine. |

## Ambiguous Files Left in REVIEW

These should not be moved until one of the following is checked: Slurm logs, last successful run, import dependencies, and whether the output is still used in figures.

- `scripts/run_kh_subsonic_riccati_core1d.py`
- `scripts/run_kh_subsonic_riccati_core1d_windowed_sparse.py`
- `scripts/run_kh_subsonic_riccati_lowalpha_repair.py`
- `scripts/run_kh_subsonic_riccati_mode_repair_edges.py`
- `scripts/run_kh_subsonic_first_order_real.py`
- `scripts/run_kh_subsonic_first_order_real_stabilized.py`
- `scripts/run_kh_subsonic_modefocus_lowalpha.py`
- `scripts/run_kh_subsonic_edge_repair_sweep.py`
- `scripts/run_kh_subsonic_hybrid_alpha_sweep.py`
- `scripts/run_kh_subsonic_alpha_sweep_spectral.sh`
- `scripts/run_kh_subsonic_alpha_sweep_external_ci.sh`
- `scripts/search_kh_subsonic_alpha_sweep_external_ci.py`
- `scripts/search_kh_subsonic_singlecase_external_ci.py`
- `scripts/run_kh_subsonic_singlecase_*`
- `scripts/submit_kh_subsonic_singlecase_phaseA*.sh`
- `scripts/train_kh_subsonic_pinn_2d.py`
- `scripts/run_kh_subsonic_pinn_2d_*`
- `scripts/evaluate_kh_subsonic_pinn_2d_ci_modes.py`
- `classical_solver/gep/adaptive_gep_guided_by_shooting.py`
- `classical_solver/gep/sweep_alpha_gep_notebook_style.py`
- `protocole_experimental.md`
- `pinn_subsonic.md`
- `article.md`

## Archive Policy

Everything under `archive/` is already effectively quarantined. Proposed category: `KEEP` as archive, no action.

Do not reclassify archived files one by one unless the cleanup plan later includes deleting old archives. For now:

- `archive/legacy_classical_solver/**`: KEEP as archive.
- `archive/gep_prototypes/**`: KEEP as archive.
- `archive/pinn_prototype/**`: KEEP as archive.
- `archive/repo_cleanup_2026-04-24/**`: KEEP as archive.
- `archive/old_runs/**`: KEEP as archive.

## Risk Points Before Any Move

1. `scripts/audit_supersonic_shooting_point_batch.py` is both an executable and an imported helper module. Moving it directly will break many imports.
2. `scripts/track_supersonic_shooting_multistart.py` is a backend helper for several shooting pipelines. Do not move without import refactor.
3. `classical_solver/subsonic/robust_subsonic_shooting.py` is imported by PINN data/training code. Do not move.
4. `classical_solver/subsonic/mstab17_subsonic_solver.py` is imported by many plot and trainer scripts. Do not move.
5. `src/` is package code, not scripts. Keep stable.
6. Several Slurm files launch old scripts that may be broken due to deleted/untracked files. Validate before quarantine.
7. Do not use `git mv` on a dirty tree without first isolating user changes.

## Suggested Next Cleanup Steps

Step 1: Create directories only:

- `scripts/reference_classique/subsonique/`
- `scripts/reference_classique/supersonique/`
- `_quarantine/`

Step 2: Add wrapper scripts in target folders, instead of moving imported modules immediately.

Step 3: Update only a small set of Slurm launchers to call wrappers.

Step 4: Run smoke checks:

- `python3 -m py_compile scripts/audit_supersonic_shooting_point_batch.py`
- `python3 -m py_compile scripts/audit_supersonic_shooting_point_batch_branch_guided.py`
- `python3 -m py_compile scripts/train_kh_subsonic_pinn.py`
- `bash -n scripts/run_kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock.sh`
- `bash -n scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh`
- `bash -n scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined.sh`

Step 5: Only after smoke checks, quarantine clearly obsolete modal-supervision scripts and broken budget wrappers.

## Summary Decision

Immediate safe cleanup action later: add wrapper organization for reference-classic scripts, not direct moves.

Highest priority `KEEP` files:

- `scripts/train_kh_subsonic_pinn.py`
- `scripts/train_kh_subsonic_ci_stage0_anchor_lock.py`
- `scripts/run_kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock.sh`
- `scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh`
- `scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1bis_physics_refined.sh`
- `scripts/run_kh_subsonic_M05_alpha010_080_pure_physics_reference.sh`
- `scripts/audit_supersonic_shooting_point_batch.py`
- `scripts/audit_supersonic_shooting_point_batch_branch_guided.py`
- `scripts/track_supersonic_shooting_multistart.py`
- `classical_solver/subsonic/*` core solver files
- `classical_solver/supersonic/*` core solver/reference files
- `src/**`

Highest priority `QUARANTINE` later:

- modal-supervision PINN experiment scripts and launchers;
- broken CI budget wrapper/launcher unless restored;
- old anchor/modefocus/amplitude-phase experiments after provenance check.
