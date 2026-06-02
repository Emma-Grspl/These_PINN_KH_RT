# Subsonic `M=0.6` Protocol

This protocol no longer starts from the unsupervised `M=0.6` spectral sweep. That stage was numerically stable but scientifically unusable: `c_i` stayed around `3e-1`, so it does not provide a valid warmstart.

The working strategy is now a direct transfer from the frozen `M=0.5` Riccati reference, followed by the same edge-focused modal repair that worked at `M=0.5`.

## Reference Base

Frozen source:

- [frozen_M05_riccati_reference_current](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current)

Checkpoint used by default:

- [model_best.pt](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/model_best.pt)

## Rationale

The `M=0.5` frozen reference is currently the best controlled subsonic Riccati model in the repository:

- `c_i` is already well locked;
- modal reconstruction is good on the core and improved on both edges;
- `M=0.6` is close enough that a transfer warmstart is more informative than a cold or weakly constrained spectral sweep.

The local classical hybrid map still motivates the active interval:

- `M=0.56` remains active up to about `alpha=0.79`;
- `M=0.63` remains active up to about `alpha=0.72`;
- `M=0.60` is therefore targeted on `alpha in [0.05, 0.75]`.

## Stage 1: Transfer Core 1D

Launcher:

- [jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm)

Default warmstart:

- [frozen_M05_riccati_reference_current](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current)

Default output:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_core1d`

What this stage does:

- keeps the model normalization inherited from the `M=0.5` warmstart;
- trains only on the active `M=0.6` interval `alpha in [0.05, 0.75]`;
- jointly reconstructs `c_i` and the modal fields at `M=0.6`.

Expected log signature:

- `mach=0.600`
- `alpha-range(model)=[0.050, 0.800] active=[0.050, 0.750]`

Validation rule:

- keep the post-train candidate only if it improves modal metrics without materially degrading `c_i`;
- otherwise keep the transferred warmstart checkpoint.

## Stage 2: Edge-Focused Modal Repair

Launcher:

- [jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm)

Default warmstart:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_core1d`

Default output:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_mode_repair_edges`

What this stage does:

- freezes the spectral branch;
- focuses the optimization on modal structure over `alpha in [0.20, 0.75]`;
- over-samples the lower edge `alpha <= 0.30` and upper edge `alpha >= 0.65`.

Validation rule:

- `c_i` should remain unchanged or nearly unchanged;
- `p_rel`, `u_rel`, `v_rel` should improve globally or at least on both edges;
- phase near `alpha in [0.65, 0.75]` remains the main item to inspect before promotion.

## Deprecated Stage

The launcher below is no longer the recommended entry point for `M=0.6`:

- [jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm)

It can still be kept for exploratory comparisons, but not as the main protocol.

## Recommended Jean Zay Sequence

```bash
git pull
sbatch launch/jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm
```

Then, after checking the `core1d` summary:

```bash
git pull
sbatch launch/jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm
```

## Promotion Rule

Promote the final `M=0.6` reference only after the local readback confirms:

- spectral quality is preserved;
- modal improvement is real on `p`, `u`, and `v`;
- no stronger phase regression appears near `alpha in [0.65, 0.75]`.
