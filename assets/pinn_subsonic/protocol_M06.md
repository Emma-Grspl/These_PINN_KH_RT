# Subsonic `M=0.6` Protocol

This protocol mirrors the successful `M=0.5` sequence while adapting the active alpha range and edge targets to the `M=0.6` branch.

## Rationale

The local classical hybrid map in [subsonic_hybrid_growth_map.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv) shows:

- at `M=0.56`, non-stable points up to `alpha=0.79`;
- at `M=0.63`, non-stable points up to `alpha=0.72`;
- by interpolation, `M=0.60` should remain active up to about `alpha=0.75`.

For this reason, the protocol uses:

- spectral warmstart on `alpha in [0.02, 0.75]`;
- joint `core1d` reconstruction on `alpha in [0.05, 0.75]`;
- edge-focused modal repair on `alpha in [0.20, 0.75]`;
- upper-edge emphasis starting at `alpha=0.65`.

## Stage 1: Spectral Backbone

Launcher:

- [jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm)

Default output:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_spectral_alpha_sweep`

Goal:

- obtain a stable `M=0.6` Riccati PINN with good `c_i(alpha)` behavior over the full active range;
- provide the warmstart used by `core1d`.

Checks:

- no crash;
- `model_best.pt` exists;
- `ci_curve_vs_reference.csv` and `ci_error_heatmap.csv` are written;
- `ci` remains coherent over `alpha in [0.05, 0.75]`.

## Stage 2: Core 1D Joint Reconstruction

Launcher:

- [jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm)

Default warmstart:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_spectral_alpha_sweep`

Default output:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_core1d`

Goal:

- reconstruct `c_i` and the modal fields jointly on the active `M=0.6` interval;
- produce the warm candidate for edge-focused repair.

Checks:

- compare `warmstart_eval` and `posttrain_eval`;
- retain the post-train model only if it improves modal metrics without materially degrading `ci`;
- if the post-train candidate degrades the mode while marginally improving `ci`, keep the warmstart and do not promote `model_best.pt`.

## Stage 3: Edge-Focused Modal Repair

Launcher:

- [jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm)

Default warmstart:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_core1d`

Default output:

- `model_saved/kh_subsonic_fixed_mach_M06_riccati_mode_repair_edges`

Goal:

- freeze the spectral branch;
- improve modal reconstruction on the lower edge and, especially, near the high-alpha transition region.

Default edge bands:

- lower edge: `alpha <= 0.30`;
- upper edge: `alpha >= 0.65`.

Checks:

- `ci_mae` should remain unchanged or effectively unchanged;
- lower/core/upper `p_rel` should improve;
- `u_rel`, `v_rel`, and `phase_rmse` should be checked before promotion;
- promote the repaired model only if the gains are global or at least do not introduce a stronger regression at the upper edge.

## Recommended Jean Zay Sequence

```bash
git pull
sbatch launch/jz_submit_kh_subsonic_pinn_M06_riccati_spectral_alpha_sweep.slurm
```

Then:

```bash
git pull
sbatch launch/jz_submit_kh_subsonic_pinn_M06_riccati_core1d.slurm
```

Then:

```bash
git pull
sbatch launch/jz_submit_kh_subsonic_pinn_M06_riccati_mode_repair_edges.slurm
```

## Promotion Rule

Promote the final `M=0.6` reference only after the local readback confirms:

- spectral stability is preserved;
- modal improvement is real on at least `p`, `u`, `v`;
- no unacceptable phase drift appears near `alpha in [0.65, 0.75]`.
