# Baseline PINN pure physics single-case, alpha=0.65, M=0.60

Frozen local snapshot of the stabilized Phase A-ter pure-physics single-case run.

## Purpose

This baseline is kept as the pure-physics reference case showing that the PINN can improve the modal reconstruction while leaving the eigenvalue `ci` essentially unconstrained.

It should be used as a negative/diagnostic baseline against hybrid runs where `ci` is locked by sparse classical supervision before mode training.

## Source

- Source run directory: `model_saved/kh_subsonic_singlecase_phaseA_ter_a0650_m0600_stabilized`
- Job id seen in logs: `726796`
- Local frozen snapshot date: `2026-06-21`
- Frozen snapshot epoch available locally: `500`
- Note: the terminal log shown by the user reached epoch `900`, but the local archive currently contains only `history.csv` up to epoch `500` and `checkpoint_epoch_500.pt`. This baseline therefore freezes exactly the local artifacts, not the later terminal-only state.

## Configuration

- Problem: subsonic Kelvin-Helmholtz single case
- `alpha = 0.65`
- `Mach = 0.60`
- Mode representation: Riccati
- Classical `ci` supervision: disabled
- `fixed_scalar_ci = True`
- Initial/frozen effective `ci`: `ci_mid = 0.200001`
- Physics losses: PDE, Riccati boundary conditions, center/band constraints, shooting match/path schedule

## Snapshot Metrics

Audit metrics at epoch `500`:

- `audit_ci_mae = 9.4834e-02`
- `audit_ci_max_abs = 9.4834e-02`
- `audit_ci_mean_rel = 9.0175e-01`
- `audit_p_rel_l2_mean = 3.0419e-01`
- `audit_env_rel_mean = 2.9329e-01`
- `audit_phase_rel_mean = 3.5163e-02`
- `audit_peak_shift_mean = 3.3333e-02`
- `audit_checkpoint_metric = 1.267548`

Interpretation:

- `ci` is not learned by the pure-physics setup: `ci_mid` stays around `0.2` and the absolute error remains around `0.095`.
- The pressure/modal audit improves substantially compared with epoch 1, so this run is useful to demonstrate mode improvement without eigenvalue locking.
- This baseline should not be used as a successful spectral solver.

## Frozen Files

- `model_saved/config.csv`
- `model_saved/history.csv`
- `model_saved/model_best.pt`
- `model_saved/checkpoint_epoch_500.pt`
- `baseline_manifest.csv`

## Checksums

- `model_best.pt`: `508ae4c2471fbde2eeb1a910f5c82cf6486953071d8192e1bde38171ff4c1ff3`
- `checkpoint_epoch_500.pt`: `63635592782c80ac727360de20e81cf220b64e13780fa4a758c93feb947eda77`
