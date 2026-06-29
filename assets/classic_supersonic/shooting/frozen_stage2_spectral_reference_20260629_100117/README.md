# Frozen sparse supersonic spectral reference

Status:
- classical shooting spectral reference
- selected points satisfy stage2_mismatch <= 1e-8
- ci constrained to the unstable range
- trusted_spectral = True
- trusted_modal = False for shooting_root_only_no_box rows

Interpretation:
- suitable as sparse classical spectral reference
- not a gold modal reference
- modal field validation remains pending

Excluded:
- M=1.1--1.5 stage1-only candidates rejected by stage2
- quasi-neutral or ci ~ 0 candidates
