from __future__ import annotations

from collections.abc import Mapping

import torch

from src.models.kh_subsonic_pinn import KHSubsonicMultiMachPINN


class KHSubsonicPINN2D(KHSubsonicMultiMachPINN):
    """
    Thin dedicated wrapper for the parametric subsonic 2D PINN.

    The scientific implementation remains the validated multi-Mach architecture
    already present in `src.models.kh_subsonic_pinn`. This wrapper exists to
    expose an explicit 2D entry point for the new hybrid4ci workflow without
    changing the 1D code path.
    """


def _config_get(config: Mapping[str, object] | object, key: str, default: object = None) -> object:
    if isinstance(config, Mapping):
        value = config.get(key, default)
    elif hasattr(config, "index"):
        value = config[key] if key in config.index else default
    else:
        value = getattr(config, key, default)
    if value is None:
        return default
    try:
        if bool(torch.isnan(torch.tensor(value, dtype=torch.float32)).item()):
            return default
    except Exception:
        pass
    return value


def build_kh_subsonic_pinn_2d_from_config(config: Mapping[str, object] | object) -> KHSubsonicPINN2D:
    mode_hidden_dim = _config_get(config, "mode_hidden_dim")
    ci_hidden_dim = _config_get(config, "ci_hidden_dim")
    return KHSubsonicPINN2D(
        alpha_min=float(_config_get(config, "alpha_min")),
        alpha_max=float(_config_get(config, "alpha_max")),
        mach_min=float(_config_get(config, "mach_min")),
        mach_max=float(_config_get(config, "mach_max")),
        hidden_dim=int(_config_get(config, "hidden_dim", 128)),
        mode_hidden_dim=None if mode_hidden_dim is None else int(mode_hidden_dim),
        ci_hidden_dim=None if ci_hidden_dim is None else int(ci_hidden_dim),
        mode_depth=int(_config_get(config, "mode_depth", 4)),
        ci_depth=int(_config_get(config, "ci_depth", 2)),
        activation=str(_config_get(config, "activation", "tanh")),
        fourier_features=int(_config_get(config, "fourier_features", 0)),
        fourier_scale=float(_config_get(config, "fourier_scale", 2.0)),
        initial_ci=float(_config_get(config, "initial_ci", 0.2)),
        mapping_scale=float(_config_get(config, "mapping_scale", 3.0)),
        trainable_mapping_scale=bool(_config_get(config, "trainable_mapping_scale", False)),
        mode_representation=str(_config_get(config, "mode_representation", "riccati")),
    )


def freeze_all_parameters(model: KHSubsonicPINN2D) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_ci_head(model: KHSubsonicPINN2D) -> None:
    if model.ci_net is not None:
        for parameter in model.ci_net.parameters():
            parameter.requires_grad_(True)
    model.raw_ci_bias.requires_grad_(True)
