from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.kh_subsonic_sampling import reference_point, sample_boundary_points, sample_interior_points
from src.data.kh_subsonic_sampling_2d import (
    build_anchor_table,
    dumps_float_list,
    normalize_float_list,
)
from src.physics.kh_subsonic_riccati_matching_2d import (
    riccati_left_right_matching_loss_2d,
    y_to_xi,
)
from src.physics.kh_subsonic_residual_2d import (
    normalization_loss_2d,
    phase_loss_2d,
    pressure_ode_residual_2d,
    riccati_boundary_loss_components_2d,
    riccati_normalization_loss_2d,
    riccati_phase_loss_2d,
)
from src.training.kh_subsonic_trainer_2d_stage1 import (
    _compute_anchor_metrics,
    load_stage0_model_from_checkpoint,
)


@dataclass
class KHSubsonic2DStage1terMatchingConfig:
    stage0_checkpoint: str = (
        "model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt"
    )
    mach_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    alpha_min: float = 0.10
    alpha_max: float = 0.80
    anchor_alphas: tuple[float, ...] = (0.10, 0.30, 0.55, 0.80)
    output_dir: str = "model_saved/kh_subsonic_2d_hybrid4ci_stage1ter_riccati_matching"
    epochs: int = 8000
    learning_rate: float = 5e-5
    grad_clip_norm: float = 1.0
    n_interior: int = 512
    n_boundary: int = 96
    n_center: int = 256
    center_width: float = 2.0
    center_fraction: float = 0.5
    n_alpha_samples: int = 12
    n_mach_samples: int | None = None
    audit_every: int = 100
    checkpoint_every: int = 500
    device: str = "cpu"
    seed: int = 1234
    freeze_ci: bool = True
    detach_ci_in_mode_branch: bool = True
    reference_cache: str | None = None
    w_pde: float = 1.0
    w_bc_kappa: float = 20.0
    w_bc_q: float = 60.0
    w_match: float = 1.0
    w_center_pde: float = 1.0
    w_norm: float = 0.0
    w_phase: float = 0.0
    w_ci_anchor: float = 1.0
    match_y_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    shoot_ymax: float = 40.0
    shoot_steps: int = 512
    match_warmup_epochs: int = 1000
    best_metric: str = "loss_total"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if str(device).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_serialized_float_list(config: dict[str, object], key_json: str, key_plain: str) -> tuple[float, ...]:
    value = config.get(key_json) or config.get(key_plain)
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        return tuple(float(item) for item in json.loads(text))
    return tuple(float(item) for item in text.replace(",", " ").split())


def _make_serializable_config(cfg: KHSubsonic2DStage1terMatchingConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["mach_min"] = float(min(cfg.mach_values))
    data["mach_max"] = float(max(cfg.mach_values))
    data["mach_values_json"] = dumps_float_list(cfg.mach_values)
    data["anchor_alphas_json"] = dumps_float_list(cfg.anchor_alphas)
    data["match_y_values_json"] = dumps_float_list(cfg.match_y_values)
    data["mach_values"] = " ".join(f"{value:.12g}" for value in cfg.mach_values)
    data["anchor_alphas"] = " ".join(f"{value:.12g}" for value in cfg.anchor_alphas)
    data["match_y_values"] = " ".join(f"{value:.12g}" for value in cfg.match_y_values)
    return data


def _choose_mach_samples(
    mach_values: tuple[float, ...],
    *,
    n_samples: int | None,
) -> np.ndarray:
    values = np.asarray(mach_values, dtype=float)
    if values.size == 0:
        raise ValueError("mach_values cannot be empty.")
    target = values.size if n_samples is None else max(1, int(n_samples))
    if target >= values.size:
        return values.copy()
    indices = np.random.choice(values.size, size=target, replace=False)
    return np.sort(values[indices])


def _sample_parameter_mesh(
    cfg: KHSubsonic2DStage1terMatchingConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_samples = torch.empty(int(cfg.n_alpha_samples), 1, device=device).uniform_(
        float(cfg.alpha_min),
        float(cfg.alpha_max),
    )
    mach_np = _choose_mach_samples(cfg.mach_values, n_samples=cfg.n_mach_samples)
    mach_samples = torch.tensor(mach_np, dtype=torch.float32, device=device).view(-1, 1)
    aa, mm = torch.meshgrid(alpha_samples[:, 0], mach_samples[:, 0], indexing="ij")
    return aa.reshape(-1, 1), mm.reshape(-1, 1)


def _repeat_spatial_over_param_pairs(
    xi: torch.Tensor,
    alpha_pairs: torch.Tensor,
    mach_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pairs = int(alpha_pairs.shape[0])
    n_xi = int(xi.shape[0])
    xi_rep = xi.repeat(n_pairs, 1)
    alpha_rep = alpha_pairs.repeat_interleave(n_xi, dim=0)
    mach_rep = mach_pairs.repeat_interleave(n_xi, dim=0)
    return xi_rep, alpha_rep, mach_rep


def _freeze_ci_branch(model) -> None:
    if getattr(model, "ci_net", None) is not None:
        for parameter in model.ci_net.parameters():
            parameter.requires_grad_(False)
    if hasattr(model, "raw_ci_bias"):
        model.raw_ci_bias.requires_grad_(False)


def _unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _build_optimizer(model, cfg: KHSubsonic2DStage1terMatchingConfig) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters available for Stage 1ter.")
    return torch.optim.Adam(trainable, lr=float(cfg.learning_rate))


def _save_anchor_predictions(
    model,
    anchors_df: pd.DataFrame,
    *,
    device: torch.device,
    output_path: Path,
) -> dict[str, float]:
    metrics, work = _compute_anchor_metrics(model, anchors_df, device=device)
    work.to_csv(output_path, index=False)
    return metrics


def _sample_center_points(
    *,
    n_points: int,
    center_width: float,
    center_fraction: float,
    mapping_scale: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty(0, 1, device=device, dtype=dtype)

    width = max(float(center_width), 1.0e-6)
    fraction = min(max(float(center_fraction), 0.0), 1.0)
    n_gaussian = int(round(fraction * int(n_points)))
    n_uniform = int(n_points) - n_gaussian

    chunks: list[torch.Tensor] = []
    if n_uniform > 0:
        y_uniform = torch.empty(n_uniform, 1, device=device, dtype=dtype).uniform_(-width, width)
        chunks.append(y_uniform)
    if n_gaussian > 0:
        y_gaussian = torch.randn(n_gaussian, 1, device=device, dtype=dtype) * (width / 3.0)
        y_gaussian = torch.clamp(y_gaussian, min=-width, max=width)
        chunks.append(y_gaussian)
    y_center = torch.cat(chunks, dim=0)
    permutation = torch.randperm(y_center.shape[0], device=device)
    y_center = y_center[permutation]
    xi_center = y_to_xi(y_center, mapping_scale.detach())
    xi_center.requires_grad_(True)
    return xi_center


def _match_weight(epoch: int, *, base_weight: float, warmup_epochs: int) -> float:
    if float(base_weight) <= 0.0:
        return 0.0
    if int(warmup_epochs) <= 0:
        return float(base_weight)
    if int(warmup_epochs) == 1:
        return float(base_weight)
    ratio = min(max((int(epoch) - 1) / float(int(warmup_epochs) - 1), 0.0), 1.0)
    return float(base_weight) * float(ratio)


def _best_metric_value(row: dict[str, float | int], metric: str) -> float:
    if metric not in row:
        raise KeyError(f"best_metric={metric!r} not present in training row.")
    value = float(row[metric])
    if not np.isfinite(value):
        return float("inf")
    return value


def _format_stage1ter_report(
    cfg: KHSubsonic2DStage1terMatchingConfig,
    *,
    stage0_checkpoint_path: Path,
    best_epoch: int,
    best_history_row: dict[str, float | int],
) -> str:
    lines = [
        "2D subsonic Stage 1ter — Riccati matching modal reconstruction",
        f"stage0_checkpoint={stage0_checkpoint_path}",
        f"freeze_ci_default={cfg.freeze_ci}",
        f"detach_ci_in_mode_branch_default={cfg.detach_ci_in_mode_branch}",
        "direct_modal_supervision=False",
        f"mach_values={json.dumps([float(value) for value in cfg.mach_values])}",
        f"alpha_range=[{cfg.alpha_min}, {cfg.alpha_max}]",
        f"anchor_alphas={json.dumps([float(value) for value in cfg.anchor_alphas])}",
        f"match_y_values={json.dumps([float(value) for value in cfg.match_y_values])}",
        f"epochs={cfg.epochs}",
        f"learning_rate={cfg.learning_rate}",
        (
            f"loss_weights: pde={cfg.w_pde} bc_kappa={cfg.w_bc_kappa} bc_q={cfg.w_bc_q} "
            f"match={cfg.w_match} center_pde={cfg.w_center_pde} norm={cfg.w_norm} phase={cfg.w_phase} "
            f"ci_anchor={cfg.w_ci_anchor}"
        ),
        f"match_warmup_epochs={cfg.match_warmup_epochs}",
        f"shoot_ymax={cfg.shoot_ymax}",
        f"shoot_steps={cfg.shoot_steps}",
        f"best_metric={cfg.best_metric}",
        f"best_epoch={best_epoch}",
        f"best_loss_total={float(best_history_row.get('loss_total', float('nan'))):.8e}",
        f"best_loss_match={float(best_history_row.get('loss_match', float('nan'))):.8e}",
        f"best_ci_anchor_max_abs={float(best_history_row.get('ci_anchor_max_abs', float('nan'))):.8e}",
        f"best_ci_anchor_max_rel_unstable={float(best_history_row.get('ci_anchor_max_rel_unstable', float('nan'))):.8e}",
        f"best_ci_neutral_max_abs={float(best_history_row.get('ci_neutral_max_abs', float('nan'))):.8e}",
        "",
        "Training losses:",
        "- pressure PDE residual on interior points",
        "- Riccati far-field boundary losses on kappa and q",
        "- left/right Riccati matching from physical ODE integration",
        "- center-focused PDE residual near y=0",
        "- optional ci anchor preservation only if freeze_ci is disabled",
        "- no direct modal supervision",
        "",
        "Diagnostics may use the classical solver after training, but the training loss does not.",
    ]
    return "\n".join(lines) + "\n"


def train_kh_subsonic_2d_stage1ter_matching(
    cfg: KHSubsonic2DStage1terMatchingConfig,
) -> int:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    model, stage0_config, _, stage0_checkpoint_path = load_stage0_model_from_checkpoint(
        cfg.stage0_checkpoint,
        device=device,
    )
    if getattr(model, "mode_representation", "cartesian") != "riccati":
        raise RuntimeError("Stage 1ter matching currently supports only Riccati mode representation.")

    if not cfg.mach_values:
        cfg.mach_values = _read_serialized_float_list(stage0_config, "mach_values_json", "mach_values")
    if not cfg.anchor_alphas:
        cfg.anchor_alphas = _read_serialized_float_list(stage0_config, "anchor_alphas_json", "anchor_alphas")
    if not cfg.match_y_values:
        cfg.match_y_values = (-1.0, -0.5, 0.0, 0.5, 1.0)
    if not cfg.mach_values:
        raise RuntimeError("No mach_values available for Stage 1ter.")
    if not cfg.anchor_alphas:
        raise RuntimeError("No anchor_alphas available for Stage 1ter.")

    anchors_path = stage0_checkpoint_path.parent / "anchors_used.csv"
    if anchors_path.exists():
        anchors_df = pd.read_csv(anchors_path)
    else:
        anchors_df = build_anchor_table(
            mach_values=cfg.mach_values,
            anchor_alphas=cfg.anchor_alphas,
            alpha_min=float(cfg.alpha_min),
            alpha_max=float(cfg.alpha_max),
            reference_cache=cfg.reference_cache,
        )

    _unfreeze_all(model)
    if bool(cfg.freeze_ci):
        _freeze_ci_branch(model)
    optimizer = _build_optimizer(model, cfg)

    serializable_cfg = _make_serializable_config(cfg)
    model_config = dict(stage0_config)
    model_config["alpha_min"] = float(cfg.alpha_min)
    model_config["alpha_max"] = float(cfg.alpha_max)
    model_config["mach_min"] = float(min(cfg.mach_values))
    model_config["mach_max"] = float(max(cfg.mach_values))
    pd.DataFrame([serializable_cfg]).to_csv(output_dir / "config.csv", index=False)

    best_metric_value = float("inf")
    best_epoch = 0
    best_state = None
    best_row: dict[str, float | int] = {}
    history_rows: list[dict[str, float | int]] = []

    print("Stage 1ter subsonic 2D Riccati matching", flush=True)
    print(f"stage0_checkpoint={stage0_checkpoint_path}", flush=True)
    print(f"freeze_ci={cfg.freeze_ci}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"device={device}", flush=True)

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        alpha_pairs, mach_pairs = _sample_parameter_mesh(cfg, device=device)
        xi_interior_base = sample_interior_points(int(cfg.n_interior), device=device)
        xi_left_base, xi_right_base = sample_boundary_points(int(cfg.n_boundary), device=device)
        xi_ref_base = reference_point(device=device)
        xi_center_base = _sample_center_points(
            n_points=int(cfg.n_center),
            center_width=float(cfg.center_width),
            center_fraction=float(cfg.center_fraction),
            mapping_scale=model.get_mapping_scale(),
            device=device,
            dtype=xi_interior_base.dtype,
        )

        xi_interior, alpha_interior, mach_interior = _repeat_spatial_over_param_pairs(
            xi_interior_base,
            alpha_pairs,
            mach_pairs,
        )
        xi_left, alpha_left, mach_left = _repeat_spatial_over_param_pairs(
            xi_left_base,
            alpha_pairs,
            mach_pairs,
        )
        xi_right, alpha_right, mach_right = _repeat_spatial_over_param_pairs(
            xi_right_base,
            alpha_pairs,
            mach_pairs,
        )
        xi_ref, alpha_ref, mach_ref = _repeat_spatial_over_param_pairs(
            xi_ref_base,
            alpha_pairs,
            mach_pairs,
        )

        if xi_center_base.numel() > 0:
            xi_center, alpha_center, mach_center = _repeat_spatial_over_param_pairs(
                xi_center_base,
                alpha_pairs,
                mach_pairs,
            )
        else:
            xi_center = torch.empty(0, 1, device=device, dtype=xi_interior.dtype)
            alpha_center = torch.empty(0, 1, device=device, dtype=alpha_pairs.dtype)
            mach_center = torch.empty(0, 1, device=device, dtype=mach_pairs.dtype)

        ci_pairs = model.get_ci(alpha_pairs, mach_pairs)
        ci_override_pairs = ci_pairs.detach() if bool(cfg.detach_ci_in_mode_branch) else ci_pairs

        ci_interior = ci_override_pairs.repeat_interleave(int(xi_interior_base.shape[0]), dim=0)
        res_r, res_i, _ = pressure_ode_residual_2d(
            model,
            xi_interior,
            alpha_interior,
            mach_interior,
            ci_override=ci_interior,
        )
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))

        zero = torch.zeros(1, device=device, dtype=loss_pde.dtype).mean()
        loss_center_pde = zero
        if xi_center.numel() > 0:
            ci_center = ci_override_pairs.repeat_interleave(int(xi_center_base.shape[0]), dim=0)
            res_center_r, res_center_i, _ = pressure_ode_residual_2d(
                model,
                xi_center,
                alpha_center,
                mach_center,
                ci_override=ci_center,
            )
            loss_center_pde = torch.mean(res_center_r.pow(2) + res_center_i.pow(2))

        ci_boundary = ci_override_pairs.repeat_interleave(int(xi_left_base.shape[0]), dim=0)
        loss_bc_kappa, loss_bc_q = riccati_boundary_loss_components_2d(
            model,
            xi_left,
            xi_right,
            alpha_left,
            mach_left,
            ci_override=ci_boundary,
        )
        loss_bc = float(cfg.w_bc_kappa) * loss_bc_kappa + float(cfg.w_bc_q) * loss_bc_q

        loss_norm = zero
        loss_phase = zero
        if float(cfg.w_norm) > 0.0:
            loss_norm = riccati_normalization_loss_2d(model, xi_ref, alpha_ref, mach_ref)
        if float(cfg.w_phase) > 0.0:
            loss_phase = riccati_phase_loss_2d(model, xi_ref, alpha_ref, mach_ref)

        loss_match, match_metrics = riccati_left_right_matching_loss_2d(
            model,
            alpha_pairs,
            mach_pairs,
            match_y_values=cfg.match_y_values,
            y_left=-float(cfg.shoot_ymax),
            y_right=float(cfg.shoot_ymax),
            n_steps=int(cfg.shoot_steps),
            ci_override=ci_override_pairs,
            detach_ci=bool(cfg.detach_ci_in_mode_branch),
        )
        w_match_effective = _match_weight(
            epoch,
            base_weight=float(cfg.w_match),
            warmup_epochs=int(cfg.match_warmup_epochs),
        )

        loss_ci_anchor = zero
        if not bool(cfg.freeze_ci):
            alpha_anchor = torch.tensor(
                anchors_df["alpha"].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            mach_anchor = torch.tensor(
                anchors_df["Mach"].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            ci_target = torch.tensor(
                anchors_df["ci_reference"].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            ci_pred_anchor = model.get_ci(alpha_anchor, mach_anchor)
            loss_ci_anchor = torch.mean((ci_pred_anchor - ci_target).pow(2))

        loss_total = (
            float(cfg.w_pde) * loss_pde
            + loss_bc
            + float(cfg.w_center_pde) * loss_center_pde
            + float(w_match_effective) * loss_match
            + float(cfg.w_norm) * loss_norm
            + float(cfg.w_phase) * loss_phase
            + (float(cfg.w_ci_anchor) * loss_ci_anchor if not bool(cfg.freeze_ci) else zero)
        )

        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite Stage 1ter loss at epoch={epoch}.")
        loss_total.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
        optimizer.step()

        model.eval()
        anchor_metrics, anchor_predictions = _compute_anchor_metrics(model, anchors_df, device=device)
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "loss_total": float(loss_total.detach().cpu()),
            "loss_pde": float(loss_pde.detach().cpu()),
            "loss_bc": float(loss_bc.detach().cpu()),
            "loss_bc_kappa": float(loss_bc_kappa.detach().cpu()),
            "loss_bc_q": float(loss_bc_q.detach().cpu()),
            "loss_match": float(loss_match.detach().cpu()),
            "loss_match_net_left": float(match_metrics["loss_match_net_left"].detach().cpu()),
            "loss_match_net_right": float(match_metrics["loss_match_net_right"].detach().cpu()),
            "loss_match_left_right": float(match_metrics["loss_match_left_right"].detach().cpu()),
            "loss_center_pde": float(loss_center_pde.detach().cpu()),
            "loss_norm": float(loss_norm.detach().cpu()),
            "loss_phase": float(loss_phase.detach().cpu()),
            "loss_ci_anchor": float(loss_ci_anchor.detach().cpu()),
            "w_match_effective": float(w_match_effective),
            "gamma_left_right_abs_mean": float(match_metrics["gamma_left_right_abs_mean"].detach().cpu()),
            "gamma_left_right_abs_max": float(match_metrics["gamma_left_right_abs_max"].detach().cpu()),
            **anchor_metrics,
        }
        history_rows.append(row)

        current_metric = _best_metric_value(row, cfg.best_metric)
        if current_metric < best_metric_value:
            best_metric_value = current_metric
            best_epoch = int(epoch)
            best_row = row
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": serializable_cfg,
                    "model_config": model_config,
                    "best_epoch": best_epoch,
                    "best_row": best_row,
                    "stage0_checkpoint": str(stage0_checkpoint_path),
                    "best_metric": str(cfg.best_metric),
                },
                output_dir / "model_best.pt",
            )
            anchor_predictions.to_csv(output_dir / "anchor_predictions_stage1ter.csv", index=False)

        if epoch == 1 or epoch % int(cfg.audit_every) == 0 or epoch == int(cfg.epochs):
            print(
                "Epoch {epoch:5d} | loss={loss_total:.3e} | pde={loss_pde:.3e} | "
                "bc={loss_bc:.3e} | match={loss_match:.3e} | center={loss_center_pde:.3e} | "
                "w_match={w_match_effective:.3e} | ci_max_abs={ci_anchor_max_abs:.3e} | "
                "ci_max_rel_unstable={ci_anchor_max_rel_unstable:.3e}".format(**row),
                flush=True,
            )

        if epoch % int(cfg.checkpoint_every) == 0 or epoch == int(cfg.epochs):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": serializable_cfg,
                    "model_config": model_config,
                    "epoch": int(epoch),
                    "stage0_checkpoint": str(stage0_checkpoint_path),
                    "best_metric": str(cfg.best_metric),
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )
            pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_anchor_metrics = _save_anchor_predictions(
        model,
        anchors_df,
        device=device,
        output_path=output_dir / "anchor_predictions_stage1ter.csv",
    )
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": serializable_cfg,
            "model_config": model_config,
            "final_anchor_metrics": final_anchor_metrics,
            "stage0_checkpoint": str(stage0_checkpoint_path),
            "best_metric": str(cfg.best_metric),
        },
        output_dir / "model_final.pt",
    )
    report_text = _format_stage1ter_report(
        cfg,
        stage0_checkpoint_path=stage0_checkpoint_path,
        best_epoch=best_epoch,
        best_history_row=best_row,
    )
    (output_dir / "stage1ter_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text, flush=True)
    return 0

