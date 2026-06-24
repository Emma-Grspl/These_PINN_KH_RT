from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.kh_subsonic_sampling_2d import build_anchor_table, dumps_float_list
from src.physics.kh_subsonic_pressure_first_2d import (
    build_pressure_first_model_from_stage0,
    pressure_first_gauge_loss,
    pressure_first_ode_residual,
    pressure_first_robin_boundary_loss,
)
from src.training.kh_subsonic_trainer_2d_stage1 import (
    _compute_anchor_metrics,
    load_stage0_model_from_checkpoint,
)


@dataclass
class KHSubsonic2DStage1quaterPressureConfig:
    stage0_checkpoint: str = (
        "model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt"
    )
    mach_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    alpha_min: float = 0.10
    alpha_max: float = 0.80
    anchor_alphas: tuple[float, ...] = (0.10, 0.30, 0.55, 0.80)
    output_dir: str = "model_saved/kh_subsonic_2d_hybrid4ci_stage1quater_pressure"
    epochs: int = 1500
    learning_rate: float = 2e-5
    grad_clip_norm: float = 1.0
    n_interior: int = 256
    n_boundary: int = 64
    n_center: int = 192
    center_width: float = 3.0
    center_fraction: float = 0.6
    n_alpha_samples: int = 8
    n_mach_samples: int | None = None
    hidden_dim: int = 192
    depth: int = 4
    activation: str = "tanh"
    audit_every: int = 100
    checkpoint_every: int = 500
    device: str = "cpu"
    seed: int = 1234
    freeze_ci: bool = True
    detach_ci_in_mode_branch: bool = True
    reference_cache: str | None = None
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_gauge: float = 100.0
    w_center_pde: float = 1.0
    w_ci_anchor: float = 1.0
    ymax: float = 75.0
    envelope_eps: float = 1.0
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


def _make_serializable_config(cfg: KHSubsonic2DStage1quaterPressureConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["mach_min"] = float(min(cfg.mach_values))
    data["mach_max"] = float(max(cfg.mach_values))
    data["mach_values_json"] = dumps_float_list(cfg.mach_values)
    data["anchor_alphas_json"] = dumps_float_list(cfg.anchor_alphas)
    data["mach_values"] = " ".join(f"{value:.12g}" for value in cfg.mach_values)
    data["anchor_alphas"] = " ".join(f"{value:.12g}" for value in cfg.anchor_alphas)
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
    cfg: KHSubsonic2DStage1quaterPressureConfig,
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
    y: torch.Tensor,
    alpha_pairs: torch.Tensor,
    mach_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pairs = int(alpha_pairs.shape[0])
    n_y = int(y.shape[0])
    y_rep = y.repeat(n_pairs, 1)
    alpha_rep = alpha_pairs.repeat_interleave(n_y, dim=0)
    mach_rep = mach_pairs.repeat_interleave(n_y, dim=0)
    return y_rep, alpha_rep, mach_rep


def _sample_interior_y(n_points: int, *, ymax: float, device: torch.device) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty(0, 1, device=device)
    y = torch.empty(int(n_points), 1, device=device).uniform_(-float(ymax), float(ymax))
    y.requires_grad_(True)
    return y


def _sample_boundary_y(n_points: int, *, ymax: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(int(n_points), 1)
    left = torch.full((count, 1), -float(ymax), dtype=torch.float32, device=device, requires_grad=True)
    right = torch.full((count, 1), float(ymax), dtype=torch.float32, device=device, requires_grad=True)
    return left, right


def _sample_center_y(
    *,
    n_points: int,
    center_width: float,
    center_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty(0, 1, device=device)
    width = max(float(center_width), 1.0e-6)
    fraction = min(max(float(center_fraction), 0.0), 1.0)
    n_gaussian = int(round(fraction * int(n_points)))
    n_uniform = int(n_points) - n_gaussian

    chunks: list[torch.Tensor] = []
    if n_uniform > 0:
        chunks.append(torch.empty(n_uniform, 1, device=device).uniform_(-width, width))
    if n_gaussian > 0:
        y_gaussian = torch.randn(n_gaussian, 1, device=device) * (width / 3.0)
        chunks.append(torch.clamp(y_gaussian, min=-width, max=width))
    y = torch.cat(chunks, dim=0)
    y = y[torch.randperm(y.shape[0], device=device)]
    y.requires_grad_(True)
    return y


def _freeze_ci_branch(model) -> None:
    if getattr(model, "ci_net", None) is not None:
        for parameter in model.ci_net.parameters():
            parameter.requires_grad_(False)
    if hasattr(model, "raw_ci_bias"):
        model.raw_ci_bias.requires_grad_(False)


def _unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _build_optimizer(model, cfg: KHSubsonic2DStage1quaterPressureConfig) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters available for Stage 1quater.")
    return torch.optim.Adam(trainable, lr=float(cfg.learning_rate))


def _best_metric_value(row: dict[str, float | int], metric: str) -> float:
    if metric not in row:
        raise KeyError(f"best_metric={metric!r} not present in training row.")
    value = float(row[metric])
    if not np.isfinite(value):
        return float("inf")
    return value


def _format_stage1quater_report(
    cfg: KHSubsonic2DStage1quaterPressureConfig,
    *,
    stage0_checkpoint_path: Path,
    best_epoch: int,
    best_history_row: dict[str, float | int],
) -> str:
    lines = [
        "2D subsonic Stage 1quater — pressure-first modal reconstruction",
        f"stage0_checkpoint={stage0_checkpoint_path}",
        f"freeze_ci_default={cfg.freeze_ci}",
        f"detach_ci_in_mode_branch_default={cfg.detach_ci_in_mode_branch}",
        "direct_modal_supervision=False",
        "trained_fields=pressure_only",
        f"mach_values={json.dumps([float(value) for value in cfg.mach_values])}",
        f"alpha_range=[{cfg.alpha_min}, {cfg.alpha_max}]",
        f"anchor_alphas={json.dumps([float(value) for value in cfg.anchor_alphas])}",
        f"epochs={cfg.epochs}",
        f"learning_rate={cfg.learning_rate}",
        (
            f"loss_weights: pde={cfg.w_pde} bc={cfg.w_bc} gauge={cfg.w_gauge} "
            f"center_pde={cfg.w_center_pde} ci_anchor={cfg.w_ci_anchor}"
        ),
        f"ymax={cfg.ymax}",
        f"envelope_eps={cfg.envelope_eps}",
        f"best_metric={cfg.best_metric}",
        f"best_epoch={best_epoch}",
        f"best_loss_total={float(best_history_row.get('loss_total', float('nan'))):.8e}",
        f"best_loss_pde={float(best_history_row.get('loss_pde_pressure', float('nan'))):.8e}",
        f"best_loss_bc={float(best_history_row.get('loss_bc_robin', float('nan'))):.8e}",
        f"best_loss_gauge={float(best_history_row.get('loss_gauge', float('nan'))):.8e}",
        f"best_ci_anchor_max_abs={float(best_history_row.get('ci_anchor_max_abs', float('nan'))):.8e}",
        f"best_ci_anchor_max_rel_unstable={float(best_history_row.get('ci_anchor_max_rel_unstable', float('nan'))):.8e}",
        "",
        "Training losses:",
        "- pressure ODE residual on interior points",
        "- pressure Robin far-field boundary conditions",
        "- center gauge Re p(0)=1 and Im p(0)=0",
        "- optional center-focused pressure ODE residual",
        "- optional ci anchor preservation only if freeze_ci is disabled",
        "- no classical p, rho, u, v, gamma or p_y inside the loss",
    ]
    return "\n".join(lines) + "\n"


def train_kh_subsonic_2d_stage1quater_pressure(
    cfg: KHSubsonic2DStage1quaterPressureConfig,
) -> int:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    stage0_model, stage0_config, _, stage0_checkpoint_path = load_stage0_model_from_checkpoint(
        cfg.stage0_checkpoint,
        device=device,
    )

    if not cfg.mach_values:
        cfg.mach_values = _read_serialized_float_list(stage0_config, "mach_values_json", "mach_values")
    if not cfg.anchor_alphas:
        cfg.anchor_alphas = _read_serialized_float_list(stage0_config, "anchor_alphas_json", "anchor_alphas")
    if not cfg.mach_values:
        raise RuntimeError("No mach_values available for Stage 1quater.")
    if not cfg.anchor_alphas:
        raise RuntimeError("No anchor_alphas available for Stage 1quater.")

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

    model = build_pressure_first_model_from_stage0(
        stage0_model,
        pressure_hidden_dim=int(cfg.hidden_dim),
        pressure_depth=int(cfg.depth),
        activation=str(cfg.activation),
        ymax=float(cfg.ymax),
        envelope_eps=float(cfg.envelope_eps),
    )
    model.to(device)

    _unfreeze_all(model)
    if bool(cfg.freeze_ci):
        _freeze_ci_branch(model)
    optimizer = _build_optimizer(model, cfg)

    serializable_cfg = _make_serializable_config(cfg)
    pd.DataFrame([serializable_cfg]).to_csv(output_dir / "config.csv", index=False)

    best_metric_value = float("inf")
    best_epoch = 0
    best_state = None
    best_row: dict[str, float | int] = {}
    history_rows: list[dict[str, float | int]] = []

    print("Stage 1quater subsonic 2D pressure-first", flush=True)
    print(f"stage0_checkpoint={stage0_checkpoint_path}", flush=True)
    print(f"freeze_ci={cfg.freeze_ci}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"device={device}", flush=True)

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        alpha_pairs, mach_pairs = _sample_parameter_mesh(cfg, device=device)
        y_interior_base = _sample_interior_y(int(cfg.n_interior), ymax=float(cfg.ymax), device=device)
        y_left_base, y_right_base = _sample_boundary_y(int(cfg.n_boundary), ymax=float(cfg.ymax), device=device)
        y_center_base = _sample_center_y(
            n_points=int(cfg.n_center),
            center_width=min(float(cfg.center_width), 0.5 * float(cfg.ymax)),
            center_fraction=float(cfg.center_fraction),
            device=device,
        )

        y_interior, alpha_interior, mach_interior = _repeat_spatial_over_param_pairs(
            y_interior_base, alpha_pairs, mach_pairs
        )
        y_left, alpha_left, mach_left = _repeat_spatial_over_param_pairs(y_left_base, alpha_pairs, mach_pairs)
        y_right, alpha_right, mach_right = _repeat_spatial_over_param_pairs(y_right_base, alpha_pairs, mach_pairs)
        if y_center_base.numel() > 0:
            y_center, alpha_center, mach_center = _repeat_spatial_over_param_pairs(
                y_center_base, alpha_pairs, mach_pairs
            )
        else:
            y_center = torch.empty(0, 1, device=device)
            alpha_center = torch.empty(0, 1, device=device)
            mach_center = torch.empty(0, 1, device=device)

        ci_pairs = model.get_ci(alpha_pairs, mach_pairs)
        ci_override_pairs = ci_pairs.detach() if bool(cfg.detach_ci_in_mode_branch) else ci_pairs

        zero = torch.zeros(1, device=device, dtype=torch.float32).mean()

        ci_interior = ci_override_pairs.repeat_interleave(max(int(y_interior_base.shape[0]), 1), dim=0)
        res_r, res_i, _ = pressure_first_ode_residual(
            model,
            y_interior,
            alpha_interior,
            mach_interior,
            ci_override=ci_interior,
        )
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))

        loss_center_pde = zero
        if y_center.numel() > 0:
            ci_center = ci_override_pairs.repeat_interleave(int(y_center_base.shape[0]), dim=0)
            res_center_r, res_center_i, _ = pressure_first_ode_residual(
                model,
                y_center,
                alpha_center,
                mach_center,
                ci_override=ci_center,
            )
            loss_center_pde = torch.mean(res_center_r.pow(2) + res_center_i.pow(2))

        ci_boundary = ci_override_pairs.repeat_interleave(max(int(y_left_base.shape[0]), 1), dim=0)
        loss_bc = pressure_first_robin_boundary_loss(
            model,
            y_left,
            y_right,
            alpha_left,
            mach_left,
            ci_override=ci_boundary,
        )

        loss_gauge = pressure_first_gauge_loss(
            model,
            alpha_pairs,
            mach_pairs,
            ci_override=ci_override_pairs,
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
            + float(cfg.w_bc) * loss_bc
            + float(cfg.w_gauge) * loss_gauge
            + float(cfg.w_center_pde) * loss_center_pde
            + (float(cfg.w_ci_anchor) * loss_ci_anchor if not bool(cfg.freeze_ci) else zero)
        )
        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite Stage 1quater loss at epoch={epoch}.")

        loss_total.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
        optimizer.step()

        model.eval()
        anchor_metrics, anchor_predictions = _compute_anchor_metrics(model, anchors_df, device=device)
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "loss_total": float(loss_total.detach().cpu()),
            "loss_pde_pressure": float(loss_pde.detach().cpu()),
            "loss_bc_robin": float(loss_bc.detach().cpu()),
            "loss_gauge": float(loss_gauge.detach().cpu()),
            "loss_center_pde": float(loss_center_pde.detach().cpu()),
            "loss_ci_anchor": float(loss_ci_anchor.detach().cpu()),
            **anchor_metrics,
        }
        history_rows.append(row)

        metric_value = _best_metric_value(row, str(cfg.best_metric))
        if metric_value < best_metric_value:
            best_metric_value = metric_value
            best_epoch = int(epoch)
            best_row = dict(row)
            best_state = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": list(history_rows),
                "config": dict(serializable_cfg),
                "model_config": model.export_model_config(),
                "stage0_checkpoint": str(stage0_checkpoint_path),
                "best_epoch": int(epoch),
                "best_loss_total": float(row["loss_total"]),
                "best_anchor_metrics": dict(anchor_metrics),
            }
            torch.save(best_state, output_dir / "model_best.pt")
            anchor_predictions.to_csv(output_dir / "anchor_predictions_stage1quater.csv", index=False)

        if epoch == 1 or epoch % int(cfg.audit_every) == 0 or epoch == int(cfg.epochs):
            print(
                "Epoch "
                f"{epoch:5d} | loss={row['loss_total']:.3e} | pde={row['loss_pde_pressure']:.3e} "
                f"| bc={row['loss_bc_robin']:.3e} | gauge={row['loss_gauge']:.3e} "
                f"| center={row['loss_center_pde']:.3e} | ci_max_abs={row['ci_anchor_max_abs']:.3e} "
                f"| ci_max_rel_unstable={row['ci_anchor_max_rel_unstable']:.3e}",
                flush=True,
            )

        if epoch % int(cfg.checkpoint_every) == 0 or epoch == int(cfg.epochs):
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": list(history_rows),
                    "config": dict(serializable_cfg),
                    "model_config": model.export_model_config(),
                    "stage0_checkpoint": str(stage0_checkpoint_path),
                    "best_epoch": int(best_epoch),
                    "best_loss_total": float(best_row.get("loss_total", float("nan"))),
                    "best_anchor_metrics": {
                        key: float(best_row[key])
                        for key in ("ci_anchor_mae_abs", "ci_anchor_max_abs", "ci_anchor_mae_rel_unstable", "ci_anchor_max_rel_unstable", "ci_neutral_max_abs")
                        if key in best_row
                    },
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    if best_state is None:
        raise RuntimeError("Stage 1quater training did not produce a best checkpoint.")

    final_state = {
        "epoch": int(cfg.epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history_rows),
        "config": dict(serializable_cfg),
        "model_config": model.export_model_config(),
        "stage0_checkpoint": str(stage0_checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_loss_total": float(best_row.get("loss_total", float("nan"))),
        "best_anchor_metrics": {
            key: float(best_row[key])
            for key in ("ci_anchor_mae_abs", "ci_anchor_max_abs", "ci_anchor_mae_rel_unstable", "ci_anchor_max_rel_unstable", "ci_neutral_max_abs")
            if key in best_row
        },
    }
    torch.save(final_state, output_dir / "model_final.pt")
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    report_text = _format_stage1quater_report(
        cfg,
        stage0_checkpoint_path=stage0_checkpoint_path,
        best_epoch=int(best_epoch),
        best_history_row=best_row,
    )
    (output_dir / "stage1quater_report.txt").write_text(report_text)
    print(report_text, flush=True)
    return 0


__all__ = [
    "KHSubsonic2DStage1quaterPressureConfig",
    "train_kh_subsonic_2d_stage1quater_pressure",
]
