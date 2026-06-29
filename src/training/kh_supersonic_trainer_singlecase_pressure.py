from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.physics.kh_supersonic_pressure_first_1d import (
    KHSupersonicPressureFirst1D,
    supersonic_pressure_gauge_loss,
    supersonic_pressure_ode_residual,
    supersonic_pressure_robin_boundary_loss,
)


@dataclass
class KHSupersonicSingleCasePressureConfig:
    alpha: float = 0.26
    mach: float = 1.8
    cr: float = 0.38
    ci: float = 0.024
    output_dir: str = "model_saved/kh_supersonic_singlecase_pressure_fixed_c"
    epochs: int = 3000
    learning_rate: float = 2e-5
    grad_clip_norm: float = 1.0
    n_interior: int = 512
    n_boundary: int = 64
    n_center: int = 256
    hidden_dim: int = 192
    depth: int = 6
    activation: str = "tanh"
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_gauge: float = 100.0
    w_center_pde: float = 1.0
    ymax: float = 120.0
    envelope_eps: float = 1.0
    device: str = "cpu"
    seed: int = 1234
    audit_every: int = 100
    checkpoint_every: int = 500
    center_width: float = 6.0
    center_fraction: float = 0.6
    interior_center_fraction: float = 0.35
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


def _make_serializable_config(cfg: KHSupersonicSingleCasePressureConfig) -> dict[str, object]:
    return asdict(cfg)


def _build_optimizer(model: torch.nn.Module, cfg: KHSupersonicSingleCasePressureConfig) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters available for supersonic single-case training.")
    return torch.optim.Adam(trainable, lr=float(cfg.learning_rate))


def _sample_center_y(
    *,
    n_points: int,
    center_width: float,
    center_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty(0, 1, device=device, dtype=torch.float32)
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


def _sample_interior_y(
    *,
    n_points: int,
    ymax: float,
    center_width: float,
    center_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty(0, 1, device=device, dtype=torch.float32)
    fraction = min(max(float(center_fraction), 0.0), 1.0)
    n_center = int(round(fraction * int(n_points)))
    n_uniform = int(n_points) - n_center

    chunks: list[torch.Tensor] = []
    if n_uniform > 0:
        chunks.append(torch.empty(n_uniform, 1, device=device).uniform_(-float(ymax), float(ymax)))
    if n_center > 0:
        width = min(max(float(center_width), 1.0e-6), float(ymax))
        y_center = torch.randn(n_center, 1, device=device) * (width / 2.5)
        chunks.append(torch.clamp(y_center, min=-width, max=width))

    y = torch.cat(chunks, dim=0)
    y = y[torch.randperm(y.shape[0], device=device)]
    y.requires_grad_(True)
    return y


def _sample_boundary_y(n_points: int, *, ymax: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(int(n_points), 1)
    left = torch.full((count, 1), -float(ymax), dtype=torch.float32, device=device, requires_grad=True)
    right = torch.full((count, 1), float(ymax), dtype=torch.float32, device=device, requires_grad=True)
    return left, right


def _best_metric_value(row: dict[str, float | int], metric: str) -> float:
    if metric not in row:
        raise KeyError(f"best_metric={metric!r} not present in training row.")
    value = float(row[metric])
    if not np.isfinite(value):
        return float("inf")
    return value


def train_kh_supersonic_singlecase_pressure(
    cfg: KHSupersonicSingleCasePressureConfig,
) -> int:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    model = KHSupersonicPressureFirst1D(
        alpha=float(cfg.alpha),
        mach=float(cfg.mach),
        cr=float(cfg.cr),
        ci=float(cfg.ci),
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
        activation=str(cfg.activation),
        ymax=float(cfg.ymax),
        envelope_eps=float(cfg.envelope_eps),
    ).to(device)
    optimizer = _build_optimizer(model, cfg)

    serializable_cfg = _make_serializable_config(cfg)
    pd.DataFrame([serializable_cfg]).to_csv(output_dir / "config.csv", index=False)

    best_metric_value = float("inf")
    best_epoch = 0
    best_row: dict[str, float | int] = {}
    best_state = None
    history_rows: list[dict[str, float | int]] = []

    print("Supersonic single-case pressure-first PINN", flush=True)
    print(
        f"alpha={cfg.alpha} mach={cfg.mach} cr={cfg.cr} ci={cfg.ci} "
        f"output_dir={output_dir} device={device}",
        flush=True,
    )

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        y_interior = _sample_interior_y(
            n_points=int(cfg.n_interior),
            ymax=float(cfg.ymax),
            center_width=float(cfg.center_width),
            center_fraction=float(cfg.interior_center_fraction),
            device=device,
        )
        y_left, y_right = _sample_boundary_y(int(cfg.n_boundary), ymax=float(cfg.ymax), device=device)
        y_center = _sample_center_y(
            n_points=int(cfg.n_center),
            center_width=float(cfg.center_width),
            center_fraction=float(cfg.center_fraction),
            device=device,
        )

        zero = torch.zeros(1, device=device, dtype=torch.float32).mean()

        res_r, res_i, _ = supersonic_pressure_ode_residual(model, y_interior)
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))

        loss_bc = supersonic_pressure_robin_boundary_loss(model, y_left, y_right)
        loss_gauge = supersonic_pressure_gauge_loss(model)

        loss_center_pde = zero
        if y_center.numel() > 0:
            res_center_r, res_center_i, _ = supersonic_pressure_ode_residual(model, y_center)
            loss_center_pde = torch.mean(res_center_r.pow(2) + res_center_i.pow(2))

        loss_total = (
            float(cfg.w_pde) * loss_pde
            + float(cfg.w_bc) * loss_bc
            + float(cfg.w_gauge) * loss_gauge
            + float(cfg.w_center_pde) * loss_center_pde
        )
        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite supersonic single-case loss at epoch={epoch}.")

        loss_total.backward()
        if float(cfg.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
        optimizer.step()

        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "loss_total": float(loss_total.detach().cpu()),
            "loss_pde_pressure": float(loss_pde.detach().cpu()),
            "loss_bc_robin": float(loss_bc.detach().cpu()),
            "loss_gauge": float(loss_gauge.detach().cpu()),
            "loss_center_pde": float(loss_center_pde.detach().cpu()),
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
                "best_epoch": int(epoch),
                "best_loss_total": float(row["loss_total"]),
                "best_loss_pde": float(row["loss_pde_pressure"]),
                "best_loss_bc": float(row["loss_bc_robin"]),
                "best_loss_gauge": float(row["loss_gauge"]),
            }
            torch.save(best_state, output_dir / "model_best.pt")

        if epoch == 1 or epoch % int(cfg.audit_every) == 0 or epoch == int(cfg.epochs):
            print(
                "Epoch "
                f"{epoch:5d} | loss={row['loss_total']:.3e} | pde={row['loss_pde_pressure']:.3e} "
                f"| bc={row['loss_bc_robin']:.3e} | gauge={row['loss_gauge']:.3e} "
                f"| center={row['loss_center_pde']:.3e}",
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
                    "best_epoch": int(best_epoch),
                    "best_loss_total": float(best_row.get("loss_total", float("nan"))),
                    "best_loss_pde": float(best_row.get("loss_pde_pressure", float("nan"))),
                    "best_loss_bc": float(best_row.get("loss_bc_robin", float("nan"))),
                    "best_loss_gauge": float(best_row.get("loss_gauge", float("nan"))),
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    if best_state is None:
        raise RuntimeError("Supersonic single-case training did not produce a best checkpoint.")

    final_state = {
        "epoch": int(cfg.epochs),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": list(history_rows),
        "config": dict(serializable_cfg),
        "model_config": model.export_model_config(),
        "best_epoch": int(best_epoch),
        "best_loss_total": float(best_row.get("loss_total", float("nan"))),
        "best_loss_pde": float(best_row.get("loss_pde_pressure", float("nan"))),
        "best_loss_bc": float(best_row.get("loss_bc_robin", float("nan"))),
        "best_loss_gauge": float(best_row.get("loss_gauge", float("nan"))),
    }
    torch.save(final_state, output_dir / "model_final.pt")
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    return 0


__all__ = [
    "KHSupersonicSingleCasePressureConfig",
    "train_kh_supersonic_singlecase_pressure",
]
