from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.kh_subsonic_sampling_2d import (
    DEFAULT_ANCHOR_ALPHAS,
    build_anchor_table,
    dumps_float_list,
    normalize_float_list,
)
from src.models.kh_subsonic_pinn_2d import (
    KHSubsonicPINN2D,
    build_kh_subsonic_pinn_2d_from_config,
    freeze_all_parameters,
    unfreeze_ci_head,
)


@dataclass
class KHSubsonic2DHybrid4CIStage0Config:
    mach_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    alpha_min: float = 0.10
    alpha_max: float = 0.80
    anchor_alphas: tuple[float, ...] = DEFAULT_ANCHOR_ALPHAS
    output_dir: str = "model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock"
    epochs: int = 2000
    learning_rate: float = 1e-3
    hidden_dim: int = 160
    mode_hidden_dim: int | None = None
    ci_hidden_dim: int | None = None
    mode_depth: int = 4
    ci_depth: int = 2
    activation: str = "tanh"
    fourier_features: int = 0
    fourier_scale: float = 2.0
    initial_ci: float = 0.2
    mapping_scale: float = 3.0
    mode_representation: str = "riccati"
    device: str = "cpu"
    seed: int = 1234
    reference_cache: str | None = None
    w_anchor: float = 100.0
    w_monotone_alpha: float = 1.0
    w_smooth_alpha: float = 0.0
    w_smooth_mach: float = 0.0
    n_shape_alpha: int = 65
    n_shape_mach: int = 25
    audit_every: int = 50
    lbfgs_steps: int = 0
    lbfgs_lr: float = 0.5
    target_max_abs: float = 1e-3
    target_max_rel: float = 5e-2
    fail_on_target: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(cfg: KHSubsonic2DHybrid4CIStage0Config) -> torch.device:
    if str(cfg.device).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_stage0_model(
    cfg: KHSubsonic2DHybrid4CIStage0Config,
    *,
    device: torch.device,
) -> KHSubsonicPINN2D:
    model = KHSubsonicPINN2D(
        alpha_min=float(cfg.alpha_min),
        alpha_max=float(cfg.alpha_max),
        mach_min=float(min(cfg.mach_values)),
        mach_max=float(max(cfg.mach_values)),
        hidden_dim=int(cfg.hidden_dim),
        mode_hidden_dim=cfg.mode_hidden_dim,
        ci_hidden_dim=cfg.ci_hidden_dim,
        mode_depth=int(cfg.mode_depth),
        ci_depth=int(cfg.ci_depth),
        activation=str(cfg.activation),
        fourier_features=int(cfg.fourier_features),
        fourier_scale=float(cfg.fourier_scale),
        initial_ci=float(cfg.initial_ci),
        mapping_scale=float(cfg.mapping_scale),
        trainable_mapping_scale=False,
        mode_representation=str(cfg.mode_representation),
    ).to(device)
    freeze_all_parameters(model)
    unfreeze_ci_head(model)
    return model


def shape_losses(
    model: KHSubsonicPINN2D,
    cfg: KHSubsonic2DHybrid4CIStage0Config,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha_grid = torch.linspace(float(cfg.alpha_min), float(cfg.alpha_max), int(cfg.n_shape_alpha), device=device)
    mach_grid = torch.linspace(float(min(cfg.mach_values)), float(max(cfg.mach_values)), int(cfg.n_shape_mach), device=device)
    aa, mm = torch.meshgrid(alpha_grid, mach_grid, indexing="ij")
    alpha = aa.reshape(-1, 1).clone().detach().requires_grad_(True)
    mach = mm.reshape(-1, 1).clone().detach().requires_grad_(True)

    ci = model.get_ci(alpha, mach)
    dci_dalpha = torch.autograd.grad(ci.sum(), alpha, create_graph=True)[0]
    dci_dmach = torch.autograd.grad(ci.sum(), mach, create_graph=True)[0]
    d2ci_dalpha2 = torch.autograd.grad(dci_dalpha.sum(), alpha, create_graph=True)[0]
    d2ci_dmach2 = torch.autograd.grad(dci_dmach.sum(), mach, create_graph=True)[0]

    loss_monotone_alpha = torch.relu(dci_dalpha).pow(2).mean()
    loss_smooth_alpha = d2ci_dalpha2.pow(2).mean()
    loss_smooth_mach = d2ci_dmach2.pow(2).mean()
    return loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach


@torch.no_grad()
def audit_anchor_metrics(
    model: KHSubsonicPINN2D,
    alpha_anchor: torch.Tensor,
    mach_anchor: torch.Tensor,
    ci_target: torch.Tensor,
) -> dict[str, float]:
    ci_pred = model.get_ci(alpha_anchor, mach_anchor)
    abs_err = torch.abs(ci_pred - ci_target)
    rel_err = abs_err / torch.clamp(torch.abs(ci_target), min=1e-12)
    return {
        "anchor_mae_abs": float(abs_err.mean().detach().cpu()),
        "anchor_max_abs": float(abs_err.max().detach().cpu()),
        "anchor_mae_rel": float(rel_err.mean().detach().cpu()),
        "anchor_max_rel": float(rel_err.max().detach().cpu()),
    }


@torch.no_grad()
def save_prediction_tables(
    model: KHSubsonicPINN2D,
    cfg: KHSubsonic2DHybrid4CIStage0Config,
    anchors_df: pd.DataFrame,
    *,
    device: torch.device,
    output_dir: Path,
) -> None:
    alpha_anchor = torch.tensor(anchors_df["alpha"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
    mach_anchor = torch.tensor(anchors_df["Mach"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
    ci_pred_anchor = model.get_ci(alpha_anchor, mach_anchor).detach().cpu().numpy().reshape(-1)

    anchor_predictions = anchors_df.copy()
    anchor_predictions["ci_pred_stage0"] = ci_pred_anchor
    anchor_predictions["ci_abs_err"] = np.abs(anchor_predictions["ci_pred_stage0"] - anchor_predictions["ci_reference"])
    anchor_predictions["ci_rel_err"] = anchor_predictions["ci_abs_err"] / np.maximum(
        np.abs(anchor_predictions["ci_reference"].to_numpy(dtype=float)),
        1e-12,
    )
    anchor_predictions.to_csv(output_dir / "anchor_predictions_best.csv", index=False)

    alpha_eval = np.linspace(float(cfg.alpha_min), float(cfg.alpha_max), 121, dtype=float)
    mach_eval = np.linspace(float(min(cfg.mach_values)), float(max(cfg.mach_values)), max(31, len(cfg.mach_values)), dtype=float)
    aa, mm = np.meshgrid(alpha_eval, mach_eval)
    alpha_tensor = torch.tensor(aa.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    mach_tensor = torch.tensor(mm.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    ci_pred = model.get_ci(alpha_tensor, mach_tensor).detach().cpu().numpy().reshape(-1)

    pd.DataFrame(
        {
            "alpha": aa.reshape(-1),
            "Mach": mm.reshape(-1),
            "ci_pred_stage0": ci_pred,
        }
    ).to_csv(output_dir / "ci_stage0_surface.csv", index=False)


def _serializable_config_dict(cfg: KHSubsonic2DHybrid4CIStage0Config) -> dict[str, object]:
    data = asdict(cfg)
    data["mach_min"] = float(min(cfg.mach_values))
    data["mach_max"] = float(max(cfg.mach_values))
    data["mach_values_json"] = dumps_float_list(cfg.mach_values)
    data["anchor_alphas_json"] = dumps_float_list(cfg.anchor_alphas)
    data["mach_values"] = " ".join(f"{value:.12g}" for value in cfg.mach_values)
    data["anchor_alphas"] = " ".join(f"{value:.12g}" for value in cfg.anchor_alphas)
    return data


def write_stage0_report(
    cfg: KHSubsonic2DHybrid4CIStage0Config,
    *,
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    pass_abs = best_metrics["anchor_max_abs"] <= float(cfg.target_max_abs)
    pass_rel = best_metrics["anchor_max_rel"] <= float(cfg.target_max_rel)
    status = "PASS" if pass_abs or pass_rel else "FAIL"
    lines = [
        f"Stage 0 subsonic 2D hybrid4ci: {status}",
        f"mach_values={json.dumps([float(value) for value in cfg.mach_values])}",
        f"alpha_range=[{cfg.alpha_min}, {cfg.alpha_max}]",
        f"anchor_alphas={json.dumps([float(value) for value in cfg.anchor_alphas])}",
        f"epochs={cfg.epochs}",
        f"learning_rate={cfg.learning_rate}",
        f"weights: anchor={cfg.w_anchor} monotone_alpha={cfg.w_monotone_alpha} smooth_alpha={cfg.w_smooth_alpha} smooth_mach={cfg.w_smooth_mach}",
        f"best_epoch={best_epoch}",
        f"best_anchor_mae_abs={best_metrics['anchor_mae_abs']:.8e}",
        f"best_anchor_max_abs={best_metrics['anchor_max_abs']:.8e}",
        f"best_anchor_mae_rel={best_metrics['anchor_mae_rel']:.8e}",
        f"best_anchor_max_rel={best_metrics['anchor_max_rel']:.8e}",
        f"final_anchor_mae_abs={final_metrics['anchor_mae_abs']:.8e}",
        f"final_anchor_max_abs={final_metrics['anchor_max_abs']:.8e}",
        f"final_anchor_mae_rel={final_metrics['anchor_mae_rel']:.8e}",
        f"final_anchor_max_rel={final_metrics['anchor_max_rel']:.8e}",
        "",
        "Interpretation:",
        "- PASS: the spectral head is sufficiently locked on the sparse classical ci anchors.",
        "- FAIL: keep Stage 0 as a diagnostic and do not warm-start a physics stage from it.",
    ]
    (output_dir / "stage0_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_kh_subsonic_2d_stage0_anchor_lock(
    cfg: KHSubsonic2DHybrid4CIStage0Config,
) -> int:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    device = device_from_config(cfg)

    config_dict = _serializable_config_dict(cfg)
    pd.DataFrame([config_dict]).to_csv(output_dir / "config.csv", index=False)

    anchors_df = build_anchor_table(
        mach_values=cfg.mach_values,
        anchor_alphas=cfg.anchor_alphas,
        alpha_min=float(cfg.alpha_min),
        alpha_max=float(cfg.alpha_max),
        reference_cache=cfg.reference_cache,
    )
    anchors_df.to_csv(output_dir / "anchors_used.csv", index=False)

    model = build_stage0_model(cfg, device=device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable spectral parameters were found for Stage 0.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(cfg.learning_rate))

    alpha_anchor = torch.tensor(anchors_df["alpha"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
    mach_anchor = torch.tensor(anchors_df["Mach"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
    ci_target = torch.tensor(anchors_df["ci_reference"].to_numpy(dtype=np.float32), device=device).view(-1, 1)

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_state = None
    best_metrics = {
        "anchor_mae_abs": float("inf"),
        "anchor_max_abs": float("inf"),
        "anchor_mae_rel": float("inf"),
        "anchor_max_rel": float("inf"),
    }

    def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ci_pred = model.get_ci(alpha_anchor, mach_anchor)
        loss_anchor = torch.mean((ci_pred - ci_target).pow(2))
        loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach = shape_losses(model, cfg, device=device)
        loss = (
            float(cfg.w_anchor) * loss_anchor
            + float(cfg.w_monotone_alpha) * loss_monotone_alpha
            + float(cfg.w_smooth_alpha) * loss_smooth_alpha
            + float(cfg.w_smooth_mach) * loss_smooth_mach
        )
        return loss, loss_anchor, loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach

    def is_better(metrics: dict[str, float]) -> bool:
        return (
            metrics["anchor_max_rel"] < best_metrics["anchor_max_rel"] - 1e-12
            or (
                abs(metrics["anchor_max_rel"] - best_metrics["anchor_max_rel"]) <= 1e-12
                and metrics["anchor_max_abs"] < best_metrics["anchor_max_abs"] - 1e-12
            )
            or (
                abs(metrics["anchor_max_rel"] - best_metrics["anchor_max_rel"]) <= 1e-12
                and abs(metrics["anchor_max_abs"] - best_metrics["anchor_max_abs"]) <= 1e-12
                and metrics["anchor_mae_abs"] < best_metrics["anchor_mae_abs"]
            )
        )

    def record(epoch: int, loss: torch.Tensor, loss_anchor: torch.Tensor, loss_monotone_alpha: torch.Tensor, loss_smooth_alpha: torch.Tensor, loss_smooth_mach: torch.Tensor) -> None:
        nonlocal best_epoch, best_state, best_metrics
        metrics = audit_anchor_metrics(model, alpha_anchor, mach_anchor, ci_target)
        row = {
            "epoch": int(epoch),
            "loss": float(loss.detach().cpu()),
            "loss_anchor": float(loss_anchor.detach().cpu()),
            "loss_monotone_alpha": float(loss_monotone_alpha.detach().cpu()),
            "loss_smooth_alpha": float(loss_smooth_alpha.detach().cpu()),
            "loss_smooth_mach": float(loss_smooth_mach.detach().cpu()),
            **metrics,
        }
        history.append(row)
        print(
            "Epoch {epoch:5d} | loss={loss:.3e} | anchor_mae_abs={anchor_mae_abs:.3e} | "
            "anchor_max_abs={anchor_max_abs:.3e} | anchor_mae_rel={anchor_mae_rel:.3e} | "
            "anchor_max_rel={anchor_max_rel:.3e}".format(**row),
            flush=True,
        )
        if is_better(metrics):
            best_epoch = int(epoch)
            best_metrics = metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_dict,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                },
                output_dir / "model_best.pt",
            )

    print("Stage 0 subsonic 2D hybrid4ci anchor lock", flush=True)
    print(f"mach_values={list(cfg.mach_values)}", flush=True)
    print(f"anchor_alphas={list(cfg.anchor_alphas)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"device={device}", flush=True)

    for epoch in range(1, int(cfg.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, loss_anchor, loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach = compute_losses()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage 0 loss at epoch={epoch}.")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % int(cfg.audit_every) == 0 or epoch == int(cfg.epochs):
            record(epoch, loss, loss_anchor, loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach)

    if int(cfg.lbfgs_steps) > 0:
        lbfgs = torch.optim.LBFGS(trainable_parameters, lr=float(cfg.lbfgs_lr), max_iter=1, line_search_fn="strong_wolfe")
        for step in range(1, int(cfg.lbfgs_steps) + 1):
            epoch = int(cfg.epochs) + step

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                closure_loss, _, _, _, _ = compute_losses()
                if not torch.isfinite(closure_loss):
                    raise FloatingPointError(f"Non-finite Stage 0 LBFGS loss at epoch={epoch}.")
                closure_loss.backward()
                return closure_loss

            lbfgs.step(closure)
            if step == 1 or step % int(cfg.audit_every) == 0 or step == int(cfg.lbfgs_steps):
                loss, loss_anchor, loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach = compute_losses()
                record(epoch, loss, loss_anchor, loss_monotone_alpha, loss_smooth_alpha, loss_smooth_mach)

    final_metrics = audit_anchor_metrics(model, alpha_anchor, mach_anchor, ci_target)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "final_metrics": final_metrics,
        },
        output_dir / "model_final.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    save_prediction_tables(model, cfg, anchors_df, device=device, output_dir=output_dir)
    write_stage0_report(cfg, best_epoch=best_epoch, best_metrics=best_metrics, final_metrics=final_metrics, output_dir=output_dir)
    print((output_dir / "stage0_report.txt").read_text(encoding="utf-8"), flush=True)

    pass_abs = best_metrics["anchor_max_abs"] <= float(cfg.target_max_abs)
    pass_rel = best_metrics["anchor_max_rel"] <= float(cfg.target_max_rel)
    if bool(cfg.fail_on_target) and not (pass_abs or pass_rel):
        return 2
    return 0


def load_stage0_checkpoint(run_dir: str | Path, *, device: torch.device) -> tuple[KHSubsonicPINN2D, dict[str, object], dict[str, object]]:
    run_path = Path(run_dir)
    checkpoint = torch.load(run_path / "model_best.pt", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config_df = pd.read_csv(run_path / "config.csv")
        config = config_df.iloc[0].to_dict()
    model = build_kh_subsonic_pinn_2d_from_config(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, checkpoint if isinstance(checkpoint, dict) else {}
