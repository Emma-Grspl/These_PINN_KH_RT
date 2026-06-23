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
from src.models.kh_subsonic_pinn_2d import build_kh_subsonic_pinn_2d_from_config
from src.physics.kh_subsonic_residual_2d import (
    boundary_decay_loss_2d,
    normalization_loss_2d,
    phase_loss_2d,
    pressure_ode_residual_2d,
    riccati_boundary_loss_components_2d,
    riccati_normalization_loss_2d,
    riccati_phase_loss_2d,
)


@dataclass
class KHSubsonic2DHybrid4CIStage1Config:
    stage0_checkpoint: str = (
        "model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt"
    )
    mach_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    alpha_min: float = 0.10
    alpha_max: float = 0.80
    anchor_alphas: tuple[float, ...] = (0.10, 0.30, 0.55, 0.80)
    output_dir: str = "model_saved/kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0"
    epochs: int = 5000
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    n_interior: int = 128
    n_boundary: int = 32
    n_alpha_samples: int = 8
    n_mach_samples: int | None = None
    audit_every: int = 100
    checkpoint_every: int = 500
    device: str = "cpu"
    seed: int = 1234
    freeze_ci: bool = True
    detach_ci_in_mode_branch: bool = True
    reference_cache: str | None = None
    w_pde: float = 1.0
    w_bc_kappa: float = 10.0
    w_bc_q: float = 25.0
    w_norm: float = 1.0
    w_phase: float = 1.0
    w_shooting: float = 0.0
    w_ci_anchor: float = 1.0
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


def load_stage0_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], dict[str, object], Path]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Stage 0 checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Unsupported Stage 0 checkpoint format in {checkpoint_file}. Expected a dict with model_state_dict."
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config_path = checkpoint_file.parent / "config.csv"
        if not config_path.exists():
            raise RuntimeError(f"Missing Stage 0 config next to checkpoint: {config_path}")
        config_df = pd.read_csv(config_path)
        if config_df.empty:
            raise RuntimeError(f"Empty Stage 0 config: {config_path}")
        config = config_df.iloc[0].to_dict()

    model = build_kh_subsonic_pinn_2d_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config, checkpoint, checkpoint_file


def _make_serializable_stage1_config(cfg: KHSubsonic2DHybrid4CIStage1Config) -> dict[str, object]:
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
    cfg: KHSubsonic2DHybrid4CIStage1Config,
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


@torch.no_grad()
def _compute_anchor_metrics(
    model,
    anchors_df: pd.DataFrame,
    *,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = anchors_df.copy()
    alpha_anchor = torch.tensor(work["alpha"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    mach_anchor = torch.tensor(work["Mach"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
    ci_ref = work["ci_reference"].to_numpy(dtype=float)
    ci_pred = model.get_ci(alpha_anchor, mach_anchor).detach().cpu().numpy().reshape(-1)
    abs_err = np.abs(ci_pred - ci_ref)
    unstable_mask = np.abs(ci_ref) > 1e-4
    neutral_mask = ~unstable_mask
    rel_err_unstable = np.full_like(abs_err, np.nan, dtype=float)
    if np.any(unstable_mask):
        rel_err_unstable[unstable_mask] = abs_err[unstable_mask] / np.abs(ci_ref[unstable_mask])

    work["ci_pred_stage1"] = ci_pred
    work["ci_abs_err"] = abs_err
    work["ci_rel_err_unstable"] = rel_err_unstable
    work["ci_is_unstable"] = unstable_mask.astype(int)
    metrics = {
        "ci_anchor_mae_abs": float(abs_err.mean()) if abs_err.size else 0.0,
        "ci_anchor_max_abs": float(abs_err.max()) if abs_err.size else 0.0,
        "ci_anchor_mae_rel_unstable": float(np.nanmean(rel_err_unstable)) if np.any(unstable_mask) else 0.0,
        "ci_anchor_max_rel_unstable": float(np.nanmax(rel_err_unstable)) if np.any(unstable_mask) else 0.0,
        "ci_neutral_max_abs": float(abs_err[neutral_mask].max()) if np.any(neutral_mask) else 0.0,
    }
    return metrics, work


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


def _freeze_ci_branch(model) -> None:
    if getattr(model, "ci_net", None) is not None:
        for parameter in model.ci_net.parameters():
            parameter.requires_grad_(False)
    if hasattr(model, "raw_ci_bias"):
        model.raw_ci_bias.requires_grad_(False)


def _unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _build_optimizer(model, cfg: KHSubsonic2DHybrid4CIStage1Config) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters available for Stage 1.")
    return torch.optim.Adam(trainable, lr=float(cfg.learning_rate))


def _format_stage1_report(
    cfg: KHSubsonic2DHybrid4CIStage1Config,
    *,
    stage0_checkpoint_path: Path,
    best_epoch: int,
    best_history_row: dict[str, float | int],
) -> str:
    lines = [
        "2D hybrid4ci Stage 1 — physics-only modal reconstruction from sparse spectral supervision",
        f"stage0_checkpoint={stage0_checkpoint_path}",
        f"freeze_ci_default={cfg.freeze_ci}",
        "direct_modal_supervision=False",
        f"mach_values={json.dumps([float(value) for value in cfg.mach_values])}",
        f"alpha_range=[{cfg.alpha_min}, {cfg.alpha_max}]",
        f"anchor_alphas={json.dumps([float(value) for value in cfg.anchor_alphas])}",
        f"epochs={cfg.epochs}",
        f"learning_rate={cfg.learning_rate}",
        f"loss_weights: pde={cfg.w_pde} bc_kappa={cfg.w_bc_kappa} bc_q={cfg.w_bc_q} norm={cfg.w_norm} phase={cfg.w_phase} shooting={cfg.w_shooting} ci_anchor={cfg.w_ci_anchor}",
        f"best_epoch={best_epoch}",
        f"best_loss_total={float(best_history_row.get('loss_total', float('nan'))):.8e}",
        f"best_ci_anchor_max_abs={float(best_history_row.get('ci_anchor_max_abs', float('nan'))):.8e}",
        f"best_ci_anchor_max_rel_unstable={float(best_history_row.get('ci_anchor_max_rel_unstable', float('nan'))):.8e}",
        f"best_ci_neutral_max_abs={float(best_history_row.get('ci_neutral_max_abs', float('nan'))):.8e}",
        "",
        "Implemented physical losses:",
        "- pressure PDE residual",
        "- Riccati far-field boundary loss",
        "- modal normalization",
        "- phase constraint",
        "- shooting placeholder logged but disabled by default in this first Stage 1 implementation",
        "- spectral anchor preservation only when freeze_ci is disabled",
    ]
    return "\n".join(lines) + "\n"


def train_kh_subsonic_2d_hybrid4ci_stage1(
    cfg: KHSubsonic2DHybrid4CIStage1Config,
) -> int:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    model, stage0_config, _, stage0_checkpoint_path = load_stage0_model_from_checkpoint(
        cfg.stage0_checkpoint,
        device=device,
    )

    if not cfg.mach_values:
        cfg.mach_values = _read_serialized_float_list(stage0_config, "mach_values_json", "mach_values")
    if not cfg.anchor_alphas:
        cfg.anchor_alphas = _read_serialized_float_list(stage0_config, "anchor_alphas_json", "anchor_alphas")
    if not cfg.mach_values:
        raise RuntimeError("No mach_values available for Stage 1.")
    if not cfg.anchor_alphas:
        raise RuntimeError("No anchor_alphas available for Stage 1.")

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

    serializable_cfg = _make_serializable_stage1_config(cfg)
    model_config = dict(stage0_config)
    model_config["alpha_min"] = float(cfg.alpha_min)
    model_config["alpha_max"] = float(cfg.alpha_max)
    model_config["mach_min"] = float(min(cfg.mach_values))
    model_config["mach_max"] = float(max(cfg.mach_values))
    pd.DataFrame([serializable_cfg]).to_csv(output_dir / "config.csv", index=False)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_row: dict[str, float | int] = {}
    history_rows: list[dict[str, float | int]] = []

    print("Stage 1 subsonic 2D hybrid4ci", flush=True)
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

        ci_override = None
        if bool(cfg.detach_ci_in_mode_branch):
            ci_override = model.get_ci(alpha_interior, mach_interior).detach()

        res_r, res_i, _ = pressure_ode_residual_2d(
            model,
            xi_interior,
            alpha_interior,
            mach_interior,
            ci_override=ci_override,
        )
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))

        zero = torch.zeros(1, device=device, dtype=loss_pde.dtype).mean()
        loss_riccati = zero
        loss_bc = zero
        loss_norm = zero
        loss_phase = zero
        loss_shooting = zero

        if getattr(model, "mode_representation", "cartesian") == "riccati":
            ci_boundary = model.get_ci(alpha_left, mach_left).detach() if bool(cfg.detach_ci_in_mode_branch) else None
            loss_bc_kappa, loss_bc_q = riccati_boundary_loss_components_2d(
                model,
                xi_left,
                xi_right,
                alpha_left,
                mach_left,
                ci_override=ci_boundary,
            )
            loss_riccati = loss_bc_kappa + loss_bc_q
            loss_bc = float(cfg.w_bc_kappa) * loss_bc_kappa + float(cfg.w_bc_q) * loss_bc_q
            loss_norm = riccati_normalization_loss_2d(model, xi_ref, alpha_ref, mach_ref)
            loss_phase = riccati_phase_loss_2d(model, xi_ref, alpha_ref, mach_ref)
        else:
            loss_bc = boundary_decay_loss_2d(model, xi_left, xi_right, alpha_left, mach_left)
            loss_norm = normalization_loss_2d(model, xi_ref, alpha_ref, mach_ref)
            loss_phase = phase_loss_2d(model, xi_ref, alpha_ref, mach_ref)

        loss_ci_anchor = zero
        if not bool(cfg.freeze_ci):
            alpha_anchor = torch.tensor(anchors_df["alpha"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
            mach_anchor = torch.tensor(anchors_df["Mach"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
            ci_target = torch.tensor(anchors_df["ci_reference"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)
            ci_pred = model.get_ci(alpha_anchor, mach_anchor)
            loss_ci_anchor = torch.mean((ci_pred - ci_target).pow(2))

        loss_total = (
            float(cfg.w_pde) * loss_pde
            + loss_bc
            + float(cfg.w_norm) * loss_norm
            + float(cfg.w_phase) * loss_phase
            + float(cfg.w_shooting) * loss_shooting
            + (float(cfg.w_ci_anchor) * loss_ci_anchor if not bool(cfg.freeze_ci) else zero)
        )

        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite Stage 1 loss at epoch={epoch}.")
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
            "loss_riccati": float(loss_riccati.detach().cpu()),
            "loss_bc": float(loss_bc.detach().cpu()),
            "loss_norm": float(loss_norm.detach().cpu()),
            "loss_phase": float(loss_phase.detach().cpu()),
            "loss_shooting": float(loss_shooting.detach().cpu()),
            "loss_ci_anchor": float(loss_ci_anchor.detach().cpu()),
            **anchor_metrics,
        }
        history_rows.append(row)

        if row["loss_total"] < best_loss:
            best_loss = float(row["loss_total"])
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
                },
                output_dir / "model_best.pt",
            )
            anchor_predictions.to_csv(output_dir / "anchor_predictions_stage1.csv", index=False)

        if epoch == 1 or epoch % int(cfg.audit_every) == 0 or epoch == int(cfg.epochs):
            print(
                "Epoch {epoch:5d} | loss={loss_total:.3e} | pde={loss_pde:.3e} | "
                "bc={loss_bc:.3e} | norm={loss_norm:.3e} | phase={loss_phase:.3e} | "
                "ci_max_abs={ci_anchor_max_abs:.3e} | ci_max_rel_unstable={ci_anchor_max_rel_unstable:.3e}".format(**row),
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
        output_path=output_dir / "anchor_predictions_stage1.csv",
    )
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": serializable_cfg,
            "model_config": model_config,
            "final_anchor_metrics": final_anchor_metrics,
            "stage0_checkpoint": str(stage0_checkpoint_path),
        },
        output_dir / "model_final.pt",
    )
    report_text = _format_stage1_report(
        cfg,
        stage0_checkpoint_path=stage0_checkpoint_path,
        best_epoch=best_epoch,
        best_history_row=best_row,
    )
    (output_dir / "stage1_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text, flush=True)
    return 0
