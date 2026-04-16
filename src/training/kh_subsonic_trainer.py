from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from src.data.kh_subsonic_sampling import (
    SubsonicReferenceCache,
    sample_alpha_adaptive_batch,
    sample_boundary_points,
    sample_mode_interior_points,
)
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN
from src.physics.kh_subsonic_residual import (
    boundary_decay_loss,
    integral_normalization_loss,
    local_peak_envelope_losses,
    localization_moment_losses,
    normalization_loss,
    phase_loss,
    pressure_ode_residual,
    riccati_boundary_band_loss_components,
    riccati_boundary_loss_components,
    riccati_center_constraints,
    riccati_shooting_match_loss,
    reconstruct_pressure_from_riccati,
    xi_to_y,
)


@dataclass
class KHSubsonicTrainingConfig:
    mach: float = 0.5
    alpha_min: float = 0.05
    alpha_max: float = 0.85
    epochs: int = 5000
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    mode_hidden_dim: int | None = None
    ci_hidden_dim: int | None = None
    mode_depth: int = 4
    ci_depth: int = 2
    fixed_scalar_ci: bool = False
    freeze_ci: bool = False
    activation: str = "tanh"
    fourier_features: int = 0
    fourier_scale: float = 2.0
    initial_ci: float = 0.2
    mapping_scale: float = 3.0
    trainable_mapping_scale: bool = False
    n_interior: int = 512
    n_boundary: int = 64
    n_alpha_supervision: int = 128
    n_anchor_alpha: int = 32
    n_norm_interior: int = 256
    n_reference_alpha: int = 81
    n_audit_alpha: int = 21
    n_mode_audit_alpha: int = 7
    n_mode_audit_y: int = 801
    audit_every: int = 250
    checkpoint_every: int = 500
    enable_classic_ci_supervision: bool = True
    enable_classic_mode_audit: bool = True
    stage_split_epoch: int = 0
    stage2_freeze_ci: bool = False
    stage2_ci_lr_scale: float = 1.0
    separate_branch_optimizers: bool = False
    detach_ci_in_mode_branch: bool = False
    ci_branch_lr: float | None = None
    mode_branch_lr: float | None = None
    stage1_w_ci_supervision: float | None = None
    stage2_w_ci_supervision: float | None = None
    stage1_neutral_fraction: float | None = None
    stage2_neutral_fraction: float | None = None
    focus_fraction: float = 0.6
    focus_half_width: float = 0.03
    neutral_fraction: float = 0.0
    ci_supervision_neutral_boost: float = 0.0
    neutral_half_width: float = 0.03
    error_threshold: float = 0.01
    mode_error_threshold: float = 0.12
    max_focus_points: int = 8
    anchor_strategy: str = "band"
    anchor_half_width: float = 0.12
    anchor_max_candidates: int = 257
    mode_center_fraction: float = 0.5
    mode_center_half_width: float = 0.3
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_bc_kappa: float = 10.0
    w_bc_q: float = 10.0
    w_norm: float = 1.0
    w_integral_norm: float = 1.0
    w_phase: float = 1.0
    w_peak_slope: float = 0.0
    w_peak_curvature: float = 0.0
    w_loc_center: float = 0.0
    w_loc_spread: float = 0.0
    w_ci_supervision: float = 5.0
    w_ci_stability_outside: float = 0.0
    w_ci_neutrality: float = 0.0
    w_ci_low_alpha_zero: float = 0.0
    w_ci_smoothness: float = 0.0
    n_ci_spectral_grid: int = 129
    audit_ci_weight: float = 10.0
    audit_mode_weight: float = 1.0
    audit_env_weight: float = 1.0
    audit_phase_weight: float = 0.5
    audit_peak_weight: float = 0.25
    phase_mask_fraction: float = 0.15
    classic_n_points: int = 561
    classic_mapping_scale: float = 3.0
    classic_xi_max: float = 0.99
    enforce_mode_symmetry: bool = False
    mode_representation: str = "cartesian"
    mode_experts: int = 1
    alpha_split_threshold: float = 0.4
    riccati_anchor_supervision: bool = False
    riccati_anchor_n_xi: int = 97
    riccati_anchor_every: int = 20
    riccati_anchor_alphas: tuple[float, ...] = ()
    w_riccati_anchor: float = 1.0
    w_q_supervision: float = 0.0
    q_supervision_n_xi: int = 97
    q_supervision_every: int = 20
    q_supervision_alpha_count: int = 6
    w_riccati_center_kappa: float = 0.0
    w_riccati_center_peak: float = 0.0
    w_riccati_boundary_band_kappa: float = 0.0
    w_riccati_boundary_band_q: float = 0.0
    riccati_center_xi: float = 0.0
    riccati_boundary_band_points: int = 0
    riccati_boundary_band_start: float = 0.94
    riccati_boundary_band_end: float = 0.995
    w_riccati_shooting_match: float = 0.0
    riccati_shooting_steps: int = 256
    riccati_shooting_xi_boundary: float = 0.995
    mode_low_alpha_threshold: float = 0.25
    mode_low_alpha_weight: float = 1.0
    mode_low_alpha_audit_fraction: float = 0.6
    output_dir: str = "model_saved/kh_subsonic_fixed_mach"
    device: str = "cpu"


class KingOfTheHill:
    def __init__(self, model: torch.nn.Module):
        self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.best_metric = float("inf")

    def update(self, model: torch.nn.Module, metric: float) -> bool:
        if metric < self.best_metric:
            self.best_metric = float(metric)
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return True
        return False


def safe_torch_save(state_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path, _use_new_zipfile_serialization=False)


def anchor_pressure_mode(y: np.ndarray, p: np.ndarray, *, anchor_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.interp(anchor_y, y, np.real(p)) + 1j * np.interp(anchor_y, y, np.imag(p))
    if abs(p0) > 1e-12:
        p = p / p0
    return y, p


def build_anchor_points(
    model: KHSubsonicFixedMachPINN,
    alpha_ref: torch.Tensor,
    cfg: KHSubsonicTrainingConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.anchor_strategy == "point":
        xi = torch.zeros(alpha_ref.shape[0], 1, device=device, requires_grad=True)
        return xi, alpha_ref

    if cfg.anchor_strategy == "band":
        xi = sample_mode_interior_points(
            alpha_ref.shape[0],
            center_fraction=1.0,
            center_half_width=cfg.anchor_half_width,
            device=device,
        )
        return xi, alpha_ref

    if cfg.anchor_strategy == "max":
        candidate_xi = torch.linspace(
            -cfg.mode_center_half_width,
            cfg.mode_center_half_width,
            cfg.anchor_max_candidates,
            device=device,
        ).view(-1, 1)
        xi_out = torch.empty(alpha_ref.shape[0], 1, device=device)
        with torch.no_grad():
            for idx in range(alpha_ref.shape[0]):
                alpha_i = alpha_ref[idx : idx + 1].repeat(candidate_xi.shape[0], 1)
                pred = model(candidate_xi, alpha_i)
                amp2 = pred[:, 0].pow(2) + pred[:, 1].pow(2)
                max_idx = int(torch.argmax(amp2).item())
                xi_out[idx, 0] = candidate_xi[max_idx, 0]
        xi_out.requires_grad_(True)
        return xi_out, alpha_ref

    if cfg.anchor_strategy == "point_max":
        xi_point = torch.zeros(alpha_ref.shape[0], 1, device=device)
        candidate_xi = torch.linspace(
            -cfg.mode_center_half_width,
            cfg.mode_center_half_width,
            cfg.anchor_max_candidates,
            device=device,
        ).view(-1, 1)
        xi_max = torch.empty(alpha_ref.shape[0], 1, device=device)
        with torch.no_grad():
            for idx in range(alpha_ref.shape[0]):
                alpha_i = alpha_ref[idx : idx + 1].repeat(candidate_xi.shape[0], 1)
                pred = model(candidate_xi, alpha_i)
                amp2 = pred[:, 0].pow(2) + pred[:, 1].pow(2)
                max_idx = int(torch.argmax(amp2).item())
                xi_max[idx, 0] = candidate_xi[max_idx, 0]
        xi_out = torch.cat([xi_point, xi_max], dim=0)
        alpha_out = torch.cat([alpha_ref, alpha_ref], dim=0)
        xi_out.requires_grad_(True)
        return xi_out, alpha_out

    raise ValueError(f"Unsupported anchor_strategy={cfg.anchor_strategy!r}.")


class PressureModeReferenceCache:
    def __init__(self, *, mach: float, n_points: int, mapping_scale: float, xi_max: float):
        self.mach = float(mach)
        self.n_points = int(n_points)
        self.mapping_scale = float(mapping_scale)
        self.xi_max = float(xi_max)
        self.cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        alpha_key = round(float(alpha), 8)
        if alpha_key not in self.cache:
            solver = NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=self.mach,
                n_points=self.n_points,
                mapping_scale=self.mapping_scale,
                xi_max=self.xi_max,
            )
            mode, _, _ = solver.get_selected_mode()
            if mode is None:
                raise RuntimeError(f"Aucun mode classique selectionne pour alpha={alpha:.6f}, M={self.mach:.6f}.")
            p_ref = mode["vector"][2 * solver.n_points : 3 * solver.n_points]
            self.cache[alpha_key] = anchor_pressure_mode(solver.y, p_ref, anchor_y=0.0)
        return self.cache[alpha_key]


def riccati_anchor_supervision_loss(
    model: KHSubsonicFixedMachPINN,
    alpha_samples: torch.Tensor,
    reference_cache: PressureModeReferenceCache,
    *,
    n_xi: int,
    xi_half_width: float,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    xi_template = torch.linspace(-float(xi_half_width), float(xi_half_width), int(n_xi), device=device).view(-1, 1)
    for alpha_value in alpha_samples[:, 0].detach().cpu().numpy():
        alpha_scalar = float(alpha_value)
        alpha_line = torch.full_like(xi_template, alpha_scalar)
        pr, pi, y_pred_t = reconstruct_pressure_from_riccati(model, xi_template, alpha_line, anchor_xi=0.0)
        y_pred = y_pred_t[:, 0].detach().cpu().numpy()

        y_ref, p_ref = reference_cache.get(alpha_scalar)
        p_ref_interp = np.interp(y_pred, y_ref, np.real(p_ref)) + 1j * np.interp(y_pred, y_ref, np.imag(p_ref))
        target = torch.tensor(
            np.column_stack([np.real(p_ref_interp), np.imag(p_ref_interp)]),
            dtype=pr.dtype,
            device=device,
        )
        pred = torch.cat([pr, pi], dim=-1)
        losses.append(torch.mean((pred - target) ** 2))

    if not losses:
        return torch.zeros(1, device=device, dtype=alpha_samples.dtype).mean()
    return torch.stack(losses).mean()


def build_riccati_anchor_alphas(
    cfg: KHSubsonicTrainingConfig,
    alpha_ref: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if len(cfg.riccati_anchor_alphas):
        return torch.tensor(cfg.riccati_anchor_alphas, dtype=alpha_ref.dtype, device=device).view(-1, 1)
    count = min(3, alpha_ref.shape[0])
    return alpha_ref[:count]


def build_mode_audit_alphas(cfg: KHSubsonicTrainingConfig) -> np.ndarray:
    n_total = max(int(cfg.n_mode_audit_alpha), 1)
    alpha_min = float(cfg.alpha_min)
    alpha_max = float(cfg.alpha_max)
    threshold = float(np.clip(cfg.mode_low_alpha_threshold, alpha_min, alpha_max))

    if n_total == 1 or threshold <= alpha_min or cfg.mode_low_alpha_audit_fraction <= 0.0:
        return np.linspace(alpha_min, alpha_max, n_total, dtype=float)
    if threshold >= alpha_max:
        return np.linspace(alpha_min, alpha_max, n_total, dtype=float)

    n_low = int(round(float(cfg.mode_low_alpha_audit_fraction) * n_total))
    n_low = min(max(n_low, 1), n_total - 1)
    n_high = n_total - n_low

    low = np.linspace(alpha_min, threshold, n_low, endpoint=False, dtype=float)
    high = np.linspace(threshold, alpha_max, n_high, dtype=float)
    grid = np.concatenate([low, high])
    if grid.shape[0] != n_total:
        return np.linspace(alpha_min, alpha_max, n_total, dtype=float)
    return grid


def mode_alpha_weights(alphas: np.ndarray, cfg: KHSubsonicTrainingConfig) -> np.ndarray:
    weights = np.ones_like(alphas, dtype=float)
    threshold = float(cfg.mode_low_alpha_threshold)
    if cfg.mode_low_alpha_weight > 1.0:
        weights[alphas <= threshold] = float(cfg.mode_low_alpha_weight)
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def compute_reference_q(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    dp_dy = np.gradient(p, y, edge_order=2)
    denom = np.where(np.abs(p) > 1e-10, p, 1e-10 + 0.0j)
    return np.imag(dp_dy / denom)


def build_q_supervision_alphas(
    cfg: KHSubsonicTrainingConfig,
    alpha_ref: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    count = min(int(cfg.q_supervision_alpha_count), int(alpha_ref.shape[0]))
    if count <= 0:
        return alpha_ref[:0]

    threshold = min(max(float(cfg.mode_low_alpha_threshold), float(cfg.alpha_min)), float(cfg.alpha_max))
    n_low = count // 2 if threshold > float(cfg.alpha_min) else 0
    n_base = count - n_low

    chunks: list[torch.Tensor] = []
    if n_base > 0:
        if alpha_ref.shape[0] <= n_base:
            chunks.append(alpha_ref[:n_base])
        else:
            perm = torch.randperm(alpha_ref.shape[0], device=device)[:n_base]
            chunks.append(alpha_ref[perm])
    if n_low > 0:
        low = torch.rand(n_low, 1, device=device, dtype=alpha_ref.dtype)
        low = float(cfg.alpha_min) + (threshold - float(cfg.alpha_min)) * low
        chunks.append(low)

    alpha_samples = torch.cat(chunks, dim=0)
    perm = torch.randperm(alpha_samples.shape[0], device=device)
    return alpha_samples[perm]


def riccati_q_supervision_loss(
    model: KHSubsonicFixedMachPINN,
    alpha_samples: torch.Tensor,
    reference_cache: PressureModeReferenceCache,
    *,
    n_xi: int,
    phase_mask_fraction: float,
    low_alpha_threshold: float,
    low_alpha_weight: float,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    xi_template = torch.linspace(-0.98, 0.98, int(n_xi), device=device).view(-1, 1)
    y_pred = xi_to_y(xi_template, model.get_mapping_scale().detach())[:, 0].detach().cpu().numpy()

    for alpha_value in alpha_samples[:, 0].detach().cpu().numpy():
        alpha_scalar = float(alpha_value)
        alpha_line = torch.full_like(xi_template, alpha_scalar)
        q_pred = model(xi_template, alpha_line)[:, 1]

        y_ref, p_ref = reference_cache.get(alpha_scalar)
        q_ref = compute_reference_q(y_ref, p_ref)
        overlap = (y_pred >= float(np.min(y_ref))) & (y_pred <= float(np.max(y_ref)))
        if not np.any(overlap):
            continue

        y_overlap = y_pred[overlap]
        p_ref_real = np.interp(y_overlap, y_ref, np.real(p_ref))
        p_ref_imag = np.interp(y_overlap, y_ref, np.imag(p_ref))
        env_ref = np.hypot(p_ref_real, p_ref_imag)
        q_ref_interp = np.interp(y_overlap, y_ref, q_ref)
        amp_threshold = float(phase_mask_fraction) * max(float(np.max(np.abs(p_ref))), 1e-12)
        mask = env_ref >= amp_threshold
        if not np.any(mask):
            mask = np.ones_like(env_ref, dtype=bool)

        target = torch.tensor(q_ref_interp[mask], dtype=q_pred.dtype, device=device)
        q_local = q_pred[torch.tensor(overlap, device=device)][torch.tensor(mask, device=device)]
        alpha_weight = float(low_alpha_weight) if alpha_scalar <= float(low_alpha_threshold) else 1.0
        losses.append(alpha_weight * torch.mean((q_local - target) ** 2))

    if not losses:
        return torch.zeros(1, device=device, dtype=alpha_samples.dtype).mean()
    return torch.stack(losses).mean()


def compute_mode_diagnostics(
    model: KHSubsonicFixedMachPINN,
    *,
    alpha: float,
    device: torch.device,
    n_y: int,
    reference_cache: PressureModeReferenceCache,
    phase_mask_fraction: float,
) -> dict[str, float]:
    y_ref, p_ref = reference_cache.get(alpha)

    xi = torch.linspace(-0.98, 0.98, n_y, device=device).view(-1, 1)
    alpha_tensor = torch.full_like(xi, float(alpha))
    with torch.no_grad():
        if model.mode_representation == "riccati":
            pr, pi, y_pred_t = reconstruct_pressure_from_riccati(model, xi, alpha_tensor, anchor_xi=0.0)
            y_pred = y_pred_t.cpu().numpy().reshape(-1)
            p_pred = (pr[:, 0] + 1j * pi[:, 0]).cpu().numpy().reshape(-1)
        else:
            pred = model(xi, alpha_tensor)
            y_pred = xi_to_y(xi, model.get_mapping_scale().detach()).cpu().numpy().reshape(-1)
            p_pred = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy().reshape(-1)
    y_pred, p_pred = anchor_pressure_mode(y_pred, p_pred, anchor_y=0.0)

    y_min = max(float(np.min(y_ref)), float(np.min(y_pred)))
    y_max = min(float(np.max(y_ref)), float(np.max(y_pred)))
    if y_max <= y_min:
        return float("inf")
    y_common = np.linspace(y_min, y_max, n_y, dtype=float)
    p_ref_interp = np.interp(y_common, y_ref, np.real(p_ref)) + 1j * np.interp(y_common, y_ref, np.imag(p_ref))
    p_pred_interp = np.interp(y_common, y_pred, np.real(p_pred)) + 1j * np.interp(y_common, y_pred, np.imag(p_pred))
    denom = np.linalg.norm(p_ref_interp)
    if denom <= 1e-12:
        p_rel = float(np.linalg.norm(p_pred_interp - p_ref_interp))
    else:
        p_rel = float(np.linalg.norm(p_pred_interp - p_ref_interp) / denom)

    env_ref = np.abs(p_ref_interp)
    env_pred = np.abs(p_pred_interp)
    env_denom = np.linalg.norm(env_ref)
    if env_denom <= 1e-12:
        env_rel = float(np.linalg.norm(env_pred - env_ref))
    else:
        env_rel = float(np.linalg.norm(env_pred - env_ref) / env_denom)

    env_threshold = float(phase_mask_fraction) * max(float(env_ref.max()), 1e-12)
    mask = env_ref >= env_threshold
    if np.any(mask):
        phase_ref = np.angle(p_ref_interp[mask])
        phase_pred = np.angle(p_pred_interp[mask])
        phase_diff = np.angle(np.exp(1j * (phase_pred - phase_ref)))
        phase_rel = float(np.sqrt(np.mean(phase_diff**2)) / np.pi)
    else:
        phase_rel = float("inf")

    peak_ref_idx = int(np.argmax(env_ref))
    peak_pred_idx = int(np.argmax(env_pred))
    peak_shift = float(abs(y_common[peak_pred_idx] - y_common[peak_ref_idx]))

    return {
        "p_rel": p_rel,
        "env_rel": env_rel,
        "phase_rel": phase_rel,
        "peak_shift": peak_shift,
    }


def audit_ci_and_mode(
    model: KHSubsonicFixedMachPINN,
    reference_cache: SubsonicReferenceCache,
    mode_reference_cache: PressureModeReferenceCache,
    cfg: KHSubsonicTrainingConfig,
    *,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    alphas_np, ci_true_np = reference_cache.audit_grid(num_points=cfg.n_audit_alpha)
    alpha_tensor = torch.tensor(alphas_np, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        ci_pred = model.get_ci(alpha_tensor).cpu().numpy().reshape(-1)
    ci_abs_err = abs(ci_pred - ci_true_np)
    denom = abs(ci_true_np) + 1e-8
    ci_rel_err = ci_abs_err / denom
    mode_alphas = build_mode_audit_alphas(cfg)
    mode_diag = [
        compute_mode_diagnostics(
            model,
            alpha=float(alpha),
            device=device,
            n_y=cfg.n_mode_audit_y,
            reference_cache=mode_reference_cache,
            phase_mask_fraction=cfg.phase_mask_fraction,
        )
        for alpha in mode_alphas
    ]
    mode_rel_err = np.array([row["p_rel"] for row in mode_diag], dtype=float)
    env_rel_err = np.array([row["env_rel"] for row in mode_diag], dtype=float)
    phase_rel_err = np.array([row["phase_rel"] for row in mode_diag], dtype=float)
    peak_shift_err = np.array([row["peak_shift"] for row in mode_diag], dtype=float)
    mode_weights = mode_alpha_weights(mode_alphas, cfg)
    mode_focus_err = np.maximum.reduce([mode_rel_err, env_rel_err, phase_rel_err]) * mode_weights
    metrics = {
        "audit_ci_mae": float(ci_abs_err.mean()),
        "audit_ci_max_abs": float(ci_abs_err.max()),
        "audit_ci_mean_rel": float(ci_rel_err.mean()),
        "audit_ci_max_rel": float(ci_rel_err.max()),
        "audit_p_rel_l2_mean": float(mode_rel_err.mean()),
        "audit_p_rel_l2_max": float(mode_rel_err.max()),
        "audit_env_rel_mean": float(env_rel_err.mean()),
        "audit_env_rel_max": float(env_rel_err.max()),
        "audit_phase_rel_mean": float(phase_rel_err.mean()),
        "audit_phase_rel_max": float(phase_rel_err.max()),
        "audit_peak_shift_mean": float(peak_shift_err.mean()),
        "audit_peak_shift_max": float(peak_shift_err.max()),
        "audit_env_rel_weighted_mean": weighted_mean(env_rel_err, mode_weights),
        "audit_phase_rel_weighted_mean": weighted_mean(phase_rel_err, mode_weights),
        "audit_peak_shift_weighted_mean": weighted_mean(peak_shift_err, mode_weights),
    }
    metrics["audit_checkpoint_metric"] = (
        cfg.audit_ci_weight * metrics["audit_ci_mae"]
        + cfg.audit_env_weight * metrics["audit_env_rel_weighted_mean"]
        + cfg.audit_phase_weight * metrics["audit_phase_rel_weighted_mean"]
        + cfg.audit_peak_weight * metrics["audit_peak_shift_weighted_mean"]
    )
    return (
        metrics,
        np.asarray(alphas_np, dtype=float),
        np.asarray(ci_abs_err, dtype=float),
        mode_alphas,
        mode_focus_err,
    )


def train_fixed_mach_subsonic_pinn(cfg: KHSubsonicTrainingConfig) -> tuple[KHSubsonicFixedMachPINN, pd.DataFrame]:
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_classic_ci_supervision = bool(cfg.enable_classic_ci_supervision)
    use_classic_mode_audit = bool(cfg.enable_classic_mode_audit)
    use_classic_mode_supervision = bool(
        (cfg.riccati_anchor_supervision and cfg.w_riccati_anchor > 0.0) or cfg.w_q_supervision > 0.0
    )
    use_classic_references = use_classic_ci_supervision or use_classic_mode_audit or use_classic_mode_supervision

    reference_cache = None
    if use_classic_references:
        reference_cache = SubsonicReferenceCache.build(
            mach=cfg.mach,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            num_alpha=cfg.n_reference_alpha,
        )

    mode_reference_cache = None
    if use_classic_mode_audit or use_classic_mode_supervision:
        mode_reference_cache = PressureModeReferenceCache(
            mach=cfg.mach,
            n_points=cfg.classic_n_points,
            mapping_scale=cfg.classic_mapping_scale,
            xi_max=cfg.classic_xi_max,
        )

    model = KHSubsonicFixedMachPINN(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.hidden_dim,
        mode_hidden_dim=cfg.mode_hidden_dim,
        ci_hidden_dim=cfg.ci_hidden_dim,
        mode_depth=cfg.mode_depth,
        ci_depth=cfg.ci_depth,
        fixed_scalar_ci=cfg.fixed_scalar_ci,
        activation=cfg.activation,
        fourier_features=cfg.fourier_features,
        fourier_scale=cfg.fourier_scale,
        initial_ci=cfg.initial_ci,
        mapping_scale=cfg.mapping_scale,
        trainable_mapping_scale=cfg.trainable_mapping_scale,
        enforce_mode_symmetry=cfg.enforce_mode_symmetry,
        mode_representation=cfg.mode_representation,
        mode_experts=cfg.mode_experts,
        alpha_split_threshold=cfg.alpha_split_threshold,
    ).to(device)
    model.mach = float(cfg.mach)
    if cfg.freeze_ci:
        if model.ci_net is not None:
            for param in model.ci_net.parameters():
                param.requires_grad_(False)
        model.raw_ci_bias.requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ci_optimizer = None
    mode_optimizer = None
    if cfg.separate_branch_optimizers:
        ci_params = ([model.raw_ci_bias] if model.ci_net is None else list(model.ci_net.parameters()) + [model.raw_ci_bias])
        ci_param_ids = {id(param) for param in ci_params}
        mode_params = [param for param in model.parameters() if id(param) not in ci_param_ids]
        trainable_ci_params = [param for param in ci_params if param.requires_grad]
        if trainable_ci_params:
            ci_optimizer = optim.Adam(trainable_ci_params, lr=cfg.ci_branch_lr or cfg.learning_rate)
        mode_optimizer = optim.Adam(mode_params, lr=cfg.mode_branch_lr or cfg.learning_rate)
    king = KingOfTheHill(model)
    focus_alphas: np.ndarray | None = None

    history: list[dict] = []
    neutral_alpha = None
    if cfg.mach < 1.0:
        neutral_alpha = float(np.sqrt(max(1.0 - cfg.mach**2, 0.0)))
    stage2_started = False
    for epoch in range(1, cfg.epochs + 1):
        if cfg.stage_split_epoch > 0 and epoch == cfg.stage_split_epoch + 1 and not stage2_started:
            stage2_started = True
            if cfg.stage2_freeze_ci:
                if model.ci_net is not None:
                    for param in model.ci_net.parameters():
                        param.requires_grad_(False)
                model.raw_ci_bias.requires_grad_(False)
                if cfg.separate_branch_optimizers:
                    mode_params = [p for p in model.parameters() if p.requires_grad]
                    mode_optimizer = optim.Adam(mode_params, lr=cfg.mode_branch_lr or cfg.learning_rate)
                    ci_optimizer = None
                else:
                    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
            elif cfg.stage2_ci_lr_scale != 1.0:
                if cfg.separate_branch_optimizers:
                    if ci_optimizer is not None:
                        for group in ci_optimizer.param_groups:
                            group["lr"] = (cfg.ci_branch_lr or cfg.learning_rate) * cfg.stage2_ci_lr_scale
                else:
                    ci_params = ([model.raw_ci_bias] if model.ci_net is None else list(model.ci_net.parameters()) + [model.raw_ci_bias])
                    ci_param_ids = {id(param) for param in ci_params}
                    other_params = [param for param in model.parameters() if id(param) not in ci_param_ids]
                    optimizer = optim.Adam(
                        [
                            {"params": other_params, "lr": cfg.learning_rate},
                            {"params": ci_params, "lr": cfg.learning_rate * cfg.stage2_ci_lr_scale},
                        ]
                    )

        stage_w_ci_supervision = cfg.w_ci_supervision
        if not stage2_started and cfg.stage1_w_ci_supervision is not None:
            stage_w_ci_supervision = cfg.stage1_w_ci_supervision
        if stage2_started and cfg.stage2_w_ci_supervision is not None:
            stage_w_ci_supervision = cfg.stage2_w_ci_supervision

        stage_neutral_fraction = cfg.neutral_fraction
        if not stage2_started and cfg.stage1_neutral_fraction is not None:
            stage_neutral_fraction = cfg.stage1_neutral_fraction
        if stage2_started and cfg.stage2_neutral_fraction is not None:
            stage_neutral_fraction = cfg.stage2_neutral_fraction

        model.train()
        optimizer.zero_grad()

        xi_interior = sample_mode_interior_points(
            cfg.n_interior,
            center_fraction=cfg.mode_center_fraction,
            center_half_width=cfg.mode_center_half_width,
            device=device,
        )
        alpha_interior = sample_alpha_adaptive_batch(
            cfg.n_interior,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            neutral_fraction=stage_neutral_fraction,
            neutral_alpha=neutral_alpha,
            neutral_half_width=cfg.neutral_half_width,
            device=device,
        )
        xi_left, xi_right = sample_boundary_points(cfg.n_boundary, device=device)
        alpha_boundary = sample_alpha_adaptive_batch(
            cfg.n_boundary,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            neutral_fraction=stage_neutral_fraction,
            neutral_alpha=neutral_alpha,
            neutral_half_width=cfg.neutral_half_width,
            device=device,
        )
        alpha_ref = sample_alpha_adaptive_batch(
            cfg.n_anchor_alpha,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            neutral_fraction=stage_neutral_fraction,
            neutral_alpha=neutral_alpha,
            neutral_half_width=cfg.neutral_half_width,
            device=device,
        )
        xi_ref, alpha_anchor = build_anchor_points(model, alpha_ref, cfg, device=device)
        xi_norm = sample_mode_interior_points(
            cfg.n_norm_interior,
            center_fraction=cfg.mode_center_fraction,
            center_half_width=cfg.mode_center_half_width,
            device=device,
        )
        alpha_norm = sample_alpha_adaptive_batch(
            cfg.n_norm_interior,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            neutral_fraction=stage_neutral_fraction,
            neutral_alpha=neutral_alpha,
            neutral_half_width=cfg.neutral_half_width,
            device=device,
        )

        ci_supervision_neutral_fraction = min(1.0, stage_neutral_fraction + max(cfg.ci_supervision_neutral_boost, 0.0))
        alpha_supervision = sample_alpha_adaptive_batch(
            cfg.n_alpha_supervision,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            focus_alphas=focus_alphas,
            focus_fraction=cfg.focus_fraction,
            focus_half_width=cfg.focus_half_width,
            neutral_fraction=ci_supervision_neutral_fraction,
            neutral_alpha=neutral_alpha,
            neutral_half_width=cfg.neutral_half_width,
            device=device,
        )
        if use_classic_ci_supervision:
            assert reference_cache is not None
            ci_target = reference_cache.interpolate(alpha_supervision)
            ci_pred = model.get_ci(alpha_supervision)
            loss_ci = torch.mean((ci_pred - ci_target).pow(2))
        else:
            loss_ci = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_bc_kappa = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_bc_q = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_norm = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_integral_norm = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_phase = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_peak_slope = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_peak_curvature = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_loc_center = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_loc_spread = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_anchor = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_q_supervision = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_center_kappa = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_center_peak = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_boundary_band_kappa = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_boundary_band_q = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_riccati_shooting_match = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_ci_stability_outside = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_ci_neutrality = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_ci_low_alpha_zero = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()
        loss_ci_smoothness = torch.zeros(1, device=device, dtype=xi_interior.dtype).mean()

        if (
            cfg.w_ci_stability_outside > 0.0
            or cfg.w_ci_neutrality > 0.0
            or cfg.w_ci_low_alpha_zero > 0.0
            or cfg.w_ci_smoothness > 0.0
        ):
            alpha_spectral = torch.linspace(
                float(cfg.alpha_min),
                float(cfg.alpha_max),
                max(int(cfg.n_ci_spectral_grid), 3),
                device=device,
                dtype=xi_interior.dtype,
            ).view(-1, 1)
            alpha_spectral.requires_grad_(cfg.w_ci_smoothness > 0.0)
            ci_spectral = model.get_ci(alpha_spectral)

            if cfg.w_ci_stability_outside > 0.0 and neutral_alpha is not None:
                outside_mask = alpha_spectral[:, 0] >= float(neutral_alpha)
                if torch.any(outside_mask):
                    loss_ci_stability_outside = torch.mean(ci_spectral[outside_mask].pow(2))

            if cfg.w_ci_neutrality > 0.0 and neutral_alpha is not None:
                alpha_neutral_t = torch.tensor([[float(neutral_alpha)]], device=device, dtype=xi_interior.dtype)
                ci_neutral = model.get_ci(alpha_neutral_t)
                loss_ci_neutrality = torch.mean(ci_neutral.pow(2))

            if cfg.w_ci_low_alpha_zero > 0.0:
                alpha_low_t = torch.tensor([[float(cfg.alpha_min)]], device=device, dtype=xi_interior.dtype)
                ci_low = model.get_ci(alpha_low_t)
                loss_ci_low_alpha_zero = torch.mean(ci_low.pow(2))

            if cfg.w_ci_smoothness > 0.0:
                dci_dalpha = torch.autograd.grad(ci_spectral.sum(), alpha_spectral, create_graph=True)[0]
                loss_ci_smoothness = torch.mean(dci_dalpha.pow(2))

        if cfg.separate_branch_optimizers:
            if ci_optimizer is not None and stage_w_ci_supervision > 0.0:
                ci_optimizer.zero_grad()
                loss_ci_branch = stage_w_ci_supervision * loss_ci
                loss_ci_branch.backward()
                ci_optimizer.step()
            else:
                loss_ci_branch = stage_w_ci_supervision * loss_ci.detach()

            mode_optimizer.zero_grad()
            ci_for_mode = model.get_ci(alpha_interior).detach() if cfg.detach_ci_in_mode_branch else None
            ci_for_boundary = model.get_ci(alpha_boundary).detach() if cfg.detach_ci_in_mode_branch else None
            res_r, res_i, _ = pressure_ode_residual(model, xi_interior, alpha_interior, cfg.mach, ci_override=ci_for_mode)
            loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))
            if model.mode_representation == "riccati":
                loss_bc_kappa, loss_bc_q = riccati_boundary_loss_components(
                    model,
                    xi_left,
                    xi_right,
                    alpha_boundary,
                    ci_override=ci_for_boundary,
                )
                loss_bc = cfg.w_bc_kappa * loss_bc_kappa + cfg.w_bc_q * loss_bc_q
                if cfg.w_riccati_center_kappa > 0.0 or cfg.w_riccati_center_peak > 0.0:
                    loss_riccati_center_kappa, loss_riccati_center_peak = riccati_center_constraints(
                        model,
                        alpha_ref,
                        center_xi=cfg.riccati_center_xi,
                    )
                if (
                    cfg.riccati_boundary_band_points > 0
                    and (cfg.w_riccati_boundary_band_kappa > 0.0 or cfg.w_riccati_boundary_band_q > 0.0)
                ):
                    loss_riccati_boundary_band_kappa, loss_riccati_boundary_band_q = riccati_boundary_band_loss_components(
                        model,
                        alpha_boundary,
                        n_points=cfg.riccati_boundary_band_points,
                        xi_start=cfg.riccati_boundary_band_start,
                        xi_end=cfg.riccati_boundary_band_end,
                        ci_override=ci_for_boundary,
                    )
                if cfg.w_riccati_shooting_match > 0.0:
                    loss_riccati_shooting_match = riccati_shooting_match_loss(
                        model,
                        alpha_ref,
                        cfg.mach,
                        n_steps=cfg.riccati_shooting_steps,
                        xi_boundary=cfg.riccati_shooting_xi_boundary,
                    )
                if (
                    cfg.riccati_anchor_supervision
                    and cfg.w_riccati_anchor > 0.0
                    and cfg.riccati_anchor_every > 0
                    and epoch % cfg.riccati_anchor_every == 0
                ):
                    alpha_anchor_loss = build_riccati_anchor_alphas(cfg, alpha_ref, device=device)
                    assert mode_reference_cache is not None
                    loss_riccati_anchor = riccati_anchor_supervision_loss(
                        model,
                        alpha_anchor_loss,
                        mode_reference_cache,
                        n_xi=cfg.riccati_anchor_n_xi,
                        xi_half_width=cfg.anchor_half_width,
                        device=device,
                    )
                if cfg.w_q_supervision > 0.0 and cfg.q_supervision_every > 0 and epoch % cfg.q_supervision_every == 0:
                    alpha_q = build_q_supervision_alphas(cfg, alpha_ref, device=device)
                    assert mode_reference_cache is not None
                    loss_q_supervision = riccati_q_supervision_loss(
                        model,
                        alpha_q,
                        mode_reference_cache,
                        n_xi=cfg.q_supervision_n_xi,
                        phase_mask_fraction=cfg.phase_mask_fraction,
                        low_alpha_threshold=cfg.mode_low_alpha_threshold,
                        low_alpha_weight=cfg.mode_low_alpha_weight,
                        device=device,
                    )
            else:
                loss_bc = boundary_decay_loss(model, xi_left, xi_right, alpha_boundary)
                if cfg.anchor_strategy in {"point", "max", "point_max"}:
                    loss_norm = normalization_loss(model, xi_ref, alpha_anchor)
                else:
                    loss_norm = integral_normalization_loss(model, xi_ref, alpha_anchor)
                loss_integral_norm = integral_normalization_loss(model, xi_norm, alpha_norm)
                loss_phase = phase_loss(model, xi_ref, alpha_anchor)
                loss_peak_slope, loss_peak_curvature = local_peak_envelope_losses(model, xi_ref, alpha_anchor)
                loss_loc_center, loss_loc_spread = localization_moment_losses(model, xi_norm, alpha_norm)
            loss_mode = (
                cfg.w_pde * loss_pde
                + (loss_bc if model.mode_representation == "riccati" else cfg.w_bc * loss_bc)
                + cfg.w_norm * loss_norm
                + cfg.w_integral_norm * loss_integral_norm
                + cfg.w_phase * loss_phase
                + cfg.w_peak_slope * loss_peak_slope
                + cfg.w_peak_curvature * loss_peak_curvature
                + cfg.w_loc_center * loss_loc_center
                + cfg.w_loc_spread * loss_loc_spread
                + cfg.w_riccati_anchor * loss_riccati_anchor
                + cfg.w_q_supervision * loss_q_supervision
                + cfg.w_riccati_center_kappa * loss_riccati_center_kappa
                + cfg.w_riccati_center_peak * loss_riccati_center_peak
                + cfg.w_riccati_boundary_band_kappa * loss_riccati_boundary_band_kappa
                + cfg.w_riccati_boundary_band_q * loss_riccati_boundary_band_q
                + cfg.w_riccati_shooting_match * loss_riccati_shooting_match
                + cfg.w_ci_stability_outside * loss_ci_stability_outside
                + cfg.w_ci_neutrality * loss_ci_neutrality
                + cfg.w_ci_low_alpha_zero * loss_ci_low_alpha_zero
                + cfg.w_ci_smoothness * loss_ci_smoothness
            )
            loss_mode.backward()
            mode_optimizer.step()
            loss = loss_mode + loss_ci_branch
        else:
            res_r, res_i, _ = pressure_ode_residual(model, xi_interior, alpha_interior, cfg.mach)
            loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))
            loss_bc = boundary_decay_loss(model, xi_left, xi_right, alpha_boundary)
            if model.mode_representation == "riccati":
                loss_bc_kappa, loss_bc_q = riccati_boundary_loss_components(model, xi_left, xi_right, alpha_boundary)
                loss_bc = cfg.w_bc_kappa * loss_bc_kappa + cfg.w_bc_q * loss_bc_q
                if cfg.w_riccati_center_kappa > 0.0 or cfg.w_riccati_center_peak > 0.0:
                    loss_riccati_center_kappa, loss_riccati_center_peak = riccati_center_constraints(
                        model,
                        alpha_ref,
                        center_xi=cfg.riccati_center_xi,
                    )
                if (
                    cfg.riccati_boundary_band_points > 0
                    and (cfg.w_riccati_boundary_band_kappa > 0.0 or cfg.w_riccati_boundary_band_q > 0.0)
                ):
                    loss_riccati_boundary_band_kappa, loss_riccati_boundary_band_q = riccati_boundary_band_loss_components(
                        model,
                        alpha_boundary,
                        n_points=cfg.riccati_boundary_band_points,
                        xi_start=cfg.riccati_boundary_band_start,
                        xi_end=cfg.riccati_boundary_band_end,
                    )
                if cfg.w_riccati_shooting_match > 0.0:
                    loss_riccati_shooting_match = riccati_shooting_match_loss(
                        model,
                        alpha_ref,
                        cfg.mach,
                        n_steps=cfg.riccati_shooting_steps,
                        xi_boundary=cfg.riccati_shooting_xi_boundary,
                    )
                if (
                    cfg.riccati_anchor_supervision
                    and cfg.w_riccati_anchor > 0.0
                    and cfg.riccati_anchor_every > 0
                    and epoch % cfg.riccati_anchor_every == 0
                ):
                    alpha_anchor_loss = build_riccati_anchor_alphas(cfg, alpha_ref, device=device)
                    assert mode_reference_cache is not None
                    loss_riccati_anchor = riccati_anchor_supervision_loss(
                        model,
                        alpha_anchor_loss,
                        mode_reference_cache,
                        n_xi=cfg.riccati_anchor_n_xi,
                        xi_half_width=cfg.anchor_half_width,
                        device=device,
                    )
                if cfg.w_q_supervision > 0.0 and cfg.q_supervision_every > 0 and epoch % cfg.q_supervision_every == 0:
                    alpha_q = build_q_supervision_alphas(cfg, alpha_ref, device=device)
                    assert mode_reference_cache is not None
                    loss_q_supervision = riccati_q_supervision_loss(
                        model,
                        alpha_q,
                        mode_reference_cache,
                        n_xi=cfg.q_supervision_n_xi,
                        phase_mask_fraction=cfg.phase_mask_fraction,
                        low_alpha_threshold=cfg.mode_low_alpha_threshold,
                        low_alpha_weight=cfg.mode_low_alpha_weight,
                        device=device,
                    )
            else:
                if cfg.anchor_strategy in {"point", "max", "point_max"}:
                    loss_norm = normalization_loss(model, xi_ref, alpha_anchor)
                else:
                    loss_norm = integral_normalization_loss(model, xi_ref, alpha_anchor)
                loss_integral_norm = integral_normalization_loss(model, xi_norm, alpha_norm)
                loss_phase = phase_loss(model, xi_ref, alpha_anchor)
                loss_peak_slope, loss_peak_curvature = local_peak_envelope_losses(model, xi_ref, alpha_anchor)
                loss_loc_center, loss_loc_spread = localization_moment_losses(model, xi_norm, alpha_norm)
            loss = (
                cfg.w_pde * loss_pde
                + (loss_bc if model.mode_representation == "riccati" else cfg.w_bc * loss_bc)
                + cfg.w_norm * loss_norm
                + cfg.w_integral_norm * loss_integral_norm
                + cfg.w_phase * loss_phase
                + cfg.w_peak_slope * loss_peak_slope
                + cfg.w_peak_curvature * loss_peak_curvature
                + cfg.w_loc_center * loss_loc_center
                + cfg.w_loc_spread * loss_loc_spread
                + cfg.w_riccati_anchor * loss_riccati_anchor
                + cfg.w_q_supervision * loss_q_supervision
                + cfg.w_riccati_center_kappa * loss_riccati_center_kappa
                + cfg.w_riccati_center_peak * loss_riccati_center_peak
                + cfg.w_riccati_boundary_band_kappa * loss_riccati_boundary_band_kappa
                + cfg.w_riccati_boundary_band_q * loss_riccati_boundary_band_q
                + cfg.w_riccati_shooting_match * loss_riccati_shooting_match
                + cfg.w_ci_stability_outside * loss_ci_stability_outside
                + cfg.w_ci_neutrality * loss_ci_neutrality
                + cfg.w_ci_low_alpha_zero * loss_ci_low_alpha_zero
                + cfg.w_ci_smoothness * loss_ci_smoothness
                + stage_w_ci_supervision * loss_ci
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_bc_kappa": float(loss_bc_kappa.item()),
            "loss_bc_q": float(loss_bc_q.item()),
            "loss_norm": float(loss_norm.item()),
            "loss_integral_norm": float(loss_integral_norm.item()),
            "loss_phase": float(loss_phase.item()),
            "loss_peak_slope": float(loss_peak_slope.item()),
            "loss_peak_curvature": float(loss_peak_curvature.item()),
            "loss_loc_center": float(loss_loc_center.item()),
            "loss_loc_spread": float(loss_loc_spread.item()),
            "loss_riccati_anchor": float(loss_riccati_anchor.item()),
            "loss_q_supervision": float(loss_q_supervision.item()),
            "loss_riccati_center_kappa": float(loss_riccati_center_kappa.item()),
            "loss_riccati_center_peak": float(loss_riccati_center_peak.item()),
            "loss_riccati_boundary_band_kappa": float(loss_riccati_boundary_band_kappa.item()),
            "loss_riccati_boundary_band_q": float(loss_riccati_boundary_band_q.item()),
            "loss_riccati_shooting_match": float(loss_riccati_shooting_match.item()),
            "loss_ci_stability_outside": float(loss_ci_stability_outside.item()),
            "loss_ci_neutrality": float(loss_ci_neutrality.item()),
            "loss_ci_low_alpha_zero": float(loss_ci_low_alpha_zero.item()),
            "loss_ci_smoothness": float(loss_ci_smoothness.item()),
            "loss_ci_supervision": float(loss_ci.item()),
            "stage_w_ci_supervision": float(stage_w_ci_supervision),
            "stage_neutral_fraction": float(stage_neutral_fraction),
            "ci_supervision_neutral_fraction": float(ci_supervision_neutral_fraction),
            "stage2_started": int(stage2_started),
            "mapping_scale": float(model.get_mapping_scale().item()),
            "ci_mid": float(model.get_ci(torch.tensor([[0.5 * (cfg.alpha_min + cfg.alpha_max)]], device=device)).item()),
        }

        should_audit = use_classic_mode_audit and cfg.audit_every > 0 and (epoch == 1 or epoch % cfg.audit_every == 0)
        if should_audit:
            assert reference_cache is not None
            assert mode_reference_cache is not None
            audit_metrics, alpha_grid, ci_abs_err, mode_alpha_grid, mode_rel_err = audit_ci_and_mode(
                model,
                reference_cache,
                mode_reference_cache,
                cfg,
                device=device,
            )
            record.update(audit_metrics)

            failing_mask = ci_abs_err > cfg.error_threshold
            failing_alphas = alpha_grid[failing_mask]
            failing_mode_alphas = mode_alpha_grid[mode_rel_err > cfg.mode_error_threshold]
            if len(failing_mode_alphas):
                failing_alphas = np.unique(np.concatenate([failing_alphas, failing_mode_alphas]))
            if len(failing_alphas) > cfg.max_focus_points:
                severity = []
                for alpha in failing_alphas:
                    ci_idx = int(np.argmin(np.abs(alpha_grid - alpha)))
                    mode_idx = int(np.argmin(np.abs(mode_alpha_grid - alpha)))
                    score = cfg.audit_ci_weight * float(ci_abs_err[ci_idx]) + cfg.audit_mode_weight * float(mode_rel_err[mode_idx])
                    severity.append(score)
                worst_idx = np.argsort(np.asarray(severity, dtype=float))[-cfg.max_focus_points :]
                failing_alphas = failing_alphas[worst_idx]
            focus_alphas = np.asarray(failing_alphas, dtype=float) if len(failing_alphas) else None

            record["n_focus_alphas"] = 0 if focus_alphas is None else int(len(focus_alphas))
            record["focus_alpha_min"] = np.nan if focus_alphas is None else float(np.min(focus_alphas))
            record["focus_alpha_max"] = np.nan if focus_alphas is None else float(np.max(focus_alphas))

            king.update(model, record["audit_checkpoint_metric"])
            print(
                f"Epoch {epoch:5d} | loss={record['loss']:.3e} | "
                f"ci_mae={record['audit_ci_mae']:.3e} | "
                f"p_rel={record['audit_p_rel_l2_mean']:.3e} | "
                f"env={record['audit_env_rel_mean']:.3e} | "
                f"phase={record['audit_phase_rel_mean']:.3e} | "
                f"peak={record['audit_peak_shift_mean']:.3e} | "
                f"n_focus={record['n_focus_alphas']} | "
                f"L={record['mapping_scale']:.3f}"
            )
        else:
            focus_alphas = None
            record["n_focus_alphas"] = 0
            record["focus_alpha_min"] = np.nan
            record["focus_alpha_max"] = np.nan
            record["audit_checkpoint_metric"] = float(record["loss"])
            king.update(model, record["audit_checkpoint_metric"])
            if epoch == 1 or (cfg.audit_every > 0 and epoch % cfg.audit_every == 0):
                print(
                    f"Epoch {epoch:5d} | loss={record['loss']:.3e} | "
                    f"pde={record['loss_pde']:.3e} | "
                    f"bc_k={record['loss_bc_kappa']:.3e} | "
                    f"bc_q={record['loss_bc_q']:.3e} | "
                    f"ci_sup={record['loss_ci_supervision']:.3e} | "
                    f"q_sup={record['loss_q_supervision']:.3e} | "
                    f"L={record['mapping_scale']:.3f}"
                )
        history.append(record)

        if epoch % cfg.checkpoint_every == 0:
            safe_torch_save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")

    model.load_state_dict(king.best_state)
    return model, pd.DataFrame(history)


def save_training_artifacts(
    model: KHSubsonicFixedMachPINN,
    history: pd.DataFrame,
    cfg: KHSubsonicTrainingConfig,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_torch_save(model.state_dict(), output_dir / "model_best.pt")
    history.to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame([asdict(cfg)]).to_csv(output_dir / "config.csv", index=False)
