from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.models.kh_subsonic_pinn import KHSubsonicFixedMachPINN


@dataclass
class Stage0Config:
    mach: float
    alpha_min: float
    alpha_max: float
    anchors: tuple[float, ...]
    output_dir: Path
    epochs: int
    learning_rate: float
    hidden_dim: int
    mode_hidden_dim: int | None
    ci_hidden_dim: int | None
    mode_depth: int
    ci_depth: int
    activation: str
    fourier_features: int
    fourier_scale: float
    initial_ci: float
    mapping_scale: float
    device: str
    reference_csv: Path | None
    w_anchor: float
    w_monotone: float
    w_smooth: float
    lbfgs_steps: int
    lbfgs_lr: float
    n_shape_grid: int
    n_curve: int
    audit_every: int
    target_max_abs: float
    fail_on_target: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 0 subsonique: verrouille uniquement ci(alpha) sur des ancres "
            "classiques avant tout entrainement PDE/mode."
        )
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.80)
    parser.add_argument("--anchors", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--mode-hidden-dim", type=int, default=None)
    parser.add_argument("--ci-hidden-dim", type=int, default=None)
    parser.add_argument("--mode-depth", type=int, default=4)
    parser.add_argument("--ci-depth", type=int, default=2)
    parser.add_argument("--activation", choices=["tanh", "silu"], default="tanh")
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--initial-ci", type=float, default=0.2)
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--w-anchor", type=float, default=100.0)
    parser.add_argument("--w-monotone", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.0)
    parser.add_argument("--lbfgs-steps", type=int, default=500)
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument("--n-shape-grid", type=int, default=129)
    parser.add_argument("--n-curve", type=int, default=301)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--target-max-abs", type=float, default=1e-3)
    parser.add_argument("--no-fail-on-target", action="store_true")
    return parser


def parse_config() -> Stage0Config:
    args = build_parser().parse_args()
    anchors = tuple(sorted(float(alpha) for alpha in args.anchors))
    if not anchors:
        raise ValueError("Il faut fournir au moins une ancre alpha.")
    if min(anchors) < args.alpha_min - 1e-12 or max(anchors) > args.alpha_max + 1e-12:
        raise ValueError(
            f"Les ancres doivent etre dans [{args.alpha_min}, {args.alpha_max}], recu {anchors}."
        )
    if args.n_shape_grid < 5:
        raise ValueError("--n-shape-grid doit etre >= 5.")
    return Stage0Config(
        mach=float(args.mach),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        anchors=anchors,
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        mode_hidden_dim=args.mode_hidden_dim,
        ci_hidden_dim=args.ci_hidden_dim,
        mode_depth=int(args.mode_depth),
        ci_depth=int(args.ci_depth),
        activation=str(args.activation),
        fourier_features=int(args.fourier_features),
        fourier_scale=float(args.fourier_scale),
        initial_ci=float(args.initial_ci),
        mapping_scale=float(args.mapping_scale),
        device=str(args.device),
        reference_csv=args.reference_csv,
        w_anchor=float(args.w_anchor),
        w_monotone=float(args.w_monotone),
        w_smooth=float(args.w_smooth),
        lbfgs_steps=max(0, int(args.lbfgs_steps)),
        lbfgs_lr=float(args.lbfgs_lr),
        n_shape_grid=int(args.n_shape_grid),
        n_curve=int(args.n_curve),
        audit_every=max(1, int(args.audit_every)),
        target_max_abs=float(args.target_max_abs),
        fail_on_target=not bool(args.no_fail_on_target),
    )


def read_reference_csv(path: Path, anchors: tuple[float, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "alpha" not in df.columns:
        raise ValueError(f"{path} ne contient pas de colonne alpha.")
    ci_column = None
    for candidate in ("ci_reference", "ci_classic", "ci", "ci_true"):
        if candidate in df.columns:
            ci_column = candidate
            break
    if ci_column is None:
        raise ValueError(f"{path} ne contient aucune colonne ci exploitable.")

    src = df[["alpha", ci_column]].dropna().sort_values("alpha")
    alpha_src = src["alpha"].to_numpy(dtype=float)
    ci_src = src[ci_column].to_numpy(dtype=float)
    ci_targets = np.interp(np.asarray(anchors, dtype=float), alpha_src, ci_src)
    return pd.DataFrame(
        {
            "alpha": np.asarray(anchors, dtype=float),
            "Mach": np.full(len(anchors), np.nan),
            "ci_reference": ci_targets,
            "source": f"csv:{path}",
            "success": True,
        }
    )


def compute_anchor_targets(cfg: Stage0Config) -> pd.DataFrame:
    if cfg.reference_csv is not None:
        return read_reference_csv(cfg.reference_csv, cfg.anchors)

    rows: list[dict[str, object]] = []
    for alpha in cfg.anchors:
        solver = RobustSubsonicShootingSolver(alpha=alpha, Mach=cfg.mach)
        result = solver.solve(force_cross_check=True)
        if not result.success:
            raise RuntimeError(f"Reference classique non convergee pour alpha={alpha:.6f}, M={cfg.mach:.6f}.")
        rows.append(
            {
                "alpha": result.alpha,
                "Mach": result.Mach,
                "ci_reference": result.ci,
                "omega_i": result.omega_i,
                "source": result.source,
                "success": result.success,
                "primary_ci": result.primary_ci,
                "primary_success": result.primary_success,
                "primary_mismatch": result.primary_mismatch,
                "secondary_ci": result.secondary_ci,
                "secondary_success": result.secondary_success,
                "secondary_stage1_mismatch": result.secondary_stage1_mismatch,
                "secondary_stage2_mismatch": result.secondary_stage2_mismatch,
                "ci_abs_diff": result.ci_abs_diff,
            }
        )
    return pd.DataFrame(rows)


def build_model(cfg: Stage0Config, device: torch.device) -> KHSubsonicFixedMachPINN:
    model = KHSubsonicFixedMachPINN(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        hidden_dim=cfg.hidden_dim,
        mode_hidden_dim=cfg.mode_hidden_dim,
        ci_hidden_dim=cfg.ci_hidden_dim,
        mode_depth=cfg.mode_depth,
        ci_depth=cfg.ci_depth,
        activation=cfg.activation,
        fourier_features=cfg.fourier_features,
        fourier_scale=cfg.fourier_scale,
        initial_ci=cfg.initial_ci,
        mapping_scale=cfg.mapping_scale,
        trainable_mapping_scale=False,
        mode_representation="riccati",
    ).to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.ci_net is not None:
        for parameter in model.ci_net.parameters():
            parameter.requires_grad_(True)
    model.raw_ci_bias.requires_grad_(True)
    return model


def ci_shape_losses(
    model: KHSubsonicFixedMachPINN,
    cfg: Stage0Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_grid = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_shape_grid, device=device).view(-1, 1)
    alpha_grid.requires_grad_(True)
    ci_grid = model.get_ci(alpha_grid)
    dci = torch.autograd.grad(ci_grid.sum(), alpha_grid, create_graph=True)[0]
    d2ci = torch.autograd.grad(dci.sum(), alpha_grid, create_graph=True)[0]

    # Branche subsonique de Blumen: ci decroit quand alpha augmente.
    loss_monotone = torch.relu(dci).pow(2).mean()
    loss_smooth = d2ci.pow(2).mean()
    return loss_monotone, loss_smooth


@torch.no_grad()
def audit_model(
    model: KHSubsonicFixedMachPINN,
    alpha_anchor: torch.Tensor,
    ci_target: torch.Tensor,
) -> dict[str, float]:
    ci_pred = model.get_ci(alpha_anchor)
    abs_err = torch.abs(ci_pred - ci_target)
    rel_err = abs_err / torch.clamp(torch.abs(ci_target), min=1e-12)
    return {
        "anchor_mae": float(abs_err.mean().detach().cpu()),
        "anchor_max_abs": float(abs_err.max().detach().cpu()),
        "anchor_mean_rel": float(rel_err.mean().detach().cpu()),
        "anchor_max_rel": float(rel_err.max().detach().cpu()),
    }


def save_curve(
    model: KHSubsonicFixedMachPINN,
    cfg: Stage0Config,
    targets: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> None:
    alphas = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_curve)
    with torch.no_grad():
        alpha_t = torch.tensor(alphas, dtype=torch.float32, device=device).view(-1, 1)
        ci_pinn = model.get_ci(alpha_t).detach().cpu().numpy().reshape(-1)

    anchor_alpha = targets["alpha"].to_numpy(dtype=float)
    anchor_ci = targets["ci_reference"].to_numpy(dtype=float)
    ci_interp = np.interp(alphas, anchor_alpha, anchor_ci)
    curve = pd.DataFrame(
        {
            "alpha": alphas,
            "Mach": cfg.mach,
            "ci_pinn_stage0": ci_pinn,
            "ci_reference_anchor_interp": ci_interp,
            "ci_abs_err_anchor_interp": np.abs(ci_pinn - ci_interp),
        }
    )
    curve.to_csv(output_dir / "ci_stage0_curve.csv", index=False)


def write_report(
    cfg: Stage0Config,
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    status = "PASS" if best_metrics["anchor_max_abs"] <= cfg.target_max_abs else "FAIL"
    lines = [
        f"Stage 0 CI anchor lock: {status}",
        f"Mach={cfg.mach}",
        f"alpha_range=[{cfg.alpha_min}, {cfg.alpha_max}]",
        "anchors=" + " ".join(f"{alpha:.6f}" for alpha in cfg.anchors),
        f"epochs={cfg.epochs}",
        f"learning_rate={cfg.learning_rate}",
        f"weights: anchor={cfg.w_anchor} monotone={cfg.w_monotone} smooth={cfg.w_smooth}",
        f"lbfgs_steps={cfg.lbfgs_steps}",
        f"target_max_abs={cfg.target_max_abs}",
        f"best_epoch={best_epoch}",
        f"best_anchor_mae={best_metrics['anchor_mae']:.8e}",
        f"best_anchor_max_abs={best_metrics['anchor_max_abs']:.8e}",
        f"best_anchor_mean_rel={best_metrics['anchor_mean_rel']:.8e}",
        f"best_anchor_max_rel={best_metrics['anchor_max_rel']:.8e}",
        f"final_anchor_mae={final_metrics['anchor_mae']:.8e}",
        f"final_anchor_max_abs={final_metrics['anchor_max_abs']:.8e}",
        "",
        "Interpretation:",
        "- PASS: la branche ci(alpha) apprend les ancres; on peut passer au stage PDE/mode en warmstart.",
        "- FAIL: ne pas lancer le stage PDE/mode; corriger d'abord le verrouillage de ci.",
    ]
    (output_dir / "stage0_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_stage0(cfg: Stage0Config) -> int:
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**asdict(cfg), "output_dir": str(cfg.output_dir), "reference_csv": str(cfg.reference_csv or "")}]).to_csv(
        output_dir / "config.csv", index=False
    )

    targets = compute_anchor_targets(cfg)
    targets.to_csv(output_dir / "anchor_targets.csv", index=False)

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Aucun parametre ci entrainable.")
    optimizer = torch.optim.Adam(trainable, lr=cfg.learning_rate)

    alpha_anchor = torch.tensor(targets["alpha"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
    ci_target = torch.tensor(targets["ci_reference"].to_numpy(dtype=np.float32), device=device).view(-1, 1)

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_metrics = {
        "anchor_mae": float("inf"),
        "anchor_max_abs": float("inf"),
        "anchor_mean_rel": float("inf"),
        "anchor_max_rel": float("inf"),
    }
    best_state = None

    print("Stage 0 CI anchor lock")
    print(f"Mach={cfg.mach} alpha=[{cfg.alpha_min}, {cfg.alpha_max}]")
    print("anchors=" + " ".join(f"{alpha:.6f}" for alpha in cfg.anchors))
    print(f"output_dir={output_dir}")
    print(f"device={device}")

    def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ci_pred = model.get_ci(alpha_anchor)
        loss_anchor = torch.mean((ci_pred - ci_target).pow(2))
        loss_monotone, loss_smooth = ci_shape_losses(model, cfg, device)
        loss = cfg.w_anchor * loss_anchor + cfg.w_monotone * loss_monotone + cfg.w_smooth * loss_smooth
        return loss, loss_anchor, loss_monotone, loss_smooth

    def record_audit(
        epoch: int,
        loss: torch.Tensor,
        loss_anchor: torch.Tensor,
        loss_monotone: torch.Tensor,
        loss_smooth: torch.Tensor,
    ) -> None:
        nonlocal best_epoch, best_metrics, best_state
        metrics = audit_model(model, alpha_anchor, ci_target)
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_anchor": float(loss_anchor.detach().cpu()),
            "loss_monotone": float(loss_monotone.detach().cpu()),
            "loss_smooth": float(loss_smooth.detach().cpu()),
            **metrics,
        }
        history.append(row)
        print(
            "Epoch {epoch:5d} | loss={loss:.3e} | anchor_mae={anchor_mae:.3e} | "
            "anchor_max={anchor_max_abs:.3e} | mono={loss_monotone:.3e} | smooth={loss_smooth:.3e}".format(
                **row
            ),
            flush=True,
        )

        is_better = (
            metrics["anchor_max_abs"] < best_metrics["anchor_max_abs"] - 1e-12
            or (
                abs(metrics["anchor_max_abs"] - best_metrics["anchor_max_abs"]) <= 1e-12
                and metrics["anchor_mae"] < best_metrics["anchor_mae"]
            )
        )
        if is_better:
            best_epoch = epoch
            best_metrics = metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {**asdict(cfg), "output_dir": str(cfg.output_dir), "reference_csv": str(cfg.reference_csv or "")},
                    "anchor_targets": targets.to_dict(orient="records"),
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                },
                output_dir / "model_best.pt",
            )

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, loss_anchor, loss_monotone, loss_smooth = compute_losses()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss non finie a epoch={epoch}.")
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % cfg.audit_every == 0 or epoch == cfg.epochs:
            record_audit(epoch, loss, loss_anchor, loss_monotone, loss_smooth)

    if cfg.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(trainable, lr=cfg.lbfgs_lr, max_iter=1, line_search_fn="strong_wolfe")
        for step in range(1, cfg.lbfgs_steps + 1):
            epoch = cfg.epochs + step

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                closure_loss, _, _, _ = compute_losses()
                if not torch.isfinite(closure_loss):
                    raise FloatingPointError(f"Loss LBFGS non finie a epoch={epoch}.")
                closure_loss.backward()
                return closure_loss

            lbfgs.step(closure)
            if step == 1 or step % cfg.audit_every == 0 or step == cfg.lbfgs_steps:
                loss, loss_anchor, loss_monotone, loss_smooth = compute_losses()
                record_audit(epoch, loss, loss_anchor, loss_monotone, loss_smooth)

    final_metrics = audit_model(model, alpha_anchor, ci_target)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {**asdict(cfg), "output_dir": str(cfg.output_dir), "reference_csv": str(cfg.reference_csv or "")},
            "anchor_targets": targets.to_dict(orient="records"),
            "final_metrics": final_metrics,
        },
        output_dir / "model_final.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)
    save_curve(model, cfg, targets, output_dir, device)
    write_report(cfg, best_epoch, best_metrics, final_metrics, output_dir)

    print((output_dir / "stage0_report.txt").read_text(encoding="utf-8"), flush=True)
    if cfg.fail_on_target and best_metrics["anchor_max_abs"] > cfg.target_max_abs:
        return 2
    return 0


def main() -> None:
    cfg = parse_config()
    raise SystemExit(train_stage0(cfg))


if __name__ == "__main__":
    main()
