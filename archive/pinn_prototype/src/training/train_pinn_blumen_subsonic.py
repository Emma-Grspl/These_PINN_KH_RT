from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from src.data.collocation_blumen import reference_point, sample_boundary_points, sample_interior_points
from src.models.pinn_blumen_subsonic import BlumenSubsonicPINN
from src.physics.residual_blumen import boundary_decay_loss, normalization_loss, pressure_ode_residual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training d'un PINN subsonique minimal pour l'equation de Blumen.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--initial-ci", type=float, default=0.3)
    parser.add_argument("--initial-L", type=float, default=10.0)
    parser.add_argument("--n-interior", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=64)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-bc", type=float, default=10.0)
    parser.add_argument("--w-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("model_saved/blumen_subsonic_pinn"))
    return parser


def train_single_case(args: argparse.Namespace) -> tuple[BlumenSubsonicPINN, list[dict]]:
    device = torch.device(args.device)
    model = BlumenSubsonicPINN(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        fourier_features=args.fourier_features,
        initial_ci=args.initial_ci,
        initial_L=args.initial_L,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        xi_interior = sample_interior_points(args.n_interior, device=device.type)
        xi_left, xi_right = sample_boundary_points(args.n_boundary, device=device.type)
        xi_ref = reference_point(device=device.type)

        res_r, res_i, _ = pressure_ode_residual(model, xi_interior, alpha=args.alpha, mach=args.mach)
        loss_pde = torch.mean(res_r.pow(2) + res_i.pow(2))
        loss_bc = boundary_decay_loss(model, xi_left, xi_right)
        loss_norm = normalization_loss(model, xi_ref)

        loss = args.w_pde * loss_pde + args.w_bc * loss_bc + args.w_norm * loss_norm
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_norm": float(loss_norm.item()),
            "ci": float(model.get_ci().item()),
            "L": float(model.get_L().item()),
        }
        history.append(record)

        if epoch == 1 or epoch % 250 == 0:
            print(
                f"Epoch {epoch:5d} | loss={record['loss']:.3e} | "
                f"pde={record['loss_pde']:.3e} | bc={record['loss_bc']:.3e} | "
                f"norm={record['loss_norm']:.3e} | ci={record['ci']:.5f} | L={record['L']:.3f}"
            )

    return model, history


def save_artifacts(model: BlumenSubsonicPINN, history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "history.csv").open("w", encoding="utf-8") as f:
        f.write("epoch,loss,loss_pde,loss_bc,loss_norm,ci,L\n")
        for item in history:
            f.write(
                f"{item['epoch']},{item['loss']},{item['loss_pde']},"
                f"{item['loss_bc']},{item['loss_norm']},{item['ci']},{item['L']}\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    model, history = train_single_case(args)
    save_artifacts(model, history, args.output_dir)
    print(f"Modele enregistre dans {args.output_dir}")


if __name__ == "__main__":
    main()
