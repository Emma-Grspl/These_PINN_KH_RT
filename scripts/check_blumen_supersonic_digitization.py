from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.supersonic.blumen_reference import (  # noqa: E402
    estimate_blumen_ci,
    load_digitized_curves,
    load_supersonic_blumen_csv,
)


BLUMEN_DIR = ROOT_DIR / "KH_RT_Blumen" / "supersonic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controle de calibration des courbes supersoniques de Blumen.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--mach-values", type=float, nargs="+", default=[1.20, 1.25, 1.275, 1.30])
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "assets" / "blumen" / "supersonic_digitization_check.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    curves = load_digitized_curves(BLUMEN_DIR)

    print("Blumen targets from calibrated supersonic digitization")
    print(f"alpha={args.alpha:.6f}")
    for mach in args.mach_values:
        ci = estimate_blumen_ci(args.alpha, mach, curves)
        print(f"Mach={mach:.6f} ci_target={ci:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, calibrate, title in [
        (axes[0], False, "CSV brut: abscisse numerisee"),
        (axes[1], True, "Calibration: Mach = x + 0.9"),
    ]:
        for csv_file in sorted(BLUMEN_DIR.glob("*.csv")):
            df = load_supersonic_blumen_csv(csv_file, calibrate_mach=calibrate)
            level = csv_file.stem
            if level in {"ci01", "ci02", "cr0"}:
                ax.scatter(df["Mach"], df["alpha"], color="0.35", s=10, marker="x", alpha=0.75)
            else:
                ax.scatter(df["Mach"], df["alpha"], s=16, alpha=0.85, label=level.replace(" (1)", ""))
        ax.axhline(args.alpha, color="tab:red", linewidth=1.0, linestyle=":")
        for mach in args.mach_values:
            ax.axvline(mach, color="tab:blue", linewidth=0.8, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("Mach" if calibrate else "x CSV")
        ax.grid(True, linestyle=":", alpha=0.3)
    axes[0].set_ylabel(r"$\alpha$")
    axes[1].set_xlim(0.9, 2.1)
    axes[1].set_ylim(0.0, 0.5)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
