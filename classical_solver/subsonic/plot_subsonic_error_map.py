from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path("/Users/emma.grospellier/Thèse/These_PINN_KH_RT")
ERROR_CSV = ROOT_DIR / "assets" / "blumen_shooting" / "subsonic_shooting_error_by_point.csv"
OUTPUT_PNG = ROOT_DIR / "assets" / "blumen_shooting" / "subsonic_shooting_error_map.png"


def make_panel(ax, df: pd.DataFrame, value_column: str, title: str, colorbar_label: str) -> None:
    values = df[value_column].to_numpy(dtype=float)
    scatter = ax.scatter(
        df["Mach"],
        df["alpha"],
        c=values,
        s=24,
        cmap="inferno",
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label=colorbar_label)

    worst_threshold = np.nanpercentile(values, 90)
    worst = df[values >= worst_threshold]
    ax.scatter(
        worst["Mach"],
        worst["alpha"],
        s=42,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.9,
        label="Top 10% erreurs",
    )

    mach_line = np.linspace(0.0, 1.0, 500)
    alpha_line = np.sqrt(np.clip(1.0 - mach_line**2, 0.0, None))
    ax.plot(mach_line, alpha_line, "--", color="white", linewidth=1.2, alpha=0.9, label=r"$\alpha^2 + M^2 = 1$")

    ax.set_title(title)
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel(r"Nombre d'onde ($\alpha$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    df = pd.read_csv(ERROR_CSV)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Mach", "alpha"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    make_panel(
        axes[0],
        df.dropna(subset=["abs_omega_residual"]),
        value_column="abs_omega_residual",
        title=r"Erreur sur $\omega_i$ aux points de Blumen",
        colorbar_label=r"$|\omega_{solver} - \omega_{Blumen}|$",
    )
    make_panel(
        axes[1],
        df.dropna(subset=["distance_to_solver_isoline"]),
        value_column="distance_to_solver_isoline",
        title="Distance point -> isoligne du solveur",
        colorbar_label="Distance geometrique dans le plan (M, alpha)",
    )

    fig.suptitle("Carte des erreurs subsoniques de la méthode du tir", fontsize=14)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Carte d'erreur enregistree dans {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
