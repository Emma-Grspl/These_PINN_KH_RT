from __future__ import annotations

from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "KH_RT_Blumen"
OUTPUT_DIR = ROOT_DIR / "assets" / "blumen"


def parse_growth_label(csv_path: str, regime: str) -> tuple[float, str]:
    stem = Path(csv_path).stem.strip()
    normalized = stem.replace("_", ".").replace(",", ".")

    if normalized.lower().startswith("ci"):
        value = float(normalized[2:]) / 100.0
        if regime == "supersonic":
            return 100.0 + value, fr"$c_r = 0,\; c_i = {value:.2f}$"
        return value, f"{value:.2f}"

    if normalized.lower().startswith("cr"):
        suffix = normalized[2:] or "0"
        value = float(suffix)
        if regime == "supersonic":
            return 200.0 + value, fr"$c_r = {value:.2f}$"
        return value, f"{value:.2f}"

    numeric_part = "".join(ch for ch in normalized if ch.isdigit() or ch == ".")
    if not numeric_part:
        raise ValueError(f"Nom de fichier non interpretable: {csv_path}")

    value = float(numeric_part)
    formatted_value = f"{value:.3f}".rstrip("0").rstrip(".")
    if regime == "supersonic":
        return value, fr"$c_i = 0,\; c_r = {formatted_value}$"
    return value, formatted_value


def load_curve(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["Mach", "alpha"],
        sep=";",
        decimal=",",
        engine="python",
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def preprocess_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde l'ordre naturel du fichier CSV pour respecter le tracé d'origine.

    Les isocontours numérisés ne sont pas nécessairement des fonctions
    alpha = f(M). On évite donc tout tri par Mach, qui casserait la géométrie.
    """
    cleaned_df = df[["Mach", "alpha"]].dropna().copy().reset_index(drop=True)
    if len(cleaned_df) < 2:
        raise ValueError("Au moins deux points sont necessaires pour tracer une courbe.")
    return cleaned_df


def plot_regime(
    input_dir: Path,
    output_path: Path,
    title: str,
    regime: str,
    theoretical_limit: bool = False,
) -> None:
    csv_files = sorted(glob.glob(str(input_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouve dans {input_dir}")

    curves: list[tuple[float, str, pd.DataFrame]] = []
    for csv_file in csv_files:
        growth_value, growth_label = parse_growth_label(csv_file, regime=regime)
        curves.append((growth_value, growth_label, load_curve(csv_file)))

    curves.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, growth_label, df in curves:
        df_ordered = preprocess_curve(df)
        ax.scatter(
            df_ordered["Mach"],
            df_ordered["alpha"],
            s=18,
            alpha=0.9,
            label=growth_label if regime == "supersonic" else fr"$\omega_i={growth_label}$",
        )

    if theoretical_limit:
        mach = np.linspace(0.0, 1.0, 500)
        alpha = np.sqrt(np.clip(1.0 - mach**2, 0.0, None))
        ax.plot(
            mach,
            alpha,
            linestyle="--",
            color="blue",
            linewidth=2,
            label=r"Limite théorique $\alpha=\sqrt{1-M^2}$",
        )
        ax.set_xlim(0.0, 1.0)

    ax.set_title(title)
    ax.set_xlabel("Nombre de Mach (M)")
    ax.set_ylabel("Nombre d'onde (alpha)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_regime(
        input_dir=DATA_DIR / "subsonic",
        output_path=OUTPUT_DIR / "blumen_subsonic.png",
        title="Isocontours de Blumen (Régime Subsonique)",
        regime="subsonic",
        theoretical_limit=True,
    )
    plot_regime(
        input_dir=DATA_DIR / "supersonic",
        output_path=OUTPUT_DIR / "blumen_supersonic.png",
        title="Isocontours de Blumen (Régime Supersonique)",
        regime="supersonic",
        theoretical_limit=False,
    )

    print(f"Graphiques enregistres dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
