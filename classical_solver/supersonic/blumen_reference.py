from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SUPERSONIC_MACH_OFFSET = 0.9


def supersonic_digitized_x_to_mach(x: float | np.ndarray) -> float | np.ndarray:
    """Convertit l'abscisse numerisee de Blumen en Mach physique.

    Les CSV supersoniques ont ete digitalises depuis une figure dont l'axe des
    Mach visible commence vers M=0.9. La premiere colonne encode donc l'abscisse
    locale de la planche, pas directement le Mach physique.
    """
    return np.asarray(x) + SUPERSONIC_MACH_OFFSET


def parse_reference_level(csv_path: str | Path) -> tuple[float | None, str, str]:
    stem = Path(csv_path).stem.strip().replace("_", ".").replace(",", ".")
    lower = stem.lower()
    if lower.endswith(".datasets"):
        return None, stem, "dataset"
    if lower.startswith("ci"):
        value = float(lower[2:] or "0") / 100.0
        return value, fr"$c_r = 0,\; c_i = {value:.2f}$", "ci_special"
    if lower.startswith("cr"):
        value = float(lower[2:] or "0")
        return value, fr"$c_i = 0,\; c_r = {value:.2f}$", "cr_special"
    numeric = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    if not numeric:
        return None, stem, "unknown"
    value = float(numeric)
    return value, fr"$c_i = {value:.2f}$", "ci_level"


def load_supersonic_blumen_csv(csv_path: str | Path, *, calibrate_mach: bool = True) -> pd.DataFrame:
    df = (
        pd.read_csv(
            csv_path,
            header=None,
            names=["Mach", "alpha"],
            sep=";",
            decimal=",",
            engine="python",
        )
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )
    if calibrate_mach:
        df["Mach_digitized"] = df["Mach"].astype(float)
        df["Mach"] = supersonic_digitized_x_to_mach(df["Mach"].to_numpy(dtype=float))
    return df


def is_wide_digitized_dataset(csv_path: str | Path) -> bool:
    lower = Path(csv_path).stem.strip().replace("_", ".").replace(",", ".").lower()
    return lower.endswith(".datasets")


def parse_wide_dataset_curve(label: str) -> tuple[float | None, str, str]:
    normalized = str(label).strip().replace(",", ".")
    lower = normalized.lower()
    if not normalized or lower == "nan":
        return None, normalized, "unknown"
    if lower == "ci=0":
        return 0.0, r"$c_i = 0$", "ci_special"
    if lower == "ci_sup=0":
        return 0.0, r"$c_i^{sup} = 0$", "ci_special"
    if lower == "cr=0":
        return 0.0, r"$c_r = 0$", "cr_special"
    try:
        value = float(normalized)
    except ValueError:
        return None, normalized, "unknown"
    return value, fr"$c_i = {value:.2f}$", "ci_level"


def load_wide_digitized_curves(csv_path: str | Path) -> list[dict]:
    raw = pd.read_csv(csv_path, header=None)
    levels = raw.iloc[0].tolist()
    coords = raw.iloc[1].tolist()
    data = raw.iloc[2:].reset_index(drop=True)
    curves: list[dict] = []

    for index in range(0, len(levels) - 1, 2):
        raw_label = str(levels[index]).strip()
        x_coord = str(coords[index]).strip().upper()
        y_coord = str(coords[index + 1]).strip().upper()
        if not raw_label or raw_label.lower() == "nan" or x_coord != "X" or y_coord != "Y":
            continue

        x = pd.to_numeric(data.iloc[:, index], errors="coerce")
        y = pd.to_numeric(data.iloc[:, index + 1], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            continue

        level, label, family = parse_wide_dataset_curve(raw_label)
        df = pd.DataFrame(
            {
                "Mach": x[mask].to_numpy(dtype=float),
                "alpha": y[mask].to_numpy(dtype=float),
            }
        ).reset_index(drop=True)
        curves.append(
            {
                "csv_path": str(csv_path),
                "stem": raw_label,
                "level": None if level is None else float(level),
                "label": label,
                "family": family,
                "data": df,
            }
        )

    return curves


def load_digitized_curves(data_dir: str | Path, *, calibrate_mach: bool = True) -> list[dict]:
    curves: list[dict] = []
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    single_curve_files = [csv_file for csv_file in csv_files if not is_wide_digitized_dataset(csv_file)]
    dataset_files = [csv_file for csv_file in csv_files if is_wide_digitized_dataset(csv_file)]

    if single_curve_files:
        for csv_file in single_curve_files:
            level, label, family = parse_reference_level(csv_file)
            curves.append(
                {
                    "csv_path": str(csv_file),
                    "stem": csv_file.stem,
                    "level": None if level is None else float(level),
                    "label": label,
                    "family": family,
                    "data": load_supersonic_blumen_csv(csv_file, calibrate_mach=calibrate_mach),
                }
            )
        return curves

    for csv_file in dataset_files:
        curves.extend(load_wide_digitized_curves(csv_file))

    return curves


def estimate_blumen_ci(alpha: float, mach: float, curves: list[dict]) -> float:
    """Estime c_i depuis les isolignes principales de Blumen.

    Pour chaque niveau c_i, on interpole alpha(M) a Mach fixe, puis on interpole
    le niveau c_i tel que alpha(M, c_i)=alpha. Cela evite de trier une courbe
    fermee/non-monotone par alpha, ce qui cassait l'estimation precedente.
    """
    anchors: list[tuple[float, float]] = []
    for curve in curves:
        if curve.get("family") != "ci_level" or curve.get("level") is None:
            continue
        df = curve["data"][["Mach", "alpha"]].dropna().sort_values("Mach").reset_index(drop=True)
        if len(df) < 2:
            continue
        mach_values = df["Mach"].to_numpy(dtype=float)
        alpha_values = df["alpha"].to_numpy(dtype=float)
        if mach < float(np.min(mach_values)) or mach > float(np.max(mach_values)):
            continue
        alpha_on_curve = float(np.interp(mach, mach_values, alpha_values))
        anchors.append((alpha_on_curve, float(curve["level"])))

    if len(anchors) < 2:
        return float("nan")

    alpha_grid = np.array([item[0] for item in anchors], dtype=float)
    ci_grid = np.array([item[1] for item in anchors], dtype=float)
    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    ci_grid = ci_grid[order]
    if alpha < float(np.min(alpha_grid)) or alpha > float(np.max(alpha_grid)):
        return float("nan")
    return float(np.interp(alpha, alpha_grid, ci_grid))
