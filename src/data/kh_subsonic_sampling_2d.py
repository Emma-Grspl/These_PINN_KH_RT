from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.subsonic.mstab17_subsonic_solver import Mstab17SubsonicSolver
from src.data.kh_subsonic_sampling import sample_boundary_points, sample_interior_points, reference_point


DEFAULT_SUBSONIC_REFERENCE_CANDIDATES = (
    Path("assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv"),
)

DEFAULT_ANCHOR_ALPHAS = (0.10, 0.30, 0.55, 0.80)


@dataclass(frozen=True)
class ClassicalReferencePoint2D:
    alpha: float
    mach: float
    ci_reference: float
    omega_i: float | None
    source: str
    success: bool
    stage1_mismatch: float | None = None
    stage2_mismatch: float | None = None


def normalize_float_list(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def dumps_float_list(values: list[float] | tuple[float, ...]) -> str:
    return json.dumps([float(value) for value in values])


def resolve_reference_cache_path(reference_cache: str | Path | None = None) -> Path | None:
    if reference_cache is not None:
        path = Path(reference_cache)
        if not path.exists():
            raise FileNotFoundError(f"Explicit subsonic reference cache not found: {path}")
        return path
    for candidate in DEFAULT_SUBSONIC_REFERENCE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_reference_table(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    required = {"alpha", "Mach", "ci"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Reference cache {csv_path} is missing columns: {sorted(missing)}")
    work = df.copy()
    work["alpha"] = work["alpha"].astype(float)
    work["Mach"] = work["Mach"].astype(float)
    work["ci"] = work["ci"].astype(float)
    return work.sort_values(["Mach", "alpha"]).reset_index(drop=True)


def _interp_ci_along_alpha(df: pd.DataFrame, *, alpha: float) -> float | None:
    if df.empty:
        return None
    alpha_values = df["alpha"].to_numpy(dtype=float)
    ci_values = df["ci"].to_numpy(dtype=float)
    if alpha < float(alpha_values.min()) - 1e-12 or alpha > float(alpha_values.max()) + 1e-12:
        return None
    return float(np.interp(float(alpha), alpha_values, ci_values))


def interpolate_reference_ci(reference_df: pd.DataFrame, *, alpha: float, mach: float) -> float | None:
    same_point = reference_df[
        (
            np.isclose(reference_df["alpha"].to_numpy(dtype=float), float(alpha), atol=1e-12)
            & np.isclose(reference_df["Mach"].to_numpy(dtype=float), float(mach), atol=1e-12)
        )
    ]
    if not same_point.empty:
        return float(same_point.iloc[0]["ci"])

    mach_values = np.sort(reference_df["Mach"].drop_duplicates().to_numpy(dtype=float))
    if mach_values.size == 0:
        return None

    if mach <= float(mach_values.min()) + 1e-12:
        local = reference_df[np.isclose(reference_df["Mach"], mach_values.min(), atol=1e-12)]
        return _interp_ci_along_alpha(local, alpha=alpha)
    if mach >= float(mach_values.max()) - 1e-12:
        local = reference_df[np.isclose(reference_df["Mach"], mach_values.max(), atol=1e-12)]
        return _interp_ci_along_alpha(local, alpha=alpha)

    upper_idx = int(np.searchsorted(mach_values, mach, side="left"))
    lower_idx = max(upper_idx - 1, 0)
    upper_idx = min(upper_idx, len(mach_values) - 1)
    m0 = float(mach_values[lower_idx])
    m1 = float(mach_values[upper_idx])
    local0 = reference_df[np.isclose(reference_df["Mach"], m0, atol=1e-12)]
    local1 = reference_df[np.isclose(reference_df["Mach"], m1, atol=1e-12)]
    ci0 = _interp_ci_along_alpha(local0, alpha=alpha)
    ci1 = _interp_ci_along_alpha(local1, alpha=alpha)
    if ci0 is None or ci1 is None:
        return None
    if abs(m1 - m0) <= 1e-12:
        return float(ci0)
    weight = (float(mach) - m0) / (m1 - m0)
    return float((1.0 - weight) * ci0 + weight * ci1)


def solve_classical_reference_point(alpha: float, mach: float) -> ClassicalReferencePoint2D:
    solver = Mstab17SubsonicSolver(alpha=float(alpha), Mach=float(mach))
    result = solver.solve()
    return ClassicalReferencePoint2D(
        alpha=float(result.alpha),
        mach=float(result.Mach),
        ci_reference=float(result.ci),
        omega_i=float(result.omega_i),
        source="mstab17_subsonic_solver",
        success=bool(result.success),
        stage1_mismatch=float(result.stage1_mismatch),
        stage2_mismatch=float(result.stage2_mismatch),
    )


def lookup_classical_reference_point(
    *,
    alpha: float,
    mach: float,
    reference_df: pd.DataFrame | None,
) -> ClassicalReferencePoint2D:
    if reference_df is not None:
        ci_value = interpolate_reference_ci(reference_df, alpha=float(alpha), mach=float(mach))
        if ci_value is not None:
            return ClassicalReferencePoint2D(
                alpha=float(alpha),
                mach=float(mach),
                ci_reference=float(ci_value),
                omega_i=float(alpha) * float(ci_value),
                source="assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv",
                success=True,
            )
    return solve_classical_reference_point(float(alpha), float(mach))


def build_anchor_table(
    *,
    mach_values: list[float] | tuple[float, ...],
    anchor_alphas: list[float] | tuple[float, ...],
    alpha_min: float,
    alpha_max: float,
    reference_cache: str | Path | None = None,
) -> pd.DataFrame:
    mach_values = normalize_float_list(mach_values)
    anchor_alphas = normalize_float_list(anchor_alphas)
    for alpha in anchor_alphas:
        if alpha < float(alpha_min) - 1e-12 or alpha > float(alpha_max) + 1e-12:
            raise ValueError(
                f"Anchor alpha={alpha:.6f} is outside the training interval [{alpha_min}, {alpha_max}]."
            )

    reference_df = None
    reference_path = resolve_reference_cache_path(reference_cache)
    if reference_path is not None:
        reference_df = load_reference_table(reference_path)

    rows: list[dict[str, object]] = []
    for mach in mach_values:
        for alpha in anchor_alphas:
            point = lookup_classical_reference_point(alpha=float(alpha), mach=float(mach), reference_df=reference_df)
            rows.append(
                {
                    "alpha": float(point.alpha),
                    "Mach": float(point.mach),
                    "ci_reference": float(point.ci_reference),
                    "omega_i": np.nan if point.omega_i is None else float(point.omega_i),
                    "source": point.source,
                    "success": bool(point.success),
                    "stage1_mismatch": np.nan if point.stage1_mismatch is None else float(point.stage1_mismatch),
                    "stage2_mismatch": np.nan if point.stage2_mismatch is None else float(point.stage2_mismatch),
                }
            )
    df = pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    return df


def build_reference_surface(
    *,
    alpha_values: np.ndarray,
    mach_values: np.ndarray,
    reference_cache: str | Path | None = None,
) -> pd.DataFrame:
    reference_df = None
    reference_path = resolve_reference_cache_path(reference_cache)
    if reference_path is not None:
        reference_df = load_reference_table(reference_path)

    rows: list[dict[str, float | str | bool]] = []
    for mach in np.asarray(mach_values, dtype=float):
        for alpha in np.asarray(alpha_values, dtype=float):
            point = lookup_classical_reference_point(alpha=float(alpha), mach=float(mach), reference_df=reference_df)
            rows.append(
                {
                    "alpha": float(point.alpha),
                    "Mach": float(point.mach),
                    "ci_reference": float(point.ci_reference),
                    "source": point.source,
                    "success": bool(point.success),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_ANCHOR_ALPHAS",
    "DEFAULT_SUBSONIC_REFERENCE_CANDIDATES",
    "build_anchor_table",
    "build_reference_surface",
    "dumps_float_list",
    "normalize_float_list",
    "reference_point",
    "sample_boundary_points",
    "sample_interior_points",
]
