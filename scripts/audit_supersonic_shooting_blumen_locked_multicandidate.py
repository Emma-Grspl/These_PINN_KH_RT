from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
    build_blumen_targets,
    load_candidates_from_csv,
    load_digitized_long,
)
from scripts.audit_supersonic_shooting_blumen_locked import (  # noqa: E402
    DEFAULT_ANCHOR_CSV,
    anchor_seeds,
    load_anchor_points,
)
from scripts.audit_supersonic_shooting_point_batch import (  # noqa: E402
    apply_box_rejection,
    default_box_robustness_metrics,
    reconstruct_candidate_fields_and_diagnostics,
    success_label,
)
from scripts.audit_supersonic_shooting_point_batch_branch_guided import (  # noqa: E402
    DEFAULT_MODAL_REFERENCE_CSV,
    DEFAULT_REFERENCE_CSV,
    build_guided_target,
    guided_seed_list,
    pick_bracketing_guides,
)
from scripts.track_supersonic_shooting_multistart import multistart_single_box  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "assets" / "classic_supersonic" / "multicandidate_audits"
DEFAULT_EXISTING_CANDIDATE_ROOT = ROOT_DIR / "assets" / "classic_supersonic" / "shooting"
REJECT_REASON_ORDER = [
    "exception",
    "solver_failure",
    "missing_reference",
    "stage1",
    "stage2",
    "ci_abs",
    "ci_rel",
    "cr_abs",
    "box",
    "peak_shift",
    "no_candidate",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit multi-candidats supersonique verrouille autour de Blumen et/ou de la branche guidee. "
            "Le script sauvegarde apres chaque candidat pour survivre aux time limits Slurm."
        )
    )
    parser.add_argument("--points", type=str, nargs="*", default=None, help="Liste de couples alpha:Mach.")
    parser.add_argument("--mach", type=float, default=None, help="Mach fixe si --alphas est fourni.")
    parser.add_argument("--alphas", type=str, default=None, help="Liste CSV d'alpha si Mach est fixe.")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha fixe si --mach-values est fourni.")
    parser.add_argument("--mach-values", type=str, default=None, help="Liste CSV de Mach si alpha est fixe.")
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHOR_CSV)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--modal-reference-csv", type=Path, default=DEFAULT_MODAL_REFERENCE_CSV)
    parser.add_argument("--existing-candidate-root", type=Path, default=DEFAULT_EXISTING_CANDIDATE_ROOT)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-stem", type=str, required=True)
    parser.add_argument(
        "--candidate-source",
        type=str,
        choices=["auto", "existing", "blumen_perturb", "branch_guided", "all", "manual_grid"],
        default="auto",
    )
    parser.add_argument("--max-candidates-per-point", type=int, default=12)
    parser.add_argument("--manual-seed-grid", default="", help="Manual seed list as cr:ci pairs separated by spaces or commas, e.g. 0.36:0.024 0.40:0.028.")
    parser.add_argument("--stop-after-accepted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush-every-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-validated", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run-candidates", action="store_true")

    parser.add_argument("--box-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--box-min", type=float, default=10.0)
    parser.add_argument("--box-max", type=float, default=1200.0)
    parser.add_argument("--box-factor", type=float, default=10.0, help="Equivalent du y_limit_factor.")
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--retry-growth", type=float, default=1.5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--alpha-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--existing-mach-window", type=float, default=0.25)

    parser.add_argument("--strict-stage1", type=float, default=5.0e-2)
    parser.add_argument("--strict-stage2", type=float, default=1.0e-4)
    parser.add_argument("--strict-ci-abs", type=float, default=8.0e-3)
    parser.add_argument("--strict-ci-rel", type=float, default=0.10)
    parser.add_argument("--strict-cr-abs", type=float, default=3.5e-2)
    parser.add_argument("--strict-box-max-rel-l2", type=float, default=0.15)
    parser.add_argument("--strict-peak-shift", type=float, default=0.75)
    parser.add_argument("--strict-box-max-center8-delta", type=float, default=0.10)
    parser.add_argument("--strict-box-max-edge-growth", type=float, default=1.25)
    return parser


def parse_csv_floats(raw_value: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        key = round(value, 10)
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("Liste vide.")
    return values


def parse_points(raw_points: list[str]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for item in raw_points:
        alpha_raw, mach_raw = item.split(":")
        key = (round(float(alpha_raw), 10), round(float(mach_raw), 10))
        if key in seen:
            continue
        seen.add(key)
        points.append((float(alpha_raw), float(mach_raw)))
    return points


def resolve_points(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.points:
        return parse_points(list(args.points))
    if args.mach is not None and args.alphas is not None:
        return [(float(alpha), float(args.mach)) for alpha in parse_csv_floats(args.alphas)]
    if args.alpha is not None and args.mach_values is not None:
        return [(float(args.alpha), float(mach)) for mach in parse_csv_floats(args.mach_values)]
    raise ValueError("Fournir --points, ou --mach avec --alphas, ou --alpha avec --mach-values.")


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def bool_from_env_like(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def point_key(alpha: float, mach: float) -> str:
    return f"a{alpha:.6f}_M{mach:.6f}"


def candidate_key(cr: float, ci: float) -> tuple[float, float]:
    return (round(float(cr), 8), round(float(ci), 8))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def discover_existing_candidate_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    include_tokens = ("candidate", "summary", "reference", "surface", "manual", "branch", "point_batch")
    exclude_tokens = ("fields", "levels", "overlay", "diagnostic", "status_map", "modes", "points", "plot")
    paths: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        lower = path.name.lower()
        if any(token in lower for token in exclude_tokens):
            continue
        if not any(token in lower for token in include_tokens):
            continue
        paths.append(path)
    return paths


def seed_distance_score(
    *,
    cr: float,
    ci: float,
    target_cr: float,
    target_ci: float,
    ci_only_target: float | None = None,
) -> float:
    if np.isfinite(target_cr) and np.isfinite(target_ci):
        return float(np.hypot((float(cr) - target_cr) / 0.05, (float(ci) - target_ci) / 0.02))
    if ci_only_target is not None and np.isfinite(float(ci_only_target)):
        return float(abs(float(ci) - float(ci_only_target)) / 0.02)
    return float(-float(ci))


def build_existing_seed_records(
    *,
    alpha: float,
    mach: float,
    anchors: pd.DataFrame,
    candidate_paths: list[Path],
    blumen_cr: float,
    blumen_ci: float,
    mach_window: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for seed_name, cr, ci in anchor_seeds(anchors, alpha=float(alpha), mach=float(mach)):
        records.append(
            {
                "candidate_bucket": "existing",
                "candidate_source": f"anchor_{seed_name}",
                "candidate_label": seed_name,
                "source_file": str(DEFAULT_ANCHOR_CSV.relative_to(ROOT_DIR)),
                "source_mach": float(mach),
                "source_alpha": float(alpha),
                "cr_init": float(cr),
                "ci_init": float(ci),
                "priority_score": seed_distance_score(
                    cr=float(cr),
                    ci=float(ci),
                    target_cr=float(blumen_cr),
                    target_ci=float(blumen_ci),
                    ci_only_target=float(blumen_ci) if np.isfinite(blumen_ci) else None,
                ),
            }
        )

    for path in candidate_paths:
        try:
            df = load_candidates_from_csv(path, alpha_filter=float(alpha))
        except Exception:
            continue
        if df.empty or "Mach" not in df.columns:
            continue
        df = df.copy()
        df["mach_distance"] = np.abs(df["Mach"].astype(float) - float(mach))
        df = df[df["mach_distance"] <= float(mach_window)].sort_values(
            ["mach_distance", "candidate_ci"], ascending=[True, False]
        )
        if df.empty:
            continue
        for _, row in df.iterrows():
            cr = safe_float(row.get("candidate_cr"))
            ci = safe_float(row.get("candidate_ci"))
            if not np.isfinite(cr) or not np.isfinite(ci) or ci <= 0.0:
                continue
            label = str(row.get("label", row.get("source_kind", "existing_catalog")))
            source_kind = str(row.get("source_kind", "existing_catalog"))
            records.append(
                {
                    "candidate_bucket": "existing",
                    "candidate_source": f"existing_{source_kind}",
                    "candidate_label": label,
                    "source_file": str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path),
                    "source_mach": safe_float(row.get("Mach")),
                    "source_alpha": safe_float(row.get("alpha")),
                    "cr_init": float(cr),
                    "ci_init": float(ci),
                    "priority_score": seed_distance_score(
                        cr=float(cr),
                        ci=float(ci),
                        target_cr=float(blumen_cr),
                        target_ci=float(blumen_ci),
                        ci_only_target=float(blumen_ci) if np.isfinite(blumen_ci) else None,
                    )
                    + 0.5 * float(row["mach_distance"]),
                }
            )
    return records


def build_branch_guided_seed_records(
    *,
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    spectral_df = pd.read_csv(args.reference_csv)
    modal_df = pd.read_csv(args.modal_reference_csv)
    lower_bundle, upper_bundle = pick_bracketing_guides(
        spectral_df=spectral_df,
        modal_df=modal_df,
        target_mach=float(mach),
        target_alpha=float(alpha),
        alpha_tolerance=float(args.alpha_tolerance),
    )
    guide = build_guided_target(
        target_mach=float(mach),
        target_alpha=float(alpha),
        lower_bundle=lower_bundle,
        upper_bundle=upper_bundle,
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
    )
    records: list[dict[str, object]] = []
    for seed_name, cr, ci in guided_seed_list(guide=guide, include_generic_seeds=False):
        records.append(
            {
                "candidate_bucket": "branch_guided",
                "candidate_source": f"branch_guided_{seed_name}",
                "candidate_label": seed_name,
                "source_file": str(args.reference_csv),
                "source_mach": float(mach),
                "source_alpha": float(alpha),
                "cr_init": float(cr),
                "ci_init": float(ci),
                "priority_score": seed_distance_score(
                    cr=float(cr),
                    ci=float(ci),
                    target_cr=float(guide["guide_target_cr"]),
                    target_ci=float(guide["guide_target_ci"]),
                    ci_only_target=float(guide["guide_target_ci"]),
                ),
            }
        )
    return records, guide


def build_blumen_perturb_seed_records(
    *,
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    guide: dict[str, object] | None,
) -> list[dict[str, object]]:
    target_cr = float(blumen_cr) if np.isfinite(blumen_cr) else safe_float(guide.get("guide_target_cr") if guide else np.nan)
    target_ci = float(blumen_ci) if np.isfinite(blumen_ci) else safe_float(guide.get("guide_target_ci") if guide else np.nan)
    if not np.isfinite(target_ci) or target_ci <= 0.0:
        return []
    if not np.isfinite(target_cr):
        target_cr = safe_float(guide.get("interp_cr") if guide else np.nan, default=0.0)

    cr_offsets = [0.0, -0.015, 0.015, -0.03, 0.03, -0.05, 0.05]
    ci_scales = [1.0, 0.92, 1.08, 0.85, 1.15]
    ci_offsets = [0.0, -0.004, 0.004, -0.008, 0.008]
    seeds: list[dict[str, object]] = []

    seeds.append(
        {
            "candidate_bucket": "blumen_perturb",
            "candidate_source": "blumen_target",
            "candidate_label": "blumen_target",
            "source_file": "blumen_digitized",
            "source_mach": float(mach),
            "source_alpha": float(alpha),
            "cr_init": float(target_cr),
            "ci_init": float(target_ci),
            "priority_score": 0.0,
        }
    )

    if guide is not None:
        seeds.append(
            {
                "candidate_bucket": "blumen_perturb",
                "candidate_source": "branch_interp_target",
                "candidate_label": "branch_interp_target",
                "source_file": "branch_guided_interp",
                "source_mach": float(mach),
                "source_alpha": float(alpha),
                "cr_init": float(guide["interp_cr"]),
                "ci_init": float(guide["interp_ci"]),
                "priority_score": seed_distance_score(
                    cr=float(guide["interp_cr"]),
                    ci=float(guide["interp_ci"]),
                    target_cr=float(target_cr),
                    target_ci=float(target_ci),
                    ci_only_target=float(target_ci),
                ),
            }
        )

    for cr_offset in cr_offsets:
        for ci_scale in ci_scales:
            ci_value = max(1.0e-6, float(target_ci) * float(ci_scale))
            seeds.append(
                {
                    "candidate_bucket": "blumen_perturb",
                    "candidate_source": "blumen_perturb_scale",
                    "candidate_label": f"dcr={cr_offset:+.3f},sci={ci_scale:.2f}",
                    "source_file": "blumen_digitized",
                    "source_mach": float(mach),
                    "source_alpha": float(alpha),
                    "cr_init": float(max(0.0, target_cr + cr_offset)),
                    "ci_init": float(ci_value),
                    "priority_score": seed_distance_score(
                        cr=float(max(0.0, target_cr + cr_offset)),
                        ci=float(ci_value),
                        target_cr=float(target_cr),
                        target_ci=float(target_ci),
                        ci_only_target=float(target_ci),
                    ),
                }
            )
        for ci_offset in ci_offsets:
            ci_value = max(1.0e-6, float(target_ci) + float(ci_offset))
            seeds.append(
                {
                    "candidate_bucket": "blumen_perturb",
                    "candidate_source": "blumen_perturb_shift",
                    "candidate_label": f"dcr={cr_offset:+.3f},dci={ci_offset:+.3f}",
                    "source_file": "blumen_digitized",
                    "source_mach": float(mach),
                    "source_alpha": float(alpha),
                    "cr_init": float(max(0.0, target_cr + cr_offset)),
                    "ci_init": float(ci_value),
                    "priority_score": seed_distance_score(
                        cr=float(max(0.0, target_cr + cr_offset)),
                        ci=float(ci_value),
                        target_cr=float(target_cr),
                        target_ci=float(target_ci),
                        ci_only_target=float(target_ci),
                    ),
                }
            )
    return seeds


def dedup_seed_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        records,
        key=lambda row: (
            {"existing": 0, "branch_guided": 1, "blumen_perturb": 2}.get(str(row["candidate_bucket"]), 99),
            float(row.get("priority_score", np.inf)),
            str(row.get("candidate_source", "")),
            str(row.get("candidate_label", "")),
        ),
    )
    seen: set[tuple[float, float]] = set()
    out: list[dict[str, object]] = []
    for row in ordered:
        cr = safe_float(row.get("cr_init"))
        ci = safe_float(row.get("ci_init"))
        if not np.isfinite(cr) or not np.isfinite(ci) or ci <= 0.0:
            continue
        key = candidate_key(cr, ci)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def interleave_seed_buckets(
    bucket_map: dict[str, list[dict[str, object]]],
    *,
    order: list[str],
    max_candidates: int,
) -> list[dict[str, object]]:
    sorted_buckets = {
        name: sorted(
            bucket_map.get(name, []),
            key=lambda row: (
                float(row.get("priority_score", np.inf)),
                str(row.get("candidate_source", "")),
                str(row.get("candidate_label", "")),
            ),
        )
        for name in order
    }
    indices = {name: 0 for name in order}
    seen: set[tuple[float, float]] = set()
    selected: list[dict[str, object]] = []

    while True:
        advanced = False
        for name in order:
            rows = sorted_buckets[name]
            idx = indices[name]
            while idx < len(rows):
                row = rows[idx]
                idx += 1
                key = candidate_key(safe_float(row.get("cr_init")), safe_float(row.get("ci_init")))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                indices[name] = idx
                advanced = True
                break
            else:
                indices[name] = idx
            if 0 < max_candidates <= len(selected):
                return selected
        if not advanced:
            return selected



def build_manual_grid_seed_records(
    *,
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Build explicit manual seeds from args.manual_seed_grid.

    Format: "cr:ci cr:ci ..." or "cr:ci,cr:ci,...".
    These are search seeds only; they are not validated references.
    """
    raw = str(getattr(args, "manual_seed_grid", "") or "").strip()
    if not raw:
        return []

    tokens = raw.replace(",", " ").split()
    records: list[dict[str, object]] = []

    for idx, token in enumerate(tokens, start=1):
        if ":" not in token:
            raise ValueError(
                f"Invalid manual seed token {token!r}; expected cr:ci, for example 0.38:0.024"
            )
        cr_s, ci_s = token.split(":", 1)
        cr = float(cr_s)
        ci = float(ci_s)
        if not np.isfinite(cr) or not np.isfinite(ci) or ci <= 0.0:
            continue

        records.append(
            {
                "candidate_bucket": "manual_grid",
                "candidate_source": "manual_grid",
                "candidate_label": f"manual_cr{cr:.6f}_ci{ci:.6f}",
                "source_file": "manual_seed_grid",
                "source_mach": float(mach),
                "source_alpha": float(alpha),
                "cr_init": float(cr),
                "ci_init": float(ci),
                "priority_score": seed_distance_score(
                    cr=float(cr),
                    ci=float(ci),
                    target_cr=float(blumen_cr),
                    target_ci=float(blumen_ci),
                    ci_only_target=float(blumen_ci) if np.isfinite(blumen_ci) else None,
                ) - 100.0 + 1.0e-6 * float(idx),
            }
        )

    return records


def collect_candidate_records(
    *,
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    anchors: pd.DataFrame,
    existing_candidate_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    notes: dict[str, object] = {
        "existing_candidate_paths_considered": int(len(existing_candidate_paths)),
        "existing_candidates_found": 0,
        "branch_guided_available": False,
        "blumen_ci_available": bool(np.isfinite(blumen_ci)),
        "blumen_cr_available": bool(np.isfinite(blumen_cr)),
    }

    manual_records = build_manual_grid_seed_records(
        alpha=float(alpha),
        mach=float(mach),
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
        args=args,
    )
    notes["manual_grid_candidates_found"] = int(len(manual_records))

    existing_records = build_existing_seed_records(
        alpha=float(alpha),
        mach=float(mach),
        anchors=anchors,
        candidate_paths=existing_candidate_paths,
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
        mach_window=float(args.existing_mach_window),
    )
    notes["existing_candidates_found"] = int(len(existing_records))

    branch_records: list[dict[str, object]] = []
    guide: dict[str, object] | None = None
    try:
        branch_records, guide = build_branch_guided_seed_records(
            alpha=float(alpha),
            mach=float(mach),
            blumen_cr=float(blumen_cr),
            blumen_ci=float(blumen_ci),
            args=args,
        )
        notes["branch_guided_available"] = True
        notes["branch_guided_candidates_found"] = int(len(branch_records))
    except Exception as exc:
        notes["branch_guided_error"] = repr(exc)
        notes["branch_guided_candidates_found"] = 0

    blumen_records = build_blumen_perturb_seed_records(
        alpha=float(alpha),
        mach=float(mach),
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
        guide=guide,
    )
    notes["blumen_perturb_candidates_found"] = int(len(blumen_records))

    source_to_records = {
        "manual_grid": manual_records,
        "existing": existing_records,
        "branch_guided": branch_records,
        "blumen_perturb": blumen_records,
    }
    if args.candidate_source == "existing":
        selected = existing_records
    elif args.candidate_source == "branch_guided":
        selected = branch_records
    elif args.candidate_source == "blumen_perturb":
        selected = blumen_records
    elif args.candidate_source == "all":
        selected = manual_records + existing_records + branch_records + blumen_records
    else:
        deduped = interleave_seed_buckets(
            source_to_records,
            order=["manual_grid", "existing", "branch_guided", "blumen_perturb"],
            max_candidates=int(args.max_candidates_per_point),
        )
        for idx, row in enumerate(deduped, start=1):
            row["candidate_order"] = int(idx)
        notes["candidate_source_mode"] = str(args.candidate_source)
        notes["candidates_after_dedup"] = int(len(deduped))
        return deduped, notes

    deduped = dedup_seed_records(selected)
    if int(args.max_candidates_per_point) > 0:
        deduped = deduped[: int(args.max_candidates_per_point)]
    for idx, row in enumerate(deduped, start=1):
        row["candidate_order"] = int(idx)
    notes["candidate_source_mode"] = str(args.candidate_source)
    notes["candidates_after_dedup"] = int(len(deduped))
    return deduped, notes


def evaluate_reject_reasons(
    *,
    candidate_row: dict[str, object],
    args: argparse.Namespace,
) -> tuple[bool, str, str]:
    reasons: list[str] = []
    exception_text = str(candidate_row.get("exception", "")).strip()
    if exception_text:
        reasons.append("exception")
    elif bool_from_env_like(candidate_row.get("solver_failed", False)):
        reasons.append("solver_failure")

    ci_available = bool(candidate_row.get("ci_ref_available", False))
    if not ci_available:
        reasons.append("missing_reference")

    stage1 = safe_float(candidate_row.get("stage1_mismatch"))
    stage2 = safe_float(candidate_row.get("stage2_mismatch"))
    if not np.isfinite(stage1) or not np.isfinite(stage2):
        if "solver_failure" not in reasons and "exception" not in reasons:
            reasons.append("solver_failure")
    else:
        if stage1 > float(args.strict_stage1):
            reasons.append("stage1")
        if stage2 > float(args.strict_stage2):
            reasons.append("stage2")

    ci_abs_err = safe_float(candidate_row.get("ci_abs_err"))
    ci_rel_err = safe_float(candidate_row.get("ci_rel_err"))
    if ci_available:
        if not np.isfinite(ci_abs_err) or not np.isfinite(ci_rel_err):
            reasons.append("solver_failure")
        else:
            if ci_abs_err > float(args.strict_ci_abs):
                reasons.append("ci_abs")
            if ci_rel_err > float(args.strict_ci_rel):
                reasons.append("ci_rel")

    cr_available = bool(candidate_row.get("cr_ref_available", False))
    cr_abs_err = safe_float(candidate_row.get("cr_abs_err"))
    if cr_available:
        if not np.isfinite(cr_abs_err):
            reasons.append("solver_failure")
        elif cr_abs_err > float(args.strict_cr_abs):
            reasons.append("cr_abs")

    box_required = bool(args.box_required)
    box_reject = bool(candidate_row.get("box_truncation_suspect_any_field", False)) or not bool(
        candidate_row.get("box_robustness_pass", not box_required)
    )
    if box_required and box_reject:
        reasons.append("box")

    peak_shift = safe_float(candidate_row.get("box_robustness_max_peak_shift"))
    if np.isfinite(peak_shift) and peak_shift > float(args.strict_peak_shift):
        reasons.append("peak_shift")

    if not bool(candidate_row.get("spectral_success", False)) and "solver_failure" not in reasons and "exception" not in reasons:
        reasons.append("solver_failure")
    if not bool(candidate_row.get("mode_success", False)) and "solver_failure" not in reasons and "exception" not in reasons:
        reasons.append("solver_failure")

    deduped: list[str] = []
    for reason in REJECT_REASON_ORDER:
        if reason in reasons and reason not in deduped:
            deduped.append(reason)
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    if not deduped:
        return True, "", ""
    return False, deduped[0], ";".join(deduped)


def candidate_summary_rank(row: dict[str, object]) -> tuple[float, ...]:
    accept = bool(row.get("accept", False))
    n_reasons = 0 if accept else len([item for item in str(row.get("reject_reasons_all", "")).split(";") if item])
    ci_abs = safe_float(row.get("ci_abs_err"), default=1.0e9)
    ci_rel = safe_float(row.get("ci_rel_err"), default=1.0e9)
    stage1 = safe_float(row.get("stage1_mismatch"), default=1.0e9)
    stage2 = safe_float(row.get("stage2_mismatch"), default=1.0e9)
    box_rel_l2 = safe_float(row.get("box_robustness_max_rel_l2"), default=1.0e9)
    order = safe_float(row.get("candidate_order"), default=1.0e9)
    return (
        0.0 if accept else 1.0,
        float(n_reasons),
        ci_abs,
        ci_rel,
        stage1,
        stage2,
        box_rel_l2,
        order,
    )


def build_candidate_row_base(
    *,
    candidate_id: str,
    candidate_record: dict[str, object],
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
) -> dict[str, object]:
    cr_init = safe_float(candidate_record.get("cr_init"))
    ci_init = safe_float(candidate_record.get("ci_init"))
    return {
        "candidate_id": str(candidate_id),
        "candidate_bucket": str(candidate_record.get("candidate_bucket", "")),
        "candidate_source": str(candidate_record.get("candidate_source", "")),
        "candidate_label": str(candidate_record.get("candidate_label", "")),
        "candidate_order": int(candidate_record.get("candidate_order", 0)),
        "alpha": float(alpha),
        "Mach": float(mach),
        "source_file": str(candidate_record.get("source_file", "")),
        "source_alpha": safe_float(candidate_record.get("source_alpha")),
        "source_mach": safe_float(candidate_record.get("source_mach")),
        "cr_init": float(cr_init),
        "ci_init": float(ci_init),
        "cr_final": np.nan,
        "ci_final": np.nan,
        "omega_i_final": np.nan,
        "ci_ref": float(blumen_ci) if np.isfinite(blumen_ci) else np.nan,
        "cr_ref": float(blumen_cr) if np.isfinite(blumen_cr) else np.nan,
        "ci_ref_available": bool(np.isfinite(blumen_ci)),
        "cr_ref_available": bool(np.isfinite(blumen_cr)),
        "ci_abs_err": np.nan,
        "ci_rel_err": np.nan,
        "cr_abs_err": np.nan,
        "stage1_mismatch": np.nan,
        "stage2_mismatch": np.nan,
        "requested_cr_half_window": np.nan,
        "requested_ci_half_window": np.nan,
        "used_cr_half_window": np.nan,
        "used_ci_half_window": np.nan,
        "retry_index": np.nan,
        "y_limit": np.nan,
        "ln_p_start_right": np.nan,
        "spectral_success": False,
        "mode_success": False,
        "raw_success": False,
        "raw_status": "not_run",
        "strict_status": "not_run",
        "accept": False,
        "reject_reason_primary": "",
        "reject_reasons_all": "",
        "solver_failed": False,
        "mode_reconstructed": False,
        "peak_shift": np.nan,
        "exception": "",
        **default_box_robustness_metrics([1.5, 2.0]),
    }


def run_candidate(
    *,
    candidate_id: str,
    candidate_record: dict[str, object],
    alpha: float,
    mach: float,
    blumen_cr: float,
    blumen_ci: float,
    args: argparse.Namespace,
    cfg: dict[str, object],
) -> dict[str, object]:
    row = build_candidate_row_base(
        candidate_id=str(candidate_id),
        candidate_record=candidate_record,
        alpha=float(alpha),
        mach=float(mach),
        blumen_cr=float(blumen_cr),
        blumen_ci=float(blumen_ci),
    )
    row.update(
        default_box_robustness_metrics(
            sorted(float(value) for value in cfg["box_robustness_factors"] if float(value) > 1.0)
        )
    )
    try:
        cr_half = float(cfg["cr_half_windows"][0])
        ci_half = float(cfg["ci_half_windows"][0])
        solver, result, retry_idx, used_cr_half, used_ci_half = multistart_single_box(
            alpha=float(alpha),
            mach=float(mach),
            match_y=float(cfg["match_y"]),
            use_mapping=bool(cfg["use_mapping"]),
            mapping_scale=float(cfg["mapping_scale"]),
            min_y_limit=float(cfg["min_y_limit"]),
            max_y_limit=float(cfg["max_y_limit"]),
            y_limit_factor=float(cfg["y_limit_factor"]),
            amp_lower_bound=float(cfg["amp_lower_bound"]),
            amp_upper_bound=float(cfg["amp_upper_bound"]),
            cr_center=float(candidate_record["cr_init"]),
            ci_center=float(candidate_record["ci_init"]),
            cr_half_window=cr_half,
            ci_half_window=ci_half,
            retry_growth=float(cfg["retry_growth"]),
            max_retries=int(cfg["max_retries"]),
            max_iter=int(cfg["max_iter"]),
            grid_size=int(cfg["grid_size"]),
        )
        del solver
        row["requested_cr_half_window"] = float(cr_half)
        row["requested_ci_half_window"] = float(ci_half)
        row["used_cr_half_window"] = float(used_cr_half)
        row["used_ci_half_window"] = float(used_ci_half)
        row["retry_index"] = int(retry_idx)
        row["cr_final"] = float(result.cr)
        row["ci_final"] = float(result.ci)
        row["omega_i_final"] = float(result.omega_i)
        row["stage1_mismatch"] = float(result.stage1_mismatch)
        row["stage2_mismatch"] = float(result.stage2_mismatch)
        row["ln_p_start_right"] = float(result.ln_p_start_right)
        row["y_limit"] = float(result.y_limit)
        row["spectral_success"] = bool(result.spectral_success)
        row["mode_success"] = bool(result.mode_success)
        row["raw_success"] = bool(result.success)
        row["raw_status"] = str(success_label(bool(result.spectral_success), bool(result.mode_success)))
        if row["cr_ref_available"]:
            row["cr_abs_err"] = abs(float(result.cr) - float(blumen_cr))
        if row["ci_ref_available"]:
            row["ci_abs_err"] = abs(float(result.ci) - float(blumen_ci))
            row["ci_rel_err"] = abs(float(result.ci) - float(blumen_ci)) / max(abs(float(blumen_ci)), 1.0e-12)

        if bool(result.mode_success) and np.isfinite(float(result.ln_p_start_right)):
            _, diag = reconstruct_candidate_fields_and_diagnostics(
                alpha=float(alpha),
                mach=float(mach),
                candidate={
                    "shooting_cr": float(result.cr),
                    "shooting_ci": float(result.ci),
                    "ln_p_start_right": float(result.ln_p_start_right),
                },
                cfg=cfg,
            )
            row.update(diag)
            row["mode_reconstructed"] = True
            row["peak_shift"] = safe_float(diag.get("box_robustness_max_peak_shift"))
            final_status, _ = apply_box_rejection(str(row["raw_status"]), diag)
            row["strict_status"] = str(final_status)
        else:
            row["strict_status"] = str(row["raw_status"])
            row["solver_failed"] = not bool(result.success)

    except Exception as exc:  # noqa: BLE001
        row["exception"] = repr(exc)
        row["solver_failed"] = True
        row["raw_status"] = "exception"
        row["strict_status"] = "exception"

    accept, reject_primary, reject_all = evaluate_reject_reasons(candidate_row=row, args=args)
    row["accept"] = bool(accept)
    row["reject_reason_primary"] = str(reject_primary)
    row["reject_reasons_all"] = str(reject_all)
    row["strict_status"] = "strict_accepted" if accept else ("exception" if row["exception"] else "strict_rejected")
    return row


def build_point_summary(
    *,
    alpha: float,
    mach: float,
    notes: dict[str, object],
    candidate_records: list[dict[str, object]],
    tested_rows: list[dict[str, object]],
) -> dict[str, object]:
    accepted = [row for row in tested_rows if bool(row.get("accept", False))]
    best_row = min(tested_rows, key=candidate_summary_rank) if tested_rows else None
    summary = {
        "alpha": float(alpha),
        "Mach": float(mach),
        "n_candidates_discovered": int(notes.get("candidates_after_dedup", len(candidate_records))),
        "n_candidates_tested": int(len(tested_rows)),
        "n_accepted": int(len(accepted)),
        "candidate_source_mode": str(notes.get("candidate_source_mode", "")),
        "branch_guided_available": bool(notes.get("branch_guided_available", False)),
        "existing_candidates_found": int(notes.get("existing_candidates_found", 0)),
        "branch_guided_candidates_found": int(notes.get("branch_guided_candidates_found", 0)),
        "blumen_perturb_candidates_found": int(notes.get("blumen_perturb_candidates_found", 0)),
        "best_status": "no_candidate",
        "best_reject_reason": "no_candidate",
        "best_ci_abs_err": np.nan,
        "best_ci_rel_err": np.nan,
        "best_box_robustness": np.nan,
        "best_peak_shift": np.nan,
        "best_stage1_mismatch": np.nan,
        "best_stage2_mismatch": np.nan,
        "accepted_candidate_id": "",
        "accepted_candidate_source": "",
        "accepted_cr_final": np.nan,
        "accepted_ci_final": np.nan,
        "best_candidate_id": "",
        "best_candidate_source": "",
    }
    if best_row is None:
        return summary

    summary["best_status"] = str(best_row.get("strict_status", "strict_rejected"))
    summary["best_reject_reason"] = str(best_row.get("reject_reason_primary", ""))
    summary["best_ci_abs_err"] = safe_float(best_row.get("ci_abs_err"))
    summary["best_ci_rel_err"] = safe_float(best_row.get("ci_rel_err"))
    summary["best_box_robustness"] = safe_float(best_row.get("box_robustness_max_rel_l2"))
    summary["best_peak_shift"] = safe_float(best_row.get("box_robustness_max_peak_shift"))
    summary["best_stage1_mismatch"] = safe_float(best_row.get("stage1_mismatch"))
    summary["best_stage2_mismatch"] = safe_float(best_row.get("stage2_mismatch"))
    summary["best_candidate_id"] = str(best_row.get("candidate_id", ""))
    summary["best_candidate_source"] = str(best_row.get("candidate_source", ""))

    if accepted:
        first_accepted = min(accepted, key=lambda row: safe_float(row.get("candidate_order"), default=1.0e9))
        summary["accepted_candidate_id"] = str(first_accepted.get("candidate_id", ""))
        summary["accepted_candidate_source"] = str(first_accepted.get("candidate_source", ""))
        summary["accepted_cr_final"] = safe_float(first_accepted.get("cr_final"))
        summary["accepted_ci_final"] = safe_float(first_accepted.get("ci_final"))
    return summary


def rows_to_dataframe(rows: list[dict[str, object]], columns: list[str] | None = None) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns or [])


def write_run_readme(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    points: list[tuple[float, float]],
    point_summaries: list[dict[str, object]],
    candidate_paths: list[Path],
) -> None:
    accepted_count = int(sum(bool(row.get("accepted_candidate_id")) for row in point_summaries))
    lines = [
        "# Supersonic Multicandidate Audit",
        "",
        "Ce dossier contient un audit multi-candidats supersonique avec verrouillage Blumen,",
        "selection stricte et robustesse de boite.",
        "",
        "## Configuration",
        "",
        f"- `output_stem`: `{args.output_stem}`",
        f"- `candidate_source`: `{args.candidate_source}`",
        f"- `max_candidates_per_point`: `{int(args.max_candidates_per_point)}`",
        f"- `box_required`: `{bool(args.box_required)}`",
        f"- `dry_run_candidates`: `{bool(args.dry_run_candidates)}`",
        f"- `append_validated`: `{bool(args.append_validated)}`",
        "",
        "## Critere strict",
        "",
        f"- `stage1 <= {args.strict_stage1:g}`",
        f"- `stage2 <= {args.strict_stage2:g}`",
        f"- `|delta ci| <= {args.strict_ci_abs:g}`",
        f"- `|delta ci| / ci_ref <= {args.strict_ci_rel:g}`",
        f"- `|delta cr| <= {args.strict_cr_abs:g}` si `c_r` de reference est disponible",
        f"- `box_robustness_max_rel_l2 <= {args.strict_box_max_rel_l2:g}`",
        f"- `peak_shift <= {args.strict_peak_shift:g}` si disponible",
        "",
        "## Points demandes",
        "",
    ]
    lines.extend([f"- `{alpha:.6f}:{mach:.6f}`" for alpha, mach in points])
    lines.extend(
        [
            "",
            "## Sorties",
            "",
            "- `all_candidates.csv` : tous les candidats testes ou prepares",
            "- `accepted_points.csv` : un candidat accepte par point quand il existe",
            "- `rejected_candidates.csv` : tous les candidats rejetes",
            "- `point_summary.csv` : resume une ligne par point",
            "- `run_config.json` : configuration du run",
            "",
            "## Etat courant",
            "",
            f"- points resumes : `{len(point_summaries)}`",
            f"- points avec candidat accepte : `{accepted_count}`",
            f"- catalogues existants considers : `{len(candidate_paths)}`",
            "",
            "## Garantie de securite",
            "",
            "- aucun point n'est ajoute automatiquement a `validated_modal_points`",
            "- si `--append-validated` est active, seule une proposition locale est ecrite dans ce dossier",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def flush_outputs(
    *,
    output_dir: Path,
    all_candidate_rows: list[dict[str, object]],
    point_summary_rows: list[dict[str, object]],
    args: argparse.Namespace,
    points: list[tuple[float, float]],
    candidate_paths: list[Path],
) -> None:
    all_df = rows_to_dataframe(all_candidate_rows)
    accepted_df = rows_to_dataframe([row for row in all_candidate_rows if bool(row.get("accept", False))])
    rejected_df = rows_to_dataframe(
        [
            row
            for row in all_candidate_rows
            if not bool(row.get("accept", False)) and str(row.get("strict_status", "")) not in {"dry_run", "not_run"}
        ]
    )
    summary_df = rows_to_dataframe(point_summary_rows)
    all_df.to_csv(output_dir / "all_candidates.csv", index=False)
    accepted_df.to_csv(output_dir / "accepted_points.csv", index=False)
    rejected_df.to_csv(output_dir / "rejected_candidates.csv", index=False)
    summary_df.to_csv(output_dir / "point_summary.csv", index=False)
    if bool(args.append_validated):
        proposal_df = accepted_df.copy()
        proposal_df.to_csv(output_dir / "validated_append_proposal.csv", index=False)
    write_run_readme(
        output_dir=output_dir,
        args=args,
        points=points,
        point_summaries=point_summary_rows,
        candidate_paths=candidate_paths,
    )


def main() -> None:
    args = build_parser().parse_args()
    points = resolve_points(args)
    output_dir = Path(args.output_root) / str(args.output_stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = discover_existing_candidate_paths(Path(args.existing_candidate_root))

    anchors = load_anchor_points(Path(args.anchor_csv))
    cr_points = load_digitized_long(Path(args.cr_points))
    ci_points = load_digitized_long(Path(args.ci_points))

    cfg = {
        "match_y": float(args.match_y),
        "use_mapping": bool(args.use_mapping),
        "mapping_scale": float(args.mapping_scale),
        "min_y_limit": float(args.box_min),
        "max_y_limit": float(args.box_max),
        "y_limit_factor": float(args.box_factor),
        "amp_lower_bound": float(args.amp_lower_bound),
        "amp_upper_bound": float(args.amp_upper_bound),
        "cr_half_windows": [float(value) for value in args.cr_half_windows],
        "ci_half_windows": [float(value) for value in args.ci_half_windows],
        "retry_growth": float(args.retry_growth),
        "max_retries": int(args.max_retries),
        "max_iter": int(args.max_iter),
        "grid_size": int(args.grid_size),
        "edge_amp_threshold": 5.0e-2,
        "box_robustness_factors": [1.5, 2.0],
        "box_robustness_max_rel_l2": float(args.strict_box_max_rel_l2),
        "box_robustness_max_peak_shift": float(args.strict_peak_shift),
        "box_robustness_max_center8_delta": float(args.strict_box_max_center8_delta),
        "box_robustness_max_edge_growth": float(args.strict_box_max_edge_growth),
    }

    run_config = {
        "points": [f"{alpha:.6f}:{mach:.6f}" for alpha, mach in points],
        "anchor_csv": str(args.anchor_csv),
        "reference_csv": str(args.reference_csv),
        "modal_reference_csv": str(args.modal_reference_csv),
        "existing_candidate_root": str(args.existing_candidate_root),
        "existing_candidate_paths": [str(path) for path in candidate_paths],
        "args": vars(args),
        "cfg": cfg,
    }
    (output_dir / "run_config.json").write_text(json.dumps(json_safe(run_config), indent=2), encoding="utf-8")

    print("Supersonic shooting Blumen-locked multicandidate audit")
    print(f"points: {' '.join(f'{alpha:.5f}:{mach:.5f}' for alpha, mach in points)}")
    print(f"candidate_source={args.candidate_source}")
    print(f"max_candidates_per_point={int(args.max_candidates_per_point)}")
    print(f"box_required={bool(args.box_required)} dry_run={bool(args.dry_run_candidates)}")

    all_candidate_rows: list[dict[str, object]] = []
    point_summary_rows: list[dict[str, object]] = []

    for alpha, mach in points:
        target_df = build_blumen_targets([float(mach)], float(alpha), cr_points, ci_points)
        target = target_df.iloc[0]
        blumen_cr = safe_float(target["blumen_cr"])
        blumen_ci = safe_float(target["blumen_ci"])
        candidate_records, notes = collect_candidate_records(
            alpha=float(alpha),
            mach=float(mach),
            blumen_cr=float(blumen_cr),
            blumen_ci=float(blumen_ci),
            anchors=anchors,
            existing_candidate_paths=candidate_paths,
            args=args,
        )

        point_rows: list[dict[str, object]] = []
        if not candidate_records:
            summary_row = build_point_summary(
                alpha=float(alpha),
                mach=float(mach),
                notes=notes,
                candidate_records=candidate_records,
                tested_rows=point_rows,
            )
            point_summary_rows.append(summary_row)
            flush_outputs(
                output_dir=output_dir,
                all_candidate_rows=all_candidate_rows,
                point_summary_rows=point_summary_rows,
                args=args,
                points=points,
                candidate_paths=candidate_paths,
            )
            print(f"[point] alpha={alpha:.5f} Mach={mach:.5f} status=no_candidate reason=no_candidate")
            continue

        for candidate_record in candidate_records:
            candidate_id = f"{point_key(float(alpha), float(mach))}__cand{int(candidate_record['candidate_order']):03d}"
            if args.dry_run_candidates:
                candidate_row = build_candidate_row_base(
                    candidate_id=candidate_id,
                    candidate_record=candidate_record,
                    alpha=float(alpha),
                    mach=float(mach),
                    blumen_cr=float(blumen_cr),
                    blumen_ci=float(blumen_ci),
                )
                candidate_row["strict_status"] = "dry_run"
                candidate_row["raw_status"] = "dry_run"
                candidate_row["reject_reason_primary"] = "dry_run"
                candidate_row["reject_reasons_all"] = "dry_run"
            else:
                candidate_row = run_candidate(
                    candidate_id=candidate_id,
                    candidate_record=candidate_record,
                    alpha=float(alpha),
                    mach=float(mach),
                    blumen_cr=float(blumen_cr),
                    blumen_ci=float(blumen_ci),
                    args=args,
                    cfg=cfg,
                )
            point_rows.append(candidate_row)
            all_candidate_rows.append(candidate_row)

            if bool(args.flush_every_candidate):
                current_summary_rows = point_summary_rows + [
                    build_point_summary(
                        alpha=float(alpha),
                        mach=float(mach),
                        notes=notes,
                        candidate_records=candidate_records,
                        tested_rows=point_rows,
                    )
                ]
                flush_outputs(
                    output_dir=output_dir,
                    all_candidate_rows=all_candidate_rows,
                    point_summary_rows=current_summary_rows,
                    args=args,
                    points=points,
                    candidate_paths=candidate_paths,
                )

            print(
                f"[candidate] alpha={alpha:.5f} Mach={mach:.5f} id={candidate_id} "
                f"source={candidate_row['candidate_source']} status={candidate_row['strict_status']} "
                f"accept={bool(candidate_row['accept'])} ci={safe_float(candidate_row['ci_final']):.5f} "
                f"reason={candidate_row['reject_reason_primary'] or 'accepted'}"
            )
            if bool(args.stop_after_accepted) and bool(candidate_row.get("accept", False)):
                break

        summary_row = build_point_summary(
            alpha=float(alpha),
            mach=float(mach),
            notes=notes,
            candidate_records=candidate_records,
            tested_rows=point_rows,
        )
        point_summary_rows.append(summary_row)
        flush_outputs(
            output_dir=output_dir,
            all_candidate_rows=all_candidate_rows,
            point_summary_rows=point_summary_rows,
            args=args,
            points=points,
            candidate_paths=candidate_paths,
        )
        print(
            f"[point] alpha={alpha:.5f} Mach={mach:.5f} status={summary_row['best_status']} "
            f"accepted={bool(summary_row['accepted_candidate_id'])} "
            f"best_reason={summary_row['best_reject_reason']}"
        )

    accepted_count = sum(bool(row.get("accepted_candidate_id")) for row in point_summary_rows)
    print(f"\nAccepted points: {accepted_count}/{len(point_summary_rows)}")
    print(f"Wrote {output_dir / 'all_candidates.csv'}")
    print(f"Wrote {output_dir / 'accepted_points.csv'}")
    print(f"Wrote {output_dir / 'rejected_candidates.csv'}")
    print(f"Wrote {output_dir / 'point_summary.csv'}")
    print(f"Wrote {output_dir / 'run_config.json'}")
    print(f"Wrote {output_dir / 'README.md'}")
    if bool(args.append_validated):
        print(f"Wrote {output_dir / 'validated_append_proposal.csv'}")
    print("\nSafety confirmation:")
    print("- no validated reference CSV was modified")
    print("- no point was appended automatically to validated_modal_points")


if __name__ == "__main__":
    main()
