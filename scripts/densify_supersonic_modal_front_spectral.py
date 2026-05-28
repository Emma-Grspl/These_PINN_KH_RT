from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_supersonic_shooting_ci_alpha_continuation import (  # noqa: E402
    LineSpec,
    build_cfg,
    evaluate_line,
    line_anchor_key,
    parse_line_spec,
    plot_continuation_errors,
    plot_continuation_lines,
)
from scripts.audit_supersonic_shooting_point_batch import DEFAULT_OUTPUT_DIR, plot_modes_pdf  # noqa: E402
from scripts.audit_supersonic_families_against_blumen import (  # noqa: E402
    DEFAULT_BLUMEN_CI_POINTS,
    DEFAULT_BLUMEN_CR_POINTS,
)


DEFAULT_POINTS_CSV = DEFAULT_OUTPUT_DIR / "supersonic_reference_core_local_spectral.csv"
DEFAULT_MODAL_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "supersonic_shooting_modal_surface_core_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Densifie localement la grille spectrale supersonique autour du front modal "
            "accepted_modal -> unresolved, puis relance une continuation spectrale sur ces segments."
        )
    )
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--modal-summary-csv", type=Path, default=DEFAULT_MODAL_SUMMARY_CSV)
    parser.add_argument("--manual-line-specs", type=str, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backtrack-points", type=int, default=1)
    parser.add_argument("--lookahead-points", type=int, default=4)
    parser.add_argument("--frontier-subdivisions", type=int, default=2)
    parser.add_argument("--other-subdivisions", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--use-mapping", action="store_true", default=True)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=500.0)
    parser.add_argument("--y-limit-factor", type=float, default=6.0)
    parser.add_argument("--amp-lower-bound", type=float, default=-30.0)
    parser.add_argument("--amp-upper-bound", type=float, default=5.0)
    parser.add_argument("--cr-half-windows", type=float, nargs="+", default=[0.015, 0.03])
    parser.add_argument("--ci-half-windows", type=float, nargs="+", default=[0.008, 0.015])
    parser.add_argument("--retry-growth", type=float, default=1.60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ci-weight", type=float, default=4.0)
    parser.add_argument("--cr-weight", type=float, default=0.35)
    parser.add_argument("--continuity-weight", type=float, default=1.0)
    parser.add_argument("--acceptance-mode", choices=["modal", "spectral"], default="spectral")
    parser.add_argument("--edge-amp-threshold", type=float, default=0.05)
    parser.add_argument("--max-stage1", type=float, default=5.0e-2)
    parser.add_argument("--max-err-ci-abs", type=float, default=1.0e-2)
    parser.add_argument("--max-delta-ci", type=float, default=2.5e-2)
    parser.add_argument("--max-delta-cr", type=float, default=8.0e-2)
    parser.add_argument("--continuation-generic-seeds", action="store_true", dest="continuation_include_generic_seeds")
    parser.set_defaults(continuation_include_generic_seeds=False, anchor_include_generic_seeds=True)
    parser.add_argument("--cr-points", type=Path, default=DEFAULT_BLUMEN_CR_POINTS)
    parser.add_argument("--ci-points", type=Path, default=DEFAULT_BLUMEN_CI_POINTS)
    parser.add_argument("--output-stem", type=str, default="supersonic_shooting_modal_front_spectral_dense")
    return parser


def load_modal_flags(points_df: pd.DataFrame, modal_summary_path: Path | None) -> pd.DataFrame:
    work = points_df.copy()
    work["accepted_modal_seed"] = work.get("trusted_modal", False).fillna(False).astype(bool)
    work["modal_flag_source"] = "points.trusted_modal"

    if modal_summary_path is not None and modal_summary_path.exists():
        modal_df = pd.read_csv(modal_summary_path)
        if {"Mach", "alpha", "accepted_modal"}.issubset(modal_df.columns):
            modal_flags = modal_df[["Mach", "alpha", "accepted_modal"]].copy()
            modal_flags["accepted_modal"] = modal_flags["accepted_modal"].fillna(False).astype(bool)
            work = work.merge(modal_flags, on=["Mach", "alpha"], how="left", suffixes=("", "_modal"))
            has_modal_flag = work["accepted_modal"].notna()
            work["accepted_modal"] = work["accepted_modal"].fillna(work["accepted_modal_seed"]).astype(bool)
            work["modal_flag_source"] = np.where(
                has_modal_flag,
                "modal_surface_summary",
                work["modal_flag_source"],
            )
            work = work.drop(columns=["accepted_modal_seed"])
            return work

    work["accepted_modal"] = work["accepted_modal_seed"].astype(bool)
    work = work.drop(columns=["accepted_modal_seed"])
    return work


def densify_alphas(
    alpha_values: list[float],
    *,
    frontier_gap_index: int,
    frontier_subdivisions: int,
    other_subdivisions: int,
) -> tuple[float, ...]:
    dense: list[float] = []
    for idx, (left, right) in enumerate(zip(alpha_values[:-1], alpha_values[1:])):
        dense.append(float(left))
        n_sub = int(frontier_subdivisions) if idx == frontier_gap_index else int(other_subdivisions)
        if n_sub > 0:
            inserted = np.linspace(float(left), float(right), n_sub + 2)[1:-1]
            dense.extend(float(value) for value in inserted)
    dense.append(float(alpha_values[-1]))
    rounded = sorted({round(float(value), 6) for value in dense})
    return tuple(float(value) for value in rounded)


def build_frontier_line_specs(
    points_df: pd.DataFrame,
    *,
    backtrack_points: int,
    lookahead_points: int,
    frontier_subdivisions: int,
    other_subdivisions: int,
) -> tuple[list[LineSpec], pd.DataFrame]:
    line_specs: list[LineSpec] = []
    frontier_rows: list[dict[str, object]] = []

    for mach, sub in points_df.groupby("Mach", sort=True):
        ordered = sub.sort_values("alpha").reset_index(drop=True)
        alphas = ordered["alpha"].to_numpy(dtype=float)
        modal_flags = ordered["accepted_modal"].to_numpy(dtype=bool)
        n_points = len(ordered)
        if n_points < 2:
            continue

        for idx in range(n_points - 1):
            left_modal = bool(modal_flags[idx])
            right_modal = bool(modal_flags[idx + 1])
            if left_modal == right_modal:
                continue

            if left_modal and not right_modal:
                direction = "right"
                anchor_index = idx
                segment_start = max(0, anchor_index - int(backtrack_points))
                segment_end = min(n_points - 1, anchor_index + int(lookahead_points))
                frontier_gap_index = idx - segment_start
            else:
                direction = "left"
                anchor_index = idx + 1
                segment_start = max(0, anchor_index - int(lookahead_points))
                segment_end = min(n_points - 1, anchor_index + int(backtrack_points))
                frontier_gap_index = idx - segment_start

            if segment_end - segment_start < 1:
                continue

            segment_alphas = ordered.iloc[segment_start : segment_end + 1]["alpha"].to_list()
            dense_alphas = densify_alphas(
                segment_alphas,
                frontier_gap_index=frontier_gap_index,
                frontier_subdivisions=frontier_subdivisions,
                other_subdivisions=other_subdivisions,
            )
            anchor_alpha = float(alphas[anchor_index])
            line_specs.append(LineSpec(mach=float(mach), anchor_alpha=anchor_alpha, alphas=dense_alphas))
            frontier_rows.append(
                {
                    "Mach": float(mach),
                    "frontier_direction": str(direction),
                    "anchor_alpha": float(anchor_alpha),
                    "frontier_left_alpha": float(alphas[idx]),
                    "frontier_right_alpha": float(alphas[idx + 1]),
                    "segment_alpha_min": float(segment_alphas[0]),
                    "segment_alpha_max": float(segment_alphas[-1]),
                    "segment_points_original": int(len(segment_alphas)),
                    "segment_points_dense": int(len(dense_alphas)),
                    "dense_alphas": " ".join(f"{value:.6f}" for value in dense_alphas),
                }
            )

    deduped_specs: list[LineSpec] = []
    seen: set[tuple[float, float, tuple[float, ...]]] = set()
    for line in line_specs:
        key = (round(float(line.mach), 8), round(float(line.anchor_alpha), 8), tuple(round(float(v), 8) for v in line.alphas))
        if key in seen:
            continue
        seen.add(key)
        deduped_specs.append(line)
    frontier_df = pd.DataFrame(frontier_rows).sort_values(["Mach", "anchor_alpha", "frontier_direction"]).reset_index(drop=True)
    return deduped_specs, frontier_df


def serialize_line_specs(line_specs: list[LineSpec]) -> list[str]:
    return [
        f"{line.mach:.2f}:{line.anchor_alpha:.6f}:{','.join(f'{alpha:.6f}' for alpha in line.alphas)}"
        for line in line_specs
    ]


def build_anchor_overrides(points_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    source_cache: dict[str, pd.DataFrame] = {}
    fields_cache: dict[str, pd.DataFrame] = {}
    overrides: dict[str, dict[str, object]] = {}

    for _, row in points_df.iterrows():
        source_csv = row.get("source_csv")
        if source_csv is None or pd.isna(source_csv):
            continue
        source_csv_path = Path(str(source_csv))
        if not source_csv_path.exists():
            continue
        source_df = source_cache.get(str(source_csv_path))
        if source_df is None:
            source_df = pd.read_csv(source_csv_path)
            source_cache[str(source_csv_path)] = source_df
        matched = source_df[
            np.isclose(source_df["Mach"].to_numpy(dtype=float), float(row["Mach"]))
            & np.isclose(source_df["alpha"].to_numpy(dtype=float), float(row["alpha"]))
        ]
        if matched.empty:
            continue
        summary_row = matched.iloc[0].to_dict()
        fields_csv_path = source_csv_path.with_name(source_csv_path.name.replace("_summary.csv", "_fields.csv"))
        reference_fields = None
        if fields_csv_path.exists():
            fields_df = fields_cache.get(str(fields_csv_path))
            if fields_df is None:
                fields_df = pd.read_csv(fields_csv_path)
                fields_cache[str(fields_csv_path)] = fields_df
            sub_fields = fields_df[
                np.isclose(fields_df["Mach"].to_numpy(dtype=float), float(row["Mach"]))
                & np.isclose(fields_df["alpha"].to_numpy(dtype=float), float(row["alpha"]))
            ].copy()
            if not sub_fields.empty:
                reference_fields = sub_fields
        overrides[line_anchor_key(float(row["Mach"]), float(row["alpha"]))] = {
            "summary_row": summary_row,
            "reference_fields": reference_fields,
        }
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_csv(args.points_csv).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    points_df = load_modal_flags(points_df, args.modal_summary_csv)

    generated_specs, frontier_df = build_frontier_line_specs(
        points_df,
        backtrack_points=int(args.backtrack_points),
        lookahead_points=int(args.lookahead_points),
        frontier_subdivisions=int(args.frontier_subdivisions),
        other_subdivisions=int(args.other_subdivisions),
    )
    manual_specs = [parse_line_spec(raw) for raw in args.manual_line_specs]
    line_specs = generated_specs + manual_specs
    if not line_specs:
        raise RuntimeError("Aucun front modal detecte pour la densification locale.")

    line_specs_text = serialize_line_specs(line_specs)
    line_specs_path = output_dir / f"{args.output_stem}_line_specs.txt"
    frontier_path = output_dir / f"{args.output_stem}_frontiers.csv"
    line_specs_path.write_text("\n".join(line_specs_text) + "\n", encoding="utf-8")
    frontier_df.to_csv(frontier_path, index=False)

    print("Supersonic modal-front spectral densification")
    print(f"points_csv={args.points_csv}")
    print(f"modal_summary_csv={args.modal_summary_csv if args.modal_summary_csv.exists() else 'fallback:trusted_modal_from_points'}")
    print(f"frontiers={len(frontier_df)} line_specs={len(line_specs)}")
    print(
        f"densification: backtrack={int(args.backtrack_points)} "
        f"lookahead={int(args.lookahead_points)} "
        f"frontier_subdivisions={int(args.frontier_subdivisions)} "
        f"other_subdivisions={int(args.other_subdivisions)}"
    )
    for raw in line_specs_text:
        print(f"line-spec={raw}")
    print(f"Wrote {line_specs_path}")
    print(f"Wrote {frontier_path}")

    if args.dry_run:
        return

    cfg = build_cfg(args)
    cfg["acceptance_mode"] = "spectral"
    cfg["anchor_overrides"] = build_anchor_overrides(points_df)

    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = {executor.submit(evaluate_line, line, cfg): line for line in line_specs}
        for future in as_completed(futures):
            line = futures[future]
            line_summary, line_candidates, line_fields = future.result()
            summary_rows.extend(line_summary)
            candidate_rows.extend(line_candidates)
            field_rows.extend(line_fields)
            accepted = sum(bool(row["continuation_accepted"]) for row in line_summary if row["continuation_state"] != "not_run_after_reject")
            print(f"[line] {line.line_id} completed | accepted={accepted}/{len(line_summary)}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows)
    fields_df = pd.DataFrame(field_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["Mach", "alpha", "continuation_direction", "continuation_step_index", "success_priority", "selection_metric"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)
    if not fields_df.empty:
        fields_df = fields_df.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

    summary_path = output_dir / f"{args.output_stem}_summary.csv"
    candidates_path = output_dir / f"{args.output_stem}_candidates.csv"
    fields_path = output_dir / f"{args.output_stem}_fields.csv"
    lines_path = output_dir / f"{args.output_stem}_ci_alpha_lines.png"
    errors_path = output_dir / f"{args.output_stem}_ci_alpha_errors.png"
    modes_path = output_dir / f"{args.output_stem}_modes.pdf"

    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)
    fields_df.to_csv(fields_path, index=False)
    plot_continuation_lines(summary_df, lines_path)
    plot_continuation_errors(summary_df, errors_path)
    if not fields_df.empty:
        plot_modes_pdf(summary_df, fields_df, threshold_ratio=0.02, min_half_width=8.0, output_path=modes_path)

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(summary_df.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {fields_path}")
    print(f"Wrote {lines_path}")
    print(f"Wrote {errors_path}")
    if not fields_df.empty:
        print(f"Wrote {modes_path}")


if __name__ == "__main__":
    main()
