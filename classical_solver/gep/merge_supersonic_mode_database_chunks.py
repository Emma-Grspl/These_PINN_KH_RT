from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import OUTPUT_DIR, plot_isolines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusionne plusieurs chunks de base de modes supersonique.")
    parser.add_argument("--chunk-stems", type=str, nargs="+", required=True)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser


def load_chunk_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_parts: list[pd.DataFrame] = []
    mode_parts: list[pd.DataFrame] = []
    y_export: np.ndarray | None = None
    npz_fields: dict[str, list[np.ndarray]] = {"u": [], "v": [], "p": [], "rho": []}
    mode_meta_parts: dict[str, list[np.ndarray]] = {
        "alpha": [],
        "Mach": [],
        "accepted": [],
        "gep_cr": [],
        "gep_ci": [],
        "distance_to_shooting": [],
    }

    for stem in args.chunk_stems:
        surface_csv = OUTPUT_DIR / f"{stem}_surface.csv"
        modes_csv = OUTPUT_DIR / f"{stem}_modes.csv"
        modes_npz = OUTPUT_DIR / f"{stem}_modes.npz"

        if surface_csv.exists():
            summary_parts.append(pd.read_csv(surface_csv))
        if modes_csv.exists():
            mode_parts.append(pd.read_csv(modes_csv))

        payload = load_chunk_npz(modes_npz)
        if payload:
            if "y" in payload and y_export is None:
                y_export = payload["y"]
            for key in ("u", "v", "p", "rho"):
                if key in payload and payload[key].size:
                    npz_fields[key].extend([arr for arr in payload[key]])
            for key in mode_meta_parts:
                if key in payload and payload[key].size:
                    mode_meta_parts[key].append(payload[key])

    if not summary_parts:
        raise FileNotFoundError("Aucun chunk surface.csv trouve.")

    summary_df = (
        pd.concat(summary_parts, ignore_index=True)
        .sort_values(["Mach", "alpha"])
        .drop_duplicates(subset=["Mach", "alpha"], keep="last")
        .reset_index(drop=True)
    )

    if mode_parts:
        modes_df = (
            pd.concat(mode_parts, ignore_index=True)
            .sort_values(["Mach", "alpha"])
            .drop_duplicates(subset=["Mach", "alpha"], keep="last")
            .reset_index(drop=True)
        )
        modes_df["mode_index"] = np.arange(len(modes_df), dtype=int)
    else:
        modes_df = pd.DataFrame()

    surface_csv = OUTPUT_DIR / f"{args.output_stem}_surface.csv"
    modes_csv = OUTPUT_DIR / f"{args.output_stem}_modes.csv"
    modes_npz = OUTPUT_DIR / f"{args.output_stem}_modes.npz"
    png_path = OUTPUT_DIR / f"{args.output_stem}_isolines.png"
    progress_path = OUTPUT_DIR / f"{args.output_stem}_progress.json"

    summary_df.to_csv(surface_csv, index=False)
    modes_df.to_csv(modes_csv, index=False)
    plot_isolines(summary_df, png_path)

    if y_export is not None and not modes_df.empty and npz_fields["u"]:
        payload = {
            "y": y_export.astype(np.float64),
            "alpha": modes_df["alpha"].to_numpy(dtype=np.float64),
            "Mach": modes_df["Mach"].to_numpy(dtype=np.float64),
            "accepted": modes_df["accepted"].to_numpy(dtype=bool),
            "gep_cr": modes_df["gep_cr"].to_numpy(dtype=np.float64),
            "gep_ci": modes_df["gep_ci"].to_numpy(dtype=np.float64),
            "distance_to_shooting": modes_df["distance_to_shooting"].to_numpy(dtype=np.float64),
            "u": np.asarray(npz_fields["u"], dtype=np.complex128),
            "v": np.asarray(npz_fields["v"], dtype=np.complex128),
            "p": np.asarray(npz_fields["p"], dtype=np.complex128),
            "rho": np.asarray(npz_fields["rho"], dtype=np.complex128),
        }
        np.savez_compressed(modes_npz, **payload)

    progress = {
        "completed_points": int(len(summary_df)),
        "stored_modes": int(len(modes_df)),
        "chunk_stems": list(args.chunk_stems),
        "output_stem": args.output_stem,
    }
    progress_path.write_text(json.dumps(progress, indent=2))

    print(f"Surface CSV: {surface_csv}")
    print(f"Modes CSV: {modes_csv}")
    if modes_npz.exists():
        print(f"Modes NPZ: {modes_npz}")
    print(f"Isoline figure: {png_path}")
    print(f"Progress JSON: {progress_path}")


if __name__ == "__main__":
    main()
