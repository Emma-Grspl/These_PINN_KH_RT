from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from classical_solver.gep.build_supersonic_mode_database import build_parser, run_supersonic_mode_database


def build_chunk_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Lance un chunk Mach du sweep supersonique GEP."
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    return parser


def main() -> None:
    parser = build_chunk_parser()
    args = parser.parse_args()

    if not (0 <= args.chunk_index < args.num_chunks):
        raise ValueError("--chunk-index doit etre dans [0, num_chunks).")

    global_machs = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    chunks = np.array_split(global_machs, args.num_chunks)
    selected = np.asarray(chunks[args.chunk_index], dtype=float)
    if selected.size == 0:
        raise ValueError("Chunk vide.")

    args.mach_values = [float(v) for v in selected]
    args.num_mach = int(selected.size)

    print(
        "Chunk supersonique "
        f"{args.chunk_index + 1}/{args.num_chunks} | "
        f"Mach in [{selected[0]:.6f}, {selected[-1]:.6f}] | "
        f"{selected.size} valeurs"
    )
    run_supersonic_mode_database(args)


if __name__ == "__main__":
    main()
