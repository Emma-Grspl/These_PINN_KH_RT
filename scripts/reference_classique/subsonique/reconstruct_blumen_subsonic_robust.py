from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
TARGET = ROOT_DIR / "classical_solver" / "subsonic" / "reconstruct_blumen_subsonic_robust.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")

