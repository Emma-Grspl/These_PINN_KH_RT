from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_kh_subsonic_pinn_2d_pilot_M05_M06 import (  # noqa: E402
    build_pilot_parser,
    run_pilot,
)


DEFAULT_MODE_POINTS = [
    "0.10:0.50",
    "0.25:0.50",
    "0.65:0.50",
    "0.10:0.55",
    "0.25:0.55",
    "0.65:0.55",
    "0.10:0.60",
    "0.25:0.60",
    "0.65:0.60",
    "0.10:0.65",
    "0.25:0.65",
    "0.65:0.65",
    "0.10:0.70",
    "0.25:0.70",
    "0.65:0.70",
]


def build_parser() -> argparse.ArgumentParser:
    return build_pilot_parser(
        description=(
            "Extension PINN subsonique 2D sur la bande Mach [0.5, 0.7], "
            "warmstart depuis le pilote multi-Mach [0.5, 0.6]."
        ),
        default_warmstart_run_dir=Path("model_saved/kh_subsonic_2d_pilot_M05_M06"),
        default_output_dir=Path("model_saved/kh_subsonic_2d_band_M05_M07"),
        default_mode_points=DEFAULT_MODE_POINTS,
        default_alpha_min=0.05,
        default_alpha_max=0.75,
        default_mach_min=0.50,
        default_mach_max=0.70,
        default_n_reference_mach=9,
        default_n_audit_mach=7,
        default_upper_mach_min=0.65,
        default_lower_mach_max=0.53,
        default_focus_mach_half_width=0.03,
        default_ci_plot_mach=9,
        default_mode_audit_mach=5,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_pilot(args, pilot_name="subsonic_2d_M05_M07_band")


if __name__ == "__main__":
    main()
