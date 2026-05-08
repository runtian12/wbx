"""Run the hardware-platform plotting experiment only."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve().parent / "204.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("chapter2_hardware_204", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    output_dir = ROOT / "outputs" / "chapter2_plots" / "hardware"
    module = load_script_module()
    files = module.run(output_dir)

    print("Hardware plotting experiment completed.")
    for file in files:
        print(f"  - {Path(file).resolve()}")


if __name__ == "__main__":
    main()

