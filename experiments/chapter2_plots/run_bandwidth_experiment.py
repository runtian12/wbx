"""Run the bandwidth plotting experiment only."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve().parent / "202.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("chapter2_bandwidth_202", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    output_dir = ROOT / "outputs" / "chapter2_plots" / "bandwidth"
    module = load_script_module()
    files = module.run(output_dir)

    print("Bandwidth plotting experiment completed.")
    for file in files:
        print(f"  - {Path(file).resolve()}")


if __name__ == "__main__":
    main()

