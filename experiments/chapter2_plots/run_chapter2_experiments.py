"""Run the Chapter 2 plotting experiments in one command.

The runner executes 202.py and 204.py as separate experiment scripts, captures
their console output, and writes a reproducible execution log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str, output_dir: Path) -> dict:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / script_name),
        "--output-dir",
        str(output_dir),
    ]

    started_at = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    finished_at = datetime.now().isoformat(timespec="seconds")

    return {
        "script": script_name,
        "command": command,
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 2 figure-generation experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "chapter2_plots",
        help="Base output directory for all generated figures, CSV files, and logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Chapter 2 experiment runner")
    print(f"Repository root: {ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Output directory: {base_output_dir.resolve()}")
    print("=" * 80)

    runs = [
        run_script("202.py", base_output_dir / "bandwidth"),
        run_script("204.py", base_output_dir / "hardware"),
    ]

    log_path = base_output_dir / "run_log.json"
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "repository_root": str(ROOT),
                "python_executable": sys.executable,
                "runs": runs,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    failed = False
    for run in runs:
        print(f"\n[{run['script']}] returncode={run['returncode']}")
        if run["stdout"]:
            print(run["stdout"].rstrip())
        if run["stderr"]:
            print("stderr:")
            print(run["stderr"].rstrip())
        if run["returncode"] != 0:
            failed = True

    print("\nGenerated outputs:")
    for path in sorted(base_output_dir.rglob("*")):
        if path.is_file():
            print(f"  - {path.resolve()}")

    print(f"\nExecution log: {log_path.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

