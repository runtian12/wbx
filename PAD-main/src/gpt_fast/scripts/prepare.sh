#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "script dir $SCRIPT_DIR"

# Run from the src directory to properly handle relative imports
cd "$(dirname "$SCRIPT_DIR")/.." && python -m gpt_fast.scripts.download --repo_id $1 && python -m gpt_fast.scripts.convert_hf_checkpoint --checkpoint_dir checkpoints/$1 && python -m gpt_fast.quantize --checkpoint_path checkpoints/$1/model.pth --mode int8
