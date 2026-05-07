# Environment setup

PAD uses two environments:

1. dataset generation with vLLM;
2. decoding and classifier training with PyTorch/Transformers or gpt-fast.

Large model weights, generated datasets, checkpoints and `wandb` logs should stay out of Git. The repository-level `.gitignore` already excludes these artifacts.

## Setup vLLM environment

Linux or WSL2 with CUDA is recommended.

```bash
uv venv --python 3.12 --seed vllm_venv
source vllm_venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install -r requirements.txt
```

Alternatively:

```bash
pip install -r requirements-vllm.txt
```

## GPT-fast / decoding environment

```bash
conda create -n pivot-dec python=3.10
conda activate pivot-dec
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --upgrade transformers accelerate bitsandbytes  
pip install -r requirements.txt
```

## Local data layout

For GSM8K and MATH, prepare local JSONL files:

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
data/math_splits/train.jsonl
data/math_splits/test.jsonl
```

Other tasks such as MBPP/AIME use the Hugging Face `datasets` package in the task utilities.

## Environment variables

Create a local `.env` file only on your machine. Do not commit it.

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

These keys are only required for optional LLM-based soundness checks.
