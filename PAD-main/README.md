# Pivot-Aware Speculative Decoding (PAD)

Code for **“Reject Only Critical Tokens: Pivot-Aware Speculative Decoding.”**

This repo currently includes core components for **dataset generation**, **pivot-classifier training**, and **PAD decoding**. 

---

## Reproducibility Guide

This directory is arranged as an experiment pipeline:

```text
PAD-main/
├── dataset_generation.py      # generate token-level training data
├── train_classifier.py        # train the pivot-token classifier
├── decode.py                  # run PAD decoding/evaluation
├── requirements.txt           # PyTorch/Transformers training and decoding env
├── requirements-vllm.txt      # vLLM data-generation env
├── setup.md                   # detailed environment notes
└── src/
    ├── classifiers.py
    ├── models.py
    ├── gpt_fast/
    └── util/
```

### 1. Install

Use separate environments for data generation and decoding/classifier training.

```bash
# decoding and classifier training
pip install -r requirements.txt

# dataset generation with vLLM
pip install -r requirements-vllm.txt
```

`requirements-vllm.txt` is intended for Linux/WSL2 CUDA environments. On Windows, use WSL2 or a Linux server for vLLM.

### 2. Prepare Datasets

Some tasks load data from Hugging Face, while `gsm8k` and `math` expect local JSONL files:

```text
data/gsm8k/train.jsonl
data/gsm8k/test.jsonl
data/math_splits/train.jsonl
data/math_splits/test.jsonl
```

GSM8K rows should include at least:

```json
{"question": "Question text", "answer": "Reasoning #### final_answer"}
```

### 3. Generate Training Data

```bash
python dataset_generation.py \
  --dataset gsm8k \
  --target_model_name Qwen/Qwen3-8B \
  --draft_model_name Qwen/Qwen3-0.6B \
  --spec_len 1 \
  --max_iter 1000 \
  --save_name v31_gpt_fast
```

Outputs are written to `output/<save_name>_<dataset>_<target>_<draft>/<spec_len>/`.

### 4. Train the Pivot Classifier

`train_classifier.py` currently uses the `Args` class at the top of the file instead of command-line arguments. Before training, edit these fields:

```python
target_model_name = "Qwen/Qwen3-8B"
draft_model_name = "Qwen/Qwen3-0.6B"
dataset = "gsm8k"
data_base_dir = "./output/vllm_v20_10_gsm8k_Qwen3-8B_Qwen3-0.6B/1"
layer_index = -8
classifier_type = "extended_v2"
quantize = True
```

Then run:

```bash
python train_classifier.py
```

### 5. Decode/Evaluate

```bash
python decode.py \
  --dataset gsm8k \
  --target_model_name Qwen/Qwen3-8B \
  --draft_model_name Qwen/Qwen3-0.6B \
  --classifier_ckp path/to/classifier.pt \
  --runner hf \
  --spec_len 10 \
  --threshold 0.5 \
  --max_iter 200
```

For `--runner gpt-fast`, first prepare checkpoints under `checkpoints/<model_name>/model.pth` with the scripts in `src/gpt_fast/scripts/`.

---

## Overview

Conventional speculative decoding preserves the target model’s distribution, but suffers from low acceptance rates. **PAD** reframes verification around **task utility** rather than exact distribution matching, using a lightweight classifier to accept all **non-pivotal** tokens and reject only **critical (pivot) tokens** that would degrade downstream performance.

**Highlights**

* **Utility-based objective:** Matches the target model’s expected utility, not its exact distribution.
* **Pivot-token classifier:** Identifies tokens likely to harm task performance.
* **Speedups in practice:** Up to **2.5× faster** with comparable accuracy.
* **Drop-in friendly:** Minimal overhead; compatible with standard SD pipelines.

---

## 📄 Reference

If you use PAD, please cite:

```bibtex
@inproceedings{
  ziashahabi2025pad,
  title={Reject Only Critical Tokens: Pivot-Aware Speculative Decoding},
  author={Amir Ziashahabi and Yavuz Faruk Bakman and Duygu Nur Yaldiz and Mostafa El-Khamy and Sai Praneeth Karimireddy and Salman Avestimehr},
  booktitle={NeurIPS 2025 Workshop on Efficient Reasoning},
  year={2025},
}
```

---

## 📬 Contact

Questions or collaborations: **Amir Ziashahabi** — [ziashaha@usc.edu](mailto:ziashaha@usc.edu)

---
