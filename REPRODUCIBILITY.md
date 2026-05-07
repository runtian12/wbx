# 复现指南

本文档按子项目说明环境准备、数据准备和运行命令。由于不同实验依赖的 Transformers、CUDA、vLLM 版本可能互相冲突，推荐为每个子项目建立独立环境。

## 0. 通用准备

推荐环境：

- Python 3.10 或 3.11；
- NVIDIA GPU 与 CUDA，用于大模型训练、推理和 vLLM 数据生成；
- Git LFS 或 Hugging Face Hub，用于管理外部模型权重；
- Linux/WSL2 更适合 vLLM 和多 GPU 实验，Windows 可用于阅读代码和运行轻量 smoke test。

检查目录完整性：

```bash
python scripts/check_project.py
```

如果使用 Hugging Face gated model，请先登录：

```bash
huggingface-cli login
```

## 1. 第二章轻量化工程

路径：

```bash
cd chapter2_project/chapter2_project
```

安装依赖：

```bash
pip install -r requirements.txt
```

运行最小演示：

```bash
python run_demo.py
```

说明：

- `run_demo.py` 使用 `hf-internal-testing/tiny-random-LlamaForCausalLM`，适合验证流程是否能跑通；
- `run_template.py` 是替换真实模型、校准集和蒸馏数据后的实验模板；
- 正式实验需要自行准备 teacher/base/reference 模型、校准数据和蒸馏数据。

## 2. EGASD

路径：

```bash
cd <repo-root>
```

安装依赖：

```bash
pip install -r egasd/requirements.txt
```

CPU smoke test：

```bash
python -m egasd.example_usage \
  --draft_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --target_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --device cpu \
  --max_new_tokens 16 \
  --no_pivot
```

GPU 示例：

```bash
python -m egasd.example_usage \
  --draft_model Qwen/Qwen2.5-0.5B \
  --target_model Qwen/Qwen2.5-7B \
  --device cuda \
  --max_new_tokens 128
```

训练接受概率预测头：

```bash
python -m egasd.train_acceptance_head \
  --draft_model_path Qwen/Qwen2.5-0.5B \
  --target_model_path Qwen/Qwen2.5-7B \
  --train_data_path egasd/data/sample_train.json \
  --output_dir egasd/output/acceptance_head \
  --num_epochs 1 \
  --batch_size 1 \
  --device cuda
```

注意：

- `egasd/data/sample_train.json` 只用于展示数据格式，不代表有效训练集；
- 正式训练数据格式见 `egasd/data/README.md`；
- 如果没有训练好的接受预测头或 Pivot 分类器，`example_usage.py` 会退化为熵启发式和标准验证流程。

## 3. PAD

路径：

```bash
cd PAD-main
```

PAD 分为两个环境：

- 数据生成环境：需要 vLLM；
- 解码和分类器训练环境：使用 PyTorch、Transformers、scikit-learn、wandb 等。

安装解码/训练环境：

```bash
pip install -r requirements.txt
```

安装 vLLM 数据生成环境：

```bash
pip install -r requirements-vllm.txt
```

准备本地数据：

```text
PAD-main/data/gsm8k/train.jsonl
PAD-main/data/gsm8k/test.jsonl
PAD-main/data/math_splits/train.jsonl
PAD-main/data/math_splits/test.jsonl
```

GSM8K 的 `jsonl` 每行至少应包含：

```json
{"question": "Question text", "answer": "Reasoning #### final_answer"}
```

生成训练数据：

```bash
python dataset_generation.py \
  --dataset gsm8k \
  --target_model_name Qwen/Qwen3-8B \
  --draft_model_name Qwen/Qwen3-0.6B \
  --spec_len 1 \
  --max_iter 1000 \
  --save_name v31_gpt_fast
```

训练 Pivot 分类器：

```bash
python train_classifier.py
```

`train_classifier.py` 当前通过文件顶部的 `Args` 类配置实验。复现时需要至少检查：

- `target_model_name`
- `draft_model_name`
- `dataset`
- `data_base_dir`
- `layer_index`
- `classifier_type`
- `quantize`

运行 PAD 解码评估：

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

## 4. SpecDec++

路径：

```bash
cd SpecDec_pp-main
```

安装：

```bash
pip install -r requirements.txt
```

数据准备、训练接受预测头和评估命令见 `SpecDec_pp-main/README.md` 与 `SpecDec_pp-main/data/readme.md`。

注意：

- 原项目依赖 `transformers==4.34.1`；
- 评估 LLaMA-2 7B/70B 需要足够显存和 Hugging Face 访问权限；
- `assets/` 中的图片用于论文方法说明，可保留上传。

## 5. 结果记录建议

建议每次实验保存：

- 命令行参数或配置文件；
- Git commit hash；
- Python、PyTorch、Transformers、CUDA、GPU 型号；
- 数据集版本和切分方式；
- 模型权重名称、下载日期和 checkpoint 路径；
- 训练日志、评估 JSON 和随机种子。

