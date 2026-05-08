# 复现指南

本文档给出本仓库的端到端复现流程，包括环境配置、外部资源准备、模块执行顺序和实验产物路径。由于不同模块依赖的 Transformers、CUDA、vLLM 版本不同，建议为每个模块建立独立环境。

## 0. 实验环境

- Python 3.10 或 3.11；
- NVIDIA GPU 与 CUDA，用于大模型训练、推理和 vLLM 数据生成；
- Git LFS 或 Hugging Face Hub，用于管理外部模型权重；
- Linux/WSL2 更适合 vLLM 和多 GPU 实验，Windows 可用于阅读代码和运行轻量 smoke test。

检查目录完整性：

```bash
python scripts/check_project.py
```

如果实验使用 gated model，先完成 Hugging Face 认证：

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

实验输入：

- teacher model、base model 和 reference model；
- calibration loader，用于结构化剪枝重要性估计；
- distillation loader，用于选择性知识蒸馏。

实验输出：

- 剪枝后的 student model；
- 资源估计结果，包括参数显存、KV cache 显存、FLOPs 和理论时延；
- 蒸馏阶段的损失日志。

## 2. EGASD

路径：

```bash
cd <repo-root>
```

安装依赖：

```bash
pip install -r egasd/requirements.txt
```

CPU 功能验证：

```bash
python -m egasd.example_usage \
  --draft_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --target_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --device cpu \
  --max_new_tokens 16 \
  --no_pivot
```

GPU 解码实验：

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

实验输入：

- draft model：小规模草稿模型；
- target model：目标大模型；
- acceptance head：接受概率预测头，可由 `train_acceptance_head.py` 训练；
- pivot classifier：Pivot token 分类器，可复用 PAD 模块训练产物或单独训练。

实验输出：

- 生成文本；
- draft token 数量、接受 token 数量、接受率；
- 平均熵、平均动态阈值、平均草稿长度；
- 推理耗时。

## 3. 第二章实验图复现

路径：

```bash
cd <repo-root>
```

安装依赖：

```bash
pip install -r experiments/chapter2_plots/requirements.txt
```

运行统一入口：

```bash
python experiments/chapter2_plots/run_chapter2_experiments.py
```

该命令会依次调用：

```text
experiments/chapter2_plots/202.py
experiments/chapter2_plots/204.py
```

实验输入：

- `202.py`：带宽约束场景，包含 10 Mbps、30 Mbps、50 Mbps；
- `204.py`：硬件平台场景，包含 Jetson Orin Nano、Jetson Orin NX、RTX 3060；
- 对比方法包括 Proposed、IACI、PSOCI、Neur 和 DeeBERT。

实验输出：

```text
outputs/chapter2_plots/
├── bandwidth/
│   ├── bw_accuracy_bar.png
│   ├── bw_latency_bar.png
│   ├── bw_throughput_bar.png
│   └── bandwidth_metrics.csv
├── hardware/
│   ├── hardware_accuracy.png
│   ├── hardware_latency.png
│   ├── hardware_throughput.png
│   └── hardware_metrics.csv
└── run_log.json
```

其中 `run_log.json` 记录每个脚本的执行命令、起止时间、返回码、标准输出和错误输出，便于录屏展示完整复现链路。

## 4. PAD

路径：

```bash
cd PAD-main
```

PAD 分为两个执行环境：

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

`train_classifier.py` 当前通过文件顶部的 `Args` 类配置实验。复现前需要设置：

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

## 5. SpecDec++

路径：

```bash
cd SpecDec_pp-main
```

安装：

```bash
pip install -r requirements.txt
```

数据准备、训练接受预测头和评估命令见 `SpecDec_pp-main/README.md` 与 `SpecDec_pp-main/data/readme.md`。

实验输入：

- Alpaca、HumanEval、GSM8K 等评估数据；
- draft model 和 target model；
- acceptance prediction head checkpoint。

实验输出：

- SpecDec++、固定长度 SpecDec 和无推测解码 baseline 的生成结果；
- `spec_time`、`target_time`、`draft_time`；
- `num_mismatched_tokens`、`num_LM_call`、`generated_length`。

## 6. 实验记录格式

每次实验保存以下信息：

- 命令行参数或配置文件；
- Git commit hash；
- Python、PyTorch、Transformers、CUDA、GPU 型号；
- 数据集版本和切分方式；
- 模型权重名称、下载日期和 checkpoint 路径；
- 训练日志、评估 JSON 和随机种子。
