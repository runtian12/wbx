# 毕业论文代码复现说明

本仓库整理了毕业论文中与大模型高效推理相关的代码，包含模型轻量化、推测解码、接受概率预测和 Pivot-Aware 验证等实验模块。目录已按子项目拆分，上传 GitHub 后建议把本文件作为读者的第一个入口。

## 项目结构

```text
代码/
├── README.md
├── REPRODUCIBILITY.md
├── requirements-common.txt
├── scripts/
│   └── check_project.py
├── experiments/
│   └── chapter2_plots/
│       ├── 202.py
│       ├── 204.py
│       ├── run_chapter2_experiments.py
│       └── requirements.txt
├── chapter2_project/
│   └── chapter2_project/
│       ├── README.md
│       ├── requirements.txt
│       ├── run_demo.py
│       └── chapter2_lightweighting/
├── egasd/
│   ├── README.md
│   ├── requirements.txt
│   ├── example_usage.py
│   ├── train_acceptance_head.py
│   └── data/
│       ├── README.md
│       └── sample_train.json
├── PAD-main/
│   ├── README.md
│   ├── setup.md
│   ├── requirements.txt
│   ├── requirements-vllm.txt
│   ├── dataset_generation.py
│   ├── train_classifier.py
│   └── decode.py
└── SpecDec_pp-main/
    ├── README.md
    ├── requirements.txt
    ├── data/
    └── specdec_pp/
```

## 子项目说明

| 目录 | 作用 | 推荐入口 |
| --- | --- | --- |
| `chapter2_project/chapter2_project` | 第二章：硬件资源感知的模型轻量化代码骨架，包含资源估计、结构化剪枝、选择性蒸馏和完整 pipeline。 | `python run_demo.py` |
| `experiments/chapter2_plots` | 第二章实验结果绘图脚本，包含带宽约束实验和硬件平台实验。 | `python experiments/chapter2_plots/run_chapter2_experiments.py` |
| `egasd` | Entropy-Guided Adaptive Speculative Decoding，融合熵引导动态草稿长度、接受概率预测和 Pivot 验证。 | `python -m egasd.example_usage` |
| `PAD-main` | Pivot-Aware Speculative Decoding，包含训练数据生成、Pivot 分类器训练和解码评估脚本。 | `dataset_generation.py`、`train_classifier.py`、`decode.py` |
| `SpecDec_pp-main` | SpecDec++ 原始实现，用于训练接受概率预测头并评估自适应候选长度。 | `SpecDec_pp-main/README.md` |

## 快速开始

建议每个子项目使用独立虚拟环境，因为 `SpecDec_pp-main` 固定了较旧的 `transformers==4.34.1`，而其他模块更适合较新的 Transformers 版本。

先检查目录是否完整：

```bash
python scripts/check_project.py
```

运行第二章最小演示：

```bash
cd chapter2_project/chapter2_project
pip install -r requirements.txt
python run_demo.py
```

运行 EGASD 小模型 smoke test：

```bash
pip install -r egasd/requirements.txt
python -m egasd.example_usage \
  --draft_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --target_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --device cpu \
  --max_new_tokens 16 \
  --no_pivot
```

完整实验请见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 复现资源准备

本仓库采用“代码公开、外部资源按路径加载”的复现方式。大模型权重、完整数据集和实验输出不直接纳入 Git 版本库，复现实验时按以下约定准备。

### 模型权重

实验中的 draft model 和 target model 均通过 Hugging Face 模型标识符或本地 checkpoint 路径加载。推荐的组织方式如下：

```text
checkpoints/
├── Qwen/
│   ├── Qwen3-0.6B/
│   └── Qwen3-8B/
└── meta-llama/
    ├── Llama-2-7b-chat-hf/
    └── Llama-2-70b-chat-hf/
```

如果直接使用 Hugging Face Hub，可在命令行参数中传入模型名称，例如：

```bash
--draft_model_name Qwen/Qwen3-0.6B
--target_model_name Qwen/Qwen3-8B
```

如果使用本地权重，则将参数替换为本地目录路径。对于需要授权访问的模型，需先完成 Hugging Face 登录：

```bash
huggingface-cli login
```

### 数据集

各模块的数据输入路径如下：

```text
PAD-main/data/gsm8k/train.jsonl
PAD-main/data/gsm8k/test.jsonl
PAD-main/data/math_splits/train.jsonl
PAD-main/data/math_splits/test.jsonl
SpecDec_pp-main/data/alpaca_data/train.json
SpecDec_pp-main/data/alpaca_data/dev.json
SpecDec_pp-main/data/alpaca_data/test.json
SpecDec_pp-main/data/humaneval_data/test.json
SpecDec_pp-main/data/gsm8k_test_data/test.json
egasd/data/train.json
```

GSM8K 的 JSONL 数据至少包含 `question` 和 `answer` 字段：

```json
{"question": "Question text", "answer": "Reasoning process #### final_answer"}
```

EGASD 接受概率预测头的训练数据为 JSON 数组，每个样本包含 `prefix`、`tokens` 和 `draft` 三个 token id 序列。格式示例见 `egasd/data/sample_train.json`。

### 训练产物

训练和评估产物按照模块分别保存：

```text
egasd/output/acceptance_head/
PAD-main/output/
PAD-main/results/
SpecDec_pp-main/exp-weight{weight}-layer{layer}/
SpecDec_pp-main/test-results-*/
```

其中，接受概率预测头、Pivot 分类器和评估结果分别作为后续解码实验的输入或论文结果统计依据。

## 标准复现流程

### 1. 检查仓库结构

在仓库根目录执行：

```bash
python scripts/check_project.py
```

输出 `Required files: OK` 表示代码文件、说明文档和示例数据结构完整。

### 2. 运行轻量化模块

```bash
cd chapter2_project/chapter2_project
pip install -r requirements.txt
python run_demo.py
```

该步骤验证资源估计、结构化剪枝和选择性蒸馏 pipeline 的基本执行流程。真实实验可在 `run_template.py` 中替换 teacher model、base model、reference model、校准集和蒸馏数据。

### 3. 生成第二章实验图

```bash
pip install -r experiments/chapter2_plots/requirements.txt
python experiments/chapter2_plots/run_chapter2_experiments.py
```

该步骤依次调用 `experiments/chapter2_plots/202.py` 和 `experiments/chapter2_plots/204.py`，生成带宽约束实验和硬件平台实验的准确率、时延、吞吐量对比图。输出目录为 `outputs/chapter2_plots/`。

### 4. 运行 EGASD 解码实验

```bash
pip install -r egasd/requirements.txt
python -m egasd.example_usage \
  --draft_model Qwen/Qwen2.5-0.5B \
  --target_model Qwen/Qwen2.5-7B \
  --device cuda \
  --max_new_tokens 128
```

该步骤执行熵引导动态草稿长度控制、接受概率估计和 Pivot 验证。若需要训练接受概率预测头，执行：

```bash
python -m egasd.train_acceptance_head \
  --draft_model_path Qwen/Qwen2.5-0.5B \
  --target_model_path Qwen/Qwen2.5-7B \
  --train_data_path egasd/data/train.json \
  --output_dir egasd/output/acceptance_head \
  --num_epochs 3 \
  --batch_size 4 \
  --device cuda \
  --bf16
```

### 5. 运行 PAD 数据生成、分类器训练和解码评估

```bash
cd PAD-main
pip install -r requirements.txt
pip install -r requirements-vllm.txt
```

生成 Pivot 分类器训练数据：

```bash
python dataset_generation.py \
  --dataset gsm8k \
  --target_model_name Qwen/Qwen3-8B \
  --draft_model_name Qwen/Qwen3-0.6B \
  --spec_len 1 \
  --max_iter 1000 \
  --save_name v31_gpt_fast
```

训练分类器：

```bash
python train_classifier.py
```

运行解码评估：

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

### 6. 运行 SpecDec++ 训练与评估

```bash
cd SpecDec_pp-main
pip install -r requirements.txt
```

数据构造、接受预测头训练和 benchmark 命令见 `SpecDec_pp-main/README.md`。核心流程为：

```text
data/gen_data.sh
specdec_pp/train.py
specdec_pp/evaluate.py
```

### 7. 汇总实验结果

最终复现实验应至少汇总以下指标：

- 解码总耗时与平均 token 延迟；
- draft token 接受率；
- target model 调用次数；
- 任务准确率或 pass rate；
- 不同 `spec_len`、`threshold`、`stop_threshold` 下的速度-质量权衡。

完整命令和参数说明见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
