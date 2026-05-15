# 研究点一：硬件资源感知模型轻量化方法

本目录是论文研究点一的独立代码入口，对应“硬件资源感知模型轻量化方法”。代码围绕车端有限算力、显存和时延约束下的大模型压缩展开，包含资源估计、结构化剪枝率搜索、结构化掩码构造、选择性知识蒸馏、实验报告导出和第 4.1.6 节图表复现。

## 目录结构

```text
lightweighting/
├── README.md
├── requirements.txt
├── configs/
│   ├── demo_cpu.json
│   └── vehicle_gpu_8g.json
├── data/
│   └── sample_corpus.txt
├── scripts/
│   ├── run_from_config.py
│   ├── run_resource_report.py
│   └── run_all_lightweighting_experiments.py
├── src/
│   └── chapter2_lightweighting/
│       ├── adapters.py
│       ├── config.py
│       ├── config_loader.py
│       ├── data_utils.py
│       ├── distillation.py
│       ├── metrics.py
│       ├── pipeline.py
│       ├── pruning.py
│       ├── reporting.py
│       └── resource_estimator.py
└── experiments/
    └── chapter4_lightweighting/
```

## 方法模块

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 模型适配 | `src/chapter2_lightweighting/adapters.py` | 将 LLaMA、Mistral、Qwen2 等 Hugging Face CausalLM 模型映射为统一的 attention、MLP、embedding 和 norm 组件 |
| 硬件资源估计 | `src/chapter2_lightweighting/resource_estimator.py` | 估计静态参数显存、KV cache 显存、prefill FLOPs、decode FLOPs 和理论时延 |
| 结构化剪枝 | `src/chapter2_lightweighting/pruning.py` | 计算一阶显著性、估计组件耦合项、搜索剪枝率、构造 emb/head/ffn 掩码并应用到模型 |
| 选择性蒸馏 | `src/chapter2_lightweighting/distillation.py` | 基于 teacher、reference 和 student 的 token-level KL 差异选择高价值词元进行蒸馏 |
| 流程封装 | `src/chapter2_lightweighting/pipeline.py` | 串联校准打分、剪枝率搜索、掩码剪枝和蒸馏训练 |
| 配置加载 | `src/chapter2_lightweighting/config_loader.py` | 从 JSON 配置文件加载模型、数据、硬件约束、搜索参数和输出目录 |
| 指标与报告 | `src/chapter2_lightweighting/metrics.py`、`reporting.py` | 统计参数量、稀疏率、验证损失，并导出 JSON/Markdown 实验报告 |

## 环境配置

从仓库根目录进入本研究点目录：

```powershell
cd "C:\Users\32437\Desktop\王冰心毕业论文\代码\lightweighting"
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果网络访问 Hugging Face 较慢，可以先运行不加载模型权重的资源报告脚本，确认配置和目录没有问题：

```powershell
python scripts\run_resource_report.py --config configs\demo_cpu.json
```

输出文件：

```text
outputs/demo_cpu/resource_report.json
```

## 配置文件说明

`configs/demo_cpu.json` 是最小可运行配置，使用 Hugging Face tiny random 模型验证完整流程。核心字段如下：

```json
{
  "model": {
    "base_model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "teacher_model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "reference_model": "hf-internal-testing/tiny-random-LlamaForCausalLM"
  },
  "vehicle": {
    "max_memory_bytes": 8589934592,
    "effective_flops_per_sec": 2000000000000.0,
    "max_latency_sec": 0.8
  },
  "search": {
    "population_size": 8,
    "iterations": 4
  },
  "distill": {
    "keep_ratio": 0.4,
    "epochs": 1,
    "lr": 0.00001
  }
}
```

`configs/vehicle_gpu_8g.json` 是正式实验模板，可将 `base_model`、`teacher_model`、`reference_model` 替换为本地 checkpoint 路径或 Hugging Face 模型名称。

## 完整轻量化流程

运行配置驱动的完整流程：

```powershell
python scripts\run_from_config.py --config configs\demo_cpu.json
```

该命令会依次完成：

1. 加载 base、teacher 和 reference 模型；
2. 读取 `data/sample_corpus.txt` 构造校准集和蒸馏集；
3. 根据车端显存、FLOPs 和时延约束搜索剪枝率；
4. 构造 embedding、attention head 和 FFN 通道掩码；
5. 执行结构化剪枝；
6. 进行选择性知识蒸馏；
7. 保存剪枝后模型权重、结构化掩码和实验报告。

输出目录：

```text
outputs/demo_cpu/
├── report.json
├── report.md
├── structured_masks.pt
└── student_pruned_distilled_state.pt
```

## 第 4.1.6 节图表实验

复现论文第 4.1.6 节图表：

```powershell
python scripts\run_all_lightweighting_experiments.py
```

也可以逐个运行：

```powershell
python experiments\chapter4_lightweighting\run_convergence_experiment.py
python experiments\chapter4_lightweighting\run_pruning_accuracy_experiment.py
python experiments\chapter4_lightweighting\run_pruning_ppl_experiment.py
python experiments\chapter4_lightweighting\run_distillation_comparison_experiment.py
python experiments\chapter4_lightweighting\run_table_4_3_hardware_constraints.py
python experiments\chapter4_lightweighting\run_table_4_4_ablation.py
python experiments\chapter4_lightweighting\run_table_4_5_kvret.py
```

输出目录：

```text
outputs/chapter4_lightweighting/
```

## 正式实验配置建议

使用真实模型复现实验时，建议新建独立虚拟环境，并将配置文件调整为：

```json
{
  "model": {
    "base_model": "checkpoints/Qwen2.5-0.5B",
    "teacher_model": "checkpoints/Qwen2.5-1.5B",
    "reference_model": "checkpoints/Qwen2.5-0.5B",
    "tokenizer": "checkpoints/Qwen2.5-0.5B",
    "trust_remote_code": true
  },
  "data": {
    "calibration_texts_path": "../data/calibration.txt",
    "distill_texts_path": "../data/distill.txt",
    "max_length": 128,
    "batch_size": 1
  },
  "runtime": {
    "device": "cuda",
    "max_calibration_batches": 8,
    "seed": 42
  }
}
```

`calibration_texts_path` 用于剪枝打分，`distill_texts_path` 用于蒸馏恢复。两者均为 UTF-8 文本文件，每行一条样本。

