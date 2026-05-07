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

## 数据和模型权重

本仓库不直接提交大模型权重、训练输出、实验结果和完整数据集。复现时需要读者自行准备：

- Hugging Face 模型权重，例如 Qwen、LLaMA 或论文中指定的 draft/target model pair；
- GSM8K、MATH、MBPP、HumanEval、Alpaca 等公开数据集或脚本生成的数据；
- PAD/EGASD/SpecDec++ 的训练得到的接受预测头或 Pivot 分类器权重。

`.gitignore` 已排除常见的大文件目录，例如 `checkpoints/`、`output/`、`results/`、`*.pt`、`*.pth`、`*.safetensors`。如果确实需要公开小样例数据，请放在明确的 `sample` 文件中。

## 上传 GitHub 前建议

1. 在根目录执行 `python scripts/check_project.py`，确认关键文件存在。
2. 删除本地缓存和 IDE 目录，或确认它们已被 `.gitignore` 排除。
3. 不要上传真实 API Key、`.env`、模型权重、训练输出和大规模数据。
4. 如果代码中包含第三方项目，请保留原项目的 README、LICENSE 和引用信息。
5. 在 GitHub README 中注明硬件环境，例如 CUDA 版本、GPU 型号、显存需求和 Python 版本。

