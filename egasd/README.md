# EGASD: Entropy-Guided Adaptive Speculative Decoding

本目录实现基于熵引导的自适应推测解码。代码将三类思想放在同一个解码器中：

- 熵引导的动态草稿长度控制；
- SpecDec++ 风格的接受概率预测；
- PAD 风格的 Pivot-Aware 验证。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `egasd_decode.py` | EGASD 主解码器，包含草稿生成、动态停止、目标模型验证和统计信息。 |
| `entropy_utils.py` | 熵计算、归一化和动态阈值管理。 |
| `models.py` | 接受概率预测头、Pivot 分类器和组合模型封装。 |
| `train_acceptance_head.py` | 接受概率预测头训练脚本。 |
| `example_usage.py` | 可直接运行的推理示例，支持小模型 smoke test 和真实模型实验。 |
| `data/` | 示例数据格式说明，不包含完整训练集。 |

## 安装

建议在仓库根目录执行：

```bash
pip install -r egasd/requirements.txt
```

## 最小 smoke test

下面命令使用 Hugging Face 的 tiny random LLaMA 模型，主要用于确认代码路径和依赖是否可用：

```bash
python -m egasd.example_usage \
  --draft_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --target_model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --device cpu \
  --max_new_tokens 16 \
  --no_pivot
```

## 真实模型示例

```bash
python -m egasd.example_usage \
  --draft_model Qwen/Qwen2.5-0.5B \
  --target_model Qwen/Qwen2.5-7B \
  --device cuda \
  --max_new_tokens 128
```

如果没有训练好的接受预测头和 Pivot 分类器，脚本会使用熵启发式接受概率和标准验证流程。正式复现实验时，建议先训练接受预测头，再加载训练好的权重。

## 训练接受概率预测头

训练数据格式见 `data/README.md`。

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

`sample_train.json` 只展示字段格式，不适合用于真实训练。

