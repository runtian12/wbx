# EGASD 数据格式

`train_acceptance_head.py` 读取一个 JSON 数组，每个元素代表一个 prompt 下的 token 对齐样本。

## 字段

```json
{
  "prefix": [1, 2, 3],
  "tokens": [4, 5, 6],
  "draft": [4, 9, 6]
}
```

- `prefix`：输入 prompt 的 token id 序列；
- `tokens`：目标模型生成的 token id 序列；
- `draft`：草稿模型生成的 token id 序列；
- `tokens` 和 `draft` 需要等长，脚本会用 `draft == tokens` 自动生成接受/拒绝标签。

## 真实数据生成建议

正式实验中应使用同一个 tokenizer 或做好跨 tokenizer 对齐，并记录：

- draft model 名称和 checkpoint；
- target model 名称和 checkpoint；
- prompt 数据集来源和切分；
- 采样参数，例如 temperature、top-p、max_new_tokens；
- 随机种子。

`sample_train.json` 只用于说明结构，不代表有效训练样本。

