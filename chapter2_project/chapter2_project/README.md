# 第二章代码工程：基于硬件资源感知的模型轻量化方法

这套代码按“工程拆分”的方式组织，不再是单文件版本，便于你后续写论文附录、展示代码结构和继续扩展。

## 目录结构

```text
chapter2_project/
├── README.md
├── requirements.txt
├── run_demo.py
├── run_template.py
└── chapter2_lightweighting/
    ├── __init__.py
    ├── adapters.py
    ├── config.py
    ├── data_utils.py
    ├── distillation.py
    ├── pipeline.py
    ├── pruning.py
    └── resource_estimator.py
```

## 各文件作用

### 1. `config.py`
统一放参数和数据结构，包含：
- 车辆硬件约束 `VehicleConstraint`
- 剪枝搜索边界 `SearchBounds`
- 进化搜索配置 `SearchConfig`
- 选择性蒸馏配置 `DistillConfig`
- 模型拓扑 `ModelTopology`
- 剪枝率向量 `RateVector`
- 候选解 `CandidateSolution`
- 结构化单元 `StructuredUnit`

### 2. `adapters.py`
做模型适配，把 HuggingFace 风格的 LLaMA/Mistral/Qwen2 之类模型统一映射成论文需要的组件：
- attention
- mlp
- q/k/v/o 投影
- gate/up/down 投影
- embedding 和 final norm

### 3. `resource_estimator.py`
对应第二章 2.3.1 节，负责：
- 静态参数显存估计
- 动态 KV Cache 显存估计
- 预填充阶段 FLOPs
- 解码阶段 FLOPs
- 理论时延估计
- 资源可行性判断

### 4. `pruning.py`
对应第二章 2.3.2 节，包含四部分：
- `StructuredPruningScorer`：一阶显著性 + Fisher 近似二阶耦合
- `EvolutionaryRateSearcher`：进化搜索最优剪枝率
- `StructuredMaskBuilder`：按显著性构造 emb/head/ffn 掩码
- `StructuredPruner`：把掩码真正作用到模型参数

### 5. `distillation.py`
对应第二章 2.3.3 节，实现基于损失偏差的选择性知识蒸馏。
核心逻辑是：
- teacher / reference / student 同步前向
- 计算 `delta = stu_loss - ref_loss`
- 选取高可学习性词元
- 仅对这些词元保留蒸馏监督

### 6. `pipeline.py`
对应第二章 2.4 节，把整个流程串起来：
1. 校准集打分
2. 搜索最优剪枝率
3. 构造结构化掩码
4. 执行剪枝
5. 进行选择性蒸馏

### 7. `data_utils.py`
给了一个最简单的示例数据加载器，方便先跑通流程。

### 8. `run_demo.py`
最小可运行示例，用一个 tiny random LLaMA 做演示。

### 9. `run_template.py`
真正给你接自己实验时用的模板。你只需要把：
- 模型路径
- calibration_loader
- distill_loader
替换成自己的内容即可。

## 运行方式

先安装依赖：

```bash
pip install -r requirements.txt
```

跑演示：

```bash
python run_demo.py
```

## 你论文里可以怎么描述这套代码结构

你后面如果需要写“算法实现”或“系统实现”部分，可以直接按下面这个思路展开：

- `config.py` 负责统一管理剪枝与蒸馏阶段的参数配置和中间数据结构；
- `adapters.py` 负责将不同大模型骨干网络映射为统一的可剪枝组件表示；
- `resource_estimator.py` 负责实现车载资源约束的显式建模；
- `pruning.py` 负责实现协同优化结构化剪枝；
- `distillation.py` 负责实现选择性知识蒸馏性能恢复；
- `pipeline.py` 负责封装完整训练流程并向上层实验脚本提供统一调用接口。

## 说明

这份工程更适合作为“论文方法代码骨架”和“答辩展示版工程结构”。
如果你接下来要做成真正可复现实验代码，我建议下一步再补三类内容：

1. 和你当前 LLaMA-7B / LLaMA-13B 权重完全对齐的适配代码；
2. BoolQ、PIQA、WikiText2、PTB 的正式数据预处理脚本；
3. 日志保存、模型导出、实验配置 yaml 化。
