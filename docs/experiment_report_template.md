# 匹克球四类击球分类的稀疏混合专家模型实验报告

## 1. 任务与数据

本实验将单次匹克球击球建模为表格型四分类任务。输入为每次击球的数值特征，例如站位、挥拍速度、球速、落点、击球高度、发射角等；输出为四类击球标签。

数据处理遵循“先划分、再拟合预处理器”的原则。若存在比赛、回合或球员 ID，使用 group-aware 划分；否则使用分层 `70/15/15` 的 train/val/test 划分。缺失值用训练集中位数填补，数值特征用训练集拟合的 `StandardScaler` 标准化。

## 2. 模型设计

### Dense MLP

稠密基线采用三层隐藏层：

```text
d -> 128 -> 64 -> 32 -> 4
```

前两层使用 `Linear + BatchNorm1d + ReLU + Dropout(0.2)`，最后一层隐藏层使用 `Linear + ReLU + Dropout(0.1)`。

### Sparse MoE

MoE 采用共享前端、门控网络、四个专家和分类头：

```text
Input -> Shared Stem -> GateNet -> Top-K Experts -> Weighted Sum -> Classifier
```

主实验设置专家数 `E=4`，分别比较 `K=1` 和 `K=2`。训练损失为：

```text
L_total = L_ce + lambda_lb * L_lb
```

其中 `L_lb` 使用 Switch-style 负载平衡项，防止门控长期偏向少数专家。

## 3. 训练配置

| 项目 | Dense MLP | Sparse MoE |
|---|---:|---:|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 128 | 128 |
| Max epochs | 80 | 80 |
| Early stopping | patience=15 | patience=15 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| lambda_lb | 0 | 0.01 |
| Seed | 42 | 42 |

## 4. 实验结果

| 模型 | 专家数 E | K | 总参数量 | 激活参数量/样本 | Test Acc | Macro F1 | Balanced Acc | CV(load) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense MLP | - | - |  |  |  |  |  | - |
| MoE | 4 | 1 |  |  |  |  |  |  |
| MoE | 4 | 2 |  |  |  |  |  |  |

每类指标可从 `runs/<实验名>/metrics.json` 的 `classification_report` 字段中填写。

| 模型 | 类别 | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Dense MLP | C0 |  |  |  |  |
| Dense MLP | C1 |  |  |  |  |
| Dense MLP | C2 |  |  |  |  |
| Dense MLP | C3 |  |  |  |  |
| MoE-K=1 | C0 |  |  |  |  |
| MoE-K=1 | C1 |  |  |  |  |
| MoE-K=1 | C2 |  |  |  |  |
| MoE-K=1 | C3 |  |  |  |  |
| MoE-K=2 | C0 |  |  |  |  |
| MoE-K=2 | C1 |  |  |  |  |
| MoE-K=2 | C2 |  |  |  |  |
| MoE-K=2 | C3 |  |  |  |  |

## 5. 混淆矩阵与专家分工

混淆矩阵图片位于：

```text
runs/<实验名>/figures/confusion_matrix_normalized.png
```

MoE 专家热图位于：

```text
runs/<实验名>/figures/class_to_expert_heatmap.png
runs/<实验名>/figures/expert_to_class_heatmap.png
```

分析时建议回答两个问题：

1. 某些类别是否稳定路由到特定专家。
2. 若没有明显类别对齐，专家是否可能按球速、站位深浅、落点区域等特征子空间分工。

## 6. 结论

总结 Dense MLP 与 Sparse MoE 在准确率、稳定性、可解释性和专家负载上的差异。若 MoE 没有明显优于 MLP，也可以给出合理解释：小样本表格任务中稠密 MLP 往往很强，而 MoE 的主要价值体现在专家分工和条件计算机制上。
