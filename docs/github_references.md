# GitHub 参考项目与借鉴点

本项目没有直接复制第三方代码，而是参考优秀开源项目的实验组织方式和模型设计取舍，并按本课程任务改写为轻量、可解释、可本地运行的版本。

## 1. yandex-research/rtdl-revisiting-models

链接：[Revisiting Deep Learning Models for Tabular Data](https://github.com/yandex-research/rtdl-revisiting-models)

借鉴点：

- 把调优充分的 MLP 作为表格深度学习强基线。
- 保持 Dense MLP 与复杂模型使用同一数据划分和训练策略。
- 报告中不预设复杂模型必胜，而是用指标和误差分析说话。

本项目对应实现：

- `configs/mlp.yaml`
- `src/pickleball_moe/models/mlp.py`
- `run_mlp.py`

## 2. manujosephv/pytorch_tabular

链接：[PyTorch Tabular](https://github.com/manujosephv/pytorch_tabular)

借鉴点：

- 用配置文件组织实验参数，减少脚本里散落的超参数。
- 自动保存训练产物，包括模型、配置、指标、预测结果和图表。
- 让训练入口保持统一，便于切换模型和数据集。

本项目对应实现：

- `configs/*.yaml`
- `src/pickleball_moe/train.py`
- `scripts/summarize_runs.py`
- `runs/<实验名>/metrics.json`

## 3. lucidrains/mixture-of-experts

链接：[Sparsely Gated Mixture of Experts - Pytorch](https://github.com/lucidrains/mixture-of-experts)

借鉴点：

- 使用稀疏门控专家层，通过只激活部分专家降低单样本计算。
- 保留 top-k 路由权重、专家分配和负载统计，方便分析专家是否塌缩。
- 将 MoE 的价值放在条件计算和专家分工，而不是简单比较总参数量。

本项目对应实现：

- `src/pickleball_moe/models/moe.py`
- `switch_style_load_balance_loss`
- `class_to_expert_heatmap.png`
- `expert_to_class_heatmap.png`

## 与本任务的适配

这些项目多面向通用表格学习或大规模 MoE。课程任务是小到中等规模的匹克球击球四分类，因此本项目做了三点收缩：

1. 只保留 Dense MLP、MoE-K=1、MoE-K=2 三组核心实验。
2. 只处理数值表格特征，默认使用中位数填补和标准化。
3. MoE 专家数默认设为 4，重点输出专家负载、路由热图和混淆矩阵，服务于实验报告解释。
