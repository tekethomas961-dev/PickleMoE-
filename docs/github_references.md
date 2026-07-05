# GitHub 参考项目与设计借鉴

本项目的代码为课程任务独立实现，没有复制第三方仓库源码。参考 GitHub 项目的目的主要有三点：确定合理的表格学习基线、借鉴实验工程组织方式、吸收稀疏 MoE 的路由与负载分析思路。最终实现已按“匹克球四类击球分类”这一小规模、可解释、可本地运行的场景重新设计。

## 1. yandex-research/rtdl-revisiting-models

项目链接：[Revisiting Deep Learning Models for Tabular Data](https://github.com/yandex-research/rtdl-revisiting-models)

该项目强调，在表格数据任务中，结构更复杂的模型需要与调优充分的 MLP 基线进行公平比较。因此本项目没有只展示 MoE，而是保留 Dense MLP 作为强基线，并让 MLP 与 MoE 共享同一套数据划分、标准化方式、训练策略和评价指标。

本项目中的对应实现：

- `configs/mlp.yaml`：MLP baseline 的默认实验配置。
- `src/pickleball_moe/models/mlp.py`：Dense MLP 模型结构。
- `run_mlp.py`：与 MoE 保持一致风格的训练入口。

## 2. manujosephv/pytorch_tabular

项目链接：[PyTorch Tabular](https://github.com/manujosephv/pytorch_tabular)

该项目提供了较成熟的表格深度学习实验组织方式，包括配置化管理、统一训练入口、自动保存指标和产物等。本项目借鉴的是这种工程组织思路：用 YAML 管理模型和训练参数，用统一脚本运行不同实验，并将指标、预测结果、模型文件和图表统一保存到 `runs/` 目录。

本项目中的对应实现：

- `configs/*.yaml`：集中管理 MLP、MoE-K=1、MoE-K=2 和负载平衡消融参数。
- `src/pickleball_moe/train.py`：统一训练、验证、测试和保存逻辑。
- `scripts/summarize_runs.py`：汇总多组实验结果。
- `runs/<实验名>/metrics.json`：保存每组实验的核心指标和专家统计。

## 3. lucidrains/mixture-of-experts

项目链接：[Sparsely Gated Mixture of Experts - Pytorch](https://github.com/lucidrains/mixture-of-experts)

该项目体现了稀疏 MoE 的核心思想：门控网络根据输入选择少数专家参与计算，而不是让所有专家同时参与。本项目参考的是 top-k 稀疏路由、专家负载统计和辅助负载平衡损失的设计方向，并针对四类击球分类任务实现了一个更轻量的表格 MoE。

本项目中的对应实现：

- `src/pickleball_moe/models/moe.py`：Expert、Gate、Top-K 路由和稀疏 logits 加权融合。
- `switch_style_load_balance_loss`：用于缓解专家坍缩的负载平衡损失。
- `class_to_expert_heatmap.png`：展示每类击球被路由到各专家的比例。
- `expert_to_class_heatmap.png`：展示每个专家主要处理的击球类型。

## 与课程任务的适配

上述参考项目多面向通用表格学习或较大规模 MoE，而本课程任务更关注“是否真正实现稀疏 MoE”以及“能否解释专家分工”。因此本项目做了如下适配：

1. 保留 Dense MLP、MoE-K=1、MoE-K=2 和负载平衡消融实验，避免只比较单一模型。
2. 将输入限定为 8 个可解释的连续技术特征，便于说明不同击球类型的技术差异。
3. 将专家数设为 4，与四类击球任务对应，方便分析专家是否出现专攻现象。
4. 同时输出准确率、Macro F1、混淆矩阵、门控权重、专家负载和专家分工热图，使实验结论不只依赖单一准确率。
5. 额外加入 Kaggle PKLMart 公开数据的 scikit-learn 外部测试，作为本地 MoE 实验之外的真实数据验证补充。
