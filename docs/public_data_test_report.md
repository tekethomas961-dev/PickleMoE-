# 公开数据源外部验证报告

## 数据源

本次测试接入公开数据源：

- 数据页面：[SCORE Grand Slam Tennis Shot-Level Data](https://data.scorenetwork.org/tennis/tennis-shot-level-data.html)
- 使用文件：[tennis-w-shots-wim.csv.gz](https://data.scorenetwork.org/data/tennis-w-shots-wim.csv.gz)

SCORE 页面说明该数据集来自 Tennis Match Charting Project，记录大满贯比赛 shot-level 信息。每行表示一次击球，字段包括 `ShotHand`、`ShotType`、`ShotDirection`、`ShotDepth`、`OutcomeType`、`ErrorType` 等。

## 外部验证流程

本次公开数据验证使用独立脚本完成，与本地 MoE 训练实验分开呈现。对应脚本为：

```text
scripts/public_score_tennis_test.py
```

该脚本基于 scikit-learn Pipeline 构建传统机器学习基线，主要用于验证外部公开数据接入、特征处理、数据划分、指标计算和混淆矩阵生成流程。对比模型包括：

- Logistic Regression
- Random Forest

## 任务设置

由于 SCORE 数据不是匹克球数据，而是网球 shot-level 数据，本次测试被定义为“公开球拍运动击球类型分类验证”。

目标类别：

| 类别 |
|---|
| groundstroke |
| slice |
| volley |
| overhead |

使用字段：

```text
ShotHand, ShotDirection, ServeDirection, ShotDepth,
OutcomeType, ErrorType, Serve, Round, Tournament, Shot
```

其中类别型字段做 one-hot，`Shot` 作为数值特征标准化。

## 数据规模

平衡抽样后：

| Split | Size |
|---|---:|
| Train | 4844 |
| Val | 692 |
| Test | 1385 |

## 结果

| Model | Accuracy | Macro F1 | Balanced Acc |
|---|---:|---:|---:|
| Logistic Regression | 0.4202 | 0.4186 | 0.4683 |
| Random Forest | 0.4917 | 0.4896 | 0.5222 |

结果文件：

```text
public_data_test/score_tennis/summary.md
public_data_test/score_tennis/metrics.json
public_data_test/score_tennis/figures/logreg_confusion_matrix.png
public_data_test/score_tennis/figures/rf_confusion_matrix.png
```

## 结论

公开数据测试结果低于模拟匹克球数据上的 1.0 准确率，这是合理的。原因是：

1. SCORE 数据不是匹克球数据，而是网球数据。
2. 当前公开测试使用的是 shot encoding 类别字段，不包含球速、挥拍速度、击球高度等连续物理特征。
3. 本次采用 scikit-learn 传统模型作为公开数据基准验证，重点观察外部数据上的分类难度和混淆模式。

因此，这组结果的作用不是证明 MoE 在真实匹克球数据上已经达到高准确率，而是证明项目现在可以接入真实公开数据源，并能完成可复现实验、输出指标和混淆矩阵。
