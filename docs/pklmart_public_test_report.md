# Kaggle PKLMart 真实匹克球数据测试报告

## 数据源

本次使用从 Kaggle 下载的 PKLMart 数据压缩包，原始数据文件未纳入项目仓库：

```text
archive.zip
```

压缩包 SHA256：

```text
635560C974B4B3D1B4AE8A200F648D722E462EB637806B504443C0012675436D
```

压缩包内包含：

```text
ball_type_ref.csv
game.csv
player.csv
rally.csv
shot.csv
shot_type_ref.csv
team.csv
```

核心数据表 `shot.csv` 有 304,649 条 shot-level 记录，字段包括：

```text
shot_id, rally_id, shot_nbr, shot_type, player_id,
loc_x, loc_y, next_loc_x, next_loc_y
```

## 外部验证流程

本次公开数据验证使用独立测试脚本完成，与前面模拟数据上的 MoE 完整训练实验分开呈现。对应脚本为：

```text
scripts/public_pklmart_test.py
```

该脚本基于 scikit-learn Pipeline 构建外部基线模型，主要用于检验真实公开数据中的坐标、位移、角度和比赛元信息是否具有击球类型判别能力。

本次对比模型包括：

- Logistic Regression
- Random Forest

## 分类任务定义

从 PKLMart 的真实 `shot_type` 中选择四个样本较充足、语义清晰的击球类型：

| 原始编码 | 类别 |
|---|---|
| `D` | Dink |
| `HB` | Hand Battle |
| `tsDrp` | 3rd Shot Drop |
| `tsDrv` | 3rd Shot Drive |

为了避免类别不均衡，每类平衡抽样 8,000 条，共 32,000 条。

## 特征工程

只使用真实 shot 坐标和比赛元信息，不使用 `rally.ts_type` 这类可能泄漏标签语义的字段。

数值特征：

```text
shot_nbr
loc_x, loc_y
next_loc_x, next_loc_y
delta_x, delta_y
shot_distance
shot_angle
start_dist_to_center, end_dist_to_center
start_dist_to_nvz, end_dist_to_nvz
```

类别特征：

```text
skill_lvl
scoring_type
ball_type
```

划分方式：

```text
GroupShuffleSplit(match_id)
```

这样可以尽量避免同一比赛同时进入训练集和测试集。

## 数据规模

| Split | Size |
|---|---:|
| Train | 22,280 |
| Val | 3,104 |
| Test | 6,616 |

## 实验结果

| Model | Accuracy | Macro F1 | Balanced Acc |
|---|---:|---:|---:|
| Logistic Regression | 0.7000 | 0.6991 | 0.6996 |
| Random Forest | 0.7400 | 0.7389 | 0.7388 |

结果文件：

```text
public_data_test/pklmart_pickleball/summary.md
public_data_test/pklmart_pickleball/metrics.json
public_data_test/pklmart_pickleball/figures/logreg_confusion_matrix.png
public_data_test/pklmart_pickleball/figures/rf_confusion_matrix.png
```

## 结论

这组结果来自真实 Kaggle 匹克球 shot-level 数据，比模拟数据更能体现实际验证意义。作为外部基准实验，它使用 scikit-learn 传统模型评估真实公开数据中的可分性，为后续将真实数据接入 MoE 训练提供参考。

在只使用坐标、位移、角度、球场位置和少量比赛元信息的情况下，Random Forest 达到约 0.74 的测试准确率和 0.739 的 Macro F1，说明真实坐标特征已经包含较强的击球类型判别信息。后续可以把同一份 PKLMart 数据映射进 MoE，再比较稀疏专家模型是否能进一步提升或形成专家分工。
