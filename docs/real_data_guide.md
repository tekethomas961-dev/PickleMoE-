# 真实数据导入与公开数据集验证指南

## 当前实验是如何完成的

当前平台里的模型对比和实验结果由本项目的本地训练流水线完成：

1. `scripts/generate_demo_data.py` 生成模拟匹克球表格数据。
2. `src/pickleball_moe/data.py` 完成数据读取、`7:1:2` 划分、中位数填补和标准化。
3. `src/pickleball_moe/models/mlp.py` 实现 Dense MLP baseline。
4. `src/pickleball_moe/models/moe.py` 实现稀疏 MoE：Expert、Gate、Top-K、稀疏专家 logits 加权融合。
5. `src/pickleball_moe/train.py` 完成训练、早停、checkpoint、测试评估。
6. `src/pickleball_moe/eval.py` 与 `src/pickleball_moe/viz.py` 输出准确率、F1、混淆矩阵、专家分工热图和负载统计。
7. `scripts/summarize_runs.py` 汇总多组实验到 `runs/summary.md` 和 `runs/summary.csv`。

也就是说，现在平台展示的是“模拟数据上的完整机制验证”，不是从网上真实数据直接训练出的结果。

## 真实 CSV 的标准格式

最直接的方式是准备一个 CSV，包含下面 9 列：

| 列名 | 含义 |
|---|---|
| `stance_x` | 球员横向站位 |
| `stance_y` | 球员纵向站位 |
| `swing_speed` | 挥拍速度 |
| `ball_speed` | 球速 |
| `landing_x` | 落点横向区域 |
| `landing_y` | 落点纵向区域 |
| `angle` | 击球角度 |
| `height` | 击球高度 |
| `label` | 0、1、2、3 四类标签 |

如果你的 CSV 已经是这个格式，可以直接训练：

```powershell
python run_mlp.py --config configs/mlp.yaml --data data/real_pickleball.csv --label-col label
python run_moe.py --config configs/moe_k1.yaml --data data/real_pickleball.csv --label-col label
python run_moe.py --config configs/moe_k2.yaml --data data/real_pickleball.csv --label-col label
```

如果标签名称不是“正手抽球、反手削球、网前吊球、高压球”，可以在命令里覆盖：

```powershell
python run_moe.py --config configs/moe_k1.yaml --data data/real_pickleball.csv --label-col label --label-names forehand backhand dink overhead
```

## 任意 CSV 的列映射导入

如果真实数据列名不同，用 `scripts/prepare_real_dataset.py` 先转成标准格式。

查看原始数据有哪些列：

```powershell
python scripts/prepare_real_dataset.py --input path/to/raw.csv --show-columns
```

示例：把真实列映射到本项目 8 个标准特征：

```powershell
python scripts/prepare_real_dataset.py `
  --input path/to/raw.csv `
  --out data/real_pickleball.csv `
  --label-col shot_type `
  --map stance_x=player_x stance_y=player_y swing_speed=racket_speed ball_speed=ball_speed landing_x=bounce_x landing_y=bounce_y angle=shot_angle height=contact_height `
  --label-map forehand=0,backhand_slice=1,dink=2,overhead=3
```

然后运行：

```powershell
python run_moe.py --config configs/moe_k1.yaml --data data/real_pickleball.csv --label-col label --label-names forehand backhand_slice dink overhead
python run_moe.py --config configs/moe_k2.yaml --data data/real_pickleball.csv --label-col label --label-names forehand backhand_slice dink overhead
```

## 推荐公开数据集

### 1. PKLMart Competitive Pickleball Extracts

链接：[Kaggle - pklmart's Competitive Pickleball Extracts](https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts/data)

这是目前最贴近本课题的公开方向。搜索结果显示它包含超过 30 万条 pickleball shot/rally 记录，覆盖比赛、回合、击球和球员信息。它适合做真实 pickleball shot-level 验证，但 Kaggle 下载通常需要账号或 Kaggle API。

建议流程：

1. 从 Kaggle 下载数据到 `data/raw/pklmart/`。
2. 先运行 `--show-columns` 查看字段。
3. 如果其中已有击球类型、落点、站位、速度等列，用 `--map` 转成标准 8 特征。
4. 如果缺少挥拍速度或角度，可以先用已有字段派生近似特征，或把模型配置改为实际可用的数值列。

### 2. SCORE Grand Slam Tennis Shot-Level Data

链接：[SCORE - Grand Slam Tennis Shot-Level Data](https://data.scorenetwork.org/tennis/tennis-shot-level-data.html)

该数据集是网球 shot-level 数据，页面说明它记录 ATP/WTA 大满贯比赛中的每次击球，包含 `ShotHand`、`ShotType`、`ShotDirection`、`ShotDepth`、`OutcomeType` 等字段，并提供多个 `.gz` 下载文件。它不完全是匹克球，也不是连续物理特征为主，但很适合做“球拍运动击球模式分类”的迁移验证。

注意：它主要是类别型字段，若直接使用当前 8 连续特征模型，需要先做 one-hot 或构造数值编码；或者扩展 `data.py` 支持类别特征。

### 3. Tennis Shot Recognition GitHub

链接：[GitHub - antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition)

该项目 README 说明其使用标注 CSV，每个样本是一帧人体姿态关键点特征，类别包括 backhand、forehand、neutral/idle、serve。它适合验证“4 类击球分类 + MLP/MoE 对比”的代码框架，但特征语义从匹克球物理特征变成了人体姿态关键点。

建议方式：

1. 保留项目中的 CSV 姿态特征。
2. 直接在命令中用 `--feature-cols` 指定关键点数值列。
3. 用 `--label-col shot --label-names backhand forehand neutral serve` 训练。

### 4. Mendeley Tennis Side-View and Top-View Dataset

链接：[Mendeley Data - Tennis Shot Side-View and Top-View Data Set](https://data.mendeley.com/datasets/75m8vz7jr2/4)

页面说明该数据集包含 472 个 clips、4 个 records CSV，并记录击球后的真实落点坐标。它适合补充 `landing_x / landing_y` 等落点特征，但它主要是网球直线/斜线击球和视频数据，若要做 4 类分类，需要额外定义标签或扩展任务。

## 实操建议

最推荐的真实验证路线是：

1. 优先下载 PKLMart pickleball 数据，因为它和题目最贴近。
2. 如果 PKLMart 字段足够，直接映射成 8 特征。
3. 如果字段不够，先用真实可用列替代模拟特征，并在报告里说明“真实数据字段约束”。
4. 再跑 MLP、MoE-K=1、MoE-K=2 和 no-lb 消融。
5. 用 `scripts/summarize_runs.py` 重新生成 `runs/summary.md`。
6. 把新 run 的图表替换进 `platform/data.js`，平台即可展示真实数据结果。
