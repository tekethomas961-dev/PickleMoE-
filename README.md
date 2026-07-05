# 匹克球四类击球分类：稀疏混合专家模型项目

这个项目根据 `docs/requirement.pdf` 搭建，目标是完成单次击球级别的四分类实验：输入每次击球的表格数值特征，输出四类击球标签。项目包含一个稠密 MLP 强基线，以及两个稀疏 MoE 实验版本：`K=1` 与 `K=2`。

## 项目结构

```text
.
├── configs/                  # MLP、MoE-K=1、MoE-K=2 默认配置
├── data/                     # 可放真实 CSV；demo 数据可自动生成
├── docs/requirement.pdf      # 原始需求文档副本
├── docs/github_references.md # GitHub 优秀项目参考与借鉴点
├── notebooks/eda_train.ipynb # EDA 与快速训练模板
├── platform/                 # 最终实验展示平台
├── scripts/                  # demo 数据生成脚本
├── src/pickleball_moe/       # 数据、模型、训练、评估、可视化代码
├── tests/                    # 路由与 demo 数据的基础测试
├── run_mlp.py                # 稠密 MLP 实验入口
└── run_moe.py                # 稀疏 MoE 实验入口
```

## 安装依赖

项目已在 Windows + Python 3.12 虚拟环境下验证通过。首次克隆后，建议在项目目录下创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

已验证的关键依赖包括：

| 依赖 | 作用 |
|---|---|
| `torch` | Dense MLP 与 Sparse MoE 训练 |
| `pandas` / `numpy` | 表格数据读取与处理 |
| `scikit-learn` | 数据划分、标准化、指标与公开数据测试 |
| `matplotlib` | 训练曲线、混淆矩阵、专家热图 |
| `PyYAML` / `joblib` | 配置读取与预处理器保存 |

安装完成后，可以先检查项目脚本和 demo 数据生成：

```powershell
python scripts/generate_demo_data.py --out data/demo_pickleball.csv --n-samples 600
```

也可以运行测试确认核心路由和 demo 数据逻辑：

```powershell
python -m pytest -q
```

## 运行内置 demo

安装依赖后，可以先用合成数据跑通完整流程：

```powershell
python run_mlp.py --config configs/mlp.yaml --demo
python run_moe.py --config configs/moe_k1.yaml --demo
python run_moe.py --config configs/moe_k2.yaml --demo
```

默认 demo 会生成 800 条均衡模拟样本，也可以手动指定：

```powershell
python run_moe.py --config configs/moe_k1.yaml --demo --demo-samples 800
```

每次实验会在 `runs/` 下生成一个目录，包含：

- `metrics.json`：准确率、Macro F1、Balanced Accuracy、混淆矩阵、参数量等
- `history.csv`：训练曲线数据
- `best_model.pt`：验证集最优模型
- `preprocessor.joblib` 与 `label_map.json`：复现实验所需预处理器和标签映射
- `figures/`：训练曲线、混淆矩阵、专家分工热图、门控熵分布图
- `class_to_expert.csv` 和 `mean_selected_gate_by_class.csv`：可直接放入报告的专家分工表

可以用下面的命令汇总多组实验：

```powershell
python scripts/summarize_runs.py runs/dense_mlp_full_demo runs/sparse_moe_k1_full_demo runs/sparse_moe_k2_full_demo --out-dir runs
```

## 使用真实 CSV

假设你的数据文件是 `data/pickleball.csv`，标签列名是 `shot_type`：

```powershell
python run_mlp.py --config configs/mlp.yaml --data data/pickleball.csv --label-col shot_type
python run_moe.py --config configs/moe_k1.yaml --data data/pickleball.csv --label-col shot_type
python run_moe.py --config configs/moe_k2.yaml --data data/pickleball.csv --label-col shot_type
```

如果有比赛、回合或球员分组列，例如 `match_id`，可以避免同组样本同时进入训练集和测试集：

```powershell
python run_moe.py --config configs/moe_k1.yaml --data data/pickleball.csv --label-col shot_type --group-col match_id
```

默认会自动选择除标签列、分组列以外的数值列作为特征。也可以手动指定特征列：

```powershell
python run_mlp.py --data data/pickleball.csv --label-col shot_type --feature-cols stance_x stance_y swing_speed ball_speed landing_x landing_y
```

如果真实 CSV 的列名和本项目标准字段不同，可以先用导入脚本规范化：

```powershell
python scripts/prepare_real_dataset.py --input path/to/raw.csv --show-columns
python scripts/prepare_real_dataset.py --input path/to/raw.csv --out data/real_pickleball.csv --label-col shot_type --map stance_x=player_x stance_y=player_y swing_speed=racket_speed ball_speed=ball_speed landing_x=bounce_x landing_y=bounce_y angle=shot_angle height=contact_height --label-map forehand=0,backhand_slice=1,dink=2,overhead=3
```

真实数据和公开数据集建议见 [docs/real_data_guide.md](docs/real_data_guide.md)。

## 公开数据源外部验证

项目提供了两条独立的外部数据测试路径。两条路径均使用 scikit-learn Pipeline，与本地 MoE 训练实验分开呈现，因此可以作为模拟数据实验之外的公开数据验证依据。

### 1. SCORE 网球 shot-level 迁移验证

SCORE Grand Slam Tennis Shot-Level Data 是公开的球拍运动击球级数据。它不属于匹克球数据集，但适合用于验证本项目的数据处理、划分、分类评估和混淆矩阵生成流程能否在外部公开数据上独立运行。

```powershell
python scripts/public_score_tennis_test.py --out-dir public_data_test/score_tennis --samples-per-class 2000
```

该脚本会直接下载 SCORE 数据源提供的公开 `.csv.gz` 文件，并使用 Logistic Regression 与 Random Forest 完成测试。

结果见 [docs/public_data_test_report.md](docs/public_data_test_report.md) 和 [public_data_test/score_tennis/summary.md](public_data_test/score_tennis/summary.md)。

### 2. Kaggle PKLMart 真实匹克球验证

PKLMart Competitive Pickleball Extracts 更贴近本课程主题，包含真实匹克球比赛中的 shot-level 记录。由于 Kaggle 数据通常需要登录后手动下载，运行时只需把下载得到的压缩包路径传给 `--zip`：

```powershell
python scripts/public_pklmart_test.py --zip "path/to/archive.zip" --out-dir public_data_test/pklmart_pickleball --samples-per-class 8000
```

该脚本从 PKLMart 的真实击球记录中构造四类击球分类任务，并同样使用 Logistic Regression 与 Random Forest 作为传统机器学习基线。

结果见 [docs/pklmart_public_test_report.md](docs/pklmart_public_test_report.md) 和 [public_data_test/pklmart_pickleball/summary.md](public_data_test/pklmart_pickleball/summary.md)。

## 实验设计

建议按 PDF 的三组实验提交：

1. Dense MLP：强基线，检查表格数据上的基本可分性。
2. MoE-K=1：Switch-style top-1 稀疏路由，观察是否出现专家塌缩或明确分工。
3. MoE-K=2：top-2 稀疏路由，观察边界样本是否获得更柔和的专家组合。
4. MoE-K=1/K=2 no-lb：把 `lambda_lb` 设为 0，作为负载平衡损失消融对照。

这些实验共享同一套数据划分、缺失值填补、标准化、训练策略和随机种子。报告中重点比较 `Test Acc`、`Macro F1`、`Balanced Acc`、混淆矩阵、专家负载和 `class -> expert` 热图。

本仓库已经跑完一组满分版 demo 实验，汇总见 [runs/summary.md](runs/summary.md)，报告稿见 [docs/final_report_full_score.md](docs/final_report_full_score.md)。

## 平台展示

最终呈现效果已经做成静态实验平台。推荐用项目内置 UTF-8 静态服务器启动，避免 Markdown 文档在浏览器中出现中文乱码：

```powershell
python scripts/serve_platform.py --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765/platform/index.html
```

平台中包含任务总览、模型结果表、K=1/K=2 专家分工热图、负载平衡消融、混淆矩阵、训练曲线、Kaggle 公开数据验证和最终提交材料入口。

## GitHub 参考项目

本项目参考了几个优秀开源项目的设计经验，详情见 [docs/github_references.md](docs/github_references.md)：

- [yandex-research/rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)：表格深度学习中强 MLP 基线的必要性。
- [manujosephv/pytorch_tabular](https://github.com/manujosephv/pytorch_tabular)：配置化实验、自动日志和统一训练入口。
- [lucidrains/mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)：稀疏 MoE 的 top-k 路由和专家负载分析思路。
