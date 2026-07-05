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

当前机器的 Python 环境缺少 PyTorch、pandas、scikit-learn、matplotlib、PyYAML 和 joblib。建议在项目目录下新建虚拟环境后安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果只想先检查项目脚本和 demo 数据生成，当前环境已有 NumPy，可以直接运行：

```powershell
python scripts/generate_demo_data.py --out data/demo_pickleball.csv --n-samples 600
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

## 公开数据源非 PyTorch 测试

根据“不要用本项目本地 PyTorch 流水线”的要求，项目已新增一个独立公开数据测试脚本：

```powershell
python scripts/public_score_tennis_test.py --out-dir public_data_test/score_tennis --samples-per-class 2000
```

该脚本直接下载 SCORE Grand Slam Tennis Shot-Level Data 的公开 `.csv.gz`，使用 scikit-learn 的 Logistic Regression 和 Random Forest 做测试，不调用 `run_mlp.py`、`run_moe.py` 或 `src/pickleball_moe/models/*`。

结果见 [docs/public_data_test_report.md](docs/public_data_test_report.md) 和 [public_data_test/score_tennis/summary.md](public_data_test/score_tennis/summary.md)。

用户下载 Kaggle PKLMart 匹克球数据后，又新增了更贴题的真实匹克球非 PyTorch 测试：

```powershell
python scripts/public_pklmart_test.py --zip "C:\Users\13342\Downloads\archive (1).zip" --out-dir public_data_test/pklmart_pickleball --samples-per-class 8000
```

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

最终呈现效果已经做成静态实验平台，入口是：

```text
platform/index.html
```

在 Windows 上可以直接双击打开，也可以用浏览器打开这个文件。平台中包含任务总览、模型结果表、K=1/K=2 专家分工热图、负载平衡消融、混淆矩阵、训练曲线和提交材料链接。

## GitHub 参考项目

本项目参考了几个优秀开源项目的设计经验，详情见 [docs/github_references.md](docs/github_references.md)：

- [yandex-research/rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)：表格深度学习中强 MLP 基线的必要性。
- [manujosephv/pytorch_tabular](https://github.com/manujosephv/pytorch_tabular)：配置化实验、自动日志和统一训练入口。
- [lucidrains/mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)：稀疏 MoE 的 top-k 路由和专家负载分析思路。
