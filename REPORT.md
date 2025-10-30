## 实验报告（分为 任务一 与 任务二，基于当前代码库）

说明：本报告直接基于仓库中的实现文件（`main.py`、`data/get_data.py`、`models/*` 等）编写。为便于评审与提交，我将内容分为：任务一（预测 PR 处理时间，回归）和任务二（预测 PR 是否被合并，二分类/概率预测），并列出数据、特征、模型、评估与复现说明。

### 目录
- 任务一：预测 Pull Request 处理时间（回归）
- 任务二：预测 Pull Request 是否被合并（分类 / 概率）
- 公共部分：特征、数据处理、实现与运行说明、可扩展方向与小结

---

## 任务一：预测 Pull Request 处理时间（回归）

### 1. 任务定义
- 目标变量：PR 从创建到合并/关闭的耗时（连续值，常用单位：小时或天）。在本项目中对应列名为 `avg_duration_y`（见 `main.py`）。
- 任务类型：监督回归问题。为应对偏态分布，实验中可选择对数变换（例如 log(1 + t)）后训练再反变换评估。

### 2. 数据与时间切分
- 数据来源：仓库根目录的 `1.xlsx`、`2.xlsx`、`3.xlsx` 被 `data/get_data.py` 合并为 `train_features.xlsx` 与 `test_features.xlsx`。
- 时间切分：`data/get_data.py` 使用 `created_at` 字段，训练截止 `2021-05-31`，测试为 `2021-06-01` 至 `2022-06-15`。该时间切分保证测试集晚于训练集，从时间维度避免泄露。

### 3. 特征（可用于回归）
- 数值特征：文件中大多数数值列（经 `StandardScaler` 标准化）；例如行数改动、文件改动数、作者/仓库历史量化指标（若存在）。
- 时间特征：创建时间拆分得到的小时、星期、月份（目前 `get_data.py` 保留 `created_at`，可在后续扩展）。
- 文本长度特征：`title_word_count`、`body_word_count`（`get_data.py` 已生成）。
- 其他：`clustering_coefficient`（已数值化或设为 0）、是否为仓库成员等可派生特征。

建议：对长尾目标做对数变换；对高基数类别（作者 id）使用 embedding 或 target encoding（在当前脚本中未实现）。

### 4. 模型与训练（回归）
- 可直接使用 `models/wide_deep.py`（Wide&Deep）在 `main.py` 中训练：回归使用 MSELoss，optimizer=Adam(lr=1e-3)，训练 50 epoch，batch_size=32。
- 也可使用树模型（XGBoost）作为 baseline（仓库当前未包含 XGBoost 实现，建议补充）。

训练细节注：`main.py` 对回归标签使用 MinMaxScaler 缩放，模型输出反变换后评估原始尺度的 MAE/RMSE/R²。

### 5. 评估指标与可视化
- 主指标：MAE（平均绝对误差）；次指标：RMSE、R²、误差分位数（P50、P90）。
- 可视化：预测 vs 真实散点图、残差直方图、学习曲线（训练/验证 loss）、按仓库/作者分组的 MAE 条形图。

### 6. 常见问题与改进方向（任务一）
- 长尾问题：对长时间未处理的 PR 建议使用截断/分段模型或鲁棒损失（Huber）。
- 文本语义：若标题/描述能显著影响耗时，建议在 `get_data.py` 中加入 sentence-BERT embedding。

---

## 任务二：预测 Pull Request 是否被合并（分类 / 概率预测）

### 1. 任务定义
- 目标变量：PR 是否最终被合并（布尔值），在数据中为 `is_merged`（`get_data.py` 将 `merged` 转为 `is_merged`）。
- 任务类型：二分类（或概率输出），可用于优先级排序与资源调度。

### 2. 数据与时间切分
- 使用与任务一相同的 `train_features.xlsx` / `test_features.xlsx`，时间切分保持一致，避免未来信息泄露。所有用于训练的特征均需为创建时刻可得的信息。

### 3. 特征（可用于分类）
- 与回归共享大部分特征：数值特征、时间特征、文本长度。
- 文本语义特征（若加入 embedding）在分类上通常很有用，例如是否包含“fix”、“bug”、“urgent”等关键词可能影响合并概率。
- 作者与仓库历史：作者过去的合并率、仓库活跃度、是否为仓库成员等在分类任务中特别重要。

### 4. 模型与训练（分类 / 多任务）
- 多任务模型：`models/shared_bottom.py` 和 `models/mmoe.py` 已在仓库中实现，能够同时输出回归与分类结果。在 `main.py` 中，SharedBottom 使用 joint loss = MSE(reg) + 0.5 * BCE(cls)；MMoE 使用 loss = MSE + BCE。
- 单任务分类：可单独训练一个二分类网络或使用 LightGBM/XGBoost（未包含但建议作为 baseline）。

训练细节注：分类 head 的输出通过 sigmoid 转为概率，threshold=0.5 可作为默认预测阈值，但应根据 Precision/Recall 曲线选择最佳阈值以匹配实际策略。

### 5. 评估指标与可视化（任务二）
- 指标：Accuracy、Precision、Recall、F1、AUC（建议记录概率输出的 AUC）。
- 可视化：ROC 曲线、Precision-Recall 曲线、概率分布直方图、按作者/仓库分组的混淆矩阵或指标表。

### 6. 模型落地建议（任务二）
- 将合并概率显示在 PR 界面，结合预测置信度和重要性指标帮助维护者优先审查。对低置信度但高影响的 PR 推荐人工复核。

---

## 公共部分：实现、运行与复现

### 项目关键文件（快速索引）
- `data/get_data.py`：读取 `1.xlsx`/`2.xlsx`/`3.xlsx`，合并并生成 `train_features.xlsx`、`test_features.xlsx`（包含标准化后的数值特征与部分衍生特征）。
- `main.py`：训练/评估入口，支持 `--model wide_deep/shared_bottom/mmoe/all`，会在训练完成后保存 `checkpoints/{model_type}_best.pth` 并在测试集上打印评估结果。
- `models/wide_deep.py`、`models/shared_bottom.py`、`models/mmoe.py`：三种可直接运行的 PyTorch 模型实现。

### 运行示例（在包含 `1.xlsx`/`2.xlsx`/`3.xlsx` 的目录下）
```powershell
# 生成特征文件
python data/get_data.py

# 训练并评估 Wide&Deep（回归）
python main.py --model wide_deep

# 训练并评估 Shared-Bottom（多任务 回归+分类）
python main.py --model shared_bottom

# 训练并评估 MMoE（多任务）或一次性运行所有
python main.py --model mmoe
python main.py --model all
```

### 依赖（请在 `requirements.txt` 中补齐版本）
- 最小依赖：Python, pandas, numpy, scikit-learn, openpyxl, torch

### 检查点与输出
- 模型权重保存在 `checkpoints/{model_type}_best.pth`，测试评估在训练后打印到控制台。建议在训练脚本中保存每 epoch 的 val loss/指标以便绘制学习曲线。

---

## 结果与分析（占位与说明）
说明：当前仓库未包含已运行过的完整实验数值或图表。本节为插入真实运行结果保留位置，并给出如何生成这些结果的说明：

- 先运行 `python main.py --model <model>` 获取基本指标（MAE/RMSE/R² 与 Accuracy/Precision/Recall/F1）。
- 为报告生成图表：在训练过程中记录训练/验证 loss 与指标并保存为 CSV；使用 `scripts/plot_results.py`（建议添加）绘制预测 vs 真实、残差图、学习曲线、SHAP 特征重要性等。

示例占位（请替换为实验输出）
- Wide&Deep: MAE = -- hours, RMSE = --, R² = --
- Shared-Bottom: MAE = --, Accuracy = --
- MMoE: MAE = --, AUC = --

错误分析建议：挑选高误差或误判样本人工审查，识别需要更多上下文或标签噪声的情况。

---

## 可扩展方向（下一步工作建议）
- 在 `get_data.py` 中加入文本 embedding（sentence-transformers）以增强语义特征。
- 将时间切分与超参写入配置文件，使用脚本批量跑实验并收集结果（实验自动化）。
- 增加树模型基线（XGBoost/LightGBM）并使用 SHAP 对模型做解释性分析。
- 若生产化：搭建在线服务输出预测并做 drift 检测与定期重训练。

---

## 小结
- 本报告已按任务一（回归）与任务二（分类）分开阐述，结合仓库中现有实现给出复现步骤、评估指标与落地建议。
- 如果你确认，我可以：
  1) 在当前环境运行一次端到端实验并把真实数值与图表填回本报告（需安装依赖并允许执行训练）；
  2) 为你添加 `eval.py` 与 `scripts/plot_results.py`，自动保存评估结果与生成图表；
  3) 将报告导出为 PDF（需 pandoc/jupyter 或本地工具支持）。

