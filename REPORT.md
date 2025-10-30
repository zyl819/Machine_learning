## 实验报告：基于多仓库的 Pull Request 处理时间与结局预测

### 1. 任务一：预测 Pull Request 处理时间（回归）
- **问题与数据**
  - **任务定义**: 预测 PR 从创建到处理完成的用时（小时）。当前实现以已合并 PR 的 `merged_at - created_at` 作为目标。
  - **数据来源与仓库数**: 通过 GitHub REST API 抓取多个开源仓库（例如 `owner1/repo1, owner2/repo2`）；亦支持本地 Excel 数据（`data/*.xlsx`）。
  - **时间切分与防止泄漏**: 按 `created_at` 排序，使用 80% 分位作为时间切分点：训练集 `< split_date`，测试集 `>= split_date`，避免未来信息泄漏到过去。

- **特征工程**
  - **文本关键词二元特征（标题/正文）**: `has_bug/document/feature/improve/refactor/test/fix/error/update`（字符串包含检测，大小写不敏感）。
  - **文本长度与词数**: `title_length, body_length, title_words, body_words`。
  - **目录/语言/文件类型结构**: `directories`（唯一目录数），`language_types`（由扩展名映射推断语言的唯一数），`file_types`（扩展名种类数）。
  - **修改规模**: `additions, deletions, changed_files, lines_added, lines_deleted, files_added, files_deleted, files_updated/files_modified`。
  - **作者经验与活跃度（占位或可扩展）**: `is_core_member(k_coreness→int), prev_PRs(experience)` 
  - **时间派生**: `created_hour, created_dayofweek, created_month, created_year`。
  - **清洗与缺失处理**:
    - 预处理：`created_at/merged_at/updated_at/closed_at` 统一转为 datetime；数值缺失填 0；`state` 统一为小写。
    - 特征阶段：将 inf→NaN，并以列中位数填充；防止从 CSV 读取造成的时间列为字符串，主流程会再次预处理保证类型正确。

- **模型与方法**
  - **模型**: RandomForestRegressor。
  - **参数**: `n_estimators=150, max_depth=8, min_samples_leaf=5, min_samples_split=10, n_jobs=-1, random_state=42`。
  - **理由**: 对特征尺度鲁棒、可拟合非线性与交互、具备特征重要性、对中小规模数据表现稳定，作为强基线。

- **结果与分析（示例）**
  - **指标**: 报告 `MAE, MSE, RMSE, R²`。
  - **可视化**（具体见程序运行结果）:
  - ![img.png](img.png)
    - 预测 vs 真实散点图（含 45 度参考线），观察整体偏差与长尾。
    - 特征重要性 Top-10 条形图，洞察关键驱动因素（通常规模与结构类特征贡献较大）。
  - **现象与风险**: 样本量过少或时间过滤过紧会导致训练集过小；时长分布长尾会抬升 RMSE，可考虑对数变换或稳健指标。

- **结论与建议**
  - **结论**: 改动规模与结构复杂度（目录/语言/文件类型数）对处理时长影响显著；文本中体现的“修复/测试”等意图和时间因素亦有影响。
  - **建议**: 将超大/跨多目录 PR 拆分；用更清晰的模板与测试说明；引入自动化检查减少返工。

# 特征映射表（仅含代码已处理数据）



| 文档特征分类             | 文档特征名       | 代码映射字段                                                 | 数据处理方式                                                 |
| ------------------------ | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. 项目（Project）       | language\_num    | `language_types`                                             | 从数据集中提取`language_num`列，填充缺失值为 0               |
|                          | change\_num      | `change_num`                                                 | 从数据集中提取`change_num`列，填充缺失值为 0                 |
| 2. PR 作者（Author）     | experience       | `prev_PRs`                                                   | 从数据集中提取`experience`列，填充缺失值为 0                 |
|                          | k\_coreness      | `is_core_member`                                             | 从数据集中提取`k_coreness`列，转换为 int 类型（0/1），缺失值补 0 |
| 3. PR 评审者（Reviewer） | -                | -                                                            | 代码中未处理评审者相关特征，暂不映射                         |
| 4. 代码变更（Change）    | directory\_num   | `directories`                                                | 从数据集中提取`directory_num`列，填充缺失值为 0              |
|                          | language\_num    | `language_types`                                             | 从数据集中提取`language_num`列，填充缺失值为 0               |
|                          | file\_type       | `file_types`                                                 | 从数据集中提取`file_type`列，填充缺失值为 0                  |
|                          | has\_test        | `has_test`                                                   | 检测`title`和`body`中是否包含 "test" 关键词，转换为 int（0/1） |
|                          | has\_feature     | `has_feature`                                                | 检测`title`和`body`中是否包含 "feature" 关键词，转换为 int（0/1） |
|                          | has\_bug         | `has_bug`                                                    | 检测`title`和`body`中是否包含 "bug" 关键词，转换为 int（0/1） |
|                          | has\_document    | `has_document`                                               | 检测`title`和`body`中是否包含 "document" 关键词，转换为 int（0/1） |
|                          | has\_improve     | `has_improve`                                                | 检测`title`和`body`中是否包含 "improve" 关键词，转换为 int（0/1） |
|                          | has\_refactor    | `has_refactor`                                               | 检测`title`和`body`中是否包含 "refactor" 关键词，转换为 int（0/1） |
|                          | subject\_length  | `title_length`/`title_words`                                 | 分别统计`title`的长度（`title_length`）和单词数（`title_words`），缺失值补 0 |
|                          | message\_length  | `body_length`/`body_words`                                   | 分别统计`body`的长度（`body_length`）和单词数（`body_words`），缺失值补 0 |
|                          | lines\_added     | `lines_added`                                                | 从数据集中提取`lines_added`列，填充缺失值为 0                |
|                          | lines\_deleted   | `lines_deleted`                                              | 从数据集中提取`lines_deleted`列，填充缺失值为 0              |
|                          | segs\_added      | `segs_added`                                                 | 从数据集中提取`segs_added`列，填充缺失值为 0                 |
|                          | segs\_deleted    | `segs_deleted`                                               | 从数据集中提取`segs_deleted`列，填充缺失值为 0               |
|                          | segs\_updated    | `segs_updated`                                               | 从数据集中提取`segs_updated`列，填充缺失值为 0               |
|                          | files\_added     | `files_added`                                                | 从数据集中提取`files_added`列，填充缺失值为 0                |
|                          | files\_deleted   | `files_deleted`                                              | 从数据集中提取`files_deleted`列，填充缺失值为 0              |
|                          | files\_updated   | `files_updated`                                              | 从数据集中提取`files_updated`列，填充缺失值为 0              |
|                          | comment\_num     | `comments`                                                   | 从数据集中提取`comments`列（对应文档`comment_num`），填充缺失值为 0 |
|                          | 时间相关衍生特征 | `created_hour`/`created_dayofweek`/`created_month`/`created_year` | 从`created_at`字段提取小时、星期、月份、年份信息，无缺失值（基于时间字段推导） |
|                          | 基础统计特征     | `commits`/`additions`/`deletions`/`changed_files`            | 从数据集中提取对应列，填充缺失值为 0（`additions`/`deletions`可辅助反映代码变更规模） |

### 2. 任务二：预测 Pull Request 结局（是否合并，二分类）
- **问题与数据**
  - **任务定义**: 预测 PR 是否被合并（标签 `merged` 布尔）。
  - **时间切分**: 与任务一一致，按时间分割避免泄漏。

- **特征工程**
  - 复用任务一全部特征；`merged` 字段兼容字符串/布尔并统一为布尔；输出阳性比例以识别不平衡。

- **模型与方法**
  - **模型**: RandomForestClassifier。
  - **参数**: `n_estimators=100, n_jobs=-1, random_state=42`；若类别不平衡（阳性比例 <0.3 或 >0.7）自动启用 `class_weight='balanced'`。
  - **理由**: 强基线、鲁棒、可解释（重要性）。

- **结果与分析（示例）**
  - **指标**: Accuracy，配合分类报告（Precision/Recall/F1）与混淆矩阵。
  - **可视化（见程序运行结果）**: ROC/PR 曲线与阈值分析；特征重要性条形图。
  - **风险**: 高不平衡下 Accuracy 可能虚高，应关注 Recall/F1；不同仓库合并偏好影响泛化，建议跨仓库验证。

- **结论与建议**
  - **结论**: 小而聚焦的改动、更少跨目录/语言、明确测试/修复意图与较高历史信誉（需完善的 `experience/k_coreness`）提升合入率。
  - **建议**: 拆分大型变更；强化贡献指南的测试与文档要求；引入审阅者推荐与模板优化。

### 3. 抓取数据说明
- **使用的 GitHub REST API（v3）**
  - PR 列表：`GET /repos/{owner}/{repo}/pulls?state=all&sort=created&direction=asc`（分页 `per_page=100`）。
  - PR 详情：`GET /repos/{owner}/{repo}/pulls/{number}`（含 commits/comment 计数）。
  - PR 文件：`GET /repos/{owner}/{repo}/pulls/{number}/files`（每文件 additions/deletions/status/path）。
  - 可选：`GET /repos/{owner}/{repo}/pulls/{number}/commits`（较慢，默认跳过）。

- **速率限制与并发策略**
  - 连接复用：`requests.Session` 降低 TLS 握手成本。
  - 并发：`ThreadPoolExecutor` 并行抓取详情与文件（默认 8 线程，可调）。
  - 分页：循环 `page=1..n` 直至不足 100 条。
  - 限流/失败重试：对 403/429/502/503/504 进行指数退避；如 `X-RateLimit-Remaining=0`，依据 `X-RateLimit-Reset` 休眠后继续。
  - 可配置：`max_prs`（限制数量）、`since/until`（时间窗口）、`skip_commits`（默认 True）、`workers`（线程数）。
  - 进度：完成每个 PR 回调一次，打印“进度 k/total”。

- **数据落盘与复用**
  - 每仓库保存：`{owner}_{repo}_raw.parquet`（或 CSV 兜底）、`{owner}_{repo}_clean.parquet`、`{owner}_{repo}_features.parquet`、`{owner}_{repo}_metrics.txt`。
  - 实验主流程（`main.py`）优先读取 `clean/features`；若缺失则由 `raw` 重新生成后缓存；即使加载了 `clean/features`，也会再次预处理，保证时间列为 datetime。


