import os
import pandas as pd
from process_data_3 import preprocess_data
from extract_feature_4 import extract_features_from_excel
from run_regression_5 import run_regression_task
from run_classification_6 import run_classification_task

# 尝试从 .env 加载环境变量（可选）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def main():
    """
    主执行函数
    """
    print("开始PR数据分析实验...\n")

    repos_env = os.getenv("GITHUB_REPOS")  # e.g. "owner1/repo1,owner2/repo2"
    output_dir = os.getenv("OUTPUT_DIR", "data_out")

    if repos_env:
        os.makedirs(output_dir, exist_ok=True)
        repos = [r.strip() for r in repos_env.split(',') if r.strip()]
        all_results = []
        total = len(repos)
        for idx, repo_full in enumerate(repos, start=1):
            owner, repo = repo_full.split('/')
            print(f"\n>>> [{idx}/{total}] 处理仓库: {owner}/{repo}")

            # 优先使用已存在的 clean/features；否则根据 raw 生成
            clean_path_parquet = os.path.join(output_dir, f"{owner}_{repo}_clean.parquet")
            feat_path_parquet = os.path.join(output_dir, f"{owner}_{repo}_features.parquet")
            raw_path_parquet = os.path.join(output_dir, f"{owner}_{repo}_raw.parquet")

            clean_path_csv = clean_path_parquet.replace('.parquet', '.csv')
            feat_path_csv = feat_path_parquet.replace('.parquet', '.csv')
            raw_path_csv = raw_path_parquet.replace('.parquet', '.csv')

            pr_df_clean = None
            X_features = None

            # 读取 clean/features
            if os.path.exists(clean_path_parquet):
                pr_df_clean = pd.read_parquet(clean_path_parquet)
            elif os.path.exists(clean_path_csv):
                pr_df_clean = pd.read_csv(clean_path_csv)

            if os.path.exists(feat_path_parquet):
                X_features = pd.read_parquet(feat_path_parquet)
            elif os.path.exists(feat_path_csv):
                X_features = pd.read_csv(feat_path_csv)

            # 如缺失，则从 raw 构建
            if pr_df_clean is None or X_features is None:
                if os.path.exists(raw_path_parquet):
                    df_raw = pd.read_parquet(raw_path_parquet)
                elif os.path.exists(raw_path_csv):
                    df_raw = pd.read_csv(raw_path_csv)
                else:
                    print(f"未找到本地数据文件：{raw_path_parquet} 或 {raw_path_csv}，跳过该仓库")
                    continue

                print("\n步骤2: 数据预处理")
                pr_df_clean = preprocess_data(df_raw)
                print("预处理后的原始数据预览：")
                print("字段名：", pr_df_clean.columns.tolist())

                print("\n步骤3: 特征工程")
                X_features = extract_features_from_excel(pr_df_clean)

                # 缓存落盘，便于下次直接使用
                try:
                    pr_df_clean.to_parquet(clean_path_parquet, index=False)
                    X_features.to_parquet(feat_path_parquet, index=False)
                except Exception:
                    pr_df_clean.to_csv(clean_path_csv, index=False)
                    X_features.to_csv(feat_path_csv)
            else:
                print("已检测到本地 clean/features 文件，重新预处理确保时间类型正确")
                # 强制重新预处理，确保时间列转换为 datetime
                pr_df_clean = preprocess_data(pr_df_clean)
                X_features = extract_features_from_excel(pr_df_clean)
                print("预处理后的原始数据预览：")
                print("字段名：", pr_df_clean.columns.tolist())
            print("\n步骤4: 运行机器学习任务")
            reg_results = run_regression_task(pr_df_clean, X_features)
            clf_results = run_classification_task(pr_df_clean, X_features)

            # 保存结果（简单示例，可扩展更多指标）
            metrics_path = os.path.join(output_dir, f"{owner}_{repo}_metrics.txt")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(f"repo: {repo_full}\n")
                try:
                    f.write("[Regression]\n")
                    if isinstance(reg_results, tuple) and len(reg_results) >= 4:
                        f.write(f"X_test_shape: {reg_results[1].shape}\n")
                        f.write(f"y_test_size: {reg_results[2].shape[0]}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Regression error: {e}\n")
                try:
                    f.write("[Classification]\n")
                    if isinstance(clf_results, dict) and 'metrics' in clf_results:
                        m = clf_results['metrics']
                        for k, v in m.items():
                            f.write(f"{k}: {v}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Classification error: {e}\n")

            repo_metrics = {
                'repo': repo_full,
                'reg_MAE': getattr(reg_results, 'mae', None) if isinstance(reg_results, dict) else None,
            }
            all_results.append(repo_metrics)

        print("\n多仓库实验完成！")

    # 传统Excel路径
    print("步骤1: 加载并合并Excel数据")
    df_features = pd.read_excel('data/PR_features.xlsx')
    df_info = pd.read_excel('data/PR_info.xlsx')
    df_author = pd.read_excel('data/author_features.xlsx')
    df = df_info.merge(df_features, on='number', how='left')
    df = df.merge(df_author, on='number', how='left')

    print("\n步骤2: 数据预处理")
    pr_df_clean = preprocess_data(df)

    print("预处理后的原始数据预览：")
    print("字段名：", pr_df_clean.columns.tolist())
    print("\n步骤3: 特征工程")
    X_features = extract_features_from_excel(pr_df_clean)

    print("\n步骤4: 运行机器学习任务")
    reg_results = run_regression_task(pr_df_clean, X_features)
    clf_results = run_classification_task(pr_df_clean, X_features)

    print("\n实验完成！")
    print("请查看上面的评估结果和可视化图表")


# 执行主函数
if __name__ == "__main__":
    main()