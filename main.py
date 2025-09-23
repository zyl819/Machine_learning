import pandas as pd
from process_data_3 import preprocess_data
from extract_feature_4 import extract_features_from_excel
from run_regression_5 import run_regression_task
from run_classification_6 import run_classification_task
def main():
    """
    主执行函数
    """
    print("开始PR数据分析实验...\n")

    # 步骤1: 加载并合并数据
    print("步骤1: 加载并合并Excel数据")
    df_features = pd.read_excel('data/PR_features.xlsx')
    df_info = pd.read_excel('data/PR_info.xlsx')
    df_author = pd.read_excel('data/author_features.xlsx')
    df = df_info.merge(df_features, on='number', how='left')
    df = df.merge(df_author, on='number', how='left')

    # 步骤2: 数据预处理
    print("\n步骤2: 数据预处理")
    pr_df_clean = preprocess_data(df)


    print("预处理后的原始数据预览：")
    print("字段名：", pr_df_clean.columns.tolist())
    # 3. 特征工程
    print("\n步骤3: 特征工程")
    X_features = extract_features_from_excel(pr_df_clean)

    # 4. 运行两个任务
    print("\n步骤4: 运行机器学习任务")

    # 任务一：回归
    reg_results = run_regression_task(pr_df_clean, X_features)

    # 任务二：分类
    clf_results = run_classification_task(pr_df_clean, X_features)

    print("\n实验完成！")
    print("请查看上面的评估结果和可视化图表")


# 执行主函数
if __name__ == "__main__":
    main()