import pandas as pd


def preprocess_data(df):
    """
    数据预处理和清洗
    """
    df_clean = df.copy()

    # 时间字段转换
    for col in ['created_at', 'merged_at', 'updated_at']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # 自动处理所有数值型缺失
    numeric_cols = df_clean.select_dtypes(include=[float, int]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(0)

    # 确保状态列一致性
    if 'state' in df_clean.columns:
        df_clean['state'] = df_clean['state'].str.lower()

    print("数据预处理完成")
    print(f"处理后的数据形状: {df_clean.shape}")

    return df_clean


# 数据预处理
# pr_df_clean = preprocess_data(pr_df)