import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
import os

# 1. 读取Excel文件（基础错误处理）
def read_excel_simple(file_path):
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        exit(1)
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"读取文件出错 {file_path}: {str(e)}")
        exit(1)

# 读取数据
df1 = read_excel_simple("1.xlsx")
df2 = read_excel_simple("2.xlsx")
df3 = read_excel_simple("3.xlsx")

# 2. 合并数据集
try:
    merged_df = pd.merge(
        pd.merge(df1, df2, on="number", how="inner"),
        df3, on="number", how="inner"
    )
    print(f"合并后的数据量: {len(merged_df)} 条")
    # 合并后删除title和author列
    cols_to_drop = ['title', 'author','name','body','state']
    # 只删除存在的列
    existing_cols = [col for col in cols_to_drop if col in merged_df.columns]
    if existing_cols:
        merged_df = merged_df.drop(columns=existing_cols)
        print(f"已删除列: {existing_cols}")
    else:
        print("未找到title或author列，无需删除")
except KeyError:
    print("错误: 数据中缺少'number'列")
    exit(1)

# 3. 基础特征处理（简化版）
def simple_feature_process(df):
    # 处理clustering_coefficient列：非数字值设为0
    if 'clustering_coefficient' in df.columns:
        # 尝试将列转换为数值类型，无法转换的设为NaN
        df['clustering_coefficient'] = pd.to_numeric(df['clustering_coefficient'], errors='coerce')
        # 统计非数字值的数量
        non_numeric_count = df['clustering_coefficient'].isna().sum()
        if non_numeric_count > 0:
            print(f"clustering_coefficient列中有 {non_numeric_count} 个非数字值，已设为0")
        # 将NaN值设为0
        df['clustering_coefficient'].fillna(0, inplace=True)

    # 时间处理（仅转换为datetime，不处理时区）
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # 简单文本特征
    text_cols = ['title', 'body']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
            df[f'{col}_word_count'] = df[col].apply(lambda x: len(str(x).split()))

    # 合并状态转换
    if 'merged' in df.columns:
        df['is_merged'] = df['merged'].astype(int, errors='ignore')

    return df

processed_df = simple_feature_process(merged_df.copy())

# 4. 分割训练集和测试集
if 'created_at' not in processed_df.columns:
    print("错误: 缺少'created_at'列，无法按时间分割")
    exit(1)

# 过滤无效时间
processed_df = processed_df[processed_df['created_at'].notna()].copy()

# 定义分割时间（不处理时区）
train_end = pd.to_datetime('2021-05-31T00:00:00Z')
test_start = pd.to_datetime('2021-06-01T00:00:00Z')
test_end = pd.to_datetime('2022-06-15T00:00:00Z')

# 分割数据
train_df = processed_df[processed_df['created_at'] <= train_end]
test_df = processed_df[(processed_df['created_at'] >= test_start) &
                       (processed_df['created_at'] <= test_end)]

print(f"训练集大小: {len(train_df)} 条")
print(f"测试集大小: {len(test_df)} 条")

#
# 5. 特征处理（只对非exclude_cols进行标准化）
# 定义不做处理但需要保留的列
exclude_cols = ['number', 'created_at', 'updated_at', 'merged_at', 'merged', 'is_merged']
# 需要标准化的列（除了exclude_cols之外的所有列）
feature_cols = [col for col in processed_df.columns if col not in exclude_cols]

if not feature_cols:
    print("错误: 没有需要标准化的特征列")
    exit(1)

print(f"将对以下列进行标准化: {feature_cols}")

# 提取需要标准化的特征并填充缺失值
X_train = train_df[feature_cols].fillna(0)  # 数值特征缺失值填0
X_test = test_df[feature_cols].fillna(0)

# 只对feature_cols进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义需要处理的时间列列表
time_cols = ['created_at', 'updated_at', 'merged_at']

for col in time_cols:
    if col in train_df.columns:
        # 先转换为datetime类型，再移除时区
        train_df[col] = pd.to_datetime(train_df[col], errors='coerce').dt.tz_localize(None)
        test_df[col] = pd.to_datetime(test_df[col], errors='coerce').dt.tz_localize(None)
# 6. 保存结果 - 包含exclude_cols（原始值）和标准化后的特征
try:

    # 训练集：合并不处理的列（原始值）和标准化后的特征
    train_processed = pd.concat([
        train_df[exclude_cols].reset_index(drop=True),  # 不处理的列，保留原始值
        pd.DataFrame(X_train_scaled, columns=feature_cols)  # 标准化后的特征
    ], axis=1)

    # 测试集：合并不处理的列（原始值）和标准化后的特征
    test_processed = pd.concat([
        test_df[exclude_cols].reset_index(drop=True),  # 不处理的列，保留原始值
        pd.DataFrame(X_test_scaled, columns=feature_cols)  # 标准化后的特征
    ], axis=1)

    # 保存到Excel
    train_processed.to_excel("train_features.xlsx", index=False)
    test_processed.to_excel("test_features.xlsx", index=False)
    print("数据已保存：train_features.xlsx 和test_features.xlsx")
    print(f"训练集包含 {len(exclude_cols)} 个不处理的列和 {len(feature_cols)} 个标准化后的特征列")
except Exception as e:
    print(f"保存文件出错: {str(e)}")
    exit(1)
