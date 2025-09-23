# 导入所有需要的库
import pandas as pd
import numpy as np

def extract_features_from_excel(df):
    """
    从Excel数据中提取特征
    """
    features = pd.DataFrame(index=df.index)

     # 1. 文本关键词特征
    keywords = ['bug', 'document', 'feature', 'improve', 'refactor', 'test', 'fix', 'error', 'update']

    for keyword in keywords:
        features[f'has_{keyword}'] = (
                df['title'].astype(str).str.lower().str.contains(keyword) |
                df['body'].astype(str).str.lower().str.contains(keyword)
        ).astype(int)

    # 2. 文本长度特征
    features['title_length'] = df['title_length'].fillna(0)
    features['body_length'] = df['body_length'].fillna(0)

    # 3. 时间特征
    if 'created_at' in df.columns:
        features['created_hour'] = df['created_at'].dt.hour
        features['created_dayofweek'] = df['created_at'].dt.dayofweek
        features['created_month'] = df['created_at'].dt.month
        features['created_year'] = df['created_at'].dt.year

    # 4. PR基础统计特征
    numeric_features = ['comments', 'commits', 'additions', 'deletions', 'changed_files']
    for feat in numeric_features:
        if feat in df.columns:
            features[feat] = df[feat].fillna(0)


    # 8. directories (统计路径中的目录数)
    features['directories'] = df['directory_num'].fillna(0)
    

    # 9. language_types
    features['language_types'] = df['language_num'].fillna(0)

    # 10. file_types
    features['file_types'] = df['file_type'].fillna(0)

    # 11-12. lines_added, lines_deleted
    if 'lines_added' in df.columns:
        features['lines_added'] = df['lines_added'].fillna(0)
    if 'lines_deleted' in df.columns:
        features['lines_deleted'] = df['lines_deleted'].fillna(0)

    # 13-15. segs_added, segs_deleted, segs_changed
    for seg in ['segs_added', 'segs_deleted', 'segs_updated']:
        if seg in df.columns:
            features[seg] = df[seg].fillna(0)

    # 16-18. files_added, files_deleted, files_changed
    for f in ['files_added', 'files_deleted', 'files_updated']:
        if f in df.columns:
            features[f] = df[f].fillna(0)

    # 19. file_developer
    if 'file_developer' in df.columns:
        features['file_developer'] = df['file_developer'].fillna(0)

    # 20. change_num
    features['change_num'] = df['change_num'].fillna(0)

    # 21. files_modified
    if 'files_modified' in df.columns:
        features['files_modified'] = df['files_modified'].fillna(0)

    # 22. is_core_member
    features['is_core_member'] = df['k_coreness'].astype(int)

    # 24. prev_PRs
    features['prev_PRs'] = df['experience'].fillna(0)

    # 25. title_words
    features['title_words'] = df['title'].astype(str).str.split().str.len()

    # 26. body_words
    features['body_words'] = df['body'].astype(str).str.split().str.len()

    # 其余特征可按类似方式添加

    # 处理无穷大和缺失
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())

    print(f"特征工程完成，共生成 {features.shape[1]} 个特征")
    print("特征列表:", features.columns.tolist())

    return features


# 提取特征
# X = extract_features_from_excel(pr_df_clean)