from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_classification_task(df, features):
    """
    运行分类任务并保存模型
    """
    print("="*50)
    print("任务二：预测PR合并与否 (分类模型)")
    print("="*50)

    # 只处理已关闭（merged_at 不为空）的PR
    classification_df = df.copy()

    # 检查 merged 列
    if 'merged' not in classification_df.columns:
        print("数据中没有 merged 列")
        return None

    # merged 字段类型转换，兼容字符串和布尔
    if classification_df['merged'].dtype == object:
        # 兼容字符串 'True'/'False'/'TRUE'/'FALSE' 及布尔
        classification_df['merged'] = classification_df['merged'].map(lambda x: True if str(x).lower() == 'true' else False if str(x).lower() == 'false' else x)

    # 再次检查类型
    print(f"转换后 merged 列类型: {classification_df['merged'].dtype}")
    print(f"转换后 merged 列唯一值: {classification_df['merged'].unique()}")

    # merged 应该是布尔类型，如果不是，尝试转换
    if classification_df['merged'].dtype != 'bool':
        classification_df['merged'] = classification_df['merged'].astype(bool)

    # 检查类别分布
    class_distribution = classification_df['merged'].value_counts()
    print(f"处理后的类别分布:\n{class_distribution}")
    print(f"merged列唯一值: {classification_df['merged'].unique()}")
    print(f"merged列取值分布:\n{classification_df['merged'].value_counts()}")

    if len(class_distribution) < 2:
        print("警告：处理后仍然只有一个类别")
        return None

    # 对齐特征
    X_clf = features.loc[classification_df.index]
    y_clf = classification_df['merged']

    print(f"特征矩阵形状: {X_clf.shape}")
    print(f"目标变量形状: {y_clf.shape}")

    # 按时间划分
    classification_df = classification_df.sort_values('created_at')
    split_date = classification_df['created_at'].quantile(0.8)
    train_mask = classification_df['created_at'] < split_date

    X_train, X_test = X_clf[train_mask], X_clf[~train_mask]
    y_train, y_test = y_clf[train_mask], y_clf[~train_mask]

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"合并比例 - 训练集: {y_train.mean():.3f}, 测试集: {y_test.mean():.3f}")

    # 处理类别不平衡
    if y_train.mean() < 0.3 or y_train.mean() > 0.7:
        print("注意：数据存在类别不平衡问题")
        class_weight = 'balanced'
    else:
        class_weight = None

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight)
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred = model.predict(X_test)

    # 安全地获取预测概率
    try:
        y_pred_proba = model.predict_proba(X_test)
        print(f"预测概率矩阵形状: {y_pred_proba.shape}")
        if y_pred_proba.shape[1] > 1:
            y_pred_proba_1 = y_pred_proba[:, 1]
        else:
            y_pred_proba_1 = y_pred_proba[:, 0]
    except Exception as e:
        print(f"获取预测概率时出错: {e}")
        y_pred_proba_1 = None

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n准确率: {accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    performance_metrics = {
        'accuracy': accuracy,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'positive_ratio_train': y_train.mean(),
        'positive_ratio_test': y_test.mean()
    }

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba_1,
        'metrics': performance_metrics
    }