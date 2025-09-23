# 导入所有需要的库
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False             # 正常显示负号

def run_regression_task(df, features):
    """
    运行回归任务：预测PR处理时间
    """
    print("=" * 50)
    print("任务一：预测PR处理时间长短 (回归模型)")
    print("=" * 50)

    # 准备回归数据
    regression_df = df[df['merged_at'].notna()].copy()
    regression_df['ttc_hours'] = (regression_df['merged_at'] - regression_df['created_at']).dt.total_seconds() / 3600

    # 过滤异常值
    regression_df = regression_df[(regression_df['ttc_hours'] > 0) & (regression_df['ttc_hours'] < 24 * 30)]  # 30天内

    # 对齐特征
    X_reg = features.loc[regression_df.index]
    y_reg = regression_df['ttc_hours']

    # 按时间划分训练测试集
    regression_df = regression_df.sort_values('created_at')
    split_date = regression_df['created_at'].quantile(0.8)
    train_mask = regression_df['created_at'] < split_date

    X_train, X_test = X_reg[train_mask], X_reg[~train_mask]
    y_train, y_test = y_reg[train_mask], y_reg[~train_mask]

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"处理时间范围: {y_reg.min():.1f} - {y_reg.max():.1f} 小时")

    # 训练模型
    model = RandomForestRegressor(
        n_estimators=150, 
        random_state=42, 
        n_jobs=-1, 
        max_depth=8, 
        min_samples_leaf=5, 
        min_samples_split=10
    )
    model.fit(X_train, y_train)

    # 预测和评估
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n回归模型评估结果:")
    print(f"MAE: {mae:.2f} 小时")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} 小时")
    print(f"R²: {r2:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 预测vs真实散点图
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('真实处理时间 (小时)')
    axes[0].set_ylabel('预测处理时间 (小时)')
    axes[0].set_title('预测值 vs 真实值')

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_reg.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    axes[1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1].set_xlabel('特征重要性')
    axes[1].set_title('Top 10 重要特征')
    plt.tight_layout()
    plt.show()

    return model, X_test, y_test, y_pred


# 运行回归任务
# reg_model, X_test_reg, y_test_reg, y_pred_reg = run_regression_task(pr_df_clean, X)