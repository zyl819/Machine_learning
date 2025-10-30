import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from models.wide_deep import WideDeep
from models.shared_bottom import SharedBottom
from models.mmoe import MMoE
import matplotlib.pyplot as plt
import os


def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"回归任务评估：MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return [mae, rmse, r2]


def evaluate_classification(y_true, y_pred):
    y_pred_cls = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_cls)
    precision = precision_score(y_true, y_pred_cls, zero_division=0)
    recall = recall_score(y_true, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true, y_pred_cls, zero_division=0)
    print(f"分类任务评估：Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return [acc, precision, recall, f1]


def plot_metrics(reg_metrics, cls_metrics, model_names, save_path="metrics_comparison.png"):
    """
    可视化回归和分类任务的指标
    """
    metrics_names_reg = ['MAE', 'RMSE', 'R²']
    metrics_names_cls = ['Accuracy', 'Precision', 'Recall', 'F1']

    plt.figure(figsize=(12, 5))
    for i, metric_name in enumerate(metrics_names_reg):
        plt.bar(np.arange(len(model_names)) + i * 0.25, [m[i] for m in reg_metrics], width=0.25, label=metric_name)
    plt.xticks(np.arange(len(model_names)) + 0.25, model_names)
    plt.title('Regression Metrics Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_metrics.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, metric_name in enumerate(metrics_names_cls):
        plt.bar(np.arange(len(model_names)) + i * 0.25, [m[i] for m in cls_metrics], width=0.25, label=metric_name)
    plt.xticks(np.arange(len(model_names)) + 0.25, model_names)
    plt.title('Classification Metrics Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig("classification_metrics.png", dpi=300)
    plt.close()

    print("✅ 指标可视化已保存为 regression_metrics.png 和 classification_metrics.png")


def main(model_type):
    # ------------------------------------------------------
    # 1. 数据加载与预处理
    # ------------------------------------------------------
    train_df = pd.read_excel("train_features.xlsx")
    test_df = pd.read_excel("test_features.xlsx")

    exclude_cols = ['number', 'created_at', 'updated_at', 'merged_at', 'merged']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols + ['is_merged', 'avg_duration_y']]

    reg_label = 'avg_duration_y'
    cls_label = 'is_merged'

    X_train = train_df[feature_cols].values
    y_train_reg = train_df[reg_label].values.reshape(-1, 1)
    y_train_cls = train_df[cls_label].values.reshape(-1, 1)
    X_test = test_df[feature_cols].values
    y_test_reg = test_df[reg_label].values.reshape(-1, 1)
    y_test_cls = test_df[cls_label].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg_scaler = MinMaxScaler()
    y_train_reg_scaled = reg_scaler.fit_transform(y_train_reg)
    y_test_reg_scaled = reg_scaler.transform(y_test_reg)

    input_dim = X_train.shape[1]

    X_train_t = to_tensor(X_train)
    X_test_t = to_tensor(X_test)
    y_train_reg_t = to_tensor(y_train_reg_scaled)
    y_test_reg_t = to_tensor(y_test_reg_scaled)
    y_train_cls_t = to_tensor(y_train_cls)
    y_test_cls_t = to_tensor(y_test_cls)

    os.makedirs("checkpoints", exist_ok=True)

    reg_results, cls_results, model_names = [], [], []

    def run_model(model_type):
        if model_type == 'wide_deep':
            print("\n=== Wide&Deep模型（回归任务） ===")
            model = WideDeep(input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_reg_t), batch_size=32, shuffle=True)
            model.train()
            for epoch in range(50):
                for x, y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                y_pred_reg_scaled = model(X_test_t).numpy()
            y_pred_reg = reg_scaler.inverse_transform(y_pred_reg_scaled)
            reg_metrics = evaluate_regression(y_test_reg, y_pred_reg)
            torch.save(model.state_dict(), f"checkpoints/{model_type}_best.pth")
            reg_results.append(reg_metrics)
            cls_results.append([0, 0, 0, 0])  # 无分类任务
            model_names.append("Wide&Deep")

        elif model_type == 'shared_bottom':
            print("\n=== Shared-Bottom模型（多任务） ===")
            model = SharedBottom(input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = [torch.nn.MSELoss(), torch.nn.BCELoss()]
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_reg_t, y_train_cls_t), batch_size=32,
                                      shuffle=True)
            model.train()
            for epoch in range(50):
                for x, y_reg, y_cls in train_loader:
                    optimizer.zero_grad()
                    reg_out, cls_out = model(x)
                    loss = criterion[0](reg_out, y_reg) + 0.5 * criterion[1](cls_out, y_cls)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                reg_out, cls_out = model(X_test_t)
                y_pred_reg = reg_scaler.inverse_transform(reg_out.numpy())
                y_pred_cls = cls_out.numpy()
            reg_metrics = evaluate_regression(y_test_reg, y_pred_reg)
            cls_metrics = evaluate_classification(y_test_cls, y_pred_cls)
            torch.save(model.state_dict(), f"checkpoints/{model_type}_best.pth")
            reg_results.append(reg_metrics)
            cls_results.append(cls_metrics)
            model_names.append("Shared-Bottom")

        elif model_type == 'mmoe':
            print("\n=== MMoE模型（多任务） ===")
            model = MMoE(input_dim, num_experts=8, num_tasks=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = [torch.nn.MSELoss(), torch.nn.BCELoss()]
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_reg_t, y_train_cls_t), batch_size=32,
                                      shuffle=True)
            model.train()
            for epoch in range(50):
                for x, y_reg, y_cls in train_loader:
                    optimizer.zero_grad()
                    reg_out, cls_out = model(x)
                    loss = criterion[0](reg_out, y_reg) + criterion[1](cls_out, y_cls)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                reg_out, cls_out = model(X_test_t)
                y_pred_reg = reg_scaler.inverse_transform(reg_out.numpy())
                y_pred_cls = cls_out.numpy()
            reg_metrics = evaluate_regression(y_test_reg, y_pred_reg)
            cls_metrics = evaluate_classification(y_test_cls, y_pred_cls)
            torch.save(model.state_dict(), f"checkpoints/{model_type}_best.pth")
            reg_results.append(reg_metrics)
            cls_results.append(cls_metrics)
            model_names.append("MMoE")

        else:
            print("未知模型类型，请选择 wide_deep、shared_bottom 或 mmoe")

    if model_type == 'all':
        for m in ['wide_deep', 'shared_bottom', 'mmoe']:
            run_model(m)
        plot_metrics(reg_results, cls_results, model_names)
    else:
        run_model(model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch多模型训练入口")
    parser.add_argument('--model', type=str, default='wide_deep', help='模型类型：wide_deep/shared_bottom/mmoe/all')
    args = parser.parse_args()
    main(args.model)
