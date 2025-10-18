# 用 PyTorch 实现 Wide&Deep、Shared-Bottom、MMoE 三个模型
# 保持与原 Keras 版本接口一致，便于对比
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# 数据加载与预处理（与原版一致）
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

# 转为 torch tensor
def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)
X_train_t = to_tensor(X_train)
X_test_t = to_tensor(X_test)
y_train_reg_t = to_tensor(y_train_reg_scaled)
y_test_reg_t = to_tensor(y_test_reg_scaled)
y_train_cls_t = to_tensor(y_train_cls)
y_test_cls_t = to_tensor(y_test_cls)

# Wide&Deep
class WideDeep(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.final = nn.Linear(2, 1)
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        concat = torch.cat([wide_out, deep_out], dim=1)
        return self.final(concat)

# Shared-Bottom
class SharedBottom(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        shared = self.shared(x)
        reg = self.reg_head(shared)
        cls = torch.sigmoid(self.cls_head(shared))
        return reg, cls

# MMoE
class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts=8, num_tasks=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        ) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_tasks)])
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch, num_experts, 32)
        outs = []
        for i in range(self.num_tasks):
            gate = F.softmax(self.gates[i](x), dim=1)  # (batch, num_experts)
            gate = gate.unsqueeze(-1)  # (batch, num_experts, 1)
            task_input = torch.sum(gate * expert_outs, dim=1)  # (batch, 32)
            if i == 0:
                outs.append(self.reg_head(task_input))
            else:
                outs.append(torch.sigmoid(self.cls_head(task_input)))
        return outs

# 训练与评估函数

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"回归任务评估：MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return mae, rmse, r2

def evaluate_classification(y_true, y_pred):
    y_pred_cls = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_cls)
    precision = precision_score(y_true, y_pred_cls, zero_division=0)
    recall = recall_score(y_true, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true, y_pred_cls, zero_division=0)
    print(f"分类任务评估：Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return acc, precision, recall, f1

# 通用训练循环
def train_model(model, train_loader, optimizer, criterion, epochs=50, task='reg', val_data=None):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0]
            y = batch[1]
            optimizer.zero_grad()
            if task == 'reg':
                out = model(x)
                loss = criterion(out, y)
            elif task == 'multi':
                reg_out, cls_out = model(x)
                loss1 = criterion[0](reg_out, y[0])
                loss2 = criterion[1](cls_out, y[1])
                loss = loss1 + 0.5 * loss2
            elif task == 'mmoe':
                reg_out, cls_out = model(x)
                loss1 = criterion[0](reg_out, y[0])
                loss2 = criterion[1](cls_out, y[1])
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    # Wide&Deep
    print("\n=== Wide&Deep模型（回归任务, PyTorch） ===")
    wd_model = WideDeep(input_dim)
    optimizer = torch.optim.Adam(wd_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_ds = TensorDataset(X_train_t, y_train_reg_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    wd_model = train_model(wd_model, train_loader, optimizer, criterion, epochs=50, task='reg')
    wd_model.eval()
    with torch.no_grad():
        y_pred_reg_scaled = wd_model(X_test_t).numpy()
    y_pred_reg = reg_scaler.inverse_transform(y_pred_reg_scaled)
    evaluate_regression(y_test_reg, y_pred_reg)

    # Shared-Bottom
    print("\n=== Shared-Bottom模型（多任务, PyTorch） ===")
    sb_model = SharedBottom(input_dim)
    optimizer = torch.optim.Adam(sb_model.parameters(), lr=0.001)
    criterion = [nn.MSELoss(), nn.BCELoss()]
    y_train_reg_t2 = y_train_reg_t
    y_train_cls_t2 = y_train_cls_t
    train_ds = TensorDataset(X_train_t, y_train_reg_t2, y_train_cls_t2)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    def sb_batch(batch):
        return batch[0], [batch[1], batch[2]]
    def sb_train_loop(model, loader, optimizer, criterion, epochs=50):
        model.train()
        for epoch in range(epochs):
            for batch in loader:
                x, ys = sb_batch(batch)
                optimizer.zero_grad()
                reg_out, cls_out = model(x)
                loss1 = criterion[0](reg_out, ys[0])
                loss2 = criterion[1](cls_out, ys[1])
                loss = loss1 + 0.5 * loss2
                loss.backward()
                optimizer.step()
        return model
    sb_model = sb_train_loop(sb_model, train_loader, optimizer, criterion, epochs=50)
    sb_model.eval()
    with torch.no_grad():
        reg_out, cls_out = sb_model(X_test_t)
        y_pred_reg_scaled = reg_out.numpy()
        y_pred_cls = cls_out.numpy()
    y_pred_reg = reg_scaler.inverse_transform(y_pred_reg_scaled)
    evaluate_regression(y_test_reg, y_pred_reg)
    evaluate_classification(y_test_cls, y_pred_cls)

    # MMoE
    print("\n=== MMoE模型（多任务, PyTorch） ===")
    mmoe_model = MMoE(input_dim, num_experts=8, num_tasks=2)
    optimizer = torch.optim.Adam(mmoe_model.parameters(), lr=0.001)
    criterion = [nn.MSELoss(), nn.BCELoss()]
    train_ds = TensorDataset(X_train_t, y_train_reg_t, y_train_cls_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    def mmoe_batch(batch):
        return batch[0], [batch[1], batch[2]]
    def mmoe_train_loop(model, loader, optimizer, criterion, epochs=50):
        model.train()
        for epoch in range(epochs):
            for batch in loader:
                x, ys = mmoe_batch(batch)
                optimizer.zero_grad()
                reg_out, cls_out = model(x)
                loss1 = criterion[0](reg_out, ys[0])
                loss2 = criterion[1](cls_out, ys[1])
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
        return model
    mmoe_model = mmoe_train_loop(mmoe_model, train_loader, optimizer, criterion, epochs=50)
    mmoe_model.eval()
    with torch.no_grad():
        reg_out, cls_out = mmoe_model(X_test_t)
        y_pred_reg_scaled = reg_out.numpy()
        y_pred_cls = cls_out.numpy()
    y_pred_reg = reg_scaler.inverse_transform(y_pred_reg_scaled)
    evaluate_regression(y_test_reg, y_pred_reg)
    evaluate_classification(y_test_cls, y_pred_cls)
