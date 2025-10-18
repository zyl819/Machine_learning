import torch
import torch.nn as nn
import torch.nn.functional as F

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
