import torch
import torch.nn as nn

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
