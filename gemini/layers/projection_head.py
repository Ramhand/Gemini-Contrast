import torch
import torch.nn as nn
import torch.nn.functional as f

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x
