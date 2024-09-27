import torch
import torch.nn as nn
import torch.nn.functional as f

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=9):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
