import torch
import torch.nn as nn
import torch.nn.functional as f

class ClassificationHead(nn.Module):
    """
    The ClassificationHead can be run in either classic or non-classic mode.  While 
    simplicity is often for the best, it can also sometimes be an obstacle to
    performance.  The tail end of the Gemini-Contrast architecture.
    """
    def __init__(self, input_dim=128, num_classes=9, classic=True):
        super(ClassificationHead, self).__init__()
        if classic:
            self.fc = nn.Linear(input_dim, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x = self.fc(x)
        return x
