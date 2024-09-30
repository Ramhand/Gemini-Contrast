import torch
import torch.nn as nn
import torch.nn.functional as f

class Encoder(nn.Module):
    """
    The Encoder is a series of ResidualBlocks meant to extract as much information as
    possible from the input image, by making sure each layer gets a taste of the original
    input, before linearizing it for classification purposes.  The eponymous twins of the
    Gemini-Contrast architecture.
    """
    def __init__(self, feature_dim=128):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(
            ResidualBlock(3, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = f.normalize(x, p=2, dim=1)
        return x