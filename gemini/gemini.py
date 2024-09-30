import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from .layers import Encoder, ClassificationHead, ProjectionHead, ResidualBlock
from .losses import supervised_contrastive_loss, contrastive_loss


class GeminiContrast(nn.Module):
    """
    The Twins, the Myths, the Legends, it's Castor and Pollux.  While Pollux is blessed 
    with the knowledge of the gods in his supervised training, Castor must make do with
    his more limited senses, honed unsupervised.  May they guide your image classification
    projects through the stormy seas of possibilities.

    When initialized, prework will kick on with the training data it gets passed, and
    pre-train Castor and Pollux individually before training them together with the class
    head.  These are the values that worked best for me with the PathMNIST dataset (89.6%)
    Feel free to subclass and reimplement the prework method if it seems to be over-training
    on your dataset.
    """
    def __init__(self, input_dim: int = 128, num_classes: int = 9, training_data: DataLoader = None, transformations: transforms.Compose = None):
        super(GeminiContrast, self).__init__()
        self.input_dim = input_dim
        self.Castor = Encoder(feature_dim=input_dim)
        self.Pollux = Encoder(feature_dim=input_dim)
        self.class_head = ClassificationHead(input_dim=input_dim * 2, num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        if transformations is None:
            transformations = transforms.Compose([
                transforms.RandomResizedCrop(28),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # brightness, contrast, saturation, hue
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if training_data is not None:
            self.prework(training_data, transformations)

    def forward(self, x):
        x = x.to(self.device)
        Castor_feat = self.Castor(x)
        Pollux_feat = self.Pollux(x)
        gemini_feat = torch.cat([Castor_feat, Pollux_feat], dim=1)
        output = self.class_head(gemini_feat)
        return output

    def prework(self, training_data, transformations):
        projector = ProjectionHead(self.input_dim, self.input_dim).to(self.device)
        optimizer_castor = optim.Adam(list(self.Castor.parameters()) + list(projector.parameters()), lr=1e-4)
        optimizer_pollux = optim.Adam(self.Pollux.parameters(), lr=5e-4)
        early_stop = 40
        epochs = 50

        for epoch in range(epochs):
            self.Castor.train()
            self.Pollux.train()
            projector.train()

            for images, labels in training_data:
                images = images.to(self.device)
                labels = labels.to(self.device)

                xi = transformations(images)
                xj = transformations(images)

                hi = self.Castor(xi)
                hj = self.Castor(xj)
                zi = projector(hi)
                zj = projector(hj)

                optimizer_castor.zero_grad()
                loss_castor = contrastive_loss(zi, zj)
                loss_castor.backward()
                optimizer_castor.step()

                if epoch <= early_stop:
                    features = self.Pollux(images)
                    optimizer_pollux.zero_grad()
                    loss_pollux = supervised_contrastive_loss(features, labels)
                    loss_pollux.backward()
                    optimizer_pollux.step()
