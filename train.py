import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gemini.layers import Encoder, ClassificationHead, ProjectionHead
from gemini.losses import supervised_contrastive_loss, contrastive_loss
from gemini.gemini import GeminiContrast

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Gemini Model')
    parser.add_argument('--data_root', type=str, default='path/to/train_data', help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
    parser.add_argument('--input_dim', type=int, default=128, help='Input dimension for the model')
    parser.add_argument('--save_model', action='store_true', help='For Saving the current Model')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = GeminiContrast(input_dim=args.input_dim, num_classes=args.num_classes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = supervised_contrastive_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{args.epochs}], Average Loss: {total_loss/len(train_loader):.4f}')

        if args.save_model:
            torch.save(model.state_dict(), f'gemini_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
