import os
import copy
import time
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from res_mlp_pytorch import ResMLP
import torch_optimizer as optim

# Dataset Class
class FoodDataset(Dataset):
    def __init__(self, data_type=None, transforms=None):
        self.path = f'../input/food5k/Food-5K/{data_type}/'
        self.images_name = os.listdir(self.path)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        data = self.images_name[idx]
        label = int(data.split('_')[0])
        label = torch.tensor(label)

        image = cv2.imread(self.path + data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            aug = self.transforms(image=image)
            image = aug['image']

        return (image, label)

# Data preparation
train_data = FoodDataset('training', A.Compose([
    A.RandomResizedCrop(256, 256),
    A.HorizontalFlip(),
    A.Normalize(),
    ToTensorV2()
]))

val_data = FoodDataset('validation', A.Compose([
    A.Resize(height=384, width=384),
    A.CenterCrop(height=256, width=256),
    A.Normalize(),
    ToTensorV2()
]))

test_data = FoodDataset('evaluation', A.Compose([
    A.Resize(height=384, width=384),
    A.CenterCrop(height=256, width=256),
    A.Normalize(),
    ToTensorV2()
]))

# DataLoaders
dataloaders = {
    'train': DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4),
    'test': DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_data),
    'val': len(val_data),
    'test': len(test_data)
}

# Training Function
def train_model(model, criterion, optimizer, epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.0

    for ep in range(epochs):
        print(f"Epoch {ep}/{epochs - 1}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Loss: {best_loss:.4f} Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResMLP(image_size=256, patch_size=16, dim=512, depth=12, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Lamb(model.parameters(), lr=0.005, weight_decay=0.2)

# Train the Model
best_model = train_model(model, criterion, optimizer, epochs=20)
