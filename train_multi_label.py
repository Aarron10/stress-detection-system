import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import time
import copy

# Configuration
DATA_DIR = './dataset' # Expects 'train' and 'val' subfolders
CLASSES = ['Focused', 'Stressed', 'Distracted'] # Folder names must match these
MODEL_SAVE_PATH = 'stress_multilabel.pth'
NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

def get_model(num_classes):
    model = models.mobilenet_v2(weights='DEFAULT')
    # Replace classifier
    # MobileNetV2 classifier[1] is the Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Data from Folders
    # Structure: dataset/train/Focused, dataset/train/Stressed...
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                          for x in ['train', 'val']}
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATA_DIR}. Ensure you have 'train' and 'val' folders.")
        return

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes found: {class_names}")
    # Verify classes match expected
    if set(class_names) != set(CLASSES):
        print(f"Warning: Dataset classes {class_names} do not match expected {CLASSES}")

    # Model
    model = get_model(len(class_names))
    model = model.to(device)

    # Loss: BCEWithLogitsLoss for independent scores
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Convert integer labels (0, 1, 2) to One-Hot Floats ([1.0, 0.0, 0.0])
                # This is tricking the multi-label loss to accept single-label training data
                target = F.one_hot(labels, num_classes=len(class_names)).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Validation Scores
            if phase == 'val':
                 # Show sample independent scores
                probs = torch.sigmoid(outputs)
                print("Sample Scores (from last batch):")
                for i in range(min(3, len(probs))):
                    print(f"Img {i}: ", end="")
                    for j, cls in enumerate(class_names):
                        print(f"{cls}: {probs[i][j]:.1%} ", end="")
                    # Show true label
                    print(f"| True: {class_names[labels[i]]}")

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
