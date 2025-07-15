import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()  # Only use pin_memory if CUDA is available
logger.info(f"Using device: {DEVICE}")

def check_dataset_structure():
    """Check if the dataset structure is correct"""
    train_path = Path("dataset/dataset/train")
    test_path = Path("dataset/dataset/test")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Dataset directories not found. Please ensure 'dataset/train' and 'dataset/test' exist")
    
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    test_classes = [d for d in test_path.iterdir() if d.is_dir()]
    
    if not train_classes or not test_classes:
        raise ValueError("No class directories found in train or test folders")
    
    logger.info(f"Found {len(train_classes)} classes in training set")
    logger.info(f"Found {len(test_classes)} classes in test set")
    
    return len(train_classes)

# Data transforms with augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transform for validation/test
val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets():
    """Load and validate datasets"""
    try:
        train_dataset = datasets.ImageFolder(
            root="dataset/dataset/train",
            transform=train_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root="dataset/dataset/test",
            transform=val_transform
        )
        
        logger.info(f"Successfully loaded datasets")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def create_data_loaders(train_dataset, test_dataset):
    """Create data loaders with error handling"""
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # Reduced for better compatibility
            pin_memory=PIN_MEMORY  # Only use pin_memory if CUDA is available
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=PIN_MEMORY  # Only use pin_memory if CUDA is available
        )
        
        return train_loader, test_loader
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

class WoundClassifier(nn.Module):
    """Neural network for wound image classification using a modified ResNet50 backbone."""
    def __init__(self, num_classes):
        super(WoundClassifier, self).__init__()
        
        # Use a pre-trained ResNet50 as the base model
        try:
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {str(e)}")
            raise
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Modify the final layer for our number of classes
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def train_epoch(model, loader, criterion, optimizer, device):
    """Training function with progress tracking"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        try:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(loader)}: Loss: {loss.item():.4f}')
                
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            continue
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validation function with error handling"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return running_loss / len(loader), 100. * correct / total

def main():
    try:
        # Check dataset structure
        num_classes = check_dataset_structure()
        
        # Load datasets
        train_dataset, test_dataset = load_datasets()
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
        
        # Initialize model
        model = WoundClassifier(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        logger.info("Starting training...")
        for epoch in range(EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        model_path = 'wound_classification_model.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Final model saved: {model_path}")
        
        # Plot training results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plot_path = 'training_history.png'
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved: {plot_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 