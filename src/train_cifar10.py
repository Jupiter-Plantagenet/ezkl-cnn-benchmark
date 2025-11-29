"""
Training utilities for composite CNN architectures on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
from pathlib import Path


def get_cifar10_loaders(batch_size=128, data_dir='./data'):
    """Load CIFAR-10 dataset with standard preprocessing"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_composite_model(model, architecture_name, config, device='cuda', save_dir='./models_pt'):
    """
    Train a composite CNN model on CIFAR-10.
    
    Args:
        model: PyTorch model to train
        architecture_name: Name of the architecture (for saving)
        config: Training configuration dictionary
        device: Device to train on
        save_dir: Directory to save trained models
    
    Returns:
        Dictionary with training history and final test accuracy
    """
    print(f"\n{'='*60}")
    print(f"Training {architecture_name}")
    print(f"{'='*60}")
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Data loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size']
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = Path(save_dir) / 'composite' / f'{architecture_name}_best.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, save_path)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print(f"\nTraining completed. Best test accuracy: {best_acc:.2f}%")
    
    # Save final model
    final_path = Path(save_dir) / 'composite' / f'{architecture_name}_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_test_acc': test_acc,
        'best_test_acc': best_acc
    }, final_path)
    
    # Save training history
    history_path = Path(save_dir) / 'composite' / f'{architecture_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'final_test_acc': test_acc,
        'best_test_acc': best_acc,
        'history': history
    }


if __name__ == '__main__':
    # Test training script
    import sys
    sys.path.append('..')
    from models import create_composite_model
    from config.experiment_config import COMPOSITE_ARCHITECTURES, TRAINING_CONFIG
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train all composite models
    for arch_name, arch_config in COMPOSITE_ARCHITECTURES.items():
        model = create_composite_model(arch_name, arch_config)
        results = train_composite_model(
            model,
            arch_name,
            TRAINING_CONFIG,
            device=device
        )
        print(f"\n{arch_name} final results:")
        print(f"  Best test accuracy: {results['best_test_acc']:.2f}%")
        print(f"  Final test accuracy: {results['final_test_acc']:.2f}%")
