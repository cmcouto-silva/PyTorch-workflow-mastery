from typing import Tuple, Dict, Any

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def create_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms from config."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['train_transform'][1]['Normalize']['mean'],
            std=config['data']['train_transform'][1]['Normalize']['std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['val_transform'][1]['Normalize']['mean'],
            std=config['data']['val_transform'][1]['Normalize']['std']
        )
    ])
    
    return train_transform, val_transform

def get_dataloaders(
    batch_size: int,
    num_workers: int,
    transforms: Tuple[transforms.Compose, transforms.Compose]
) -> Tuple[DataLoader, DataLoader]:
    """Create and return training and validation dataloaders."""
    train_transform, val_transform = transforms
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
