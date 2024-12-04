from typing import Dict, Any

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def create_model(hparams: Dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    """Create and configure the model based on hyperparameters."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify final layers with dropout
    model.fc = nn.Sequential(
        nn.Dropout(hparams["dropout_rate"]),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    model = model.to(device)
    model = torch.compile(model)
    
    return model

def get_optimizer(model: nn.Module, hparams: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on hyperparameters."""
    if hparams["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    else:
        return torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
