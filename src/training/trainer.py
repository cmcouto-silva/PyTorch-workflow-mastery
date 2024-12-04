from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import wandb

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Initialize metrics
        self.train_loss = torchmetrics.MeanMetric().to(device)
        self.val_loss = torchmetrics.MeanMetric().to(device)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_loss.reset()
        self.train_accuracy.reset()
        
        train_progress = tqdm(train_loader, desc=f'• Epoch {epoch + 1}/{num_epochs} [Train]', leave=False)
        
        for images, labels in train_progress:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.train_loss.update(loss)
            self.train_accuracy.update(outputs, labels)
            
            train_progress.set_postfix({
                'loss': f'{self.train_loss.compute():.3f}',
                'acc': f'{self.train_accuracy.compute():.1%}'
            })
        
        return self.train_loss.compute(), self.train_accuracy.compute()
    
    @torch.inference_mode()
    def validate(self, val_loader: DataLoader, epoch: int, num_epochs: int) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        self.val_loss.reset()
        self.val_accuracy.reset()
        
        val_progress = tqdm(val_loader, desc=f'• Epoch {epoch + 1}/{num_epochs} [Valid]', leave=False)
        
        for images, labels in val_progress:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.val_loss.update(loss)
            self.val_accuracy.update(outputs, labels)
            
            val_progress.set_postfix({
                'loss': f'{self.val_loss.compute():.3f}',
                'acc': f'{self.val_accuracy.compute():.1%}'
            })
        
        return self.val_loss.compute(), self.val_accuracy.compute()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        model_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Full training loop with optional model saving."""
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Train & validate
            train_loss, train_acc = self.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_acc = self.validate(val_loader, epoch, num_epochs)
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            
            wandb.log(metrics)
            logger.debug(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.3f} | "
                f"Train Acc: {train_acc:.1%} | "
                f"Val Loss: {val_loss:.3f} | "
                f"Val Acc: {val_acc:.1%}"
            )
            
            # Save best model
            if val_acc > best_accuracy and model_path:
                best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                }, model_path)
                logger.info(f"Saved best model (val_acc: {val_acc:.1%}) to {model_path}")
        
        return metrics
