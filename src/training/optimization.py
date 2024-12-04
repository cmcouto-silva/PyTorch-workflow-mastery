from typing import Dict, Any

import torch
import torch.nn as nn
import optuna
import wandb

from src.dataset import create_transforms, get_dataloaders
from src.models.resnet18 import create_model, get_optimizer
from src.training.trainer import Trainer

def create_objective(config: Dict[str, Any], device: torch.device):
    """Create an Optuna objective function with the given config."""
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hparams = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 
                config["optuna"]["parameters"]["learning_rate"]["min"],
                config["optuna"]["parameters"]["learning_rate"]["max"],
                log=config["optuna"]["parameters"]["learning_rate"]["log"]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", 
                config["optuna"]["parameters"]["batch_size"]["choices"]
            ),
            "optimizer": trial.suggest_categorical(
                "optimizer", 
                config["optuna"]["parameters"]["optimizer"]["choices"]
            ),
            "dropout_rate": trial.suggest_float(
                "dropout_rate",
                config["optuna"]["parameters"]["dropout_rate"]["min"],
                config["optuna"]["parameters"]["dropout_rate"]["max"]
            ),
            "num_epochs": config["optuna"]["num_epochs"],
            "model": "ResNet18",
            "trial_number": trial.number
        }
        
        # Initialize W&B
        run = wandb.init(
            project=config["wandb"]["project"],
            config=hparams,
            name=f"trial_{trial.number}",
            reinit=True
        )
        
        try:
            # Setup data
            transforms = create_transforms(config)
            train_loader, val_loader = get_dataloaders(
                batch_size=hparams["batch_size"],
                num_workers=config["defaults"]["num_workers"],
                transforms=transforms
            )
            
            # Setup model & training
            model = create_model(hparams, config["defaults"]["num_classes"], device)
            optimizer = get_optimizer(model, hparams)
            criterion = nn.CrossEntropyLoss()
            
            # Create trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                num_classes=config["defaults"]["num_classes"]
            )
            
            # Train and get metrics
            for epoch in range(hparams["num_epochs"]):
                train_loss, train_acc = trainer.train_epoch(train_loader, epoch, hparams["num_epochs"])
                val_loss, val_acc = trainer.validate(val_loader, epoch, hparams["num_epochs"])
                
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                })
                
                trial.report(val_acc, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return val_acc
            
        finally:
            run.finish()
    
    return objective
