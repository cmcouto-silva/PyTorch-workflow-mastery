import os
from pathlib import Path

import torch
import torch.nn as nn
import optuna
import wandb
from loguru import logger
from dotenv import load_dotenv

from src.config import load_config
from src.dataset import create_transforms, get_dataloaders
from src.models.resnet18 import create_model, get_optimizer
from src.training.trainer import Trainer
from src.training.optimization import create_objective

# [Optional] Enable TF32 for better performance on modern NVIDIA GPUs
torch.set_float32_matmul_precision('high')

def train():
    """Run training with best hyperparameters."""
    # Load environment variables
    load_dotenv()
    assert os.getenv("WANDB_API_KEY") is not None, "WANDB_API_KEY not found"

    # Load config
    config = load_config()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    Path(config["defaults"]["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(config["defaults"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["defaults"]["seed"])
        torch.backends.cudnn.deterministic = False

    # Target hyperparams
    hparams = {
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "optimizer": config["training"]["optimizer"],
        "dropout_rate": config["training"]["dropout_rate"],
        "num_epochs": config["training"]["num_epochs"],
        "model": config["model"]["name"]
    }
    
    # Initialize W&B
    wandb.init(
        project=config["wandb"]["project"],
        config=hparams,
        name="training_run",
    )
    
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
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=hparams["num_epochs"],
        model_path=config["defaults"]["model_path"]
    )
    
    wandb.finish()
    logger.info("Training completed!")

def optimize():
    """Run hyperparameter optimization."""
    # Load environment variables
    load_dotenv()
    assert os.getenv("WANDB_API_KEY") is not None, "WANDB_API_KEY not found"
    
    # Load config
    config = load_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config["optuna"]["n_startup_trials"],
            n_warmup_steps=config["optuna"]["n_warmup_steps"]
        )
    )
    
    objective = create_objective(config, device)
    
    logger.info("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])
    
    # Print results
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value:.3f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or optimize model')
    parser.add_argument('mode', choices=['train', 'optimize'], 
                       help='Whether to train with best params or run optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    else:
        optimize()
