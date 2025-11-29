"""
Utility functions for the AI vs Human writing classifier project.
"""

import yaml
import logging
import os
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_dir_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results dictionary to file."""
    ensure_dir_exists(os.path.dirname(filepath))
    
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as file:
            yaml.dump(results, file, default_flow_style=False)
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=2)
    else:
        raise ValueError("Unsupported file format. Use YAML or JSON.")

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or ['Human', 'AI'],
                yticklabels=labels or ['Human', 'AI'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_experiment_folder(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """Create a timestamped experiment folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        folder_name = f"{experiment_name}_{timestamp}"
    else:
        folder_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, folder_name)
    ensure_dir_exists(experiment_dir)
    
    return experiment_dir
