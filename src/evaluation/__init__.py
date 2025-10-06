"""
Evaluation utilities for model assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_proba: np.ndarray = None,
                           model_name: str = "model") -> Dict[str, float]:
        """Evaluate model predictions comprehensively."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        
        # Add AUC if probabilities are provided
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Could not compute AUC: {e}")
                metrics['auc_roc'] = 0.0
        
        # Store results
        self.metrics[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def print_classification_report(self, model_name: str, class_names: List[str] = None):
        """Print detailed classification report."""
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for model: {model_name}")
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        
        target_names = class_names or ['Human', 'AI-Generated']
        
        print(f"\n=== Classification Report for {model_name} ===")
        print(classification_report(y_true, y_pred, target_names=target_names))
    
    def plot_confusion_matrix(self, 
                            model_name: str, 
                            class_names: List[str] = None,
                            normalize: bool = False,
                            save_path: str = None):
        """Plot confusion matrix for a model."""
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for model: {model_name}")
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', 
                   xticklabels=class_names or ['Human', 'AI'],
                   yticklabels=class_names or ['Human', 'AI'])
        
        title = f'Confusion Matrix - {model_name}'
        if normalize:
            title += ' (Normalized)'
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self) -> pd.DataFrame:
        """Compare metrics across different models."""
        if not self.metrics:
            raise ValueError("No model metrics available for comparison")
        
        df = pd.DataFrame(self.metrics).T
        df = df.round(4)
        return df
    
    def cross_validate_model(self, 
                           model, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           cv: int = 5,
                           scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation on a model."""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        cv_results = {
            f'cv_{scoring}_mean': scores.mean(),
            f'cv_{scoring}_std': scores.std(),
            f'cv_{scoring}_scores': scores.tolist()
        }
        
        return cv_results
    
    def feature_importance_analysis(self, 
                                  model, 
                                  feature_names: List[str] = None,
                                  top_n: int = 20,
                                  save_path: str = None):
        """Analyze and plot feature importance."""
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
