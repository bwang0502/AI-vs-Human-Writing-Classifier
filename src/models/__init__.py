"""
Model definitions and training utilities.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True

class RandomForestClassifierWrapper(BaseClassifier):
    """Random Forest classifier wrapper."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

class LogisticRegressionWrapper(BaseClassifier):
    """Logistic Regression classifier wrapper."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__()
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Logistic Regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate model performance."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc'] = 0.0
    
    return metrics
