"""
Transformer-based models for text classification.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, Any, Tuple, Optional
import numpy as np

class TransformerClassifier(nn.Module):
    """BERT-based classifier for AI vs Human text classification."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, dropout: float = 0.1):
        super(TransformerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Forward pass."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

class TransformerClassifierWrapper:
    """Wrapper for transformer-based classification."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512, num_labels: int = 2):
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TransformerClassifier(model_name, num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.is_fitted = False
    
    def tokenize_texts(self, texts: list) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def predict(self, texts: list) -> np.ndarray:
        """Make predictions on texts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Tokenize inputs
        inputs = self.tokenize_texts(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Tokenize inputs
        inputs = self.tokenize_texts(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
        
        return probabilities.cpu().numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': self.num_labels
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
