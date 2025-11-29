"""
Data processing utilities for AI vs Human writing classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import re
import string
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    """Text preprocessing utilities for classification tasks."""
    
    def __init__(self, min_length: int = 50, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text.strip()
    
    def filter_by_length(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Filter texts by length constraints."""
        text_lengths = df[text_column].str.len()
        mask = (text_lengths >= self.min_length) & (text_lengths <= self.max_length)
        return df[mask].copy()

def load_data(file_path: str, text_column: str = "text", label_column: str = "label") -> pd.DataFrame:
    """Load dataset from file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               val_size: float = 0.1, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    # First split: train+val and test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df.iloc[:, -1]
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state, stratify=train_val.iloc[:, -1]
    )
    
    return train, val, test
