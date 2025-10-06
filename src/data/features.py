"""
Feature extraction utilities for text classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textstat
import re

class LinguisticFeatureExtractor:
    """Extract linguistic and stylometric features from text."""
    
    def extract_basic_stats(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        features = {}
        
        # Basic counts
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = textstat.sentence_count(text)
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Averages
        if features['sentence_count'] > 0:
            features['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_words_per_sentence'] = 0
            
        if features['word_count'] > 0:
            features['avg_chars_per_word'] = features['char_count'] / features['word_count']
        else:
            features['avg_chars_per_word'] = 0
        
        return features
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability metrics."""
        features = {}
        
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
        except:
            # Handle edge cases where readability cannot be computed
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
            features['automated_readability_index'] = 0
            features['coleman_liau_index'] = 0
            features['gunning_fog'] = 0
        
        return features
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation-based features."""
        features = {}
        
        total_chars = len(text)
        if total_chars == 0:
            return {key: 0 for key in ['comma_ratio', 'period_ratio', 'question_ratio', 'exclamation_ratio']}
        
        features['comma_ratio'] = text.count(',') / total_chars
        features['period_ratio'] = text.count('.') / total_chars
        features['question_ratio'] = text.count('?') / total_chars
        features['exclamation_ratio'] = text.count('!') / total_chars
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all linguistic features."""
        features = {}
        features.update(self.extract_basic_stats(text))
        features.update(self.extract_readability_features(text))
        features.update(self.extract_punctuation_features(text))
        
        return features

class TextVectorizer:
    """Text vectorization utilities."""
    
    def __init__(self, vectorizer_type: str = "tfidf", max_features: int = 5000):
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.vectorizer = None
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        elif self.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            raise ValueError("Unsupported vectorizer type")
        
        return self.vectorizer.fit_transform(texts).toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.transform(texts).toarray()
