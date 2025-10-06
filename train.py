#!/usr/bin/env python3
"""
Training script for AI vs Human writing classification.

This script demonstrates how to train models using the project's modules.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import TextPreprocessor, load_data, split_data
from src.data.features import LinguisticFeatureExtractor, TextVectorizer
from src.models import RandomForestClassifierWrapper, LogisticRegressionWrapper
from src.evaluation import ModelEvaluator
from src.utils import load_config, setup_logging, save_results, create_experiment_folder

def main():
    parser = argparse.ArgumentParser(description="Train AI vs Human text classifier")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--experiment-name", help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting training pipeline...")
    
    # Create experiment folder
    experiment_dir = create_experiment_folder("results", args.experiment_name)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Load data
    logger.info("Loading data...")
    df = load_data(args.data)
    logger.info(f"Loaded {len(df)} samples")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = TextPreprocessor(
        min_length=config['data']['min_text_length'],
        max_length=config['data']['max_text_length']
    )
    
    # Filter by length
    df = preprocessor.filter_by_length(df, config['data']['text_column'])
    logger.info(f"After filtering: {len(df)} samples")
    
    # Split data
    train_df, val_df, test_df = split_data(
        df, 
        test_size=config['training']['test_split'],
        val_size=config['training']['validation_split']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Feature extraction
    logger.info("Extracting features...")
    feature_extractor = LinguisticFeatureExtractor()
    
    # Extract linguistic features for all splits
    train_features = []
    val_features = []
    test_features = []
    
    for _, row in train_df.iterrows():
        features = feature_extractor.extract_all_features(row[config['data']['text_column']])
        train_features.append(features)
    
    for _, row in val_df.iterrows():
        features = feature_extractor.extract_all_features(row[config['data']['text_column']])
        val_features.append(features)
    
    for _, row in test_df.iterrows():
        features = feature_extractor.extract_all_features(row[config['data']['text_column']])
        test_features.append(features)
    
    # Convert to DataFrames
    import pandas as pd
    train_features_df = pd.DataFrame(train_features)
    val_features_df = pd.DataFrame(val_features)
    test_features_df = pd.DataFrame(test_features)
    
    # Fill NaN values
    train_features_df = train_features_df.fillna(0)
    val_features_df = val_features_df.fillna(0)
    test_features_df = test_features_df.fillna(0)
    
    # Add text vectorization if specified
    if config['features']['use_embeddings']:
        logger.info("Adding text vectorization...")
        vectorizer = TextVectorizer(vectorizer_type="tfidf", max_features=1000)
        
        train_vectors = vectorizer.fit_transform(train_df[config['data']['text_column']].tolist())
        val_vectors = vectorizer.transform(val_df[config['data']['text_column']].tolist())
        test_vectors = vectorizer.transform(test_df[config['data']['text_column']].tolist())
        
        # Combine features
        import numpy as np
        X_train = np.hstack([train_features_df.values, train_vectors])
        X_val = np.hstack([val_features_df.values, val_vectors])
        X_test = np.hstack([test_features_df.values, test_vectors])
    else:
        X_train = train_features_df.values
        X_val = val_features_df.values
        X_test = test_features_df.values
    
    # Get labels
    y_train = train_df[config['data']['label_column']].values
    y_val = val_df[config['data']['label_column']].values
    y_test = test_df[config['data']['label_column']].values
    
    # Initialize model
    logger.info(f"Training {args.model} model...")
    if args.model == "random_forest":
        model = RandomForestClassifierWrapper(n_estimators=100, random_state=42)
    elif args.model == "logistic_regression":
        model = LogisticRegressionWrapper(random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("Making predictions...")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    
    val_metrics = evaluator.evaluate_predictions(y_val, y_val_pred, y_val_proba, "validation")
    test_metrics = evaluator.evaluate_predictions(y_test, y_test_pred, y_test_proba, "test")
    
    # Print results
    print("\\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    for metric, value in val_metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    print("\\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    # Save results
    results = {
        'experiment_config': config,
        'model_type': args.model,
        'data_info': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'feature_count': X_train.shape[1]
        },
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    save_results(results, os.path.join(experiment_dir, "results.yaml"))
    
    # Save model
    model_path = os.path.join(experiment_dir, f"{args.model}_model.joblib")
    model.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Generate plots
    evaluator.plot_confusion_matrix("validation", save_path=os.path.join(experiment_dir, "confusion_matrix_val.png"))
    evaluator.plot_confusion_matrix("test", save_path=os.path.join(experiment_dir, "confusion_matrix_test.png"))
    
    if hasattr(model.model, 'feature_importances_'):
        feature_names = list(train_features_df.columns)
        if config['features']['use_embeddings']:
            feature_names += [f'tfidf_{i}' for i in range(train_vectors.shape[1])]
        
        evaluator.feature_importance_analysis(
            model.model, 
            feature_names=feature_names,
            save_path=os.path.join(experiment_dir, "feature_importance.png")
        )
    
    logger.info(f"Training completed! Results saved in: {experiment_dir}")

if __name__ == "__main__":
    main()
