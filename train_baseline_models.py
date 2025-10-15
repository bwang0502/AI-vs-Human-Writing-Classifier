#!/usr/bin/env python3
"""
Baseline model training for AI vs Human writing classification.
This script trains simple models using the linguistic features from EDA.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Create results directory
results_dir = Path('results/baseline_models')
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AI vs Human Writing Classification - Baseline Models")
print("=" * 60)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('data/raw/ai_human_content_detection_dataset.csv')
print(f"✅ Loaded {len(df):,} samples")

# Define features
linguistic_features = [
    'word_count', 'character_count', 'sentence_count', 'lexical_diversity',
    'avg_sentence_length', 'avg_word_length', 'punctuation_ratio',
    'flesch_reading_ease', 'gunning_fog_index', 'grammar_errors',
    'passive_voice_ratio', 'predictability_score', 'burstiness', 'sentiment_score'
]

# Prepare data
print("\n🔧 Preparing features...")
X = df[linguistic_features].fillna(0)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features
print("\n⚙️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, results_dir / 'scaler.pkl')
print("✅ Scaler saved")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True)
}

results = {}

print("\n" + "=" * 60)
print("🤖 Training Models...")
print("=" * 60)

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Get probability scores for AUC metrics
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Calculate AUC metrics
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    
    print(f"\n📊 Results for {name}:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")
    print(f"  Test F1 Score:       {test_f1:.4f}")
    print(f"  Test ROC-AUC:        {test_roc_auc:.4f}")
    print(f"  Test PR-AUC:         {test_pr_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"  CV Accuracy:         {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Human', 'AI-Generated']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create figure with confusion matrix, ROC curve, and PR curve
    fig = plt.figure(figsize=(18, 5))
    
    # Confusion Matrix
    ax1 = plt.subplot(131)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI-Generated'],
                yticklabels=['Human', 'AI-Generated'], ax=ax1)
    ax1.set_title(f'Confusion Matrix - {name}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # ROC Curve
    ax2 = plt.subplot(132)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {test_roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve - {name}')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax3 = plt.subplot(133)
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    ax3.plot(recall, precision, linewidth=2, label=f'PR (AUC = {test_pr_auc:.4f})')
    ax3.axhline(y=y_test.mean(), color='k', linestyle='--', linewidth=1, 
                label=f'Baseline ({y_test.mean():.4f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title(f'Precision-Recall Curve - {name}')
    ax3.legend(loc='lower left')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'evaluation_{name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    print(f"✅ Evaluation plots saved")
    
    # Save model
    model_path = results_dir / f'{name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Store results
    results[name] = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Compare models
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print("\n", comparison_df)

# Save comparison
comparison_df.to_csv(results_dir / 'model_comparison.csv')

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy comparison
comparison_df[['train_acc', 'val_acc', 'test_acc']].plot(
    kind='bar', ax=axes[0, 0], rot=0
)
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([0.5, 1.0])
axes[0, 0].legend(['Training', 'Validation', 'Test'])
axes[0, 0].grid(axis='y', alpha=0.3)

# F1 Score comparison
comparison_df['test_f1'].plot(kind='bar', ax=axes[0, 1], rot=0, color='coral')
axes[0, 1].set_title('Test F1 Score Comparison')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_ylim([0.5, 1.0])
axes[0, 1].grid(axis='y', alpha=0.3)

# ROC-AUC comparison
comparison_df['test_roc_auc'].plot(kind='bar', ax=axes[1, 0], rot=0, color='green')
axes[1, 0].set_title('Test ROC-AUC Comparison')
axes[1, 0].set_ylabel('ROC-AUC')
axes[1, 0].set_ylim([0.5, 1.0])
axes[1, 0].grid(axis='y', alpha=0.3)

# PR-AUC comparison
comparison_df['test_pr_auc'].plot(kind='bar', ax=axes[1, 1], rot=0, color='purple')
axes[1, 1].set_title('Test PR-AUC Comparison')
axes[1, 1].set_ylabel('PR-AUC')
axes[1, 1].set_ylim([0.5, 1.0])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'model_comparison.png', dpi=300)
print(f"\n✅ Comparison plot saved to {results_dir / 'model_comparison.png'}")

# Best model
best_model_name = comparison_df['test_acc'].idxmax()
best_accuracy = comparison_df.loc[best_model_name, 'test_acc']

print("\n" + "=" * 60)
print("🏆 BEST MODEL")
print("=" * 60)
print(f"Model: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"Test F1 Score: {comparison_df.loc[best_model_name, 'test_f1']:.4f}")
print(f"Test ROC-AUC: {comparison_df.loc[best_model_name, 'test_roc_auc']:.4f}")
print(f"Test PR-AUC: {comparison_df.loc[best_model_name, 'test_pr_auc']:.4f}")

print("\n✅ Training complete! Results saved to:", results_dir)
print("\n📁 Files created:")
print(f"  • Model comparison: {results_dir / 'model_comparison.csv'}")
print(f"  • Comparison plot: {results_dir / 'model_comparison.png'}")
print(f"  • Evaluation plots (CM, ROC, PR curves) for each model")
print(f"  • Trained model files (.pkl)")

print("\n🚀 Next steps:")
print("  1. Review the ROC and PR curves to understand model discrimination")
print("  2. Analyze confusion matrices to understand error patterns")
print("  3. Try feature engineering to improve performance")
print("  4. Experiment with hyperparameter tuning")
print("  5. Build more advanced models (e.g., BERT)")
