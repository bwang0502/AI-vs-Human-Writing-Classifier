"""
Train classifiers on transformer embeddings for AI vs Human text classification.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve
)
import sys
sys.path.append('src')

from models.embeddings import SentenceBERTEmbedder, DistilBERTEmbedder, HybridEmbedder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')


def load_data(data_path):
    """Load and prepare dataset."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    print(f"✓ Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def generate_embeddings(texts, embedder_type='sentence-bert', model_name=None):
    """Generate embeddings for texts."""
    print(f"\n{'=' * 60}")
    print(f"GENERATING {embedder_type.upper()} EMBEDDINGS")
    print("=" * 60)
    
    if embedder_type == 'sentence-bert':
        embedder = SentenceBERTEmbedder(model_name or 'all-MiniLM-L6-v2')
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        embeddings = embedder.encode(texts, batch_size=32, show_progress=True)
    elif embedder_type == 'distilbert':
        embedder = DistilBERTEmbedder(model_name or 'distilbert-base-uncased')
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        embeddings = embedder.encode(texts, batch_size=16)
    elif embedder_type == 'hybrid':
        sbert = SentenceBERTEmbedder()
        distilbert = DistilBERTEmbedder()
        embedder = HybridEmbedder([sbert, distilbert])
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        embeddings = embedder.encode(texts, batch_size=32)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings, embedder


def train_classifier(X_train, y_train, X_test, y_test, classifier_type='logistic'):
    """Train and evaluate a classifier."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING {classifier_type.upper()} CLASSIFIER")
    print("=" * 60)
    
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif classifier_type == 'svm':
        clf = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    elif classifier_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif classifier_type == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  CV F1:     {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    
    return clf, metrics, y_pred, y_proba


def plot_evaluation(y_test, y_pred, y_proba, metrics, model_name, save_path):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'ROC Curve - {model_name}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 0].plot(recall, precision, label=f'PR (AUC = {metrics["pr_auc"]:.3f})', linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1'], metrics['roc_auc'], metrics['pr_auc']]
    
    bars = axes[1, 1].barh(metric_names, metric_values, alpha=0.7)
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_title(f'Metrics - {model_name}')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("TRANSFORMER EMBEDDING CLASSIFIER TRAINING")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results') / 'transformer_models' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data('data/raw/ai_human_content_detection_dataset.csv')
    texts = df['text_content'].tolist()
    labels = df['label'].values
    
    X_texts_train, X_texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    embedding_configs = [
        {'type': 'sentence-bert', 'model_name': 'all-MiniLM-L6-v2', 'label': 'SentenceBERT'},
        {'type': 'distilbert', 'model_name': 'distilbert-base-uncased', 'label': 'DistilBERT'}
    ]
    
    classifiers = ['logistic', 'mlp', 'svm', 'random_forest']
    all_results = []
    
    for emb_config in embedding_configs:
        print(f"\n{'#' * 60}")
        print(f"EMBEDDING: {emb_config['label']}")
        print('#' * 60)
        
        train_embeddings, _ = generate_embeddings(
            X_texts_train, 
            embedder_type=emb_config['type'],
            model_name=emb_config['model_name']
        )
        
        test_embeddings, _ = generate_embeddings(
            X_texts_test,
            embedder_type=emb_config['type'],
            model_name=emb_config['model_name']
        )
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_embeddings)
        test_scaled = scaler.transform(test_embeddings)
        
        for clf_type in classifiers:
            model_name = f"{emb_config['label']}_{clf_type}"
            
            clf, metrics, y_pred, y_proba = train_classifier(
                train_scaled, y_train, test_scaled, y_test, classifier_type=clf_type
            )
            
            result = {'embedding': emb_config['label'], 'classifier': clf_type, 
                     'model_name': model_name, **metrics}
            all_results.append(result)
            
            plot_path = results_dir / f"eval_{emb_config['type']}_{clf_type}.png"
            plot_evaluation(y_test, y_pred, y_proba, metrics, model_name, plot_path)
            
            model_path = results_dir / f"model_{emb_config['type']}_{clf_type}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({'classifier': clf, 'scaler': scaler}, f)
    
    results_df = pd.DataFrame(all_results).sort_values('f1', ascending=False)
    results_df.to_csv(results_dir / 'results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df[['model_name', 'accuracy', 'f1', 'roc_auc']].to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n🏆 BEST: {best['model_name']} (F1: {best['f1']:.4f})")
    print(f"\n✅ Done! Results: {results_dir}")


if __name__ == "__main__":
    main()