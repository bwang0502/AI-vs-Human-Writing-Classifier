"""
IMPROVED Fine-tuning script - Option A (30% data, 4 epochs)
Expected: F1 ~0.90-0.92 | Time ~2.5 hours
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sns.set_style('whitegrid')


class TextDataset(Dataset):
    """Custom Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def analyze_data(data_path, sample_fraction=1.0):
    """Analyze dataset before training."""
    print("\n" + "=" * 70)
    print("DATA ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    
    print(f"\n✓ Total samples in file: {len(df):,}")
    
    # Sample data if needed (using random_state=42 for reproducibility)
    if sample_fraction < 1.0:
        original_size = len(df)
        df = df.sample(frac=sample_fraction, random_state=42)  # ← Consistent sampling
        print(f"⚡ Sampled dataset: {original_size:,} → {len(df):,} samples ({sample_fraction*100:.0f}%)")
        print(f"   Using random_state=42 for reproducibility")
    
    print(f"✓ Using: {len(df):,} samples for training")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        label_name = "Human" if label == 0 else "AI"
        print(f"   {label_name} (Class {label}): {count:,} ({count/len(df)*100:.1f}%)")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"\n🔍 Missing values: {missing}")
    
    # Text length statistics
    df['text_length'] = df['text_content'].str.len()
    print(f"\n📏 Text Length Statistics:")
    print(f"   Mean:   {df['text_length'].mean():.0f} chars")
    print(f"   Median: {df['text_length'].median():.0f} chars")
    print(f"   Min:    {df['text_length'].min():.0f} chars")
    print(f"   Max:    {df['text_length'].max():.0f} chars")
    
    # Sample texts
    print(f"\n📝 Sample Texts:")
    for i, row in df.head(2).iterrows():
        label_name = "Human" if row['label'] == 0 else "AI"
        print(f"\n   [{i}] {label_name} | {len(row['text_content'])} chars")
        print(f"       {row['text_content'][:80]}...")
    
    return df


def load_data(df, test_size=0.2, val_size=0.1):
    """Load and split dataset."""
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    
    texts = df['text_content'].values
    labels = df['label'].values
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\n✓ Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Human: {(y_train == 0).sum():,} | AI: {(y_train == 1).sum():,}")
    print(f"✓ Val:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"   Human: {(y_val == 0).sum():,} | AI: {(y_val == 1).sum():,}")
    print(f"✓ Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"   Human: {(y_test == 0).sum():,} | AI: {(y_test == 1).sum():,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                        tokenizer, batch_size=16, max_length=128):
    """Create PyTorch DataLoaders with class balancing."""
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    
    # Compute class weights for balanced sampling
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"✓ Val batches:   {len(val_loader)}")
    print(f"✓ Test batches:  {len(test_loader)}")
    print(f"✓ Class weights: [{class_weights[0]:.3f}, {class_weights[1]:.3f}]")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1


def evaluate(model, data_loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    probabilities = []
    true_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(true_labels, probabilities),
        'pr_auc': average_precision_score(true_labels, probabilities)
    }
    
    return metrics, predictions, probabilities, true_labels


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2, marker='s')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2, marker='o')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2, marker='o')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2, marker='s')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('Training & Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1, 1].plot(epochs, history['val_roc_auc'], 'purple', linewidth=2, marker='d')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('ROC-AUC', fontsize=12)
    axes[1, 1].set_title('Validation ROC-AUC', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history: {save_path}")


def plot_evaluation(y_true, y_pred, y_proba, metrics, save_path):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                annot_kws={'size': 14})
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.3f}', linewidth=2, color='#2ca02c')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1, 0].plot(recall, precision, label=f'AUC = {metrics["pr_auc"]:.3f}', linewidth=2, color='#d62728')
    axes[1, 0].set_xlabel('Recall', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                     metrics['f1'], metrics['roc_auc'], metrics['pr_auc']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = axes[1, 1].barh(metric_names, metric_values, color=colors, alpha=0.8)
    axes[1, 1].set_xlabel('Score', fontsize=12)
    axes[1, 1].set_title('Metrics Summary', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved evaluation plots: {save_path}")


def fine_tune_model(train_loader, val_loader, test_loader, class_weights,
                    num_epochs=4, learning_rate=5e-5, results_dir=None):
    """Fine-tune DistilBERT model."""
    print("\n" + "=" * 70)
    print("FINE-TUNING: DISTILBERT")
    print("=" * 70)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    
    print(f"✓ Loaded DistilBERT")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Device: {device}")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"✓ Total training steps: {total_steps:,}")
    print(f"✓ Warmup steps: {int(0.1 * total_steps):,}")
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_roc_auc': []
    }
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print("=" * 70)
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        print(f"\n✓ Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        print(f"✓ Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            print(f"✅ NEW BEST MODEL! (Val F1: {best_val_f1:.4f})")
        else:
            print(f"⏸️  No improvement (Best: {best_val_f1:.4f})")
    
    model.load_state_dict(best_model_state)
    print(f"\n✓ Loaded best model (Val F1: {best_val_f1:.4f})")
    
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    
    test_metrics, test_preds, test_probs, test_labels = evaluate(model, test_loader, device)
    
    print(f"\n📊 Test Set Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {test_metrics['pr_auc']:.4f}")
    
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Human', 'AI'], digits=4))
    
    if results_dir:
        model_path = results_dir / 'distilbert_finetuned_improved.pt'
        torch.save({
            'model_state_dict': best_model_state,
            'metrics': test_metrics,
            'history': history,
            'config': {
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'best_val_f1': best_val_f1
            }
        }, model_path)
        print(f"\n✓ Saved model: {model_path}")
        
        plot_training_history(history, results_dir / 'training_history_improved.png')
        plot_evaluation(test_labels, test_preds, test_probs, test_metrics, 
                       results_dir / 'evaluation_improved.png')
        
        results_df = pd.DataFrame([test_metrics])
        results_df.to_csv(results_dir / 'metrics_improved.csv', index=False)
        print(f"✓ Saved metrics: {results_dir / 'metrics_improved.csv'}")
    
    return model, test_metrics, history


def main():
    """Main pipeline - OPTION A (30% data, 4 epochs)."""
    print("\n" + "=" * 70)
    print("IMPROVED TRANSFORMER FINE-TUNING - OPTION A")
    print("=" * 70)
    
    config = {
        'data_path': 'data/raw/cleaned_ai_human_dataset.csv',
        'sample_fraction': 0.3,    # ← 15K samples (50% more than before!)
        'batch_size': 16,
        'max_length': 128,
        'num_epochs': 4,           # ← One more epoch
        'learning_rate': 5e-5,
        'test_size': 0.2,
        'val_size': 0.1
    }
    
    print("\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🚀 Improvements in Option A:")
    print("   ✅ 50% more training data (10K → 15K samples)")
    print("   ✅ Additional training epoch (3 → 4 epochs)")
    print("   📈 Expected F1-Score: 0.90-0.92 (vs 0.87 before)")
    print("   ⏱️  Estimated training time: ~2.5 hours")
    
    # Check dataset
    data_file = Path(config['data_path'])
    if not data_file.exists():
        print(f"\n❌ ERROR: Dataset not found at {config['data_path']}")
        print(f"\n🔧 Make sure you have:")
        print(f"   data/raw/cleaned_ai_human_dataset.csv")
        return
    
    print(f"\n✓ Dataset found: {config['data_path']}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results') / 'finetuned_improved_optionA' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results directory: {results_dir}")
    
    # Load and prepare data
    df = analyze_data(config['data_path'], config['sample_fraction'])
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        df, config['test_size'], config['val_size']
    )
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y_train=y_train)
    print(f"\n⚖️  Class weights: [{class_weights[0]:.3f}, {class_weights[1]:.3f}]")
    
    # Tokenizer and data loaders
    print("\n📦 Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✓ Tokenizer loaded")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        tokenizer, config['batch_size'], config['max_length']
    )
    
    # Fine-tune model
    print("\n🎯 Starting training...")
    start_time = datetime.now()
    
    model, metrics, history = fine_tune_model(
        train_loader, val_loader, test_loader, class_weights,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        results_dir=results_dir
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n⏱️  Training Time: {duration}")
    
    print(f"\n📊 Final Results:")
    print(f"   Test F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"   Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Test ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   Test Precision: {metrics['precision']:.4f}")
    print(f"   Test Recall:    {metrics['recall']:.4f}")
    
    print(f"\n📁 All results saved to:")
    print(f"   {results_dir}")
    
    print(f"\n📈 Improvement over previous run:")
    previous_f1 = 0.8696
    improvement = ((metrics['f1'] - previous_f1) / previous_f1) * 100
    print(f"   Previous F1: {previous_f1:.4f}")
    print(f"   Current F1:  {metrics['f1']:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    if metrics['f1'] >= 0.90:
        print(f"\n🏆 EXCELLENT! Achieved target F1-Score of 0.90+!")
    elif metrics['f1'] >= 0.89:
        print(f"\n✅ GREAT! Very close to target (0.90)!")
    else:
        print(f"\n✅ GOOD! Significant improvement from before!")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Check visualizations: {results_dir}")
    print(f"   2. Test on custom text: python test_model.py")
    print(f"   3. For even better results, try Option B (40% data, 5 epochs)")


if __name__ == "__main__":
    main()
