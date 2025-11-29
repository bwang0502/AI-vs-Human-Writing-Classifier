"""
IMPROVED Fine-tuning script with better hyperparameters and debugging.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sns.set_style('whitegrid')


class TextDataset(Dataset):
    """Custom Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
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


def analyze_data(data_path):
    """Analyze dataset before training."""
    print("\n" + "=" * 70)
    print("DATA ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(data_path)
    
    print(f"\n✓ Total samples: {len(df)}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        label_name = "Human" if label == 0 else "AI"
        print(f"   {label_name} (Class {label}): {count} ({count/len(df)*100:.1f}%)")
    
    # Check for missing values
    print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")
    
    # Text length statistics
    df['text_length'] = df['text_content'].str.len()
    print(f"\n📏 Text Length Statistics:")
    print(f"   Mean: {df['text_length'].mean():.0f} chars")
    print(f"   Median: {df['text_length'].median():.0f} chars")
    print(f"   Min: {df['text_length'].min():.0f} chars")
    print(f"   Max: {df['text_length'].max():.0f} chars")
    
    # Sample a few texts
    print(f"\n📝 Sample Texts:")
    for i, row in df.head(3).iterrows():
        print(f"\n   [{i}] Label: {row['label']} | Length: {len(row['text_content'])} chars")
        print(f"       Text: {row['text_content'][:100]}...")
    
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
    
    print(f"\n✓ Train set: {len(X_train)} samples")
    print(f"   Class 0 (Human): {(y_train == 0).sum()}, Class 1 (AI): {(y_train == 1).sum()}")
    print(f"✓ Val set: {len(X_val)} samples")
    print(f"   Class 0 (Human): {(y_val == 0).sum()}, Class 1 (AI): {(y_val == 1).sum()}")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"   Class 0 (Human): {(y_test == 0).sum()}, Class 1 (AI): {(y_test == 1).sum()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                        tokenizer, batch_size=32, max_length=256):
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
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    print(f"✓ Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    # Create weighted loss
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
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
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
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
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1, 1].plot(epochs, history['val_roc_auc'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history: {save_path}")


def plot_evaluation(y_true, y_pred, y_proba, metrics, model_name, save_path):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'ROC Curve - {model_name}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1, 0].plot(recall, precision, label=f'PR (AUC = {metrics["pr_auc"]:.3f})', linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                     metrics['f1'], metrics['roc_auc'], metrics['pr_auc']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = axes[1, 1].barh(metric_names, metric_values, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_title(f'Metrics Summary - {model_name}')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved evaluation plots: {save_path}")


def fine_tune_model(model_name, train_loader, val_loader, test_loader, class_weights,
                    num_epochs=5, learning_rate=5e-5, results_dir=None):
    """Fine-tune transformer model."""
    print("\n" + "=" * 70)
    print(f"FINE-TUNING: {model_name.upper()}")
    print("=" * 70)
    
    # Load model
    if 'roberta' in model_name.lower():
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    elif 'distilbert' in model_name.lower():
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    print(f"✓ Loaded {model_name}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"✓ Total training steps: {total_steps:,}")
    print(f"✓ Warmup steps: {int(0.1 * total_steps):,}")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_roc_auc': []
    }
    
    best_val_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print("=" * 70)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        print(f"\n✓ Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        print(f"✓ Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✅ NEW BEST MODEL! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"⏸️  No improvement (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\n✓ Loaded best model (Val F1: {best_val_f1:.4f})")
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    
    test_metrics, test_preds, test_probs, test_labels = evaluate(model, test_loader, device)
    
    print(f"\n📊 Test Set Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {test_metrics['pr_auc']:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Human', 'AI']))
    
    # Save results
    if results_dir:
        model_path = results_dir / f"{model_name}_finetuned.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'metrics': test_metrics,
            'history': history,
            'config': {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'best_val_f1': best_val_f1
            }
        }, model_path)
        print(f"\n✓ Saved model: {model_path}")
        
        plot_training_history(history, results_dir / f"{model_name}_history.png")
        plot_evaluation(test_labels, test_preds, test_probs, test_metrics, 
                       model_name, results_dir / f"{model_name}_eval.png")
    
    return model, test_metrics, history


def main():
    """Main pipeline."""
    print("\n" + "=" * 70)
    print("IMPROVED TRANSFORMER FINE-TUNING")
    print("=" * 70)
    
    # Configuration
    config = {
        'data_path': 'data/raw/cleaned_ai_human_dataset.csv',  # ✅ FIXED
        'batch_size': 32,
        'max_length': 256,
        'num_epochs': 5,
        'learning_rate': 5e-5,
        'test_size': 0.2,
        'val_size': 0.1
    }
    
    print("\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Check if dataset exists
    data_file = Path(config['data_path'])
    if not data_file.exists():
        print(f"\n❌ ERROR: Dataset not found!")
        print(f"   Looking for: {config['data_path']}")
        print(f"\n📁 Available files in data/raw/:")
        data_dir = Path('data/raw')
        if data_dir.exists():
            for f in data_dir.glob('*.csv'):
                print(f"   - {f.name}")
        else:
            print(f"   (data/raw/ directory not found)")
        print(f"\n🔧 Run preparation script first:")
        print(f"   python prepare_large_dataset.py")
        return
    
    print(f"✓ Dataset found: {config['data_path']}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results') / 'finetuned_models_v2' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results directory: {results_dir}")
    
    # Analyze and load data
    df = analyze_data(config['data_path'])
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        df, config['test_size'], config['val_size']
    )
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"\n⚖️  Class weights: {class_weights}")
    
    # Models to train
    models_to_train = ['distilbert']  # Start with DistilBERT
    
    all_results = []
    
    for model_name in models_to_train:
        print("\n" + "#" * 70)
        print(f"MODEL: {model_name.upper()}")
        print("#" * 70)
        
        # Load tokenizer
        if model_name == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            tokenizer, config['batch_size'], config['max_length']
        )
        
        # Fine-tune model
        model, metrics, history = fine_tune_model(
            model_name, train_loader, val_loader, test_loader, class_weights,
            num_epochs=config['num_epochs'], 
            learning_rate=config['learning_rate'],
            results_dir=results_dir
        )
        
        result = {'model': model_name, **metrics}
        all_results.append(result)
    
    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'final_results.csv', index=False)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    print(f"\n✅ Training complete!")
    print(f"📁 All results saved to: {results_dir}")


if __name__ == "__main__":
    main()