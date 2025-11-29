"""
Train a 7-class DistilBERT model for multi-model classification
Classes: Claude, GPT-3.5, GPT-4, Human, LLaMA, Mistral, Other AI
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW  # CHANGED: Import from torch.optim instead of transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
NUM_CLASSES = 7

# Paths
DATA_PATH = Path("data/raw/balanced_7class_dataset.csv")
OUTPUT_DIR = Path("results/multiclass_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamped folder for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = OUTPUT_DIR / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length):
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
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"   Total samples: {len(df):,}")
    print(f"\n📊 Class distribution:")
    class_counts = df['model_class'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count:,}")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text_content'].values,
        df['label_numeric'].values,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=df['label_numeric'].values
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )
    
    print(f"\n✂️  Data split:")
    print(f"   Train: {len(train_texts):,}")
    print(f"   Validation: {len(val_texts):,}")
    print(f"   Test: {len(test_texts):,}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), df

def create_data_loaders(train_data, val_data, test_data, tokenizer):
    """Create PyTorch data loaders"""
    print("\n🔄 Creating data loaders...")
    
    train_dataset = TextDataset(train_data[0], train_data[1], tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_data[0], val_data[1], tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_data[0], test_data[1], tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, data_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1, predictions, true_labels

def plot_training_history(history, run_dir):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(history['train_f1'], label='Train F1')
    axes[2].plot(history['val_f1'], label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training and Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(run_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"   📈 Saved training history plot")

def plot_confusion_matrix(y_true, y_pred, class_names, run_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(run_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Saved confusion matrix")

def main():
    print("="*70)
    print("TRAINING 7-CLASS MULTICLASS MODEL")
    print("="*70)
    
    # Load data
    train_data, val_data, test_data, df = load_and_prepare_data()
    
    # Get class names
    class_mapping = df[['label_numeric', 'model_class']].drop_duplicates().sort_values('label_numeric')
    class_names = class_mapping['model_class'].tolist()
    
    # Load tokenizer and model
    print("\n🤖 Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=NUM_CLASSES
    )
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🏋️  Training for {EPOCHS} epochs...")
    print("="*70)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device)
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_names': class_names
            }, run_dir / 'best_model.pt')
            print(f"   ✅ Saved best model (Val F1: {val_f1:.4f})")
    
    # Test evaluation
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(run_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"\n   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Save plots
    print("\n💾 Saving visualizations...")
    plot_training_history(history, run_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, run_dir)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_val_f1),
        'num_classes': NUM_CLASSES,
        'class_names': class_names
    }
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n✅ Training complete! Results saved to: {run_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
