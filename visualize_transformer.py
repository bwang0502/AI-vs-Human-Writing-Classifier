"""
Generate SHAP/LIME visualizations for fine-tuned DistilBERT model
Analyzes influential words/features for AI vs Human classification
"""

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
from collections import Counter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_model(model_path):
    """Load the fine-tuned DistilBERT model."""
    print(f"\n📦 Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Training metrics: F1={checkpoint['metrics']['f1']:.4f}, "
          f"Accuracy={checkpoint['metrics']['accuracy']:.4f}")
    
    return model, checkpoint


def load_test_data(data_path, sample_size=1000):
    """Load test data for visualization."""
    print(f"\n📊 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Sample for faster visualization
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    texts = df['text_content'].tolist()
    labels = df['label'].tolist()
    
    print(f"✓ Using {len(df)} samples for visualization")
    print(f"   Human (0): {(df['label'] == 0).sum()}")
    print(f"   AI (1): {(df['label'] == 1).sum()}")
    
    return texts, labels


def create_prediction_function(model, tokenizer):
    """Create prediction function for LIME."""
    def predict_proba(texts):
        model.eval()
        probs = []
        
        with torch.no_grad():
            for text in texts:
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                probs.append(prob)
        
        return np.array(probs)
    
    return predict_proba


def generate_lime_explanations(model, tokenizer, texts, labels, output_dir, num_samples=10):
    """Generate LIME explanations for sample predictions."""
    print(f"\n🔍 Generating LIME explanations for {num_samples} samples...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predict_fn = create_prediction_function(model, tokenizer)
    explainer = LimeTextExplainer(class_names=['Human', 'AI'])
    
    # Get model predictions
    print("   Getting model predictions...")
    predictions = []
    for text in tqdm(texts, desc="Predicting"):
        probs = predict_fn([text])[0]
        predictions.append(np.argmax(probs))
    
    # Select diverse samples (correct and incorrect predictions)
    correct_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) if p == l]
    wrong_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l]
    
    # Mix of correct and wrong predictions
    num_correct = min(len(correct_indices), num_samples // 2)
    num_wrong = min(len(wrong_indices), num_samples - num_correct)
    
    sample_indices = correct_indices[:num_correct] + wrong_indices[:num_wrong]
    sample_indices = sample_indices[:num_samples]
    
    print(f"   Explaining {len(sample_indices)} samples ({num_correct} correct, {num_wrong} errors)...")
    
    for idx, text_idx in enumerate(sample_indices):
        text = texts[text_idx]
        true_label = labels[text_idx]
        pred_label = predictions[text_idx]
        
        print(f"   [{idx+1}/{len(sample_indices)}] Explaining sample {text_idx}...")
        
        # Generate explanation
        exp = explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=15,
            num_samples=500
        )
        
        # Save HTML
        html_path = output_dir / f'lime_explanation_{idx}.html'
        exp.save_to_file(str(html_path))
        
        # Create plot
        fig = exp.as_pyplot_figure()
        fig.suptitle(
            f'LIME Explanation {idx}\n'
            f'True: {["Human", "AI"][true_label]} | '
            f'Predicted: {["Human", "AI"][pred_label]}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f'lime_explanation_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved LIME explanations to {output_dir}")


def extract_important_tokens(model, tokenizer, texts, labels, top_n=50):
    """Extract most important tokens using attention weights."""
    print(f"\n🎯 Extracting important tokens from attention weights...")
    
    ai_tokens = []
    human_tokens = []
    
    model.eval()
    with torch.no_grad():
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Analyzing tokens"):
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get attention weights from last layer
            attentions = outputs.attentions[-1]  # Shape: [1, num_heads, seq_len, seq_len]
            
            # Average across heads and get attention to [CLS] token
            avg_attention = attentions.mean(dim=1).squeeze()  # Shape: [seq_len, seq_len]
            cls_attention = avg_attention[0, :]  # Attention from CLS to all tokens
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Extract important tokens (high attention, excluding special tokens)
            for token, attn_score in zip(tokens, cls_attention):
                if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] and len(token) > 2:
                    # Remove subword indicators
                    clean_token = token.replace('##', '')
                    
                    if label == 1:  # AI
                        ai_tokens.append(clean_token)
                    else:  # Human
                        human_tokens.append(clean_token)
    
    # Count frequencies
    ai_token_freq = Counter(ai_tokens).most_common(top_n)
    human_token_freq = Counter(human_tokens).most_common(top_n)
    
    print(f"✓ Extracted top {top_n} tokens for each class")
    print(f"   AI tokens: {len(ai_token_freq)}")
    print(f"   Human tokens: {len(human_token_freq)}")
    
    return ai_token_freq, human_token_freq


def plot_top_tokens(ai_tokens, human_tokens, output_path, top_n=20):
    """Plot top tokens for AI vs Human."""
    print(f"\n📊 Creating token importance visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # AI tokens
    tokens_ai, counts_ai = zip(*ai_tokens[:top_n])
    y_pos = np.arange(len(tokens_ai))
    ax1.barh(y_pos, counts_ai, color='red', alpha=0.7, edgecolor='darkred')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tokens_ai, fontsize=11)
    ax1.set_xlabel('Frequency in AI Texts', fontsize=13, fontweight='bold')
    ax1.set_title(f'Top {top_n} AI-Indicative Tokens\n(High Attention Words)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Human tokens
    tokens_human, counts_human = zip(*human_tokens[:top_n])
    y_pos = np.arange(len(tokens_human))
    ax2.barh(y_pos, counts_human, color='blue', alpha=0.7, edgecolor='darkblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tokens_human, fontsize=11)
    ax2.set_xlabel('Frequency in Human Texts', fontsize=13, fontweight='bold')
    ax2.set_title(f'Top {top_n} Human-Indicative Tokens\n(High Attention Words)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved token importance plot to {output_path}")


def create_word_clouds(texts, labels, output_dir):
    """Create word clouds for AI and Human texts."""
    print(f"\n☁️  Creating word clouds...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ai_texts = ' '.join([text for text, label in zip(texts, labels) if label == 1])
    human_texts = ' '.join([text for text, label in zip(texts, labels) if label == 0])
    
    # AI word cloud
    print("   Generating AI word cloud...")
    wordcloud_ai = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Reds',
        max_words=150,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False
    ).generate(ai_texts)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud_ai, interpolation='bilinear')
    plt.title('Most Common Words in AI-Generated Text', 
              fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'wordcloud_ai_transformer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Human word cloud
    print("   Generating Human word cloud...")
    wordcloud_human = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Blues',
        max_words=150,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False
    ).generate(human_texts)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud_human, interpolation='bilinear')
    plt.title('Most Common Words in Human-Written Text', 
              fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'wordcloud_human_transformer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved word clouds to {output_dir}")


def analyze_prediction_confidence(model, tokenizer, texts, labels, output_path):
    """Analyze prediction confidence distribution."""
    print(f"\n📈 Analyzing prediction confidence...")
    
    predict_fn = create_prediction_function(model, tokenizer)
    
    all_probs = []
    for text in tqdm(texts, desc="Getting probabilities"):
        probs = predict_fn([text])[0]
        all_probs.append(probs[1])  # Probability of AI class
    
    all_probs = np.array(all_probs)
    labels = np.array(labels)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall distribution
    ax1.hist(all_probs, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('Predicted Probability (AI)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Model Confidence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # By true class
    human_probs = all_probs[labels == 0]
    ai_probs = all_probs[labels == 1]
    
    ax2.hist(human_probs, bins=30, alpha=0.7, color='blue', 
             label=f'True Human (n={len(human_probs)})', edgecolor='black')
    ax2.hist(ai_probs, bins=30, alpha=0.7, color='red', 
             label=f'True AI (n={len(ai_probs)})', edgecolor='black')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Predicted Probability (AI)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence by True Class', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confidence distribution to {output_path}")


def main():
    """Main visualization pipeline for fine-tuned transformer."""
    print("\n" + "="*70)
    print("TRANSFORMER MODEL VISUALIZATIONS")
    print("Fine-tuned DistilBERT Analysis")
    print("="*70)
    
    # Configuration
    config = {
        'model_path': 'results/finetuned_improved_optionA/20251104_220909/distilbert_finetuned_improved.pt',
        'data_path': 'data/raw/cleaned_ai_human_15k.csv',
        'output_dir': 'visualizations/transformer',
        'num_samples': 1000,
        'num_lime_explanations': 10
    }
    
    print("\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Verify paths
    model_path = Path(config['model_path'])
    data_path = Path(config['data_path'])
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        return
    
    if not data_path.exists():
        print(f"\n❌ ERROR: Data not found at {data_path}")
        return
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, checkpoint = load_model(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load data
    texts, labels = load_test_data(data_path, config['num_samples'])
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. LIME Explanations
    generate_lime_explanations(
        model, tokenizer, texts, labels,
        output_dir / 'lime',
        num_samples=config['num_lime_explanations']
    )
    
    # 2. Token Importance (Attention-based)
    ai_tokens, human_tokens = extract_important_tokens(model, tokenizer, texts, labels, top_n=50)
    plot_top_tokens(ai_tokens, human_tokens, output_dir / 'token_importance.png', top_n=20)
    
    # 3. Word Clouds
    create_word_clouds(texts, labels, output_dir)
    
    # 4. Prediction Confidence
    analyze_prediction_confidence(model, tokenizer, texts, labels, 
                                  output_dir / 'prediction_confidence.png')
    
    # Summary
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All visualizations saved to:")
    print(f"   {output_dir.absolute()}")
    print(f"\n📊 Generated:")
    print(f"   ✓ {config['num_lime_explanations']} LIME explanations (lime/)")
    print(f"   ✓ Token importance chart (token_importance.png)")
    print(f"   ✓ Word clouds for AI vs Human (wordcloud_*.png)")
    print(f"   ✓ Prediction confidence analysis (prediction_confidence.png)")
    print(f"\n💡 To view:")
    print(f"   open '{output_dir.absolute()}'")
    
    print(f"\n📈 Top 10 AI-Indicative Tokens:")
    for token, count in ai_tokens[:10]:
        print(f"   • {token}: {count}")
    
    print(f"\n📈 Top 10 Human-Indicative Tokens:")
    for token, count in human_tokens[:10]:
        print(f"   • {token}: {count}")


if __name__ == "__main__":
    main()
