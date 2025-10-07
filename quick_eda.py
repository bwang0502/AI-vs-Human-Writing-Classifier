#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    print("🔍 AI vs Human Content Detection - Dataset Analysis")
    print("=" * 60)
    
    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv('data/raw/ai_human_content_detection_dataset.csv')
    
    # Basic info
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]:,} samples × {df.shape[1]} features")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Columns
    print(f"\n📋 FEATURES ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")
    
    # Label distribution
    print(f"\n🎯 LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    for label, count in label_counts.items():
        label_name = "Human" if label == 0 else "AI-Generated"
        pct = (count / total) * 100
        print(f"  {label_name} ({label}): {count:,} samples ({pct:.1f}%)")
    
    # Check balance
    balance_ratio = min(label_counts) / max(label_counts)
    print(f"  Balance ratio: {balance_ratio:.3f}")
    if balance_ratio > 0.8:
        print("  ✅ Dataset is well balanced")
    elif balance_ratio > 0.5:
        print("  ⚠️  Dataset is moderately imbalanced")
    else:
        print("  ❌ Dataset is highly imbalanced")
    
    # Text analysis
    df['text_length'] = df['text_content'].str.len()
    print(f"\n📝 TEXT CONTENT ANALYSIS:")
    print(f"  Mean length: {df['text_length'].mean():.0f} characters")
    print(f"  Median length: {df['text_length'].median():.0f} characters") 
    print(f"  Min length: {df['text_length'].min()} characters")
    print(f"  Max length: {df['text_length'].max():,} characters")
    print(f"  Standard deviation: {df['text_length'].std():.0f} characters")
    
    # Missing values
    print(f"\n🔍 MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values found!")
    else:
        for col, count in missing[missing > 0].items():
            pct = (count / total) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Sample texts
    print(f"\n📖 SAMPLE TEXTS:")
    print(f"\n--- HUMAN SAMPLE ---")
    human_sample = df[df['label'] == 0]['text_content'].iloc[0]
    print(human_sample[:300] + ("..." if len(human_sample) > 300 else ""))
    
    print(f"\n--- AI-GENERATED SAMPLE ---") 
    ai_sample = df[df['label'] == 1]['text_content'].iloc[0]
    print(ai_sample[:300] + ("..." if len(ai_sample) > 300 else ""))
    
    # Feature stats
    print(f"\n📈 LINGUISTIC FEATURES COMPARISON:")
    numeric_cols = ['word_count', 'character_count', 'sentence_count', 
                   'lexical_diversity', 'avg_sentence_length', 'avg_word_length',
                   'flesch_reading_ease', 'sentiment_score']
    
    comparison = df.groupby('label')[numeric_cols].mean()
    print("\nMean values by label:")
    print(comparison.round(2))
    
    print(f"\n🎉 SUMMARY:")
    print(f"  • {len(df):,} total samples with rich linguistic features")
    print(f"  • {len(numeric_cols)} quantitative features for analysis")
    print(f"  • Balanced dataset ({label_counts[0]:,} human, {label_counts[1]:,} AI)")
    print(f"  • Ready for machine learning experiments!")
    
    print(f"\n✅ EDA complete! Check out the Jupyter notebook for detailed analysis:")
    print(f"   jupyter notebook notebooks/exploratory/dataset_eda.ipynb")

if __name__ == "__main__":
    main()
