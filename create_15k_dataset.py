"""
Create the EXACT same 15K dataset used in finetune_improved.py training
This ensures visualizations are consistent with the trained model
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_15k_dataset():
    """Create the exact 15K dataset used for training (30% sample with seed=42)."""
    print("\n" + "="*70)
    print("CREATING 15K DATASET (MATCHING TRAINING DATA)")
    print("="*70)
    
    # Use the SAME path as finetune_improved.py
    data_path = 'data/raw/cleaned_ai_human_dataset.csv'
    
    if not Path(data_path).exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print(f"\n💡 This should be the same file used in finetune_improved.py")
        print(f"\n📁 Available files in data/raw/:")
        data_dir = Path('data/raw')
        if data_dir.exists():
            for f in data_dir.glob('*.csv'):
                print(f"   - {f.name}")
        return
    
    print(f"\n✓ Found dataset: {data_path}")
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df):,}")
    
    # Apply EXACT same sampling as finetune_improved.py (30% with seed=42)
    print(f"\n🎯 Applying 30% sample (matching training script)...")
    np.random.seed(42)  # Same seed
    df_15k = df.sample(frac=0.3, random_state=42)  # Same parameters
    
    print(f"✓ Sampled {len(df_15k):,} rows")
    
    # Check columns and standardize
    if 'text_content' not in df_15k.columns:
        if 'text' in df_15k.columns:
            df_15k = df_15k.rename(columns={'text': 'text_content'})
    
    if 'label' not in df_15k.columns:
        if 'generated' in df_15k.columns:
            df_15k = df_15k.rename(columns={'generated': 'label'})
    
    # Verify
    print(f"\n📊 Dataset info:")
    print(f"   Columns: {df_15k.columns.tolist()}")
    print(f"   Human (0): {(df_15k['label'] == 0).sum():,}")
    print(f"   AI (1): {(df_15k['label'] == 1).sum():,}")
    
    # Save
    output_path = Path('data/raw/cleaned_ai_human_15k.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_15k.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n✨ This dataset is EXACTLY the same as used in training!")
    print(f"   (30% sample with random_state=42)")
    
    # Statistics
    df_15k['text_length'] = df_15k['text_content'].str.len()
    print(f"\n📏 Text statistics:")
    print(f"   Mean length: {df_15k['text_length'].mean():.0f} chars")
    print(f"   Median length: {df_15k['text_length'].median():.0f} chars")
    
    print(f"\n📝 First few samples:")
    print(df_15k[['text_content', 'label']].head(3))


if __name__ == "__main__":
    create_15k_dataset()
    
    # Save to data/raw/
    output_path = Path('data/raw/cleaned_ai_human_15k.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_15k.to_csv(output_path, index=False)
    print(f"\n✅ Saved 15K dataset to: {output_path}")
    
    # Show sample
    print(f"\n📝 Sample data:")
    print(df_15k.head(3)[['text_content', 'label']])
    
    # Show statistics
    print(f"\n📊 Text length statistics:")
    df_15k['text_length'] = df_15k['text_content'].str.len()
    print(f"   Mean: {df_15k['text_length'].mean():.0f} chars")
    print(f"   Median: {df_15k['text_length'].median():.0f} chars")
    print(f"   Min: {df_15k['text_length'].min():.0f} chars")
    print(f"   Max: {df_15k['text_length'].max():.0f} chars")
    
    print(f"\n💡 You can now use this dataset for:")
    print(f"   - Fine-tuning: python finetune_improved.py")
    print(f"   - Visualizations: python visualize_transformer.py")


if __name__ == "__main__":
    create_15k_dataset()
