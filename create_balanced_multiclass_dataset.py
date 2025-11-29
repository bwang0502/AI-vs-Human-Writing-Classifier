"""
Create a balanced 7-class dataset for multi-model classification
MEMORY EFFICIENT VERSION - reads in chunks and filters on-the-fly
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Target samples per class for balanced dataset
SAMPLES_PER_CLASS = 1080
TOTAL_SAMPLES = SAMPLES_PER_CLASS * 7  # 7,560 total

# Define model groupings for 7 classes
MODEL_MAPPING = {
    # Class 0: Human
    'Human': 'Human',
    
    # Class 1: GPT-3.5 (including Text-Davinci variants)
    'GPT-3.5': 'GPT-3.5',
    'Text-Davinci-003': 'GPT-3.5',
    'Text-Davinci-002': 'GPT-3.5',
    'Text-Davinci-001': 'GPT-3.5',
    
    # Class 2: GPT-4
    'GPT-4': 'GPT-4',
    
    # Class 3: Claude (all variants)
    'Claude-v1': 'Claude',
    'Claude-Instant-v1': 'Claude',
    
    # Class 4: LLaMA Family (all LLaMA variants)
    'LLaMA-7B': 'LLaMA',
    'LLaMA-13B': 'LLaMA',
    'LLaMA-30B': 'LLaMA',
    'LLaMA-65B': 'LLaMA',
    'LLaMA-2-7B': 'LLaMA',
    'LLaMA-2-70B': 'LLaMA',
    'Nous-Hermes-LLaMA-2-13B': 'LLaMA',
    'Nous-Hermes-LLaMA-2-70B': 'LLaMA',
    
    # Class 5: Mistral/Mixtral
    'Mistral-7B': 'Mistral',
    'Mistral-7B-OpenOrca': 'Mistral',
    'Mixtral-8x7B': 'Mistral',
    'Dolphin-Mixtral-8x7B': 'Mistral',
    'Dolphin-2.5-Mixtral-8x7B': 'Mistral',
    'OpenHermes-2-Mistral-7B': 'Mistral',
    'OpenHermes-2.5-Mistral-7B': 'Mistral',
    'Toppy-M-7B': 'Mistral',
}

def map_model_to_class(model_name):
    """Map a model name to one of our 7 classes"""
    return MODEL_MAPPING.get(model_name, 'Other AI')

def create_balanced_dataset_efficient(input_path, output_path, samples_per_class=SAMPLES_PER_CLASS):
    """
    Create a balanced 7-class dataset using chunk reading (memory efficient)
    """
    print("="*70)
    print("CREATING BALANCED 7-CLASS DATASET (MEMORY EFFICIENT)")
    print("="*70)
    
    # Initialize storage for each class
    class_samples = {
        'Claude': [],
        'GPT-3.5': [],
        'GPT-4': [],
        'Human': [],
        'LLaMA': [],
        'Mistral': [],
        'Other AI': []
    }
    
    # Track how many we need per class
    needed = {cls: samples_per_class for cls in class_samples.keys()}
    
    print(f"\n📂 Reading {input_path} in chunks...")
    print(f"   Target: {samples_per_class:,} samples per class\n")
    
    chunk_size = 50000  # Read 50k rows at a time
    total_read = 0
    
    # Read file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size), 1):
        total_read += len(chunk)
        
        # Standardize column names
        if 'source' in chunk.columns:
            chunk = chunk.rename(columns={'source': 'model_name'})
        if 'text' in chunk.columns:
            chunk = chunk.rename(columns={'text': 'text_content'})
        
        # Map to classes
        chunk['model_class'] = chunk['model_name'].apply(map_model_to_class)
        
        # Sample from each class in this chunk
        for class_name in class_samples.keys():
            if needed[class_name] > 0:
                class_chunk = chunk[chunk['model_class'] == class_name]
                
                if len(class_chunk) > 0:
                    # Take what we need (or all available)
                    to_take = min(needed[class_name], len(class_chunk))
                    sampled = class_chunk.sample(n=to_take, random_state=42)
                    class_samples[class_name].append(sampled)
                    needed[class_name] -= to_take
        
        # Check if we're done
        if all(n == 0 for n in needed.values()):
            print(f"✅ Collected all samples after reading {total_read:,} rows")
            break
        
        # Progress update
        collected = sum(samples_per_class - needed[cls] for cls in needed)
        print(f"   Chunk {chunk_num}: Read {total_read:,} rows, collected {collected:,}/{TOTAL_SAMPLES:,} samples")
    
    # Combine all samples
    print("\n🔀 Combining samples from all classes...")
    balanced_dfs = []
    
    print("\n📊 Final class distribution:")
    for class_name in sorted(class_samples.keys()):
        if class_samples[class_name]:
            class_df = pd.concat(class_samples[class_name], ignore_index=True)
            balanced_dfs.append(class_df)
            print(f"   {class_name}: {len(class_df):,} samples")
        else:
            print(f"   ⚠️  {class_name}: 0 samples found")
    
    # Combine and shuffle
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create numeric labels
    print("\n🏷️  Creating numeric labels...")
    label_mapping = {name: idx for idx, name in enumerate(sorted(final_df['model_class'].unique()))}
    final_df['label_numeric'] = final_df['model_class'].map(label_mapping)
    
    # Keep only necessary columns
    final_df = final_df[['text_content', 'model_class', 'label_numeric', 'model_name']]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved balanced dataset to {output_path}")
    print(f"   Total samples: {len(final_df):,}")
    
    # Save label mapping
    label_map_path = output_path.parent / "multiclass_label_mapping.txt"
    final_counts = final_df['model_class'].value_counts().sort_index()
    with open(label_map_path, 'w') as f:
        f.write("Label Mapping for 7-Class Classification\n")
        f.write("="*50 + "\n\n")
        for name, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = final_counts.get(name, 0)
            f.write(f"{idx}: {name} ({count:,} samples)\n")
    
    print(f"\n💾 Label mapping saved to {label_map_path}")
    print("\n" + "="*70)
    print("✨ DATASET CREATION COMPLETE!")
    print("="*70)
    
    return final_df, label_mapping

if __name__ == "__main__":
    # Paths
    input_csv = Path("data/raw/ai_human_writing_detection.csv")
    output_csv = Path("data/processed/balanced_7class_dataset.csv")
    
    if not input_csv.exists():
        print(f"❌ ERROR: File not found: {input_csv}")
        exit(1)
    
    # Create balanced dataset efficiently
    df, label_map = create_balanced_dataset_efficient(input_csv, output_csv)
    
    print("\n📋 Final Label Mapping:")
    final_counts = df['model_class'].value_counts().sort_index()
    for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"   {idx}: {name} ({final_counts.get(name, 0):,} samples)")
