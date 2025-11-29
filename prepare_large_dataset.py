import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("PREPARING AI DETECTION DATASET")
print("=" * 70)

# Configuration
DATASET_PATH = 'data/raw/ai_human_writing_detection.csv'
TARGET_SIZE = 50000          # 50k samples (good balance of size/quality)
HUMAN_RATIO = 0.5            # 50% human, 50% AI
MIN_SAMPLES_PER_MODEL = 100  # At least 100 samples per AI model

print(f"\n⚙️  Configuration:")
print(f"   Dataset: {DATASET_PATH}")
print(f"   Target size: {TARGET_SIZE:,} samples")
print(f"   Human ratio: {HUMAN_RATIO:.0%}")
print(f"   Min per model: {MIN_SAMPLES_PER_MODEL}")

# Load dataset
print(f"\n📂 Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"✓ Loaded {len(df):,} samples")
print(f"✓ Columns: {df.columns.tolist()}")

# Show first few rows
print("\n📊 First 3 rows:")
print(df.head(3))

# Identify columns
print("\n🔍 Analyzing structure...")

# Find text column
text_col = None
for col in ['text', 'content', 'essay', 'prompt', 'response', 'text_content', 'writing']:
    if col in df.columns:
        text_col = col
        break

# Find label/source column
label_col = None
for col in ['source', 'label', 'model', 'generator', 'origin', 'author']:
    if col in df.columns:
        label_col = col
        break

if not text_col:
    print(f"\n❌ Could not find text column!")
    print(f"   Available columns: {df.columns.tolist()}")
    print(f"\n💡 Please manually specify TEXT_COL in the script")
    exit(1)

if not label_col:
    print(f"\n❌ Could not find label column!")
    print(f"   Available columns: {df.columns.tolist()}")
    print(f"\n💡 Please manually specify LABEL_COL in the script")
    exit(1)

print(f"✓ Text column: '{text_col}'")
print(f"✓ Label column: '{label_col}'")

# Check unique labels
print(f"\n🏷️  Unique labels found:")
unique_labels = df[label_col].unique()
label_counts = df[label_col].value_counts()
print(f"   Total unique: {len(unique_labels)}")
for label, count in label_counts.head(20).items():
    print(f"   {label}: {count:,}")

if len(unique_labels) > 20:
    print(f"   ... and {len(unique_labels) - 20} more")

# Create binary labels
print(f"\n🔄 Creating binary labels...")
df['model_name'] = df[label_col]  # Preserve original

# Map to binary (Human=0, AI=1)
df['is_ai'] = df[label_col].apply(
    lambda x: 0 if str(x).lower() in ['human', 'humans', 'person'] else 1
)

print(f"\n📊 Binary distribution:")
print(f"   Human (0): {(df['is_ai'] == 0).sum():,} ({(df['is_ai'] == 0).sum()/len(df)*100:.1f}%)")
print(f"   AI (1): {(df['is_ai'] == 1).sum():,} ({(df['is_ai'] == 1).sum()/len(df)*100:.1f}%)")

# Show AI models
ai_models = df[df['is_ai'] == 1]['model_name'].value_counts()
print(f"\n🤖 AI Models ({len(ai_models)} total):")
for i, (model, count) in enumerate(ai_models.head(15).items()):
    print(f"   {i+1}. {model}: {count:,}")

if len(ai_models) > 15:
    print(f"   ... and {len(ai_models) - 15} more models")

# Stratified sampling
print(f"\n🎯 Sampling strategy:")

n_human = int(TARGET_SIZE * HUMAN_RATIO)
n_ai = TARGET_SIZE - n_human

print(f"   Target human: {n_human:,}")
print(f"   Target AI: {n_ai:,}")

# Sample humans
human_df = df[df['is_ai'] == 0].copy()
if len(human_df) > n_human:
    human_sampled = human_df.sample(n=n_human, random_state=42)
    print(f"   ✓ Sampled {len(human_sampled):,} human texts")
else:
    human_sampled = human_df
    print(f"   ⚠️  Only {len(human_sampled):,} human texts available (using all)")

# Sample AI (proportionally from each model)
ai_df = df[df['is_ai'] == 1].copy()
model_counts = ai_df['model_name'].value_counts()

print(f"\n   📦 Sampling from AI models...")

# Calculate proportional samples
samples_per_model = {}
total_ai_available = len(ai_df)

for model, count in model_counts.items():
    proportion = count / total_ai_available
    n_samples = max(int(n_ai * proportion), MIN_SAMPLES_PER_MODEL)
    n_samples = min(n_samples, count)  # Can't exceed available
    samples_per_model[model] = n_samples

# Sample from each model
ai_samples = []
for model, n_samples in sorted(samples_per_model.items(), key=lambda x: x[1], reverse=True)[:10]:
    model_df = ai_df[ai_df['model_name'] == model]
    if len(model_df) >= n_samples:
        sampled = model_df.sample(n=n_samples, random_state=42)
        ai_samples.append(sampled)
        print(f"      {model}: {n_samples:,} samples")

# If showing top 10, sample from rest proportionally
if len(samples_per_model) > 10:
    other_models = [m for m in samples_per_model.keys() if m not in [x[0] for x in sorted(samples_per_model.items(), key=lambda x: x[1], reverse=True)[:10]]]
    other_df = ai_df[ai_df['model_name'].isin(other_models)]
    other_n = sum(samples_per_model[m] for m in other_models)
    if len(other_df) > other_n:
        other_sampled = other_df.sample(n=other_n, random_state=42)
        ai_samples.append(other_sampled)
        print(f"      ... {len(other_models)} other models: {other_n:,} samples")

ai_sampled = pd.concat(ai_samples, ignore_index=True)

# Downsample if too many
if len(ai_sampled) > n_ai:
    ai_sampled = ai_sampled.sample(n=n_ai, random_state=42)
    print(f"\n   ✓ Downsampled to {len(ai_sampled):,} AI samples")

# Combine and shuffle
df_balanced = pd.concat([human_sampled, ai_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Balanced dataset:")
print(f"   Total: {len(df_balanced):,}")
print(f"   Human: {(df_balanced['is_ai'] == 0).sum():,} ({(df_balanced['is_ai'] == 0).sum()/len(df_balanced)*100:.1f}%)")
print(f"   AI: {(df_balanced['is_ai'] == 1).sum():,} ({(df_balanced['is_ai'] == 1).sum()/len(df_balanced)*100:.1f}%)")

# Create clean dataset
df_clean = pd.DataFrame({
    'text_content': df_balanced[text_col],
    'label': df_balanced['is_ai'],
    'model_name': df_balanced['model_name']
})

# Data cleaning
print(f"\n🧹 Cleaning data...")
original_size = len(df_clean)

df_clean = df_clean.dropna(subset=['text_content'])
print(f"   ✓ Removed {original_size - len(df_clean)} NaN texts")

df_clean = df_clean[df_clean['text_content'].str.len() > 50]
print(f"   ✓ Removed {original_size - len(df_clean)} short texts (<50 chars)")

df_clean = df_clean.drop_duplicates(subset=['text_content'])
print(f"   ✓ Removed {original_size - len(df_clean)} duplicates")

print(f"\n✅ Final dataset: {len(df_clean):,} samples")

# Quality check
print(f"\n📝 Quality Check (random 3 samples):")
for i, (idx, row) in enumerate(df_clean.sample(3, random_state=42).iterrows()):
    label_str = "🤖 AI" if row['label'] == 1 else "👤 Human"
    model = row['model_name']
    text_preview = row['text_content'][:200].replace('\n', ' ')
    print(f"\n   [{i+1}] {label_str} ({model})")
    print(f"       Length: {len(row['text_content'])} chars")
    print(f"       Text: {text_preview}...")

# Statistics
print(f"\n📊 Final Statistics:")
print(f"   Total samples: {len(df_clean):,}")
print(f"   Human: {len(df_clean[df_clean['label']==0]):,}")
print(f"   AI: {len(df_clean[df_clean['label']==1]):,}")
print(f"   Unique AI models: {df_clean[df_clean['label']==1]['model_name'].nunique()}")
print(f"   Avg text length: {df_clean['text_content'].str.len().mean():.0f} chars")
print(f"   Median text length: {df_clean['text_content'].str.len().median():.0f} chars")

# Save
output_path = 'data/raw/cleaned_ai_human_dataset.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n✅ Saved: {output_path}")

# Save model distribution for future reference
model_dist = df_clean[df_clean['label']==1]['model_name'].value_counts().to_frame('count')
model_dist.to_csv('data/raw/model_distribution.csv')
print(f"✅ Saved model info: data/raw/model_distribution.csv")

print(f"\n🚀 Ready for training!")
print(f"   Run: python finetune_transformers_v2.py")

