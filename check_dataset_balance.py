"""
Quick script to check the balance of the existing 15k dataset
"""

import pandas as pd
from collections import Counter

# Load the dataset
df = pd.read_csv('data/raw/cleaned_ai_human_15k.csv')

print("="*60)
print("DATASET ANALYSIS - cleaned_ai_human_15k.csv")
print("="*60)

# Basic info
print(f"\nTotal samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check label distribution (binary: human vs AI)
print("\n" + "="*60)
print("BINARY CLASSIFICATION (Human vs AI)")
print("="*60)
label_counts = df['label'].value_counts()
print(f"\nLabel distribution:")
print(label_counts)
print(f"\nPercentages:")
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    label_name = "Human" if label == 0 else "AI"
    print(f"  {label_name} (label={label}): {count} ({percentage:.2f}%)")

# Check model_name distribution (multi-class)
print("\n" + "="*60)
print("MULTI-CLASS DISTRIBUTION (By Model)")
print("="*60)
model_counts = df['model_name'].value_counts()
print(f"\nNumber of unique models: {len(model_counts)}")
print(f"\nTop 20 models:")
print(model_counts.head(20))

print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print(f"Mean samples per model: {model_counts.mean():.2f}")
print(f"Median samples per model: {model_counts.median():.2f}")
print(f"Min samples: {model_counts.min()}")
print(f"Max samples: {model_counts.max()}")

# Check for 'Human' specifically
if 'Human' in model_counts.index:
    human_count = model_counts['Human']
    print(f"\nHuman samples: {human_count}")
    print(f"AI samples (total): {len(df) - human_count}")
else:
    print("\nNote: 'Human' not found as model_name value")

print("\n" + "="*60)
