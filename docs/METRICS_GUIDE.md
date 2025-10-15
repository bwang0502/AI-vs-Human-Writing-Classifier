# Understanding ROC-AUC and PR-AUC Metrics

## 📊 Evaluation Metrics Guide

This document explains the evaluation metrics used in the baseline model training.

---

## 1. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

### What it measures:
- **Overall model discrimination ability** across all classification thresholds
- How well the model separates positive class (AI) from negative class (Human)

### Range:
- **0.5**: Random guessing (no better than flipping a coin)
- **1.0**: Perfect classification
- **Typical good score**: 0.80 - 0.95

### When to use:
- When you care about overall model performance
- When classes are **balanced** (roughly equal Human/AI samples)
- When false positives and false negatives are equally important

### ROC Curve:
- **X-axis**: False Positive Rate (FPR) = False Positives / (False Positives + True Negatives)
- **Y-axis**: True Positive Rate (TPR) = True Positives / (True Positives + False Negatives)
- **Diagonal line**: Random classifier baseline
- **Better models**: Curve closer to top-left corner

### Interpretation:
```
ROC-AUC = 0.95  →  Excellent! Model clearly separates AI from Human
ROC-AUC = 0.85  →  Very good! Strong discrimination
ROC-AUC = 0.75  →  Good, but room for improvement
ROC-AUC = 0.60  →  Weak, only slightly better than random
ROC-AUC = 0.50  →  No better than random guessing
```

---

## 2. PR-AUC (Precision-Recall - Area Under Curve)

### What it measures:
- **Trade-off between precision and recall** at different thresholds
- Particularly useful for **imbalanced datasets**

### Range:
- **Baseline**: Proportion of positive class (e.g., 0.50 if 50% are AI)
- **1.0**: Perfect precision and recall
- **Typical good score**: 0.75 - 0.95

### When to use:
- When classes are **imbalanced** (more Human than AI, or vice versa)
- When you care more about the **positive class** (AI-generated text)
- When false positives are costly

### PR Curve:
- **X-axis**: Recall = True Positives / (True Positives + False Negatives)
- **Y-axis**: Precision = True Positives / (True Positives + False Positives)
- **Horizontal line**: Random classifier baseline (= positive class proportion)
- **Better models**: Curve stays high across all recall values

### Interpretation:
```
PR-AUC = 0.95   →  Excellent! High precision and recall maintained
PR-AUC = 0.85   →  Very good! Strong performance
PR-AUC = 0.75   →  Good, reasonable precision/recall trade-off
PR-AUC = 0.60   →  Moderate, some room for improvement
PR-AUC = 0.50   →  Baseline (for balanced dataset)
```

---

## 3. Comparison: ROC-AUC vs PR-AUC

| Aspect | ROC-AUC | PR-AUC |
|--------|---------|--------|
| **Focus** | Overall discrimination | Positive class performance |
| **Best for** | Balanced datasets | Imbalanced datasets |
| **Sensitivity to imbalance** | Less sensitive | More sensitive |
| **Use when** | FP and FN equally important | FP are costly |
| **Baseline** | Always 0.5 | Equals positive class proportion |

---

## 4. Other Metrics Explained

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: 0.0 to 1.0
- **Good score**: > 0.80 for this task
- **Warning**: Can be misleading on imbalanced datasets

### F1 Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0
- **Good score**: > 0.80
- **Use**: Balances precision and recall into single metric

### Precision
- **Formula**: TP / (TP + FP)
- **Meaning**: Of all predicted AI texts, how many were actually AI?
- **High precision**: Few false alarms

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Meaning**: Of all actual AI texts, how many did we catch?
- **High recall**: Few missed AI texts

---

## 5. What to Look For in Results

### Excellent Model (Production-ready):
```
Accuracy:  > 0.90
F1 Score:  > 0.90
ROC-AUC:   > 0.95
PR-AUC:    > 0.93
```

### Good Model (Promising):
```
Accuracy:  0.80 - 0.90
F1 Score:  0.80 - 0.90
ROC-AUC:   0.85 - 0.95
PR-AUC:    0.80 - 0.93
```

### Needs Improvement:
```
Accuracy:  < 0.80
F1 Score:  < 0.80
ROC-AUC:   < 0.85
PR-AUC:    < 0.80
```

---

## 6. Interpreting Your Results

### If ROC-AUC is high but PR-AUC is low:
- Model struggles with precision or recall trade-off
- May have issues with the positive class
- Consider adjusting classification threshold

### If both ROC-AUC and PR-AUC are high:
- Excellent model! Good discrimination and precision/recall balance
- Ready for deployment considerations

### If ROC-AUC is similar across models but PR-AUC differs:
- PR-AUC better captures performance on positive class
- Choose model with higher PR-AUC for better AI detection

---

## 7. Next Steps Based on Metrics

### High metrics (> 0.90):
✅ Model is working well!
- Fine-tune hyperparameters
- Try ensemble methods
- Focus on edge cases

### Medium metrics (0.75 - 0.90):
⚠️ Room for improvement
- Feature engineering
- Try different algorithms
- Collect more data

### Low metrics (< 0.75):
❌ Significant improvement needed
- Review feature selection
- Check for data quality issues
- Consider advanced models (BERT, transformers)

---

## 📚 Quick Reference

**For your AI vs Human classifier:**

1. **ROC-AUC** → How well can the model separate AI from Human overall?
2. **PR-AUC** → How precisely can the model identify AI text?
3. **Both high** → Excellent classifier!
4. **ROC-AUC > PR-AUC** → Good overall, but check precision/recall balance
5. **PR-AUC > ROC-AUC** → Unusual, double-check results

**Target scores for this project:**
- ROC-AUC: Aim for **> 0.85**
- PR-AUC: Aim for **> 0.80**
- Both together give complete picture of model performance!
