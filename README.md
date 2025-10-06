# AI vs Human Writing Classifier

A machine learning project to classify text as either AI-generated or human-written using various NLP techniques and models.

## 🎯 Project Overview

This project implements multiple approaches to distinguish between AI-generated and human-written text:

- **Traditional ML Models**: Logistic Regression, Random Forest, SVM with linguistic features
- **Deep Learning Models**: BERT and other transformer-based models
- **Feature Engineering**: Stylometric features, readability metrics, and linguistic patterns
- **Comprehensive Evaluation**: Cross-validation, feature importance analysis, and performance metrics

## 📁 Project Structure

```
AI-vs-Human-Writing-Classifier/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Preprocessed datasets
├── src/
│   ├── data/                   # Data processing modules
│   ├── models/                 # Model implementations
│   ├── evaluation/             # Evaluation utilities
│   └── utils/                  # Utility functions
├── notebooks/
│   ├── exploratory/            # Data exploration notebooks
│   └── experiments/            # Model experiments
├── models/
│   ├── trained/                # Saved trained models
│   └── checkpoints/            # Model checkpoints
├── config/                     # Configuration files
├── results/                    # Experiment results
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv ai-human-classifier
source ai-human-classifier/bin/activate  # On Windows: ai-human-classifier\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 2. Data Preparation

Place your datasets in the `data/raw/` directory. Expected format:
- CSV file with columns: `text` and `label`
- Labels: 0 for human-written, 1 for AI-generated

### 3. Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Training settings
- Data paths
- Feature extraction options

### 4. Usage Examples

```python
from src.data import TextPreprocessor, load_data, split_data
from src.models import RandomForestClassifierWrapper
from src.evaluation import ModelEvaluator

# Load and preprocess data
df = load_data('data/raw/your_dataset.csv')
train_df, val_df, test_df = split_data(df)

# Feature extraction
preprocessor = TextPreprocessor()
# ... feature extraction code

# Train model
model = RandomForestClassifierWrapper()
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_predictions(y_test, y_pred, y_proba)
```

## 🔧 Features

### Data Processing
- Text cleaning and normalization
- Train/validation/test splitting
- Feature extraction pipeline

### Feature Engineering
- **Linguistic Features**: Word count, sentence length, vocabulary richness
- **Stylometric Features**: Punctuation patterns, readability scores
- **Text Vectorization**: TF-IDF, Count vectorization
- **Embeddings**: Support for transformer embeddings

### Models
- **Traditional ML**: Logistic Regression, Random Forest, SVM
- **Deep Learning**: BERT-based classifiers
- **Ensemble Methods**: Voting classifiers, stacking

### Evaluation
- Cross-validation
- Multiple metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrices
- Feature importance analysis
- Performance visualization

## 📊 Experiments

Use Jupyter notebooks in the `notebooks/` directory:

- **Exploratory Analysis**: Data distribution, feature correlation
- **Model Comparison**: Benchmark different approaches
- **Feature Importance**: Analyze what features matter most
- **Error Analysis**: Understand model failures

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Check style
flake8 src/

# Sort imports
isort src/
```

### Adding New Models

1. Inherit from `BaseClassifier` in `src/models/`
2. Implement required methods: `fit()`, `predict()`, `predict_proba()`
3. Add tests in `tests/`

## 📈 Results

Results are saved in the `results/` directory:
- Model performance metrics
- Confusion matrices
- Feature importance plots
- Training curves

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- scikit-learn for traditional ML models
- NLTK and spaCy for NLP utilities

## 📞 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/AI-vs-Human-Writing-Classifier](https://github.com/yourusername/AI-vs-Human-Writing-Classifier)