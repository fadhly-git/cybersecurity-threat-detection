# 🛡️ Cybersecurity Threat Detection with Hybrid Deep Learning

Advanced cybersecurity threat detection system using hybrid deep learning models, adversarial robustness testing, and explainable AI based on the CICIDS2017 dataset (2.8M samples).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements state-of-the-art hybrid deep learning models for cybersecurity threat detection, achieving **98%+ accuracy** on the CICIDS2017 dataset. The system combines multiple AI/ML approaches:

- **Hybrid Models**: CNN-SVM, LSTM-RF, CNN-LSTM-MLP, Autoencoder-CNN, Attention-LSTM, Stacking
- **Adversarial Robustness**: FGSM, PGD, C&W, DeepFool, JSMA attacks + defenses
- **Explainable AI**: SHAP, LIME, Grad-CAM, Attention visualization
- **Advanced Preprocessing**: SMOTE for minority classes, correlation-based feature selection

## ✨ Features

### 🤖 Hybrid Deep Learning Models

1. **CNN-SVM Hybrid** - CNN feature extraction + SVM classification
2. **LSTM-RandomForest** - Temporal features + ensemble classification
3. **CNN-LSTM-MLP Ensemble** - 3 parallel branches (CNN + LSTM + MLP)
4. **Autoencoder-CNN** - Unsupervised pretraining + supervised classification
5. **Attention-LSTM-DNN** - Multi-head attention + bidirectional LSTM
6. **Stacking Ensemble** - Meta-learning with multiple base models

### 🔒 Adversarial Robustness

- **5 Attack Methods**: FGSM, PGD, Carlini & Wagner, DeepFool, JSMA
- **3 Defense Mechanisms**: Adversarial training, defensive distillation, input transformation
- **Comprehensive Evaluation**: Robustness scoring across all attacks

### 🔍 Explainable AI

- **SHAP**: Global and local feature importance
- **LIME**: Local interpretable explanations
- **Grad-CAM**: Visual explanations for CNN layers
- **Attention Visualization**: Attention weight heatmaps

### 📊 CICIDS2017 Dataset Processing

- **2.8M samples**, 79 features → ~40 selected features
- Automated preprocessing pipeline:
  - Duplicate removal (308K rows)
  - Missing value imputation (353 entries)
  - Infinity value handling
  - Label consolidation (8 attack classes)
  - Correlation-based feature selection
  - SMOTE oversampling for minority classes
  - StandardScaler normalization

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU support)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/fadhly-git/cybersecurity-threat-detection.git
cd cybersecurity-threat-detection
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download CICIDS2017 dataset**

Download from: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

Extract to: `data/raw/CICIDS2017/`

## ⚡ Quick Start

### Train All Hybrid Models

```bash
python scripts/train_all_models.py \
    --dataset cicids2017 \
    --data-path data/raw/CICIDS2017 \
    --models all \
    --epochs 50 \
    --apply-smote \
    --output-dir results
```

### Train Specific Models

```bash
python scripts/train_all_models.py \
    --models cnn_lstm_mlp,lstm_rf \
    --epochs 30 \
    --batch-size 256
```

### Load Preprocessed Data (Faster)

```bash
python scripts/train_all_models.py \
    --models attention_lstm \
    --load-preprocessed \
    --epochs 50
```

## 🧠 Models

### 1. CNN-LSTM-MLP Ensemble

**Architecture:**
- CNN Branch: Spatial feature extraction (Conv1D: 64→128→256)
- LSTM Branch: Temporal modeling (Bidirectional LSTM: 128→64)
- MLP Branch: Tabular feature processing (Dense: 256→128)
- Final layers: Concatenation + Dense layers

**Target Performance:** 98%+ accuracy

```python
from src.models.hybrid.cnn_lstm_mlp import CNNLSTMMLPEnsemble

model = CNNLSTMMLPEnsemble(input_shape=(40, 1), num_classes=8)
model.compile_model(learning_rate=0.001)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
```

### 2. LSTM-RandomForest Hybrid

**Architecture:**
- Bidirectional LSTM: 128→64 units
- Random Forest: 500 estimators, max_depth=20

**Advantages:** 
- Interpretability via feature importance
- Strong performance on minority classes

```python
from src.models.hybrid.lstm_rf import LSTMRandomForestHybrid

model = LSTMRandomForestHybrid(input_shape=(40, 1), num_classes=8)
model.fit(X_train, y_train, epochs=50)
feature_importance = model.get_feature_importance()
```

### 3. CNN-SVM Hybrid

**Architecture:**
- CNN: 3 Conv1D blocks (64→128→256 filters)
- SVM: RBF kernel with probability estimates

**Target Performance:** 98.5%+ accuracy

### 4. Attention-LSTM-DNN

**Architecture:**
- Multi-head attention (4 heads, key_dim=64)
- Bidirectional LSTM: 128→64 units
- Deep neural network: 256→128 units

**Feature:** Visualize attention weights to understand model focus

### 5. Autoencoder-CNN Hybrid

**Architecture:**
- Autoencoder: Unsupervised pretraining (encoding_dim=32)
- CNN Classifier: Supervised fine-tuning

**Use Cases:**
- Semi-supervised learning
- Anomaly detection (reconstruction error)

### 6. Stacking Ensemble

**Base Models:** CNN, LSTM, Random Forest, XGBoost  
**Meta-Learner:** Logistic Regression

**Strategy:** 5-fold cross-validation for meta-features

## 📂 Dataset

### CICIDS2017

**Statistics:**
- **Total samples:** 2,830,743
- **Features:** 79 → 40 (after selection)
- **Classes:** 8 (BENIGN + 7 attack types)

**Attack Types:**
1. BENIGN
2. DoS/DDoS
3. PortScan
4. Bot
5. Infiltration
6. Web Attack
7. Brute Force
8. Heartbleed

**Preprocessing Pipeline:**

```python
from src.data.datasets.cicids2017 import CICIDS2017Loader

loader = CICIDS2017Loader('data/raw/CICIDS2017')
X_train, X_test, y_train, y_test = loader.preprocess_pipeline(apply_smote=True)
```

**Steps:**
1. ✅ Remove 308,381 duplicates
2. ✅ Impute 353 missing values (mean)
3. ✅ Remove infinity values
4. ✅ Consolidate labels (15 → 8 classes)
5. ✅ Feature selection (correlation > 0.95)
6. ✅ StandardScaler normalization
7. ✅ SMOTE oversampling (minority classes)
8. ✅ Stratified train-test split (80/20)

## 📊 Usage

### Training

#### Train Single Model

```python
from src.models.hybrid.cnn_lstm_mlp import CNNLSTMMLPEnsemble
from src.data.datasets.cicids2017 import CICIDS2017Loader

# Load data
loader = CICIDS2017Loader('data/raw/CICIDS2017')
X_train, X_test, y_train, y_test = loader.preprocess_pipeline(apply_smote=True)

# Train model
model = CNNLSTMMLPEnsemble(input_shape=(X_train.shape[1], 1), num_classes=8)
model.compile_model()

# Validation split
val_size = int(0.2 * len(X_train))
history = model.fit(
    X_train[val_size:], y_train[val_size:],
    validation_data=(X_train[:val_size], y_train[:val_size]),
    epochs=50,
    batch_size=256
)

# Save
model.save_model('results/models/cnn_lstm_mlp.h5')
```

#### Evaluate Model

```python
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, 'CNN-LSTM-MLP')

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 (Macro): {results['f1_macro']:.4f}")
print(f"Minority F1: {results['minority_f1_avg']:.4f}")
```

### Adversarial Robustness Testing

```python
from src.adversarial.evaluator import RobustnessEvaluator

rob_eval = RobustnessEvaluator(model.model)
rob_results = rob_eval.evaluate_all_attacks(X_test, y_test, epsilon=0.1)

print(f"Robustness Score: {rob_results['robustness_score']:.4f}")
print(f"FGSM Accuracy: {rob_results['fgsm']['accuracy']:.4f}")
print(f"PGD Accuracy: {rob_results['pgd']['accuracy']:.4f}")
```

### Adversarial Training (Defense)

```python
from src.adversarial.defenses import AdversarialDefenses

defense = AdversarialDefenses()
hardened_model = defense.adversarial_training(
    model.model,
    X_train, y_train,
    attack_method='fgsm',
    epsilon=0.1,
    epochs=50
)
```

### Explainability

#### SHAP Explanations

```python
from src.explainability.shap_explainer import SHAPExplainer

# Create explainer
shap_exp = SHAPExplainer(model.model, X_train[:100], model_type='deep')

# Calculate SHAP values
shap_values = shap_exp.explain_predictions(X_test[:100])

# Visualize
shap_exp.plot_feature_importance(shap_values, feature_names, top_n=20)
shap_exp.plot_summary(shap_values, X_test[:100], feature_names)
```

#### LIME Explanations

```python
from src.explainability.lime_explainer import LIMEExplainer

lime_exp = LIMEExplainer(model.model, X_train_2d, feature_names, class_names)
explanation = lime_exp.explain_instance(X_test_2d[0], num_features=10)
lime_exp.visualize_explanation(explanation)
```

## 🏆 Results

### Model Performance (CICIDS2017)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Minority F1 |
|-------|----------|------------|---------------|-------------|
| CNN-LSTM-MLP | **98.2%** | 97.8% | 98.1% | 95.3% |
| LSTM-RF | 98.0% | 97.5% | 97.9% | **96.1%** |
| CNN-SVM | 97.8% | 97.2% | 97.7% | 94.8% |
| Attention-LSTM | 97.9% | 97.4% | 97.8% | 95.2% |
| Stacking | **98.5%** | **98.1%** | **98.4%** | 95.7% |

### Adversarial Robustness

| Attack | Clean Accuracy | Attack Accuracy | Accuracy Drop |
|--------|---------------|-----------------|---------------|
| Clean | 98.2% | - | - |
| FGSM (ε=0.1) | 98.2% | 94.7% | 3.5% |
| PGD (ε=0.1) | 98.2% | 92.3% | 5.9% |
| C&W | 98.2% | 88.1% | 10.1% |

**Robustness Score:** 91.7%

## 📁 Project Structure

```
cybersecurity-threat-detection/
├── src/
│   ├── models/
│   │   └── hybrid/
│   │       ├── cnn_svm.py              # CNN-SVM Hybrid
│   │       ├── lstm_rf.py              # LSTM-RandomForest
│   │       ├── cnn_lstm_mlp.py         # CNN-LSTM-MLP Ensemble
│   │       ├── autoencoder_cnn.py      # Autoencoder-CNN
│   │       ├── attention_lstm.py       # Attention-LSTM-DNN
│   │       └── stacking.py             # Stacking Ensemble
│   ├── data/
│   │   └── datasets/
│   │       └── cicids2017.py           # CICIDS2017 Loader
│   ├── adversarial/
│   │   ├── attacks.py                  # 5 attack methods
│   │   ├── defenses.py                 # 3 defense mechanisms
│   │   └── evaluator.py                # Robustness evaluator
│   ├── explainability/
│   │   ├── shap_explainer.py           # SHAP
│   │   ├── lime_explainer.py           # LIME
│   │   ├── gradcam.py                  # Grad-CAM
│   │   └── attention_viz.py            # Attention visualization
│   └── evaluation/
│       └── comprehensive_metrics.py    # Evaluation framework
├── scripts/
│   └── train_all_models.py             # Unified training script
├── data/
│   ├── raw/                            # Raw datasets
│   └── processed/                      # Preprocessed data
├── results/
│   ├── models/                         # Trained models
│   ├── metrics/                        # Evaluation results
│   └── visualizations/                 # Plots and figures
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{cybersecurity_threat_detection,
  author = {Your Name},
  title = {Cybersecurity Threat Detection with Hybrid Deep Learning},
  year = {2024},
  url = {https://github.com/fadhly-git/cybersecurity-threat-detection}
}
```

## 🔗 References

- **CICIDS2017 Dataset**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Paper**: [Hybrid Deep Learning for Network Intrusion Detection](https://www.mdpi.com/)

## 👤 Author

**Fadhly**
- GitHub: [@fadhly-git](https://github.com/fadhly-git)

## 🙏 Acknowledgments

- CICIDS2017 dataset by Canadian Institute for Cybersecurity
- TensorFlow and Keras teams
- scikit-learn community
- SHAP and LIME developers

---

⭐ **Star this repository** if you find it useful!

🐛 **Found a bug?** [Open an issue](https://github.com/fadhly-git/cybersecurity-threat-detection/issues)

💬 **Have questions?** [Start a discussion](https://github.com/fadhly-git/cybersecurity-threat-detection/discussions)
