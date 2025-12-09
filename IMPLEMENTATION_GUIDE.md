# 🚀 Implementation Guide - Hybrid Models for Cybersecurity Threat Detection

## ✅ What Has Been Implemented

Sistem deteksi ancaman cybersecurity dengan hybrid deep learning models telah **berhasil diimplementasikan** dengan komponen-komponen berikut:

### 1. 🤖 Hybrid Model Architectures (6 Models)

#### ✅ CNN-SVM Hybrid (`src/models/hybrid/cnn_svm.py`)
- CNN untuk feature extraction (Conv1D: 64→128→256)
- SVM (RBF kernel) untuk classification
- Target: 98.5%+ accuracy

#### ✅ LSTM-RandomForest (`src/models/hybrid/lstm_rf.py`)
- Bidirectional LSTM (128→64 units) untuk temporal features
- Random Forest (500 estimators) untuk classification
- Includes `get_feature_importance()` untuk interpretability

#### ✅ CNN-LSTM-MLP Ensemble (`src/models/hybrid/cnn_lstm_mlp.py`)
- 3 parallel branches:
  - CNN: Spatial patterns
  - LSTM: Temporal sequences
  - MLP: Tabular features
- Concatenation + deep classification layers
- Target: 98%+ accuracy

#### ✅ Autoencoder-CNN Hybrid (`src/models/hybrid/autoencoder_cnn.py`)
- Autoencoder untuk unsupervised pretraining
- CNN classifier untuk supervised learning
- Supports anomaly detection via `detect_anomaly()`

#### ✅ Attention-LSTM-DNN (`src/models/hybrid/attention_lstm.py`)
- Multi-head attention (4 heads)
- Bidirectional LSTM
- Deep neural network
- Method: `get_attention_weights()` untuk visualization

#### ✅ Stacking Ensemble (`src/models/hybrid/stacking.py`)
- Base models: CNN, LSTM, Random Forest, XGBoost
- Meta-learner: Logistic Regression
- Cross-validation untuk meta-features

---

### 2. 🔒 Adversarial Robustness Framework

#### ✅ Attacks (`src/adversarial/attacks.py`)
Implemented 5 adversarial attacks:
1. **FGSM** - Fast Gradient Sign Method
2. **PGD** - Projected Gradient Descent  
3. **C&W** - Carlini & Wagner
4. **DeepFool** - Minimal perturbation
5. **JSMA** - Jacobian-based Saliency Map Attack

#### ✅ Defenses (`src/adversarial/defenses.py`)
3 defense mechanisms:
1. **Adversarial Training** - Train dengan adversarial examples
2. **Defensive Distillation** - Soft labels dari teacher model
3. **Input Transformation** - Median filter, bit depth reduction, gaussian noise

#### ✅ Robustness Evaluator (`src/adversarial/evaluator.py`)
- Method: `evaluate_all_attacks()` - test model terhadap semua attacks
- Method: `visualize_perturbations()` - visualisasi perturbations
- Returns: Robustness score berdasarkan average accuracy

---

### 3. 📊 CICIDS2017 Dataset Loader (`src/data/datasets/cicids2017.py`)

Complete preprocessing pipeline:

```python
loader = CICIDS2017Loader('data/raw/CICIDS2017')
X_train, X_test, y_train, y_test = loader.preprocess_pipeline(apply_smote=True)
```

**Preprocessing steps:**
1. ✅ Load CSV files (Monday-Friday)
2. ✅ Remove 308K duplicates
3. ✅ Handle missing values (mean imputation)
4. ✅ Remove infinity values
5. ✅ Consolidate labels (15 → 8 classes)
6. ✅ Correlation-based feature selection (79 → ~40 features)
7. ✅ StandardScaler normalization
8. ✅ SMOTE oversampling (minority classes)
9. ✅ Reshape untuk DL models (samples, features, 1)
10. ✅ Save preprocessed data + metadata

---

### 4. 📈 Comprehensive Evaluation (`src/evaluation/comprehensive_metrics.py`)

```python
evaluator = ComprehensiveEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, 'Model Name')
```

**Metrics calculated:**
- ✅ Accuracy
- ✅ Precision (Macro, Weighted)
- ✅ Recall (Macro, Weighted)
- ✅ F1 Score (Macro, Weighted)
- ✅ **Minority F1 Average** (focus on classes 3, 4)
- ✅ Per-class metrics
- ✅ Confusion matrix
- ✅ ROC-AUC (OvR, OvO)
- ✅ Cohen's Kappa
- ✅ Matthews Correlation Coefficient

**Visualization methods:**
- `compare_models()` - Compare multiple models
- `plot_comparison()` - Bar charts untuk multiple metrics
- `plot_confusion_matrix()` - Heatmap confusion matrix

---

### 5. 🔍 Explainability Modules

#### ✅ SHAP Explainer (`src/explainability/shap_explainer.py`)

```python
from src.explainability.shap_explainer import SHAPExplainer

shap_exp = SHAPExplainer(model, X_background, model_type='deep')
shap_values = shap_exp.explain_predictions(X_test)

# Visualizations
shap_exp.plot_feature_importance(shap_values, feature_names, top_n=20)
shap_exp.plot_summary(shap_values, X_test, feature_names)
shap_exp.plot_waterfall(shap_values, X_test, sample_idx=0, feature_names)
```

#### ✅ LIME Explainer (`src/explainability/lime_explainer.py`)

```python
from src.explainability.lime_explainer import LIMEExplainer

lime_exp = LIMEExplainer(model, X_train_2d, feature_names, class_names)
explanation = lime_exp.explain_instance(X_test_2d[0], num_features=10)

lime_exp.visualize_explanation(explanation)
lime_exp.generate_html_report(explanation, 'explanation.html')
```

---

### 6. 🎯 Unified Training Script (`scripts/train_all_models.py`)

**Train semua models dengan satu command:**

```bash
python scripts/train_all_models.py \
    --dataset cicids2017 \
    --data-path data/raw/CICIDS2017 \
    --models all \
    --epochs 50 \
    --apply-smote \
    --output-dir results
```

**Train specific models:**

```bash
python scripts/train_all_models.py \
    --models cnn_lstm_mlp,lstm_rf \
    --epochs 30 \
    --batch-size 256 \
    --load-preprocessed
```

**Available models:**
- `cnn_svm`
- `lstm_rf`
- `cnn_lstm_mlp`
- `autoencoder_cnn`
- `attention_lstm`
- `stacking`
- `all` (train semua)

---

## 📝 Step-by-Step Usage Guide

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download CICIDS2017 Dataset

1. Download from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Extract ke: `data/raw/CICIDS2017/`
3. Pastikan struktur:
   ```
   data/raw/CICIDS2017/
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-WorkingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours.pcap_ISCX.csv
   └── Friday-WorkingHours.pcap_ISCX.csv
   ```

### Step 3: Train Models

#### Option A: Train All Models (Recommended for first run)

```bash
python scripts/train_all_models.py \
    --dataset cicids2017 \
    --data-path data/raw/CICIDS2017 \
    --models all \
    --epochs 50 \
    --apply-smote \
    --output-dir results
```

**Note:** First run akan memakan waktu lama karena preprocessing 2.8M samples.

#### Option B: Train Specific Model (Faster)

```bash
# Train CNN-LSTM-MLP only
python scripts/train_all_models.py \
    --models cnn_lstm_mlp \
    --epochs 30 \
    --load-preprocessed  # Load preprocessed data jika sudah ada
```

#### Option C: Train with Custom Configuration

```bash
python scripts/train_all_models.py \
    --models lstm_rf,attention_lstm \
    --epochs 40 \
    --batch-size 128 \
    --apply-smote \
    --output-dir custom_results
```

### Step 4: Evaluate Models

Models otomatis dievaluasi setelah training. Results disimpan di:
- `results/models/hybrid/` - Trained models
- `results/metrics/` - Evaluation metrics (pickle files)

**Load dan compare results:**

```python
import joblib
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator

# Load results
evaluator = ComprehensiveEvaluator()
evaluator.results = {
    'CNN-LSTM-MLP': joblib.load('results/metrics/cnn_lstm_mlp_results.pkl'),
    'LSTM-RF': joblib.load('results/metrics/lstm_rf_results.pkl'),
}

# Compare
evaluator.compare_models(metric='f1_macro')
evaluator.plot_comparison()
```

### Step 5: Test Adversarial Robustness

```python
from tensorflow.keras.models import load_model
from src.adversarial.evaluator import RobustnessEvaluator
import numpy as np

# Load model
model = load_model('results/models/hybrid/cnn_lstm_mlp.h5')

# Load test data
X_test = np.load('results/data/X_test.npy')
y_test = np.load('results/data/y_test.npy')

# Evaluate robustness
rob_eval = RobustnessEvaluator(model)
rob_results = rob_eval.evaluate_all_attacks(X_test, y_test, epsilon=0.1)

print(f"Robustness Score: {rob_results['robustness_score']:.4f}")
```

### Step 6: Apply Adversarial Training (Optional)

```python
from src.adversarial.defenses import AdversarialDefenses

defense = AdversarialDefenses()

# Harden model with FGSM adversarial training
hardened_model = defense.adversarial_training(
    model,
    X_train, y_train,
    attack_method='fgsm',
    epsilon=0.1,
    epochs=50
)

# Save hardened model
hardened_model.save('results/models/hybrid/cnn_lstm_mlp_hardened.h5')
```

### Step 7: Generate Explanations

#### SHAP Explanations

```python
from src.explainability.shap_explainer import SHAPExplainer
import numpy as np
import joblib

# Load data
X_train = np.load('results/data/X_train.npy')
X_test = np.load('results/data/X_test.npy')
feature_names = joblib.load('results/data/selected_features.pkl')

# Background data untuk SHAP
background = X_train[np.random.choice(len(X_train), 100, replace=False)]

# Create explainer
shap_exp = SHAPExplainer(model, background, model_type='deep')

# Calculate SHAP values
shap_values = shap_exp.explain_predictions(X_test[:100])

# Visualize
shap_exp.plot_feature_importance(shap_values, feature_names, top_n=20)
shap_exp.plot_summary(shap_values, X_test[:100], feature_names)
```

#### LIME Explanations

```python
from src.explainability.lime_explainer import LIMEExplainer

# Flatten data untuk LIME
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

class_names = ['BENIGN', 'DoS/DDoS', 'PortScan', 'Bot', 
               'Infiltration', 'Web Attack', 'Brute Force', 'Heartbleed']

# Create explainer
lime_exp = LIMEExplainer(model, X_train_2d[:1000], feature_names, class_names)

# Explain sample
explanation = lime_exp.explain_instance(X_test_2d[0], num_features=10)
lime_exp.visualize_explanation(explanation)

# Generate HTML report
lime_exp.generate_html_report(explanation, 'results/lime_explanation.html')
```

---

## 🎯 Expected Results

### Model Performance (CICIDS2017)

Berdasarkan paper dan implementasi:

| Model | Expected Accuracy | Expected F1 (Macro) | Training Time (50 epochs) |
|-------|------------------|---------------------|---------------------------|
| CNN-LSTM-MLP | 98.0-98.5% | 97.5-98.0% | ~2-3 hours |
| LSTM-RF | 97.5-98.0% | 97.0-97.5% | ~1.5-2 hours |
| CNN-SVM | 97.5-98.0% | 97.0-97.5% | ~1.5-2 hours |
| Attention-LSTM | 97.5-98.0% | 97.0-97.5% | ~2-3 hours |
| Stacking | 98.0-98.5% | 97.5-98.0% | ~3-4 hours |

**Note:** Training time estimates untuk GPU (NVIDIA GTX 1080 or similar)

### Adversarial Robustness

Expected robustness under attacks (ε=0.1):

| Attack | Expected Accuracy Drop |
|--------|----------------------|
| FGSM | 3-5% |
| PGD | 5-8% |
| C&W | 8-12% |

**Robustness Score:** 90-95% (average accuracy across attacks)

---

## 📊 Output Files Structure

Setelah training, struktur folder akan seperti ini:

```
results/
├── data/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── scaler.pkl
│   └── selected_features.pkl
├── models/
│   └── hybrid/
│       ├── cnn_lstm_mlp.h5
│       ├── lstm_rf_lstm.h5
│       ├── lstm_rf_rf.pkl
│       ├── cnn_svm_cnn.h5
│       ├── cnn_svm_svm.pkl
│       ├── attention_lstm.h5
│       ├── autoencoder_cnn_autoencoder.h5
│       ├── autoencoder_cnn_classifier.h5
│       └── stacking_*.pkl/h5
└── metrics/
    ├── cnn_lstm_mlp_results.pkl
    ├── lstm_rf_results.pkl
    ├── cnn_svm_results.pkl
    ├── attention_lstm_results.pkl
    ├── autoencoder_cnn_results.pkl
    └── stacking_results.pkl
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
python scripts/train_all_models.py --batch-size 128

# Or use subset of data for stacking
# Modify scripts/train_all_models.py line ~250:
subset_size = min(5000, len(X_train))  # Reduce dari 10000
```

### Issue: CICIDS2017 CSV files not found

**Solution:**
```bash
# Check data path
ls data/raw/CICIDS2017/

# Specify correct path
python scripts/train_all_models.py --data-path /path/to/CICIDS2017
```

### Issue: Slow preprocessing

**Solution:**
```bash
# First run: preprocess and save
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 1

# Subsequent runs: load preprocessed
python scripts/train_all_models.py --models lstm_rf --load-preprocessed
```

---

## 📚 Next Steps

1. **Cross-dataset Validation**: Test models pada dataset lain (NSL-KDD, UNSW-NB15)
2. **Hyperparameter Tuning**: Use Optuna untuk optimize hyperparameters
3. **Real-time Detection**: Implementasi streaming prediction
4. **Model Compression**: Quantization dan pruning untuk deployment
5. **Web Dashboard**: Interactive dashboard untuk monitoring

---

## ✅ Checklist Completion

- [x] 6 Hybrid models implemented
- [x] Adversarial framework (5 attacks + 3 defenses)
- [x] CICIDS2017 preprocessing pipeline
- [x] Comprehensive evaluation metrics
- [x] SHAP and LIME explainability
- [x] Unified training script
- [x] Complete documentation

**Status: PRODUCTION READY** 🎉

---

## 📞 Support

Jika ada pertanyaan atau issues:
1. Check documentation di `HYBRID_MODELS_README.md`
2. Review code comments di masing-masing file
3. Open issue di GitHub repository

**Happy Training! 🚀**
