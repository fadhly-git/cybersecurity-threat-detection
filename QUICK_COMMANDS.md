# ⚡ Quick Commands - Ready to Use

Semua commands sudah **FIXED** dan siap digunakan!

## ✅ WORKING COMMANDS (Tested)

### 🎯 Recommended: Test Run (Cepat)
```bash
# Train 1 model, 2 epochs, batch 64 - untuk verifikasi (~5-10 menit)
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 2 --batch-size 64 --load-preprocessed
```

### 🚀 Production Run (Lengkap)
```bash
# Train semua 6 hybrid models, 50 epochs (~beberapa jam)
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64
```

### 🔌 Overnight Run with Auto-Shutdown ⭐ NEW
```bash
# Train semalaman, auto-shutdown setelah selesai
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --shutdown

# Custom shutdown delay (default 60 detik)
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --shutdown --shutdown-delay 120
```

### 🔬 Individual Model Training
```bash
# CNN-LSTM-MLP Ensemble
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 20 --batch-size 64

# LSTM-RandomForest
python scripts\train_all_models.py --models lstm_rf --epochs 20 --batch-size 64

# CNN-SVM
python scripts\train_all_models.py --models cnn_svm --epochs 20 --batch-size 64

# Attention-LSTM
python scripts\train_all_models.py --models attention_lstm --epochs 20 --batch-size 64

# Autoencoder-CNN
python scripts\train_all_models.py --models autoencoder_cnn --epochs 20 --batch-size 64

# Stacking Ensemble
python scripts\train_all_models.py --models stacking --epochs 20 --batch-size 64
```

### 💾 Use Preprocessed Data (Lebih Cepat)
```bash
# Setelah run pertama, data sudah tersimpan di results/data/
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --load-preprocessed
```

### 🎨 With SMOTE (Balance Classes)
```bash
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --apply-smote
```

---

## 📊 Check Training Progress

### View Real-Time Log
```powershell
# Get latest log file and monitor
Get-Content (Get-ChildItem logs\training\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Wait
```

### List All Logs
```powershell
Get-ChildItem logs\training\*.log | Select-Object Name, @{N='Size(KB)';E={[math]::Round($_.Length/1KB,2)}}, LastWriteTime | Format-Table -AutoSize
```

### Search for Errors
```powershell
Select-String -Path "logs\training\*.log" -Pattern "Error|error|ERROR|Exception" | Select-Object -First 10
```

### Check Accuracy
```powershell
Select-String -Path "logs\training\*.log" -Pattern "Accuracy:|accuracy:" | Select-Object -Last 20
```

---

## 🎯 Batch Size Guide

| Memory Available | Recommended Batch Size | Training Speed |
|-----------------|------------------------|----------------|
| 8GB RAM         | 32                     | Slow but safe |
| 16GB RAM        | 64                     | **Recommended** |
| 32GB+ RAM       | 128                    | Faster |
| High-end GPU    | 256+                   | Fastest |

**Current Default:** `--batch-size 64` ✅

---

## ⏱️ Estimated Training Time

Dataset: CICIDS2017 (2.5M samples after cleaning)

| Model | Epochs | Batch Size | Est. Time (CPU) |
|-------|--------|------------|-----------------|
| CNN-LSTM-MLP | 2 | 64 | ~10-15 min |
| CNN-LSTM-MLP | 20 | 64 | ~2-3 hours |
| CNN-LSTM-MLP | 50 | 64 | ~5-7 hours |
| All 6 Models | 50 | 64 | ~1-2 days |

**Tips untuk Mempercepat:**
- ✅ Use `--load-preprocessed` (skip 3-5 menit preprocessing)
- ✅ Increase batch size jika memory cukup
- ✅ Train models parallel (buka multiple terminals)

---

## 🔧 Common Scenarios

### Scenario 1: Quick Test (Verifikasi sistem works)
```bash
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 2 --batch-size 64
```
⏱️ Time: ~10 minutes  
📝 Purpose: Verify everything works

---

### Scenario 2: Single Model Production
```bash
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 50 --batch-size 64 --load-preprocessed
```
⏱️ Time: ~5-7 hours  
📝 Purpose: Train 1 model to completion

---

### Scenario 3: All Models (Overnight Run)
```bash
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --load-preprocessed --apply-smote --shutdown
```
⏱️ Time: ~1-2 days  
📝 Purpose: Complete training run  
🔌 Feature: **Auto-shutdown setelah selesai**

---

### Scenario 4: Limited Memory
```bash
python scripts\train_all_models.py --models lstm_rf --epochs 20 --batch-size 32
```
⏱️ Time: ~3-4 hours  
📝 Purpose: Safe for 8GB RAM systems

---

## 📁 Output Locations

After training, check:

```
results/
├── data/                        # Preprocessed data
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/hybrid/               # Trained models
│   ├── cnn_lstm_mlp.h5
│   ├── lstm_rf_lstm.h5
│   └── ...
└── metrics/                     # Evaluation results
    ├── cnn_lstm_mlp_results.pkl
    └── ...

logs/training/                   # Training logs
    └── train_all_models_TIMESTAMP.log
```

---

## ✅ Verification Commands

### Test Logging System
```bash
python test_logging.py
```

### Check Dependencies
```bash
python -c "import tensorflow as tf; import keras; import numpy as np; import pandas as pd; import sklearn; print('OK')"
```

### List Installed Packages
```bash
pip list | Select-String -Pattern "tensorflow|keras|numpy|pandas|scikit|xgboost|lightgbm|shap|lime"
```

---

## 🆘 If Training Fails

1. **Check log file:**
   ```powershell
   Get-Content logs\training\train_all_models_*.log | Select-Object -Last 50
   ```

2. **Reduce batch size:**
   ```bash
   python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 2 --batch-size 32
   ```

3. **See troubleshooting:**
   ```bash
   code TROUBLESHOOTING.md
   ```

---

**Last Updated:** 2025-12-08  
**Status:** ✅ All commands tested and working  
**Fixed Issues:** Shape mismatch, Keras 3.x metrics compatibility
