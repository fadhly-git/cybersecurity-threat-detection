# 🚀 Quick Start Guide - Logging System

## ✅ Logging sudah AKTIF di semua script training!

Setiap kali Anda menjalankan training, output akan otomatis tersimpan ke file `.log`.

## 📝 Command Examples

### 1. Train Hybrid Models (Recommended)
```powershell
# Train single model
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 10

# Train multiple models
python scripts\train_all_models.py --models cnn_lstm_mlp,lstm_rf,attention_lstm --epochs 20

# Train ALL models
python scripts\train_all_models.py --models all --epochs 50 --apply-smote
```
**Log Location:** `logs/training/train_all_models_YYYYMMDD_HHMMSS.log`

---

### 2. Train ML Models
```powershell
# Train Random Forest + XGBoost
python scripts\train_ml_models.py --models rf,xgb --cv 5

# Train all ML models
python scripts\train_ml_models.py --models all
```
**Log Location:** `logs/training/train_ml_models_YYYYMMDD_HHMMSS.log`

---

### 3. Train DL Models
```powershell
# Train CNN + LSTM
python scripts\train_dl_models.py --models cnn,lstm --epochs 50

# Train all DL models
python scripts\train_dl_models.py --models all --epochs 100
```
**Log Location:** `logs/training/train_dl_models_YYYYMMDD_HHMMSS.log`

---

### 4. Evaluate Models
```powershell
python scripts\evaluate_models.py --models-dir results\models
```
**Log Location:** `logs/evaluation/evaluate_models_YYYYMMDD_HHMMSS.log`

---

## 📂 Check Your Logs

### List all training logs:
```powershell
Get-ChildItem logs\training\*.log | Select-Object Name, Length, LastWriteTime
```

### View latest log:
```powershell
Get-Content (Get-ChildItem logs\training\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### Search for errors:
```powershell
Select-String -Path "logs\training\*.log" -Pattern "error|Error|ERROR|Exception"
```

### Find accuracy results:
```powershell
Select-String -Path "logs\training\*.log" -Pattern "Accuracy:|F1 Score:"
```

---

## 🎯 What Gets Logged?

✅ **Configuration** - Dataset, models, epochs, batch size, etc.  
✅ **Data Loading** - Dataset shape, preprocessing steps  
✅ **Training Progress** - Epoch progress, loss, accuracy  
✅ **Model Saving** - Checkpoint locations  
✅ **Evaluation Metrics** - Accuracy, F1, precision, recall  
✅ **Error Messages** - Full stack traces for debugging  
✅ **Timestamps** - Start time, end time, duration  

---

## 📊 Example Log Output

```
======================================================================
  CYBERSECURITY THREAT DETECTION - MODEL TRAINING
  Started: 2025-12-08 21:05:30
======================================================================

Configuration:
  Dataset: cicids2017
  Models: cnn_lstm_mlp
  Epochs: 10
  Batch size: 32
  SMOTE: True
  Output: results

============================================================
  DATA LOADING
============================================================

Loading preprocessed data...

✅ Data loaded:
   X_train: (2000000, 40)
   X_test: (500000, 40)

============================================================
  TRAINING CNN-LSTM-MLP
============================================================

Epoch 1/10
62500/62500 [======] - 234s - loss: 0.1234 - accuracy: 0.9567

...

✅ Model saved to: results/models/hybrid/cnn_lstm_mlp.h5

============================================================
  EVALUATING CNN-LSTM-MLP
============================================================

📊 Classification Metrics:
   Accuracy: 98.56%
   F1 Score (Macro): 0.9723
   Precision (Macro): 0.9689
   Recall (Macro): 0.9758

✅ Results saved to: results/metrics/cnn_lstm_mlp_results.pkl

======================================================================
  TRAINING COMPLETED
  Finished: 2025-12-08 22:15:45
======================================================================

✅ 1 model(s) trained successfully!
```

---

## ⚡ Tips

1. **Check logs in real-time** (while training is running):
   ```powershell
   Get-Content logs\training\train_all_models_*.log -Wait
   ```

2. **Compare multiple runs**:
   ```powershell
   # List all logs sorted by time
   Get-ChildItem logs\training\ -Recurse | Sort-Object LastWriteTime
   ```

3. **Archive old logs**:
   ```powershell
   # Create archive directory
   mkdir logs\archive\2025-12
   
   # Move old logs
   Move-Item logs\training\*.log logs\archive\2025-12\
   ```

---

## 📚 Full Documentation

Lihat **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** untuk dokumentasi lengkap.

---

**Quick Test:**
```powershell
python test_logging.py
```
Verifikasi bahwa output muncul di console DAN tersimpan di `logs/test/test_logging_*.log`

---

✨ **Happy Training!**
