# 📝 Logging System Documentation

Semua script training dan evaluation sekarang otomatis menyimpan output ke file `.log` dengan timestamp.

## 🎯 Features

- ✅ **Dual Output**: Output muncul di console DAN file log secara bersamaan
- ✅ **Auto Timestamping**: Setiap run diberi timestamp unik (YYYYMMDD_HHMMSS)
- ✅ **Organized Structure**: Log tersimpan di `logs/training/` dan `logs/evaluation/`
- ✅ **Complete Capture**: Semua print, errors, warnings, dan output tertangkap

## 📂 Log Directory Structure

```
logs/
├── training/
│   ├── train_all_models_20251208_210530.log
│   ├── train_ml_models_20251208_211045.log
│   └── train_dl_models_20251208_212130.log
├── evaluation/
│   └── evaluate_models_20251208_215645.log
└── test/
    └── test_logging_20251208_203015.log
```

## 🚀 Usage

### 1. Training dengan Log

**Train All Hybrid Models:**
```bash
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 10
```
Log tersimpan di: `logs/training/train_all_models_TIMESTAMP.log`

**Train ML Models:**
```bash
python scripts/train_ml_models.py --models rf,xgb --cv 5
```
Log tersimpan di: `logs/training/train_ml_models_TIMESTAMP.log`

**Train DL Models:**
```bash
python scripts/train_dl_models.py --models cnn,lstm --epochs 50
```
Log tersimpan di: `logs/training/train_dl_models_TIMESTAMP.log`

### 2. Evaluation dengan Log

```bash
python scripts/evaluate_models.py --models-dir results/models
```
Log tersimpan di: `logs/evaluation/evaluate_models_TIMESTAMP.log`

### 3. Test Logging System

```bash
python test_logging.py
```
Verifikasi bahwa output masuk ke console dan file `logs/test/test_logging_TIMESTAMP.log`

## 📋 Log Format

Setiap log file berisi:

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
   y_train: (2000000,)
   y_test: (500000,)

============================================================
  TRAINING CNN-LSTM-MLP
============================================================

Epoch 1/10
62500/62500 [==============================] - 234s 4ms/step - loss: 0.1234 - accuracy: 0.9567
...

✅ Model saved to: results/models/hybrid/cnn_lstm_mlp.h5

============================================================
  EVALUATING CNN-LSTM-MLP
============================================================

Accuracy: 98.56%
F1 Score (Macro): 0.9723
...

======================================================================
  TRAINING COMPLETED
  Finished: 2025-12-08 22:15:45
======================================================================

✅ 1 model(s) trained successfully!
   Models saved to: results/models/hybrid/
   Metrics saved to: results/metrics/
======================================================================
```

## 🔧 Implementation Details

### DualOutput Class

```python
from src.utils.logger import DualOutput
from datetime import datetime

# Setup log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/training/my_script_{timestamp}.log'

# Redirect all output
with DualOutput(log_file):
    print("This goes to both console and file")
    # ... your code here ...
```

### Benefits

1. **Reproducibility**: Full training history tersimpan
2. **Debugging**: Error messages dan stack traces tercatat
3. **Monitoring**: Review progress tanpa re-run
4. **Reporting**: Copy-paste log untuk dokumentasi
5. **Comparison**: Bandingkan hasil dari berbagai run

## 📊 Log Analysis

### Cari error dalam log:
```powershell
Select-String -Path "logs/training/*.log" -Pattern "error|Error|ERROR"
```

### Lihat summary training:
```powershell
Select-String -Path "logs/training/*.log" -Pattern "Accuracy|F1 Score"
```

### Check runtime:
```powershell
Select-String -Path "logs/training/*.log" -Pattern "Started:|Finished:"
```

## 🎨 Customization

### Ubah lokasi log:

Edit di script (e.g., `train_all_models.py`):
```python
# Default: logs/training/
log_file = f'custom/path/my_log_{timestamp}.log'
```

### Log tanpa console output:

Modifikasi `DualOutput.__init__`:
```python
# Comment out console output
# self.terminal = sys.stdout  # Remove this
```

## ⚠️ Notes

- Log files **append mode** by default (mode='a')
- Tidak ada size limit, bisa jadi besar untuk training panjang
- Rotate/delete old logs secara manual jika perlu
- TensorFlow output juga tertangkap (epoch progress, warnings, dll)

## 🔍 Troubleshooting

**Q: Log file kosong atau tidak lengkap?**
A: Pastikan program selesai normal (tidak Ctrl+C). Context manager `with DualOutput()` perlu selesai untuk flush semua output.

**Q: Encoding error di log file?**
A: DualOutput sudah set `encoding='utf-8'`, tapi pastikan terminal juga UTF-8.

**Q: Ingin log HANYA ke file (no console)?**
A: Gunakan parameter `console=False` di `setup_logger()` atau modifikasi `DualOutput`.

## 📚 Related Files

- `src/utils/logger.py` - Logger implementation + DualOutput class
- `scripts/train_all_models.py` - Hybrid models training dengan auto-log
- `scripts/train_ml_models.py` - ML models training dengan auto-log
- `scripts/train_dl_models.py` - DL models training dengan auto-log
- `scripts/evaluate_models.py` - Evaluation dengan auto-log
- `test_logging.py` - Test script untuk verifikasi logging

---

**Created**: 2025-12-08  
**Updated**: 2025-12-08  
**Version**: 1.0
