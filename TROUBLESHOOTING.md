# 🔧 Troubleshooting Guide

Common issues dan solusinya saat training models.

## ❌ Error: Shape Mismatch

### Error Message:
```
InvalidArgumentError: Incompatible shapes: [1,896] vs. [1,128]
InvalidArgumentError: Incompatible shapes: [1,1792] vs. [1,256]
```

### Penyebab:
- **Root Cause**: Precision/Recall metrics di Keras 3.x tidak kompatibel dengan `sparse_categorical_crossentropy` dan variable batch sizes
- Batch size yang tidak konsisten di batch terakhir
- Keras 3.x memiliki breaking changes dari Keras 2.x

### Solusi:
✅ **FIXED!** Sudah diperbaiki dengan menghapus Precision/Recall dari metrics saat training:
- `src/models/hybrid/cnn_lstm_mlp.py` - Sekarang hanya gunakan `['accuracy']`
- `src/models/hybrid/attention_lstm.py` - Sekarang hanya gunakan `['accuracy']`

**Note**: Precision, Recall, dan F1 Score tetap dihitung saat evaluation menggunakan `sklearn.metrics`, jadi tidak ada data yang hilang.

**Gunakan batch size yang lebih kecil untuk stability:**
```bash
python scripts/train_all_models.py --batch-size 64  # Recommended
python scripts/train_all_models.py --batch-size 32  # Safer
```

---

## ❌ Error: Out of Memory (OOM)

### Error Message:
```
ResourceExhaustedError: OOM when allocating tensor
```

### Solusi:
1. **Reduce batch size:**
   ```bash
   python scripts/train_all_models.py --batch-size 32
   ```

2. **Reduce dataset size** (untuk testing):
   Edit `src/data/datasets/cicids2017.py`:
   ```python
   # Ambil sample kecil untuk testing
   df = df.sample(n=100000, random_state=42)
   ```

3. **Train model satu per satu:**
   ```bash
   # Jangan --models all, train satu dulu
   python scripts/train_all_models.py --models cnn_lstm_mlp
   ```

---

## ❌ Error: Dataset Not Found

### Error Message:
```
FileNotFoundError: data/raw/CICIDS2017/*.csv not found
```

### Solusi:
Download CICIDS2017 dataset:
```powershell
# Create directory
mkdir data\raw\CICIDS2017

# Download from: https://www.unb.ca/cic/datasets/ids-2017.html
# Extract CSV files ke data/raw/CICIDS2017/
```

---

## ❌ Error: Import Error

### Error Message:
```
ModuleNotFoundError: No module named 'lime'
```

### Solusi:
Install missing dependencies:
```bash
pip install -r requirements.txt
```

Atau install specific package:
```bash
pip install lime shap tensorflow
```

---

## ⚠️ Warning: TensorFlow oneDNN

### Warning Message:
```
I tensorflow/core/util/port.cc:153] oneDNN custom operations are on
```

### Penjelasan:
Ini **bukan error**, hanya informasi bahwa TensorFlow menggunakan optimisasi CPU.

### Disable (optional):
```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
python scripts/train_all_models.py ...
```

---

## 🐌 Slow Training

### Gejala:
Training sangat lambat (>10 menit per epoch untuk 2M samples)

### Solusi:

1. **Gunakan GPU** (jika tersedia):
   ```bash
   pip install tensorflow-gpu
   ```

2. **Reduce epochs untuk testing:**
   ```bash
   python scripts/train_all_models.py --epochs 5
   ```

3. **Load preprocessed data:**
   ```bash
   # Setelah run pertama, data tersimpan
   python scripts/train_all_models.py --load-preprocessed
   ```

4. **Increase batch size** (jika memory cukup):
   ```bash
   python scripts/train_all_models.py --batch-size 512
   ```

---

## 📊 No Output / Frozen

### Gejala:
Script berjalan tapi tidak ada output

### Solusi:
1. **Check log file:**
   ```powershell
   Get-Content (Get-ChildItem logs\training\*.log | Sort LastWriteTime -Desc | Select -First 1).FullName -Wait
   ```

2. **Add verbose output:**
   Edit script, set `verbose=2` di model.fit():
   ```python
   model.fit(..., verbose=2)  # More detailed output
   ```

---

## 🔄 Training Interrupted

### Gejala:
Training berhenti di tengah jalan (Ctrl+C atau crash)

### Solusi:
Model checkpoint otomatis tersimpan! Resume dengan:

```python
# Edit train_all_models.py, tambahkan di train_model():
if os.path.exists('best_cnn_lstm_mlp.h5'):
    print("Loading checkpoint...")
    model.model = load_model('best_cnn_lstm_mlp.h5')
```

---

## 🧪 Verify Installation

### Test semua dependencies:
```bash
python test_logging.py
```

### Test imports:
```bash
python -c "import tensorflow as tf; import keras; import numpy as np; import pandas as pd; import sklearn; import shap; import lime; print('All imports OK!')"
```

### Check versions:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## 📞 Get Help

### Check logs:
```powershell
# Search for errors
Select-String -Path "logs\training\*.log" -Pattern "error|Error|ERROR|Exception"

# View full log
Get-Content logs\training\train_all_models_TIMESTAMP.log
```

### Debug mode:
Edit script, tambahkan debug prints:
```python
print(f"DEBUG: X_train shape = {X_train.shape}")
print(f"DEBUG: y_train shape = {y_train.shape}")
print(f"DEBUG: Model summary:")
model.summary()
```

---

## ✅ Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Shape mismatch | `--batch-size 64` |
| Out of memory | `--batch-size 32` |
| Too slow | `--epochs 5 --load-preprocessed` |
| Dataset missing | Download CICIDS2017 |
| Import error | `pip install -r requirements.txt` |
| No output | Check `logs/training/*.log` |

---

## 🎯 Recommended Settings

### For Testing (Fast):
```bash
python scripts/train_all_models.py \
  --models cnn_lstm_mlp \
  --epochs 5 \
  --batch-size 128 \
  --load-preprocessed
```

### For Production (Best Results):
```bash
python scripts/train_all_models.py \
  --models all \
  --epochs 50 \
  --batch-size 256 \
  --apply-smote
```

### For Limited Memory:
```bash
python scripts/train_all_models.py \
  --models lstm_rf \
  --epochs 20 \
  --batch-size 32
```

---

**Last Updated:** 2025-12-08
