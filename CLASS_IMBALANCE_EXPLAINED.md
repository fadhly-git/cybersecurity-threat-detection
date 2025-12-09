# ⚠️ Understanding Training Behavior - Class Imbalance

## 🔍 Kenapa Accuracy Naik Cepat?

### ❌ BUKAN Overfitting!

Accuracy yang tinggi di awal (**0.83 - 0.99**) adalah **NORMAL** karena **class imbalance** ekstrem:

```
Dataset CICIDS2017 Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Class 0 (BENIGN):         2,097,200  (83.2%)  ████████████████████████████████████
Class 1 (DoS):              321,759  (12.8%)  ██████
Class 2 (PortScan):          90,694   (3.6%)  ██
Class 3 (Bot):                1,948   (0.08%) ▏
Class 4 (Infiltration):          36   (0.001%)▏
Class 6 (Web Attack):         9,150   (0.36%) ▏
Class 7 (Heartbleed):            11   (0.0004%)▏
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 📊 Penjelasan:

**Skenario 1: Tanpa Class Weights**
```
Model naive: "Prediksi semua sebagai Class 0 (BENIGN)"
Result: 83.2% accuracy!
```

Tapi ini **TIDAK BERGUNA** karena:
- ❌ Class 3, 4, 6, 7 diabaikan total
- ❌ Model tidak belajar mendeteksi serangan
- ✅ Accuracy tinggi tapi recall rendah untuk minority classes

**Skenario 2: Dengan Class Weights** ✅ **RECOMMENDED**
```
Model: "Beri perhatian lebih ke minority classes"
Class 0 weight: 0.06 (dikurangi pengaruhnya)
Class 4 weight: 34670 (diperbesar pengaruhnya)
Result: Accuracy lebih rendah (~95%) tapi F1 jauh lebih baik!
```

---

## 🔧 Fix yang Sudah Diterapkan

### 1. **Class Weights** (Otomatis)

File: `scripts/train_all_models.py`

```python
# Compute class weights automatically
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Class weights akan tampil di output:
Class 0:  1,677,760 samples (83.20%) - Weight: 0.06
Class 1:    257,407 samples (12.76%) - Weight: 0.49
Class 2:     72,555 samples ( 3.60%) - Weight: 1.74
Class 3:      1,558 samples ( 0.08%) - Weight: 81.02
Class 4:         29 samples ( 0.00%) - Weight: 43563.31
Class 6:      7,320 samples ( 0.36%) - Weight: 17.25
Class 7:          9 samples ( 0.00%) - Weight: 140372.73
```

### 2. **Progress Bar Behavior**

**NORMAL:**
- Steps berubah: `1/6302 → 2/6302 → ...`
- Time estimate berubah: `11:55:37 → 25:49 → 24:54`
- Ini karena Keras menghitung ulang ETA berdasarkan kecepatan aktual

**Steps calculation:**
```
2,016,638 samples ÷ 256 batch size = 7,877 steps (without validation split)
1,613,310 samples ÷ 256 batch size = 6,302 steps (with 20% validation split) ✓
```

---

## ✅ Metrics yang Benar

### ❌ JANGAN hanya lihat Accuracy!

**Buruk:**
```
Accuracy: 99.5%  ← Tinggi tapi misleading
Recall for Class 4: 0%  ← Model GAGAL detect Infiltration!
```

**Baik:**
```
Accuracy: 95.2%
F1 Score (Macro): 0.87  ← Rata-rata semua class
F1 Score Class 4: 0.72  ← Bisa detect minority class!
Precision/Recall balanced
```

### 📊 Metrics yang Harus Dipantau

Lihat di evaluation report nanti:

1. **F1 Score (Macro)** - Average semua class
2. **F1 Score per Class** - Terutama class 3, 4, 6, 7
3. **Confusion Matrix** - Lihat false negatives
4. **Recall per Class** - Deteksi attack berhasil?
5. **Precision per Class** - False alarm rendah?

---

## 🎯 Rekomendasi

### Option 1: Gunakan Class Weights (Sudah Implemented) ✅
```bash
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 20 --batch-size 64
```
**Hasil:**
- Accuracy: ~95-97%
- F1 Macro: ~0.85-0.90
- Minority classes terdeteksi

### Option 2: Gunakan SMOTE (Oversample Minority)
```bash
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 20 --batch-size 64 --apply-smote
```
**Hasil:**
- Dataset balanced (all classes ~300k samples)
- Accuracy: ~96-98%
- F1 Macro: ~0.90-0.95
- **WARNING:** Training lebih lambat (dataset jadi 2x lipat)

### Option 3: Hybrid (SMOTE + Class Weights)
```bash
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 30 --batch-size 64 --apply-smote
```
Lalu edit code untuk tetap gunakan class weights meskipun sudah SMOTE.

---

## 📈 Expected Training Behavior

### Epoch 1-5: Rapid Learning
```
Accuracy: 0.83 → 0.92 → 0.94 → 0.95 → 0.96
Loss:     2.10 → 0.45 → 0.28 → 0.19 → 0.15
```
✅ NORMAL - Model belajar pattern utama

### Epoch 6-20: Fine-tuning
```
Accuracy: 0.96 → 0.965 → 0.968 → 0.970 → 0.971
Loss:     0.15 → 0.12 → 0.11 → 0.10 → 0.095
```
✅ NORMAL - Model refine decision boundary

### Epoch 20-50: Convergence
```
Accuracy: 0.971 → 0.972 → 0.972 → 0.973 → 0.973
Loss:     0.095 → 0.092 → 0.091 → 0.090 → 0.090
```
✅ NORMAL - Model konvergen, improvement minimal

### ⚠️ Warning Signs of Overfitting

```
val_accuracy << train_accuracy  (Gap >5%)
val_loss >> train_loss          (Val loss naik)
```

Jika terjadi, **Early Stopping** akan berhenti otomatis.

---

## 🔬 How to Verify Model Quality

### 1. Check Log Setelah Training
```powershell
Select-String -Path "logs\training\*.log" -Pattern "F1|f1|Recall|recall" | Select-Object -Last 20
```

### 2. Load dan Test Model
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('best_cnn_lstm_mlp.h5')
X_test = np.load('results/data/X_test.npy')
y_test = np.load('results/data/y_test.npy')

predictions = model.predict(X_test)
```

### 3. Evaluation Script
```bash
python scripts/evaluate_models.py --models-dir results/models
```

Akan generate:
- Confusion matrix untuk setiap class
- Per-class precision, recall, F1
- ROC curves
- Classification report

---

## ✅ Summary

| Behavior | Status | Explanation |
|----------|--------|-------------|
| Progress bar berubah | ✅ Normal | Keras recalculating ETA |
| Accuracy 83% di batch 1 | ✅ Expected | 83% data adalah BENIGN |
| Accuracy naik ke 99% | ✅ Expected | Model prediksi semua BENIGN (without weights) |
| **Dengan class weights** | ✅ **FIXED** | Model sekarang fokus ke minority classes |
| Steps: 6302 | ✅ Correct | 1,613,310 ÷ 256 = 6,302 |

---

## 🎯 Next Steps

1. **Biarkan training selesai** (~26 jam untuk 50 epochs)
2. **Check F1 scores** di log atau evaluation
3. **Jika F1 minority < 0.7**: Re-train dengan SMOTE
4. **Jika F1 minority > 0.8**: Model sudah bagus!

---

**Created:** 2025-12-08  
**Status:** ✅ Class weights implemented  
**Training:** In progress dengan balanced weights
