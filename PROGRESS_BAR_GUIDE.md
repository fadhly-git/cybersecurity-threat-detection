# Progress Bar Guide - SMOTE-ENN and Preprocessing

Semua proses preprocessing yang memakan waktu lama sekarang dilengkapi dengan **progress bar** menggunakan `tqdm`.

## ✅ Proses dengan Progress Bar

### 1. **Loading Data** (Loading CSV files)
```
Loading Data |████████████████████████████| 8/8 [00:15<00:00,  2.15s/it]
```

### 2. **Feature Selection** (Mutual Information)
```
Feature Selection [Training] |████████████| 1/1 [00:45<00:00, 45.23s]
Feature Selection [Test] |████████████| 1/1 [00:03<00:00,  3.12s]
```

### 3. **Scaling** (Min-Max Normalization)
```
Scaling [Training set] |████████████| 1/1 [00:02<00:00,  2.45s]
Scaling [Test set] |████████████| 1/1 [00:00<00:00,  0.89s]
```

### 4. **SMOTE-ENN** (Oversampling + Noise Cleaning) - YANG PALING LAMA ⏳
```
SMOTE-ENN [SMOTE oversampling] |████████████████| 50% [03:25<03:25, 205.32s]
SMOTE-ENN [Reshaping] |████████████████| 100% [03:42<00:00, 222.15s]
```

### 5. **Reshaping** (Reshape untuk Deep Learning)
```
Reshaping [Training set] |████████████| 1/1 [00:05<00:00,  5.23s]
Reshaping [Test set] |████████████| 1/1 [00:00<00:00,  0.67s]
```

## 📊 Output Contoh Lengkap

```
============================================================
  LOADING CICIDS2017 DATASET
============================================================

Found 8 CSV files:
  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  ...

Loading Data |████████████████████████████| 8/8 [01:23<00:00,  10.38s/it]

Combining dataframes...

✅ Total dataset shape: (2830743, 79)
   Rows: 2,830,743
   Columns: 79

[1/10] Removed 308,381 duplicate rows
      Remaining: 2,522,362 rows

[2/10] Handled 353 missing values
      Remaining: 2,522,362 rows

[3/10] Removed 0 rows with infinity values
      Remaining: 2,522,362 rows

[4/10] Consolidating labels...
      Original label distribution:
      Label
        BENIGN             2273513
        DoS Hulk             231073
        PortScan             161020
      ...

      Consolidated label distribution:
      Label
        0    2273513
        1     308186
        2     161020
      ...

[5/10] Correlation-based feature selection (threshold=0.95)...
      Removed 5 highly correlated features
      Final features: 74

[6/10] Stratified train-test split (test_size=0.2)...
      Train shape: (2009696, 75)
      Test shape: (502424, 75)

[7/10] Selecting top 60 features via mutual information...
Feature Selection [Training] |████████████████████████| 100% [00:45<00:00]
Feature Selection [Test] |████████████████████████| 100% [00:03<00:00]
      Selected features: 60

[8/10] Applying Min-Max scaling...
Scaling [Training set] |████████████████████████| 100% [00:02<00:00]
Scaling [Test set] |████████████████████████| 100% [00:00<00:00]

[9/10] Applying SMOTE-ENN (oversample + clean)...
      Processing... (this may take a few minutes)
      ⏳ Running SMOTE oversampling and ENN noise cleaning...

      Class distribution BEFORE SMOTE-ENN:
        Class 0: 1,821,168 samples
        Class 1: 186,528 samples
        ...

SMOTE-ENN |████████████████████████████████████████| 100% [05:12<00:00]

      Class distribution AFTER SMOTE-ENN:
        Class 0: 1,821,168 samples
        Class 1: 1,821,168 samples
        ...

[10/10] Reshaping for deep learning models...
Reshaping [Training set] |████████████████████████| 100% [00:08<00:00]
Reshaping [Test set] |████████████████████████| 100% [00:02<00:00]

✅ Preprocessing Complete!
   Final X_train shape: (3642336, 60, 1)
   Final X_test shape: (502424, 60, 1)
   Final y_train shape: (3642336,)
   Final y_test shape: (502424,)
```

## 🚀 Cara Menggunakan

```python
from src.data.datasets.cicids2017 import CICIDS2017Loader

# Load dan preprocess
loader = CICIDS2017Loader('data/raw/CICIDS2017/')

# Jalankan pipeline dengan progress bar
X_train, X_test, y_train, y_test = loader.preprocess_pipeline(
    test_size=0.2,
    apply_smote=True  # Progress bar akan muncul!
)
```

## 📈 Progress Bar Tips

- **Gunakan Jupyter Notebook** - Progress bar lebih bagus di notebook
- **Tidak perlu khawatir jika berhenti 5 menit** - Ini normal untuk SMOTE-ENN
- **Bisa di-cancel dengan Ctrl+C** kapan saja
- **Lihat message "Processing..."** sebagai tanda sistem sedang bekerja

## ⚙️ Optimization Tips

Jika SMOTE-ENN masih terlalu lama:

1. **Gunakan `n_jobs=-1`** (sudah diaktifkan) untuk parallel processing
2. **Kurangi sampling ratio** jika tidak perlu balancing 100%:
   ```python
   # Modify in apply_smoteenn():
   smote_enn = SMOTEENN(
       random_state=42,
       smote_kwargs={'k_neighbors': 5, 'sampling_strategy': 0.5}
   )
   ```
3. **Skip SMOTE jika dataset sudah balance**:
   ```python
   X_train, X_test, y_train, y_test = loader.preprocess_pipeline(
       test_size=0.2,
       apply_smote=False  # Skip SMOTE
   )
   ```

## 📦 Dependencies

- `tqdm>=4.65.0` (sudah included di requirements.txt)
- Semua library lain sudah ada

Install jika perlu:
```bash
pip install tqdm>=4.65.0
```
