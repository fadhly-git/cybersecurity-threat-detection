# Cybersecurity Threat Detection System
## Project Summary - WSN-DS & Cyber Security Attacks Dataset

---

## 📋 Ringkasan Proyek

Proyek ini mengimplementasikan sistem deteksi ancaman keamanan siber menggunakan model **Machine Learning (ML)** dan **Deep Learning (DL)**. Sistem ini membandingkan berbagai algoritma klasifikasi untuk mendeteksi serangan jaringan.

**Paper Referensi**: [Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of Machine and Deep Learning Techniques for Threat Detection](https://arxiv.org/abs/2407.06014)

---

## 📊 Dataset yang Digunakan

### 1. WSN-DS (Wireless Sensor Network Dataset)

| Atribut | Nilai |
|---------|-------|
| **File** | `data/raw/WSN-DS.csv` |
| **Total Sampel** | ~374,661 data points |
| **Domain** | Intrusion Detection pada Wireless Sensor Network |
| **Tipe Klasifikasi** | Binary (Normal vs Attack) |

#### Fitur Dataset WSN-DS

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | `id` | ID unik node sensor |
| 2 | `Time` | Timestamp kejadian |
| 3 | `Is_CH` | Apakah node adalah Cluster Head (0/1) |
| 4 | `who_CH` | ID Cluster Head |
| 5 | `Dist_To_CH` | Jarak ke Cluster Head |
| 6 | `ADV_S` | Advertisement messages sent |
| 7 | `ADV_R` | Advertisement messages received |
| 8 | `JOIN_S` | Join request messages sent |
| 9 | `JOIN_R` | Join request messages received |
| 10 | `SCH_S` | Schedule messages sent |
| 11 | `SCH_R` | Schedule messages received |
| 12 | `Rank` | Ranking node dalam cluster |
| 13 | `DATA_S` | Data packets sent |
| 14 | `DATA_R` | Data packets received |
| 15 | `Data_Sent_To_BS` | Data yang dikirim ke Base Station |
| 16 | `dist_CH_To_BS` | Jarak Cluster Head ke Base Station |
| 17 | `send_code` | Kode pengiriman |
| 18 | `Expaned_Energy` | Energi yang digunakan |
| 19 | `Attack type` | **TARGET**: Jenis serangan |

#### Distribusi Label WSN-DS

| Attack Type | Jumlah | Persentase | Binary Label |
|-------------|--------|------------|--------------|
| Normal | ~340,000 | ~90.7% | 0 |
| Flooding | ~7,000 | ~1.9% | 1 |
| Blackhole | ~14,000 | ~3.7% | 1 |
| Grayhole | ~6,000 | ~1.6% | 1 |
| TDMA | ~7,500 | ~2.0% | 1 |

---

### 2. Cyber Security Attacks Dataset

| Atribut | Nilai |
|---------|-------|
| **File** | `data/raw/Cyber Security Attacks.csv` |
| **Total Sampel** | 40,000 data points |
| **Domain** | Network Intrusion Detection |
| **Tipe Klasifikasi** | Multi-class (3 jenis serangan) |

#### Fitur Dataset Cyber Security Attacks

| No | Fitur | Tipe | Deskripsi |
|----|-------|------|-----------|
| 1 | `Timestamp` | datetime | Waktu kejadian |
| 2 | `Source IP Address` | string | IP address sumber |
| 3 | `Destination IP Address` | string | IP address tujuan |
| 4 | `Source Port` | int | Port sumber (0-65535) |
| 5 | `Destination Port` | int | Port tujuan (0-65535) |
| 6 | `Protocol` | categorical | TCP, UDP, ICMP |
| 7 | `Packet Length` | int | Ukuran paket (bytes) |
| 8 | `Packet Type` | categorical | Control, Data |
| 9 | `Traffic Type` | categorical | HTTP, FTP, DNS |
| 10 | `Payload Data` | text | Data payload |
| 11 | `Malware Indicators` | categorical | Indikator malware |
| 12 | `Anomaly Scores` | float | Skor anomali (0-100) |
| 13 | `Alerts/Warnings` | categorical | Alert yang di-trigger |
| 14 | `Attack Type` | categorical | **TARGET**: DDoS, Malware, Intrusion |
| 15 | `Attack Signature` | categorical | Known Pattern A/B |
| 16 | `Action Taken` | categorical | Blocked, Ignored, Logged |
| 17 | `Severity Level` | categorical | Low, Medium, High |
| 18 | `User Information` | text | Info user |
| 19 | `Device Information` | text | Browser/device info |
| 20 | `Network Segment` | categorical | Segment jaringan |
| 21 | `Geo-location Data` | text | Lokasi geografis |
| 22 | `Proxy Information` | text | Info proxy |
| 23 | `Firewall Logs` | text | Log firewall |
| 24 | `IDS/IPS Alerts` | text | Alert dari IDS/IPS |
| 25 | `Log Source` | categorical | Server, Firewall |

#### Distribusi Label Cyber Security Attacks

| Attack Type | Jumlah | Persentase |
|-------------|--------|------------|
| DDoS | 13,428 | 33.57% |
| Malware | 13,307 | 33.27% |
| Intrusion | 13,265 | 33.16% |

**Catatan**: Dataset ini tidak memiliki kelas "Normal" - semua data adalah serangan.

---

## 🔧 Preprocessing Pipeline

### Pipeline Umum

```
Raw Data → Clean Columns → Handle Missing → Feature Engineering → 
Label Encoding → Train/Test Split → Feature Scaling → Model Ready
```

### 1. Data Loading

```python
# WSN-DS
from data.data_loader_wsnds import load_wsnds, WSNDSLoader
X, y = load_wsnds(data_path="data/raw", binary=True, sample_frac=0.1)

# Cyber Security
from data.data_loader_cyber import load_cyber_security, CyberSecurityLoader
X, y = load_cyber_security(data_path="data/raw", binary=False, sample_frac=0.1)
```

### 2. Column Name Cleaning

```python
# Hapus whitespace dan standardisasi nama kolom
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')
```

### 3. Missing Value Handling

```python
# Numeric columns: isi dengan median
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)

# Replace infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Categorical columns: isi dengan 'Unknown'
for col in object_cols:
    df[col].fillna('Unknown', inplace=True)
```

### 4. Feature Engineering (Cyber Security Dataset)

#### 4.1 Numeric Features
```python
NUMERIC_FEATURES = [
    'Source Port', 'Destination Port', 
    'Packet Length', 'Anomaly Scores'
]

# Derived features
df['port_diff'] = abs(df['Source Port'] - df['Destination Port'])
df['port_sum'] = df['Source Port'] + df['Destination Port']
df['is_well_known_src'] = (df['Source Port'] < 1024).astype(int)
df['is_well_known_dst'] = (df['Destination Port'] < 1024).astype(int)
df['is_high_port_src'] = (df['Source Port'] > 49152).astype(int)
df['is_high_port_dst'] = (df['Destination Port'] > 49152).astype(int)
```

#### 4.2 Timestamp Features
```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['month'] = df['Timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
```

#### 4.3 IP Address Features
```python
# Split IP menjadi octets
ip_split = df['Source IP Address'].str.split('.', expand=True)
for i in range(4):
    df[f'src_ip_oct{i+1}'] = ip_split[i].astype(int)

# IP Class (A, B, C berdasarkan octet pertama)
df['src_ip_class'] = pd.cut(df['src_ip_oct1'], bins=[0, 127, 191, 223, 255], labels=[0, 1, 2, 3])

# Private IP detection
df['src_ip_is_private'] = (
    (oct1 == 10) | 
    ((oct1 == 172) & (oct2 >= 16) & (oct2 <= 31)) |
    ((oct1 == 192) & (oct2 == 168))
).astype(int)

# Same subnet check
df['same_subnet'] = (
    (df['src_ip_oct1'] == df['dst_ip_oct1']) &
    (df['src_ip_oct2'] == df['dst_ip_oct2']) &
    (df['src_ip_oct3'] == df['dst_ip_oct3'])
).astype(int)
```

#### 4.4 Categorical Encoding
```python
CATEGORICAL_FEATURES = [
    'Protocol', 'Packet Type', 'Traffic Type',
    'Action Taken', 'Severity Level', 'Network Segment',
    'Malware Indicators', 'Alerts/Warnings', 'Attack Signature',
    'Log Source', 'IDS/IPS Alerts'
]

# Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in CATEGORICAL_FEATURES:
    encoder = LabelEncoder()
    df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
```

#### 4.5 Text Features
```python
# Payload Data
df['payload_length'] = df['Payload Data'].str.len()
df['payload_word_count'] = df['Payload Data'].str.split().str.len()

# Device Information (Browser detection)
df['is_chrome'] = df['Device Information'].str.contains('Chrome', case=False).astype(int)
df['is_firefox'] = df['Device Information'].str.contains('Firefox', case=False).astype(int)
df['is_msie'] = df['Device Information'].str.contains('MSIE|Trident', case=False).astype(int)
df['is_mobile'] = df['Device Information'].str.contains('Mobile|Android|iPhone', case=False).astype(int)

# Firewall Logs
df['firewall_log_length'] = df['Firewall Logs'].str.len()
```

### 5. Label Creation

#### WSN-DS (Binary Classification)
```python
BINARY_MAPPING = {
    'Normal': 0,
    'Flooding': 1,
    'Blackhole': 1,
    'Grayhole': 1,
    'TDMA': 1
}
df['target'] = df['Attack type'].map(BINARY_MAPPING)
```

#### Cyber Security (Multi-class Classification)
```python
# Tidak ada kelas Normal, semua adalah attack
ATTACK_MAPPING = {
    'DDoS': 0,
    'Malware': 1,
    'Intrusion': 2
}
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['Attack Type'])
```

### 6. Data Splitting

```python
from sklearn.model_selection import train_test_split

# Split ratio: 70% train, 10% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

# Hasil:
# Train: 70% (0.8 × 0.875 = 0.7)
# Validation: 10% (0.8 × 0.125 = 0.1)  
# Test: 20%
```

### 7. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit hanya pada train
X_val_scaled = scaler.transform(X_val)          # Transform saja
X_test_scaled = scaler.transform(X_test)        # Transform saja

# Simpan scaler untuk inference
import joblib
joblib.dump(scaler, "saved_models/scaler_WSN-DS.joblib")
joblib.dump(scaler, "saved_models/scaler_Cyber-Security.joblib")
```

---

## 🤖 Model Machine Learning

### 1. Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| - | Default | Tidak ada hyperparameter utama |

**Karakteristik:**
- ✅ Sangat cepat
- ✅ Bekerja baik dengan data besar
- ❌ Asumsi independensi fitur (sering tidak realistis)
- ❌ Sensitif terhadap fitur yang berkorelasi

---

### 2. Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `max_depth` | 20 | Kedalaman maksimum tree |
| `random_state` | 42 | Untuk reproducibility |

**Karakteristik:**
- ✅ Mudah diinterpretasi
- ✅ Tidak perlu scaling (tapi kita tetap scale untuk konsistensi)
- ✅ Handle fitur categorical dan numerical
- ❌ Cenderung overfitting

---

### 3. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `n_estimators` | 100 | Jumlah decision trees |
| `max_depth` | 20 | Kedalaman maksimum tiap tree |
| `n_jobs` | -1 | Gunakan semua CPU cores |
| `random_state` | 42 | Untuk reproducibility |

**Karakteristik:**
- ✅ Robust terhadap overfitting
- ✅ Handle missing values dengan baik
- ✅ Feature importance bawaan
- ❌ Lebih lambat dari single Decision Tree
- ❌ Kurang interpretable

---

### 4. K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `n_neighbors` | 5 | Jumlah tetangga terdekat |
| `n_jobs` | -1 | Gunakan semua CPU cores |

**Karakteristik:**
- ✅ Simple dan intuitif
- ✅ Tidak ada training phase
- ❌ **Sangat lambat** untuk dataset besar
- ❌ **Wajib scaling** (distance-based)
- ❌ Sensitif terhadap curse of dimensionality

---

### 5. Support Vector Machine (SVM)

```python
from sklearn.linear_model import SGDClassifier

# Menggunakan SGD untuk efisiensi (equivalent to linear SVM)
model = SGDClassifier(
    loss='hinge',       # Hinge loss = SVM
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `loss` | 'hinge' | Equivalent to linear SVM |
| `max_iter` | 1000 | Maksimum iterasi |
| `tol` | 1e-3 | Tolerance untuk stopping |

**Karakteristik:**
- ✅ Efektif di high-dimensional space
- ✅ Memory efficient
- ❌ **Sangat lambat** dengan kernel non-linear
- ❌ **Wajib scaling**
- ❌ Tidak memberikan probabilitas langsung

---

### 6. Extra Trees

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `n_estimators` | 100 | Jumlah trees |
| `max_depth` | 20 | Kedalaman maksimum |
| `n_jobs` | -1 | Gunakan semua CPU cores |

**Karakteristik:**
- ✅ **Lebih cepat** dari Random Forest
- ✅ Mengurangi variance
- ✅ Lebih random dalam pemilihan split
- ❌ Bisa sedikit kurang akurat dari RF

---

### 7. CatBoost (Bonus untuk Cyber Security)

```python
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=8,
    loss_function='MultiClass',  # atau 'Logloss' untuk binary
    eval_metric='TotalF1',
    random_seed=42,
    verbose=False
)

# CatBoost bisa handle categorical features secara native
train_pool = Pool(X_train_df, y_train, cat_features=categorical_indices)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `iterations` | 500 | Jumlah boosting iterations |
| `learning_rate` | 0.1 | Learning rate |
| `depth` | 8 | Kedalaman tree |
| `loss_function` | 'MultiClass' | Untuk multi-class classification |

**Karakteristik:**
- ✅ **Native categorical handling** (tidak perlu encoding)
- ✅ Handle missing values otomatis
- ✅ State-of-the-art untuk tabular data
- ❌ Training lebih lama dari RF/ET

---

## 🧠 Model Deep Learning

Semua model DL diadaptasi untuk **data tabular** (bukan image). Arsitektur CNN diubah menjadi Dense layers.

### 1. VGG16 Tabular

```python
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

def create_vgg16_tabular(input_dim: int, n_classes: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Block 1 (2 layers)
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2 (2 layers)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3 (2 layers)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='VGG16_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Arsitektur:**
```
Input(features) → Dense(512)×2 → BN → Dropout(0.3)
               → Dense(256)×2 → BN → Dropout(0.3)
               → Dense(128)×2 → Dropout(0.3)
               → Dense(n_classes, softmax)
```

---

### 2. VGG19 Tabular

```python
def create_vgg19_tabular(input_dim: int, n_classes: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Block 1 (3 layers) - lebih dalam dari VGG16
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2 (3 layers)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3 (3 layers)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='VGG19_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Arsitektur:**
```
Input(features) → Dense(512)×3 → BN → Dropout(0.3)
               → Dense(256)×3 → BN → Dropout(0.3)
               → Dense(128)×3 → Dropout(0.3)
               → Dense(n_classes, softmax)
```

---

### 3. ResNet18 Tabular

```python
def create_resnet18_tabular(input_dim: int, n_classes: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial block
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks
    for units in [256, 128, 64]:
        # Shortcut connection
        shortcut = layers.Dense(units)(x)
        
        # Main path
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        
        # Add shortcut (skip connection)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet18_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Arsitektur dengan Skip Connections:**
```
Input → Dense(256) → BN
     ↓
     ├──────────────────────┐
     ↓                      │ (shortcut)
Dense(256) → BN → Dense(256) → BN
     ↓                      │
     Add ←──────────────────┘
     ↓
     ReLU → Dropout(0.3)
     ↓
     ... (repeat for 128, 64)
     ↓
Dense(n_classes, softmax)
```

---

### 4. ResNet50 Tabular

```python
def create_resnet50_tabular(input_dim: int, n_classes: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Lebih banyak residual blocks (6 blocks)
    for units in [512, 256, 256, 128, 128, 64]:
        shortcut = layers.Dense(units)(x)
        
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)  # Dropout lebih kecil
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet50_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

### 5. Inception Tabular

```python
def create_inception_tabular(input_dim: int, n_classes: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Inception module - multiple parallel paths
    # Path 1: Direct transformation
    path1 = layers.Dense(128, activation='relu')(inputs)
    path1 = layers.BatchNormalization()(path1)
    
    # Path 2: Two-layer transformation
    path2 = layers.Dense(64, activation='relu')(inputs)
    path2 = layers.Dense(128, activation='relu')(path2)
    path2 = layers.BatchNormalization()(path2)
    
    # Path 3: Three-layer transformation (deeper features)
    path3 = layers.Dense(32, activation='relu')(inputs)
    path3 = layers.Dense(64, activation='relu')(path3)
    path3 = layers.Dense(128, activation='relu')(path3)
    path3 = layers.BatchNormalization()(path3)
    
    # Concatenate all paths
    concat = layers.Concatenate()([path1, path2, path3])  # Output: 384 units
    
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='Inception_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Arsitektur Parallel Paths:**
```
              ┌→ Dense(128) → BN ──────────────────┐
              │                                     │
Input(features)─→ Dense(64) → Dense(128) → BN ────→ Concatenate → Dense(256) → Dropout → Output
              │                                     │
              └→ Dense(32) → Dense(64) → Dense(128) → BN ─┘
```

---

### Training Configuration

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_callbacks(patience=10):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

# Training parameters
EPOCHS = 30
BATCH_SIZE = 128
PATIENCE = 10
LEARNING_RATE = 0.001
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `epochs` | 30 | Maksimum training epochs |
| `batch_size` | 128 | Ukuran mini-batch |
| `patience` | 10 | Early stopping patience |
| `learning_rate` | 0.001 | Initial learning rate |
| `optimizer` | Adam | Adaptive learning rate optimizer |
| `loss` | sparse_categorical_crossentropy | Untuk integer labels |

---

## 📈 Evaluasi Model

### Metrik yang Digunakan

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # ROC-AUC hanya untuk binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics
```

### Penjelasan Metrik

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **Accuracy** | (TP + TN) / Total | Persentase prediksi benar |
| **Precision** | TP / (TP + FP) | Ketepatan prediksi positif |
| **Recall** | TP / (TP + FN) | Kemampuan mendeteksi semua positif |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean precision & recall |
| **ROC-AUC** | Area Under ROC Curve | Kemampuan ranking/membedakan kelas |

### Averaging Methods

- **macro**: Rata-rata metrik per kelas (tidak weighted)
- **weighted**: Rata-rata weighted berdasarkan jumlah sampel per kelas
- **micro**: Total TP, FP, FN across all classes

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LOADING                                │
│  ┌───────────────────────┐    ┌───────────────────────────┐         │
│  │       WSN-DS          │    │   Cyber Security Attacks  │         │
│  │  (374,661 samples)    │    │     (40,000 samples)      │         │
│  │  Binary: Normal/Attack │    │  Multi: DDoS/Malware/Int  │         │
│  └───────────┬───────────┘    └─────────────┬─────────────┘         │
└──────────────┼──────────────────────────────┼───────────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                 │
│  • Clean column names (strip whitespace)                             │
│  • Handle missing values (median for numeric, 'Unknown' for cat)     │
│  • Handle infinity values (replace with NaN → median)                │
│  • Feature engineering:                                              │
│    - WSN-DS: Use raw numeric features                                │
│    - Cyber: IP parsing, port features, timestamp, text features      │
│  • Label encoding (binary/multi-class)                               │
│  • Categorical encoding (LabelEncoder)                               │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SPLITTING                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │ Train (70%)  │   │  Val (10%)   │   │  Test (20%)  │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
│  stratify=y, random_state=42                                         │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE SCALING                                │
│  StandardScaler: z = (x - μ) / σ                                     │
│  • Fit on train set only                                             │
│  • Transform train, val, test                                        │
│  • Save scaler: saved_models/scaler_DATASET.joblib                   │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL TRAINING                                 │
│                                                                      │
│  ┌─────────────────────────┐     ┌─────────────────────────┐        │
│  │   MACHINE LEARNING      │     │     DEEP LEARNING       │        │
│  │   • Naive Bayes         │     │     • VGG16 Tabular     │        │
│  │   • Decision Tree       │     │     • VGG19 Tabular     │        │
│  │   • Random Forest       │     │     • ResNet18 Tabular  │        │
│  │   • KNN                 │     │     • ResNet50 Tabular  │        │
│  │   • SVM (SGD)           │     │     • Inception Tabular │        │
│  │   • Extra Trees         │     │                         │        │
│  │   • CatBoost (Cyber)    │     │   Callbacks:            │        │
│  │                         │     │   • EarlyStopping       │        │
│  │   Fit: model.fit(X, y)  │     │   • ReduceLROnPlateau   │        │
│  └─────────────────────────┘     └─────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALUATION                                   │
│  • Accuracy: (TP + TN) / Total                                       │
│  • Precision: TP / (TP + FP) [macro average]                         │
│  • Recall: TP / (TP + FN) [macro average]                            │
│  • F1-Score: 2 × (P × R) / (P + R) [macro average]                   │
│  • ROC-AUC: Area under ROC curve (binary only)                       │
│  • Training Time: seconds                                            │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RESULTS & SAVING                                │
│  • Save best model: saved_models/DATASET_MODEL_TIMESTAMP.joblib/keras│
│  • Save scaler: saved_models/scaler_DATASET.joblib                   │
│  • Save results: results/new_datasets_results_TIMESTAMP.csv          │
│  • Print comparison table                                            │
│  • Log to: logs/new_datasets_TIMESTAMP.log                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Persiapan Dataset

```bash
# Pastikan file berada di lokasi yang benar
data/raw/WSN-DS.csv
data/raw/Cyber Security Attacks.csv
```

### 3. Run Training

```bash
# Training WSN-DS dan Cyber Security saja
python main_new_datasets.py
```

### 4. Configuration (Edit `main_new_datasets.py`)

```python
# Baris 422-423
SAMPLE_FRAC = 0.1      # 10% sampling (None untuk full data)
DL_EPOCHS = 30         # Epoch untuk deep learning
```

| Mode | SAMPLE_FRAC | DL_EPOCHS | Waktu | Akurasi |
|------|-------------|-----------|-------|---------|
| **Quick Test** | 0.1 | 10 | ~5 min | Moderate |
| **Balanced** | 0.2 | 30 | ~15 min | Good |
| **Full** | None | 50 | ~2 hours | Best |

---

## 📊 Hasil Eksperimen

### WSN-DS Dataset (Binary Classification)

| Model | Accuracy | Precision | Recall | F1-Score | Time (s) | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|---------|
| Naive Bayes | 0.9756 | 0.8994 | 0.9821 | 0.9356 | 0.01 | 0.9816 |
| Decision Tree | 0.9931 | 0.9781 | 0.9817 | 0.9799 | 0.08 | 0.9817 |
| Random Forest | 0.9961 | 0.9909 | 0.9865 | 0.9887 | 0.52 | 0.9967 |
| KNN | 0.9963 | 0.9903 | 0.9878 | 0.9891 | 0.00 | 0.9942 |
| SVM | 0.9811 | 0.9680 | 0.9189 | 0.9417 | 0.08 | N/A |
| **Extra Trees** | **0.9965** | **0.9911** | **0.9886** | **0.9899** | 0.30 | **0.9972** |
| VGG16 | 0.9961 | 0.9872 | 0.9903 | 0.9888 | 60.36 | 0.9973 |
| VGG19 | 0.9963 | 0.9897 | 0.9885 | 0.9891 | 83.88 | 0.9965 |
| ResNet18 | 0.9961 | 0.9872 | 0.9903 | 0.9888 | 37.92 | 0.9971 |
| ResNet50 | 0.9951 | 0.9830 | 0.9884 | 0.9857 | 104.86 | 0.9963 |
| Inception | 0.9963 | 0.9903 | 0.9878 | 0.9891 | 23.75 | 0.9972 |

**🏆 Best Model WSN-DS: Extra Trees (F1-Score: 0.9899)**

---

### Cyber Security Dataset (Multi-class Classification)

| Model | Accuracy | Precision | Recall | F1-Score | Time (s) |
|-------|----------|-----------|--------|----------|----------|
| Naive Bayes | 0.3188 | 0.3143 | 0.3226 | 0.3006 | 0.00 |
| Decision Tree | 0.3400 | 0.3406 | 0.3398 | 0.3397 | 0.05 |
| Random Forest | 0.3300 | 0.3286 | 0.3291 | 0.3283 | 0.25 |
| KNN | 0.3275 | 0.3255 | 0.3253 | 0.3178 | 0.00 |
| SVM | 0.3188 | 0.3181 | 0.3180 | 0.3164 | 0.09 |
| Extra Trees | 0.3463 | 0.3448 | 0.3451 | 0.3439 | 0.21 |
| **CatBoost** | **0.3600** | **0.3611** | **0.3604** | **0.3601** | 10.14 |
| VGG16 | 0.3188 | 0.3163 | 0.3179 | 0.3091 | 5.69 |
| VGG19 | 0.3363 | 0.2914 | 0.3300 | 0.1840 | 7.24 |
| ResNet18 | 0.3263 | 0.3253 | 0.3220 | 0.2817 | 7.44 |
| ResNet50 | 0.3388 | 0.3012 | 0.3349 | 0.2002 | 17.74 |
| Inception | 0.3250 | 0.3160 | 0.3218 | 0.3033 | 4.78 |

**🏆 Best Model Cyber Security: CatBoost (F1-Score: 0.3601)**

---

## ⚠️ Catatan Penting: Cyber Security Dataset

Dataset **Cyber Security Attacks** menunjukkan performa rendah (~33-36%) karena:

1. **Dataset bersifat sintetis/random** - fitur tidak memiliki korelasi dengan target
2. **Distribusi label seimbang sempurna** (33.3% per kelas) - random guessing juga ~33%
3. **Tidak ada pola diskriminatif** yang bisa dipelajari model

**Analisis Fitur:**
```
1. SOURCE PORT by Attack Type:
   DDoS: mean=32945, Malware: mean=32979, Intrusion: mean=32987  ← Hampir identik!

2. ANOMALY SCORES by Attack Type:
   DDoS: mean=50.24, Malware: mean=50.13, Intrusion: mean=49.98  ← Hampir identik!

3. PROTOCOL distribution: ~33% TCP, ~33% UDP, ~33% ICMP untuk SEMUA attack types
```

**Rekomendasi**: Gunakan dataset nyata seperti CICIDS2017 atau NSL-KDD untuk hasil yang meaningful.

---

## 📁 Struktur Output

```
cybersecurity-threat-detection/
├── results/
│   └── new_datasets_results_YYYYMMDD_HHMMSS.csv
├── saved_models/
│   ├── WSN-DS_Extra Trees_YYYYMMDD_HHMMSS.joblib
│   ├── Cyber-Security_CatBoost_YYYYMMDD_HHMMSS.joblib
│   ├── scaler_WSN-DS.joblib
│   └── scaler_Cyber-Security.joblib
└── logs/
    └── new_datasets_YYYYMMDD_HHMMSS.log
```

---

## 🔑 Key Takeaways

1. **Tree-based models (Extra Trees, Random Forest, CatBoost)** memberikan performa terbaik untuk data tabular cybersecurity

2. **WSN-DS dataset** sangat baik untuk benchmark - F1-Score hingga 99%

3. **Cyber Security dataset** tidak cocok untuk ML karena bersifat sintetis

4. **Deep Learning** tidak selalu lebih baik dari ML untuk data tabular

5. **Feature scaling wajib** untuk KNN, SVM, dan semua model DL

6. **Early Stopping** mencegah overfitting pada DL models

7. **CatBoost** unggul untuk dataset dengan banyak fitur categorical

---

## 📚 Referensi

- Paper: [Evaluating Predictive Models in Cybersecurity (arXiv:2407.06014)](https://arxiv.org/abs/2407.06014)
- WSN-DS: [Wireless Sensor Network Dataset](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds)
- Scikit-learn: [Documentation](https://scikit-learn.org/)
- TensorFlow/Keras: [Documentation](https://www.tensorflow.org/)
- CatBoost: [Documentation](https://catboost.ai/)

---

**Generated**: January 12, 2026  
**Author**: GitHub Copilot  
**Datasets**: WSN-DS (374K samples), Cyber Security Attacks (40K samples)
