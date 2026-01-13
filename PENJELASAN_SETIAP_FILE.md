# PENJELASAN DETAIL SETIAP FILE
## Cybersecurity Threat Detection Project

---

## 📁 FILE KONFIGURASI

### 1. `config.py` - Konfigurasi Global Project

**Fungsi**: Menyimpan semua konfigurasi global yang digunakan di seluruh project.

**Isi Utama:**

```python
class Config:
    # === PATHS ===
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    MODEL_DIR = BASE_DIR / "saved_models"
    LOG_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    
    # === DATASET ===
    DATASET_NAME = "NSL-KDD"
    NSL_KDD_COLUMNS = [...]  # Nama kolom NSL-KDD
    ATTACK_CATEGORIES = {...}  # Mapping attack types
    
    # === TRAINING ===
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    BATCH_SIZE = 128
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # === MODEL SELECTION ===
    USE_ML_MODELS = True
    USE_DL_MODELS = True
    
    # === PREPROCESSING ===
    SCALING_METHOD = 'standard'
    HANDLE_MISSING = 'median'
    SMOTE_K_NEIGHBORS = 5
```

**Penggunaan:**
```python
from config import Config

# Akses path
data_path = Config.RAW_DATA_DIR / "WSN-DS.csv"

# Akses parameter
batch_size = Config.BATCH_SIZE
```

**Kapan Digunakan:**
- Di semua file yang membutuhkan path standardisasi
- Di script training untuk hyperparameter
- Di data loader untuk konfigurasi preprocessing

---

### 2. `requirements.txt` - Dependencies

**Fungsi**: Daftar semua library Python yang dibutuhkan.

**Isi:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
joblib>=1.0.0
catboost>=1.0.0
pyyaml>=6.0
```

**Install:**
```bash
pip install -r requirements.txt
```

---

### 3. `install.sh` - Installation Script

**Fungsi**: Script otomatis untuk setup environment.

**Isi:**
```bash
#!/bin/bash

echo "Installing Cybersecurity Threat Detection..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Installation complete!"
```

**Jalankan:**
```bash
bash install.sh
```

---

## 🎯 FILE EKSEKUSI UTAMA

### 1. `main_new_datasets.py` - Training Script untuk WSN-DS & Cyber Security ⭐

**Fungsi**: Script utama untuk training model pada dataset WSN-DS dan Cyber Security Attacks.

**Flow Eksekusi:**

```
START
  ↓
Setup Logging
  ↓
Configure Parameters (SAMPLE_FRAC, DL_EPOCHS)
  ↓
┌─────────────────────────┐
│ EXPERIMENT 1: WSN-DS    │
└─────────────────────────┘
  ↓
Load Data (WSNDSLoader)
  ↓
Preprocess (binary/multi-class)
  ↓
Split Data (Train 70%, Val 10%, Test 20%)
  ↓
Scale Features (StandardScaler)
  ↓
Train ML Models (6 models)
  ├─ Naive Bayes
  ├─ Decision Tree
  ├─ Random Forest
  ├─ KNN
  ├─ SVM
  └─ Extra Trees ⭐
  ↓
Train DL Models (5 models)
  ├─ VGG16
  ├─ VGG19
  ├─ ResNet18
  ├─ ResNet50
  └─ Inception
  ↓
Evaluate & Compare
  ↓
Save Best Model
  ↓
┌─────────────────────────┐
│ EXPERIMENT 2: Cyber Sec │
└─────────────────────────┘
  ↓
(Repeat same flow)
  ↓
Combine Results
  ↓
Save to CSV
  ↓
END
```

**Komponen Utama:**

#### A. Setup Logging
```python
def setup_logging():
    """Create log file and configure logger"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/new_datasets_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logger, log_file
```
- Output ke file dan console
- Format: timestamp + level + message

#### B. ModelEvaluator Class
```python
class ModelEvaluator:
    def __init__(self, dataset_name: str):
        self.results = []
        self.best_model = None
        self.best_f1 = 0
    
    def evaluate(self, name, y_true, y_pred, y_proba, train_time, model):
        """Calculate metrics and track best model"""
        metrics = {
            'Dataset': self.dataset_name,
            'Model': name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro'),
            'Recall': recall_score(y_true, y_pred, average='macro'),
            'F1-Score': f1_score(y_true, y_pred, average='macro'),
            'Training Time (s)': train_time
        }
        
        # Track best model based on F1-Score
        if metrics['F1-Score'] > self.best_f1:
            self.best_f1 = metrics['F1-Score']
            self.best_model = model
            self.best_model_name = name
        
        return metrics
```

#### C. Train ML Models
```python
def train_ml_models(X_train, y_train, X_test, y_test, evaluator):
    """Train all ML models"""
    models = get_sklearn_models(random_state=42)
    
    for name, model in models.items():
        log_print(f"\n🔄 Training {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time, model)
        
        # Log results
        log_print(f"   ✅ F1-Score: {metrics['F1-Score']:.6f}")
        log_print(f"   ⏱️  Time: {train_time:.2f}s")
```

#### D. Train DL Models
```python
def train_dl_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                    evaluator, n_classes, input_dim, epochs):
    """Train all DL models"""
    models = get_dl_models(input_dim=input_dim, n_classes=n_classes)
    
    for name, model in models.items():
        log_print(f"\n🔄 Training {name}...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Evaluate
        metrics = evaluator.evaluate(name, y_test, y_pred, y_pred_proba, train_time, model)
        
        log_print(f"   ✅ F1-Score: {metrics['F1-Score']:.6f}")
        log_print(f"   ⏱️  Time: {train_time:.2f}s")
```

#### E. Run Experiment WSN-DS
```python
def run_experiment_wsnds(sample_frac=None, dl_epochs=30):
    """Run complete experiment on WSN-DS dataset"""
    
    evaluator = ModelEvaluator(dataset_name="WSN-DS")
    
    # 1. Load data
    X, y = load_wsnds(data_path="data/raw", binary=True, sample_frac=sample_frac)
    
    # 2. Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, "saved_models/scaler_WSN-DS.joblib")
    
    # 4. Train models
    train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test, evaluator)
    train_dl_models(X_train_scaled, y_train, X_val_scaled, y_val, 
                    X_test_scaled, y_test, evaluator, n_classes, input_dim, dl_epochs)
    
    return evaluator
```

#### F. Main Function
```python
def main():
    """Main execution"""
    
    # Configuration
    SAMPLE_FRAC = 0.1   # 10% untuk testing cepat
    DL_EPOCHS = 30
    
    all_results = []
    best_models = {}
    
    # Experiment 1: WSN-DS
    evaluator_wsnds = run_experiment_wsnds(sample_frac=SAMPLE_FRAC, dl_epochs=DL_EPOCHS)
    evaluator_wsnds.print_results()
    all_results.append(evaluator_wsnds.get_results_df())
    
    # Save best model
    model_path = evaluator_wsnds.save_best_model()
    best_models['WSN-DS'] = {
        'path': model_path,
        'name': evaluator_wsnds.best_model_name,
        'f1': evaluator_wsnds.best_f1
    }
    
    # Experiment 2: Cyber Security
    # ... (same flow)
    
    # Combine and save results
    combined_results = pd.concat(all_results, ignore_index=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_results.to_csv(f"results/new_datasets_results_{timestamp}.csv", index=False)
    
    return combined_results
```

**Cara Menjalankan:**
```bash
python main_new_datasets.py
```

**Output:**
- Log file: `logs/new_datasets_20260113_120000.log`
- Results CSV: `results/new_datasets_results_20260113_120000.csv`
- Saved models: `saved_models/WSN-DS_Extra Trees_20260113.joblib`
- Saved scaler: `saved_models/scaler_WSN-DS.joblib`

---

### 2. `main.py` - Training Script untuk NSL-KDD & CICIDS2017

**Fungsi**: Script untuk training pada dataset klasik (NSL-KDD, CICIDS2017).

**Perbedaan dengan `main_new_datasets.py`:**
- Dataset berbeda (NSL-KDD, CICIDS2017)
- Preprocessing berbeda
- Tidak ada CatBoost support

**Flow sama dengan `main_new_datasets.py`**

---

## 📦 MODUL DATA (`data/`)

### 1. `data/data_loader_wsnds.py` - WSN-DS Loader ⭐

**Fungsi**: Load dan preprocess dataset WSN-DS.

**Class: WSNDSLoader**

#### Method 1: `__init__()`
```python
def __init__(self, data_path: str = "data/raw"):
    self.data_path = Path(data_path)
    self.df = None
    self.label_encoder = LabelEncoder()
    self.feature_columns = None
```
- Initialize loader
- Set path ke dataset

#### Method 2: `load_data()`
```python
def load_data(self) -> pd.DataFrame:
    """Load WSN-DS dataset from CSV"""
    
    filepath = self.data_path / "WSN-DS.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    # Read CSV
    self.df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    
    # Clean column names
    self.df.columns = self.df.columns.str.strip()
    self.df.columns = self.df.columns.str.replace(' ', '_')
    
    # Rename specific columns
    if 'who CH' in self.df.columns:
        self.df.rename(columns={'who CH': 'who_CH'}, inplace=True)
    if 'Attack type' in self.df.columns:
        self.df.rename(columns={'Attack type': 'Attack_type'}, inplace=True)
    
    return self.df
```

**Output**: DataFrame dengan 374,661 baris

#### Method 3: `preprocess()`
```python
def preprocess(self, binary: bool = True, sample_frac: Optional[float] = None):
    """
    Preprocess WSN-DS dataset
    
    Steps:
    1. Load data (if not loaded)
    2. Sample data (if sample_frac specified)
    3. Clean attack_type column
    4. Map to binary or multi-class
    5. Handle missing values
    6. Handle infinity values
    7. Return preprocessed DataFrame
    """
    
    if self.df is None:
        self.load_data()
    
    # === STEP 1: Sampling ===
    if sample_frac and sample_frac < 1.0:
        self.df = self.df.sample(frac=sample_frac, random_state=42)
    
    # === STEP 2: Clean labels ===
    self.df['Attack_type'] = self.df['Attack_type'].astype(str).str.strip()
    
    # === STEP 3: Map attack types ===
    self.df['attack_category'] = self.df['Attack_type'].map(self.ATTACK_MAPPING)
    self.df['attack_category'].fillna('Normal', inplace=True)
    
    # === STEP 4: Create target ===
    if binary:
        # Binary: Normal (0) vs Attack (1)
        self.df['target'] = self.df['attack_category'].map(self.BINARY_MAPPING)
    else:
        # Multi-class: 5 categories
        self.label_encoder.fit(self.df['attack_category'])
        self.df['target'] = self.label_encoder.transform(self.df['attack_category'])
    
    # === STEP 5: Handle missing values ===
    self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    numeric_cols = self.df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if self.df[col].isnull().any():
            median_val = self.df[col].median()
            if pd.isna(median_val):
                median_val = 0
            self.df[col].fillna(median_val, inplace=True)
    
    # === STEP 6: Drop invalid rows ===
    self.df.dropna(subset=['target'], inplace=True)
    
    return self.df
```

**Parameter:**
- `binary`: True untuk binary classification, False untuk multi-class
- `sample_frac`: Fraksi data (0.1 = 10%, None = 100%)

**Output**: Preprocessed DataFrame

#### Method 4: `get_X_y()`
```python
def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features (X) and labels (y)
    
    Returns:
        X: Features array (n_samples, 17)
        y: Labels array (n_samples,)
    """
    
    if self.df is None or 'target' not in self.df.columns:
        raise ValueError("Call preprocess() first")
    
    # Exclude columns
    excluded_cols = ['id', 'node_id', 'Attack_type', 'attack_category', 'target']
    
    # Get feature columns
    feature_cols = [col for col in self.df.columns if col not in excluded_cols]
    
    # Extract
    X = self.df[feature_cols].values
    y = self.df['target'].values
    
    return X, y
```

**Output:**
- X: numpy array shape (n_samples, 17)
- y: numpy array shape (n_samples,)

#### Method 5: `get_catboost_data()`
```python
def get_catboost_data(self):
    """
    Get data in format for CatBoost (with categorical columns)
    
    Returns:
        X: DataFrame dengan semua features
        y: Series dengan labels
        cat_indices: List of categorical column indices
    """
    
    excluded_cols = ['id', 'node_id', 'Attack_type', 'attack_category', 'target']
    feature_cols = [col for col in self.df.columns if col not in excluded_cols]
    
    X = self.df[feature_cols]
    y = self.df['target']
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    
    return X, y, cat_indices
```

#### Helper Function: `load_wsnds()`
```python
def load_wsnds(data_path: str = "data/raw", 
               binary: bool = True, 
               sample_frac: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load WSN-DS in one line
    
    Usage:
        X, y = load_wsnds(binary=True, sample_frac=0.1)
    
    Returns:
        X: Features (n_samples, 17)
        y: Labels (n_samples,)
    """
    
    loader = WSNDSLoader(data_path)
    loader.preprocess(binary=binary, sample_frac=sample_frac)
    X, y = loader.get_X_y()
    
    return X, y
```

**Contoh Penggunaan:**
```python
# Load 10% data, binary classification
X, y = load_wsnds(binary=True, sample_frac=0.1)

# Load full data, multi-class
X, y = load_wsnds(binary=False, sample_frac=None)

# Advanced usage
loader = WSNDSLoader("data/raw")
loader.load_data()
loader.preprocess(binary=True)
X, y = loader.get_X_y()
```

---

### 2. `data/data_loader_cyber.py` - Cyber Security Loader

**Fungsi**: Load dataset Cyber Security Attacks.

**Struktur sama dengan `data_loader_wsnds.py`**

**Perbedaan:**
- Dataset berbeda (Cyber Security Attacks)
- Attack types berbeda
- Support categorical features untuk CatBoost

---

### 3. `data/data_loader_nslkdd.py` - NSL-KDD Loader

**Fungsi**: Load NSL-KDD dataset (benchmark dataset klasik).

**Features:**
- 41 features (network traffic features)
- 5 classes: Normal, DoS, Probe, R2L, U2R
- Binary mapping: Normal vs Attack

---

### 4. `data/data_loader_cicids.py` - CICIDS2017 Loader

**Fungsi**: Load CICIDS2017 dataset (network intrusion dataset).

**Features:**
- Multiple CSV files (per hari)
- Network flow features
- Multiple attack types

---

### 5. `data/preprocessing.py` - Preprocessing Utilities

**Fungsi**: Utility functions untuk preprocessing data.

**Classes:**

#### A. DataPreprocessor
```python
class DataPreprocessor:
    """
    Comprehensive data preprocessing
    
    Features:
    - Categorical encoding (OneHot / Label)
    - Numerical scaling (Standard / MinMax)
    - Missing value handling
    - Automatic feature type detection
    """
    
    def __init__(self, scaling_method='standard', handle_categorical='onehot'):
        self.scaling_method = scaling_method
        self.handle_categorical = handle_categorical
        self.scaler = None
        self.encoder = None
    
    def fit(self, X: pd.DataFrame):
        """Fit preprocessor on training data"""
        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object']).columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        # Fit encoder
        if self.handle_categorical == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False)
            self.encoder.fit(X[self.categorical_columns])
        
        # Fit scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(X[self.numerical_columns])
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data"""
        # Encode categorical
        if self.categorical_columns:
            cat_encoded = self.encoder.transform(X[self.categorical_columns])
        
        # Scale numerical
        num_scaled = self.scaler.transform(X[self.numerical_columns])
        
        # Combine
        X_transformed = np.hstack([cat_encoded, num_scaled])
        
        return X_transformed
```

**Penggunaan:**
```python
preprocessor = DataPreprocessor(scaling_method='standard', handle_categorical='onehot')

# Fit on training
preprocessor.fit(X_train)

# Transform
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
```

#### B. LabelProcessor
```python
class LabelProcessor:
    """
    Process labels for classification
    
    Features:
    - Binary conversion (Normal vs Attack)
    - Label encoding
    - Custom mapping support
    """
    
    def __init__(self, binary=False, target_mapping=None):
        self.binary = binary
        self.target_mapping = target_mapping
        self.label_encoder = LabelEncoder()
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Encode labels"""
        if self.target_mapping:
            y = y.map(self.target_mapping)
        
        if self.binary:
            y = y.apply(lambda x: 'normal' if x.lower() == 'normal' else 'attack')
        
        return self.label_encoder.fit_transform(y)
```

#### C. ImbalancedDataHandler
```python
class ImbalancedDataHandler:
    """
    Handle imbalanced datasets
    
    Methods:
    - SMOTE: Synthetic Minority Over-sampling
    - ADASYN: Adaptive Synthetic Sampling
    - UnderSampling: Random under-sampling
    - SMOTE-Tomek: Combined over/under sampling
    """
    
    def __init__(self, method='smote', sampling_strategy='auto'):
        self.method = method
        if method == 'smote':
            self.sampler = SMOTE(sampling_strategy=sampling_strategy)
    
    def fit_resample(self, X, y):
        """Resample data"""
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return X_resampled, y_resampled
```

**Penggunaan:**
```python
handler = ImbalancedDataHandler(method='smote')
X_resampled, y_resampled = handler.fit_resample(X_train, y_train)
```

---

## 🤖 MODUL MODELS (`models/`)

### 1. `models/ml_models.py` - Machine Learning Models

**Fungsi**: Definisi dan factory untuk ML models.

**Models:**

#### 1. Naive Bayes
```python
class NaiveBayesModel(BaseMLModel):
    def __init__(self, random_state=42):
        self.model = GaussianNB()
        self.name = "Naive Bayes"
```

#### 2. Decision Tree
```python
class DecisionTreeModel(BaseMLModel):
    def __init__(self, max_depth=20, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
```

#### 3. Random Forest
```python
class RandomForestModel(BaseMLModel):
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
```

#### 4. KNN
```python
class KNNModel(BaseMLModel):
    def __init__(self, n_neighbors=5, random_state=42):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )
```

#### 5. SVM
```python
class SVMModel(BaseMLModel):
    def __init__(self, kernel='rbf', random_state=42):
        # Using SGDClassifier for speed
        self.model = SGDClassifier(
            loss='hinge',
            max_iter=1000,
            random_state=random_state
        )
```

#### 6. Extra Trees
```python
class ExtraTreesModel(BaseMLModel):
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
```

**Factory Function:**
```python
def get_sklearn_models(random_state=42):
    """Get all sklearn models in a dictionary"""
    return {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SGDClassifier(loss='hinge', max_iter=1000, random_state=random_state),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=20, random_state=random_state),
    }
```

**Penggunaan:**
```python
models = get_sklearn_models(random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: {score:.4f}")
```

---

### 2. `models/dl_models.py` - Deep Learning Models

**Fungsi**: Definisi arsitektur Deep Learning untuk tabular data.

**Models:**

#### 1. VGG16 Tabular
```python
def create_vgg16_tabular(input_dim, n_classes):
    """
    VGG16-style architecture for tabular data
    
    Architecture:
    - Input → Dense(512) × 2 → BatchNorm → Dropout
    - Dense(256) × 2 → BatchNorm → Dropout
    - Dense(128) × 2 → Dropout
    - Output(n_classes)
    
    Total: ~500K parameters
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='VGG16_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

#### 2. ResNet18 Tabular
```python
def create_resnet18_tabular(input_dim, n_classes):
    """
    ResNet18 with skip connections
    
    Architecture:
    - Input → Dense(256) → [Residual Block] × 3 → Output
    
    Residual Block:
        x → Dense → BatchNorm → Dense → BatchNorm → +shortcut → ReLU
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial block
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks
    for units in [256, 128, 64]:
        # Shortcut
        shortcut = layers.Dense(units)(x)
        
        # Main path
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units, activation='relu')(x)
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

#### 3. Inception Tabular
```python
def create_inception_tabular(input_dim, n_classes):
    """
    Inception-style with parallel branches
    
    Architecture:
    - Input → [Inception Module] × 2 → Output
    
    Inception Module:
        Branch 1: Dense(64)
        Branch 2: Dense(128) → Dense(64)
        Branch 3: Dense(256) → Dense(64)
        Concat → Output
    """
    # Implementation dengan parallel branches
    # untuk multi-scale feature extraction
```

**Factory Function:**
```python
def get_dl_models(input_dim, n_classes):
    """Get all DL models"""
    return {
        'VGG16': create_vgg16_tabular(input_dim, n_classes),
        'VGG19': create_vgg19_tabular(input_dim, n_classes),
        'ResNet18': create_resnet18_tabular(input_dim, n_classes),
        'ResNet50': create_resnet50_tabular(input_dim, n_classes),
        'Inception': create_inception_tabular(input_dim, n_classes),
    }
```

**Callbacks:**
```python
def get_callbacks():
    """Get training callbacks"""
    return [
        # Stop jika tidak ada improvement
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        
        # Reduce learning rate jika plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
```

**Penggunaan:**
```python
# Get models
models = get_dl_models(input_dim=17, n_classes=2)
callbacks = get_callbacks()

for name, model in models.items():
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    score = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: {score:.4f}")
```

---

## 🎯 MODUL FEATURES (`features/`)

### `features/feature_engineering.py` - Feature Engineering

**Fungsi**: Create dan select features.

**Classes:**

#### 1. FeatureEngineer
```python
class FeatureEngineer:
    """Create new features"""
    
    def create_interaction_features(self, X, max_features=10):
        """Create feature interactions (multiply, divide)"""
        interactions = []
        for i in range(max_features):
            for j in range(i+1, max_features):
                interactions.append(X[:, i] * X[:, j])
                interactions.append(X[:, i] / (X[:, j] + 1e-8))
        return np.column_stack([X] + interactions)
    
    def create_polynomial_features(self, X, degree=2):
        """Create polynomial features (x², x³)"""
        poly_features = []
        for i in range(X.shape[1]):
            for d in range(2, degree + 1):
                poly_features.append(X[:, i] ** d)
        return np.column_stack([X] + poly_features)
    
    def create_statistical_features(self, X):
        """Create row-wise statistics"""
        stats = [
            np.mean(X, axis=1, keepdims=True),
            np.std(X, axis=1, keepdims=True),
            np.min(X, axis=1, keepdims=True),
            np.max(X, axis=1, keepdims=True),
        ]
        return np.hstack([X] + stats)
```

#### 2. FeatureSelector
```python
class FeatureSelector:
    """Select most important features"""
    
    def __init__(self, method='mutual_info', n_features=40):
        self.method = method
        self.n_features = n_features
    
    def fit(self, X, y):
        """Fit selector"""
        if self.method == 'mutual_info':
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=self.n_features
            )
        elif self.method == 'rfe':
            base_model = RandomForestClassifier()
            self.selector = RFE(
                estimator=base_model,
                n_features_to_select=self.n_features
            )
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        """Select features"""
        return self.selector.transform(X)
```

**Penggunaan:**
```python
# Feature engineering
engineer = FeatureEngineer()
X_engineered = engineer.create_interaction_features(X)

# Feature selection
selector = FeatureSelector(method='mutual_info', n_features=40)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

---

## 📊 MODUL EVALUATION (`evaluation/`)

### 1. `evaluation/metrics.py` - Metrics Calculation

**Fungsi**: Calculate comprehensive metrics.

**Class: ComprehensiveMetrics**

```python
class ComprehensiveMetrics:
    """Calculate all evaluation metrics"""
    
    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """
        Calculate:
        - Accuracy
        - Precision (macro, weighted)
        - Recall (macro, weighted)
        - F1-Score (macro, weighted)
        - Cohen's Kappa
        - Matthews Correlation Coefficient
        - ROC-AUC
        - Confusion Matrix
        - Classification Report
        """
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }
        
        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        
        self.confusion_matrix_ = confusion_matrix(y_true, y_pred)
        self.classification_report_ = classification_report(y_true, y_pred, output_dict=True)
        
        return results
```

**Penggunaan:**
```python
metrics = ComprehensiveMetrics()
results = metrics.calculate_all_metrics(y_test, y_pred, y_proba)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_macro']:.4f}")
print(f"ROC-AUC: {results['roc_auc']:.4f}")
```

---

### 2. `evaluation/visualization.py` - Visualization

**Fungsi**: Visualisasi hasil training dan evaluasi.

**Functions:**

```python
def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""

def plot_roc_curve(y_true, y_proba):
    """Plot ROC curve"""

def plot_training_history(history):
    """Plot training/validation loss and accuracy"""

def plot_feature_importance(importances, feature_names):
    """Plot feature importance bar chart"""

def plot_model_comparison(results_df):
    """Plot model comparison bar chart"""
```

---

## 🛠️ MODUL UTILS (`utils/`)

### `utils/helpers.py` - Helper Functions

**Fungsi**: Utility functions.

**Functions:**

```python
def set_random_seeds(seed=42):
    """Set random seeds untuk reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def get_memory_usage():
    """Get current memory usage"""

def format_time(seconds):
    """Format seconds ke human readable"""

def save_results(results_df, filename):
    """Save results to CSV"""

def load_results(filename):
    """Load results from CSV"""

def print_system_info():
    """Print system information (CPU, RAM, etc)"""
```

---

## 📁 DIREKTORI OUTPUT

### 1. `logs/` - Training Logs

**Isi:**
```
logs/
├── new_datasets_20260113_120000.log
├── new_datasets_20260113_130000.log
└── ...
```

**Format Log:**
```
2026-01-13 12:00:00 - INFO - Starting training...
2026-01-13 12:00:05 - INFO - Loading WSN-DS dataset...
2026-01-13 12:00:10 - INFO - Training Naive Bayes...
2026-01-13 12:00:11 - INFO -    ✅ F1-Score: 0.985000
...
```

### 2. `results/` - Results CSV

**Isi:**
```
results/
├── new_datasets_results_20260113_120000.csv
├── comparison_results_20260107.csv
└── ...
```

**Format CSV:**
```csv
Dataset,Model,Accuracy,Precision,Recall,F1-Score,Training Time (s),ROC-AUC
WSN-DS,Naive Bayes,0.9850,0.9850,0.9850,0.9850,1.5,0.9900
WSN-DS,Decision Tree,0.9995,0.9995,0.9995,0.9995,8.3,0.9998
WSN-DS,Extra Trees,0.9998,0.9998,0.9998,0.9998,45.2,0.9999
...
```

### 3. `saved_models/` - Saved Models

**Isi:**
```
saved_models/
├── WSN-DS_Extra Trees_20260113.joblib
├── scaler_WSN-DS.joblib
├── Cyber-Security_CatBoost_20260113.joblib
├── scaler_Cyber-Security.joblib
└── ...
```

**Load Model:**
```python
import joblib

# Load ML model
model = joblib.load("saved_models/WSN-DS_Extra Trees_20260113.joblib")

# Load scaler
scaler = joblib.load("saved_models/scaler_WSN-DS.joblib")

# Inference
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

---

## ✅ KESIMPULAN

Setiap file memiliki peran spesifik:

1. **config.py**: Konfigurasi global
2. **main_new_datasets.py**: Orchestrator training ⭐
3. **data_loader_wsnds.py**: Load & preprocess WSN-DS ⭐
4. **preprocessing.py**: Utility preprocessing
5. **ml_models.py**: ML models (Extra Trees terbaik) ⭐
6. **dl_models.py**: DL models (VGG, ResNet, Inception)
7. **feature_engineering.py**: Feature creation & selection
8. **metrics.py**: Evaluation metrics
9. **visualization.py**: Plotting
10. **helpers.py**: Utility functions

**Flow Utama:**
```
main_new_datasets.py
  → data_loader_wsnds.py (load & preprocess)
  → preprocessing.py (scale)
  → ml_models.py (train ML)
  → dl_models.py (train DL)
  → metrics.py (evaluate)
  → Save results & models
```

**Untuk Fokus WSN-DS:**
- **Main script**: `main_new_datasets.py`
- **Data loader**: `data_loader_wsnds.py`
- **Best model**: Extra Trees (F1=0.9998)

---

**Last Updated**: 13 Januari 2026
