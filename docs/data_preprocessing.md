# Data Preprocessing Guide

## Overview

This document provides comprehensive documentation for the **7-stage preprocessing pipeline** implemented in the cybersecurity threat detection system.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Implementation Details](#implementation-details)
4. [Code Examples](#code-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Theoretical Background

### Why Preprocessing Matters

Data preprocessing is critical in cybersecurity threat detection because:

- **Data Quality**: Raw cybersecurity data often contains noise, missing values, and outliers
- **Feature Scaling**: Different network features have vastly different scales (e.g., packet size vs. port number)
- **Class Imbalance**: Attacks are typically rare compared to normal traffic (often 1:100 ratio)
- **Computational Efficiency**: Proper preprocessing reduces training time significantly
- **Model Performance**: Clean, well-prepared data leads to better model accuracy

### Research Foundation

Based on the research paper "Evaluating Predictive Models in Cybersecurity" (arXiv:2407.06014), the preprocessing pipeline implements industry best practices with several enhancements.

---

## Pipeline Architecture

The preprocessing pipeline consists of **7 sequential stages**:

```
Raw Data
    ↓
[1] Remove Redundant Columns
    ↓
[2] Encode Categorical Features
    ↓
[3] Handle Missing Values
    ↓
[4] Detect Outliers
    ↓
[5] Handle Outliers
    ↓
[6] Standardize Features
    ↓
[7] Handle Class Imbalance
    ↓
[8] Split Data
    ↓
Clean Data (Ready for Training)
```

---

## Implementation Details

### Stage 1: Remove Redundant Columns

**Purpose**: Eliminate features that don't contribute to model performance.

**Methods**:
1. **Low Variance Removal**: Remove features with variance below threshold (default: 0.01)
2. **High Correlation Removal**: Remove one of each pair of features with correlation > threshold (default: 0.95)
3. **Constant Columns**: Remove columns with single unique value

**Algorithm**:
```python
def remove_redundant_columns(df, correlation_threshold=0.95, variance_threshold=0.01):
    # Step 1: Calculate variance
    variances = df.var()
    low_variance_cols = variances[variances < variance_threshold].index
    
    # Step 2: Calculate correlation matrix
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Step 3: Find highly correlated features
    high_corr_cols = [col for col in upper_triangle.columns 
                      if any(upper_triangle[col] > correlation_threshold)]
    
    # Step 4: Remove columns
    cols_to_drop = list(set(low_variance_cols) | set(high_corr_cols))
    return df.drop(columns=cols_to_drop)
```

**Parameters**:
- `correlation_threshold`: 0.90-0.99 (default: 0.95)
- `variance_threshold`: 0.001-0.1 (default: 0.01)

**Expected Impact**: Reduces feature dimensionality by 10-30%.

---

### Stage 2: Encode Categorical Features

**Purpose**: Convert categorical variables to numerical format for ML algorithms.

**Methods**:

1. **Label Encoding** (Ordinal):
   - Use for: Ordinal features (low, medium, high)
   - Algorithm: Maps categories to integers 0, 1, 2, ...
   - Preserves order relationship

2. **One-Hot Encoding** (Nominal):
   - Use for: Nominal features (protocol types: TCP, UDP, ICMP)
   - Algorithm: Creates binary column for each category
   - Recommended for features with < 10 unique values

3. **Target Encoding**:
   - Use for: High cardinality features (IP addresses, URLs)
   - Algorithm: Replaces category with mean of target variable
   - Prevents dimensionality explosion

**Implementation**:
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical(df, categorical_columns, method='onehot'):
    if method == 'onehot':
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(categorical_columns)
        )
        return pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    
    elif method == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df
```

**When to Use**:
- `onehot`: < 10 unique values, nominal
- `label`: Ordinal features, tree-based models
- `target`: > 20 unique values, high cardinality

---

### Stage 3: Handle Missing Values

**Purpose**: Impute or remove missing data points.

**Strategies**:

1. **Simple Imputation**:
   - **Mean**: For normally distributed numerical features
   - **Median**: For skewed numerical features (robust to outliers)
   - **Mode**: For categorical features

2. **Advanced Imputation**:
   - **K-Nearest Neighbors (KNN)**: Uses similarity to other samples
   - **Iterative Imputer**: Uses other features to predict missing values
   - **Forward/Backward Fill**: For time-series data

**Implementation**:
```python
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_values(df, strategy='mean'):
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
    
    return df_imputed
```

**Decision Tree**:
```
Missing % < 5%? → Use KNN imputer
Missing % < 20%? → Use mean/median
Missing % > 20%? → Consider dropping column
```

---

### Stage 4: Detect Outliers

**Purpose**: Identify anomalous data points that could skew model training.

**Methods**:

1. **Statistical Methods**:

   a) **Z-Score**:
   ```python
   z_scores = np.abs((df - df.mean()) / df.std())
   outliers = (z_scores > 3)  # 3 standard deviations
   ```
   - Threshold: 2.5-3.5 standard deviations
   - Assumption: Normal distribution

   b) **Interquartile Range (IQR)**:
   ```python
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   outliers = (df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)
   ```
   - Robust to non-normal distributions
   - Multiplier: 1.5 (typical) or 3.0 (extreme outliers)

2. **Machine Learning Methods**:

   a) **Isolation Forest**:
   ```python
   from sklearn.ensemble import IsolationForest
   
   iso_forest = IsolationForest(contamination=0.1, random_state=42)
   outlier_mask = iso_forest.fit_predict(df) == -1
   ```
   - Contamination: Expected outlier proportion (0.05-0.15)
   - Works well for high-dimensional data

   b) **Local Outlier Factor (LOF)**:
   - Detects local density deviations
   - Good for varying density datasets

**Comparison**:

| Method | Speed | Multivariate | Non-Normal | High-Dim |
|--------|-------|--------------|------------|----------|
| Z-Score | ⚡⚡⚡ | ❌ | ❌ | ✅ |
| IQR | ⚡⚡⚡ | ❌ | ✅ | ✅ |
| Isolation Forest | ⚡⚡ | ✅ | ✅ | ✅ |
| LOF | ⚡ | ✅ | ✅ | ❌ |

**Recommendation**: Use **Isolation Forest** for cybersecurity data (multivariate, mixed distributions).

---

### Stage 5: Handle Outliers

**Purpose**: Process detected outliers appropriately.

**Strategies**:

1. **Remove**:
   ```python
   df_clean = df[~outlier_mask]
   ```
   - When: Outliers are clearly errors
   - Risk: Loses data (especially problematic if outliers = attacks!)

2. **Cap/Winsorize**:
   ```python
   lower_bound = df.quantile(0.01)
   upper_bound = df.quantile(0.99)
   df_capped = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
   ```
   - When: Want to preserve sample count
   - Percentiles: 1st/99th or 5th/95th

3. **Transform**:
   ```python
   df_transformed = np.log1p(df)  # Log transformation
   ```
   - When: Distribution is highly skewed
   - Options: log, sqrt, box-cox

**For Cybersecurity**:
⚠️ **Important**: Many "outliers" in network traffic are actually attacks!
- Recommended: **Cap** outliers instead of removing
- Preserve rare attack patterns
- Use domain knowledge to distinguish errors from attacks

---

### Stage 6: Standardize Features

**Purpose**: Scale features to similar ranges for algorithm efficiency.

**Methods**:

1. **Standard Scaler (Z-Score Normalization)**:
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   # Result: mean=0, std=1
   ```
   - Use for: SVM, Neural Networks, K-Means
   - Assumes: Gaussian distribution
   - Formula: $X_{scaled} = \frac{X - \mu}{\sigma}$

2. **Min-Max Scaler**:
   ```python
   from sklearn.preprocessing import MinMaxScaler
   
   scaler = MinMaxScaler(feature_range=(0, 1))
   X_scaled = scaler.fit_transform(X)
   # Result: min=0, max=1
   ```
   - Use for: Neural Networks (bounded activations)
   - Sensitive to outliers
   - Formula: $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$

3. **Robust Scaler**:
   ```python
   from sklearn.preprocessing import RobustScaler
   
   scaler = RobustScaler()
   X_scaled = scaler.fit_transform(X)
   # Uses median and IQR (robust to outliers)
   ```
   - Use for: Data with outliers
   - Formula: $X_{scaled} = \frac{X - Q_{median}}{IQR}$

**Algorithm Selection**:
```
Tree-based models (RF, XGBoost)? → No scaling needed
Neural Networks? → StandardScaler or MinMaxScaler
Outliers present? → RobustScaler
Distance-based (SVM, KNN)? → StandardScaler
```

---

### Stage 7: Handle Class Imbalance

**Purpose**: Balance class distribution for better model training.

**Problem**: Cybersecurity datasets often have severe imbalance (e.g., 99% normal, 1% attacks).

**Techniques**:

1. **Oversampling Minority Class**:

   a) **SMOTE (Synthetic Minority Oversampling Technique)**:
   ```python
   from imblearn.over_sampling import SMOTE
   
   smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```
   - Creates synthetic samples along line segments between neighbors
   - `k_neighbors`: 3-7 (default: 5)
   - Best for: Imbalance ratio < 1:100

   b) **ADASYN (Adaptive Synthetic Sampling)**:
   ```python
   from imblearn.over_sampling import ADASYN
   
   adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5)
   X_resampled, y_resampled = adasyn.fit_resample(X, y)
   ```
   - Generates more samples in harder-to-learn regions
   - Adaptive density distribution

2. **Undersampling Majority Class**:
   ```python
   from imblearn.under_sampling import RandomUnderSampler
   
   rus = RandomUnderSampler(sampling_strategy='auto')
   X_resampled, y_resampled = rus.fit_resample(X, y)
   ```
   - Risk: Loses potentially important data
   - Use when: Majority class has redundant samples

3. **Combined Methods**:
   ```python
   from imblearn.combine import SMOTETomek
   
   smt = SMOTETomek(sampling_strategy='auto')
   X_resampled, y_resampled = smt.fit_resample(X, y)
   ```
   - SMOTE + Tomek links removal
   - Cleans borderline samples

**Comparison**:

| Method | Data Size Change | Overfitting Risk | Recommendation |
|--------|------------------|------------------|----------------|
| SMOTE | ↑ Increase | Medium | ✅ Best for most cases |
| ADASYN | ↑ Increase | Medium | Good for complex boundaries |
| Random Undersample | ↓ Decrease | Low | Use with large datasets |
| SMOTETomek | ↔ Varies | Low | ✅ Best overall quality |

**Recommendation**: Use **SMOTE** for cybersecurity (preserves attack patterns).

---

### Stage 8: Split Data

**Purpose**: Create training and test sets for model evaluation.

**Strategy**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80/20 split
    stratify=y,           # Maintain class proportions
    random_state=42       # Reproducibility
)
```

**Parameters**:
- `test_size`: 0.2-0.3 (20-30% for testing)
- `stratify`: **Always use** for classification to maintain class balance
- `random_state`: Set for reproducibility

**Advanced**: K-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and validate
```

---

## Code Examples

### Complete Pipeline Execution

```python
from src.data.preprocessing import DataPreprocessor

# Initialize
preprocessor = DataPreprocessor()

# Run complete pipeline
X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
    df=raw_data,
    target_column='attack_type',
    correlation_threshold=0.95,
    variance_threshold=0.01,
    encoding_method='onehot',
    missing_strategy='median',
    outlier_detection_method='isolation_forest',
    outlier_handling_method='cap',
    scaling_method='standard',
    balance_method='smote',
    test_size=0.2
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Step-by-Step Execution

```python
# Stage 1: Remove redundant features
df_stage1 = preprocessor.remove_redundant_columns(df)

# Stage 2: Encode categorical
df_stage2, encoders = preprocessor.encode_categorical(
    df_stage1, 
    categorical_columns=['protocol', 'service']
)

# Stage 3: Handle missing values
df_stage3 = preprocessor.handle_missing_values(df_stage2, strategy='median')

# Stage 4-5: Detect and handle outliers
outlier_mask = preprocessor.detect_outliers(df_stage3, method='isolation_forest')
df_stage5 = preprocessor.handle_outliers(df_stage3, outlier_mask, method='cap')

# Stage 6: Standardize
X = df_stage5.drop(columns=['attack_type'])
y = df_stage5['attack_type']
X_scaled, scaler = preprocessor.standardize_features(X, method='standard')

# Stage 7: Balance classes
X_balanced, y_balanced = preprocessor.handle_class_imbalance(
    X_scaled, y, method='smote'
)

# Stage 8: Split
X_train, X_test, y_train, y_test = preprocessor.split_data(
    X_balanced, y_balanced, test_size=0.2
)
```

---

## Best Practices

### 1. **Always Use Pipelines**
- Ensures reproducibility
- Prevents data leakage
- Easier to deploy

### 2. **Save Preprocessing Artifacts**
```python
import joblib

# Save scalers and encoders
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')

# Load for inference
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

### 3. **Monitor Class Distribution**
```python
print("Original class distribution:")
print(y.value_counts())

print("\nAfter balancing:")
print(pd.Series(y_balanced).value_counts())
```

### 4. **Validate Preprocessing**
```python
# Check for NaN
assert not X_train.isnull().any().any(), "NaN values present!"

# Check scaling
assert X_train.mean().abs().max() < 1.0, "Not properly scaled!"
assert X_train.std().min() > 0.5, "Variance too low!"
```

---

## Troubleshooting

### Issue: Memory Error During SMOTE
**Solution**: Use `SMOTE` with `k_neighbors=3` or `SMOTETomek`
```python
smote = SMOTE(k_neighbors=3, n_jobs=-1)
```

### Issue: Too Many Features After One-Hot Encoding
**Solution**: Use Target Encoding or dimensionality reduction
```python
# Option 1: Target encoding
encoder = TargetEncoder()

# Option 2: PCA after encoding
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
```

### Issue: Outlier Detection Removes Attack Samples
**Solution**: Use domain knowledge to label attacks before outlier detection
```python
# Separate attacks from outliers
attack_mask = df['label'] == 'attack'
outlier_mask = outlier_mask & ~attack_mask  # Don't mark attacks as outliers
```

### Issue: Different Scale for Test Data
**Solution**: Always fit on training data, transform on test data
```python
# ❌ Wrong
scaler.fit_transform(X_test)

# ✅ Correct
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## References

1. Research Paper: "Evaluating Predictive Models in Cybersecurity" (arXiv:2407.06014)
2. Scikit-learn Documentation: https://scikit-learn.org/stable/modules/preprocessing.html
3. Imbalanced-learn: https://imbalanced-learn.org/stable/
4. SMOTE Original Paper: Chawla et al., 2002

---

## Next Steps

After preprocessing:
1. Proceed to **Model Training** (see `docs/model_architecture.md`)
2. Experiment with different preprocessing parameters
3. Validate on hold-out test set
4. Deploy preprocessing pipeline with model

---

**Last Updated**: 2024  
**Maintainer**: Fadhly  
**Version**: 1.0
