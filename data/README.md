# Data Directory

This directory contains datasets for cybersecurity threat detection.

## Structure

```
data/
├── raw/              # Original, unprocessed datasets
│   ├── cybersecurity_attacks.csv
│   └── wsn_dataset.csv
├── processed/        # Preprocessed datasets ready for training
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
└── README.md         # This file
```

## Datasets

### 1. Cybersecurity Attacks Dataset
- **Records**: ~40,000
- **Target**: attack_type
- **Description**: Contains various types of cyber attacks and network traffic patterns
- **Source**: Research paper dataset

### 2. WSN Dataset (Wireless Sensor Network)
- **Records**: ~374,661
- **Target**: label
- **Description**: Network intrusion detection dataset from wireless sensor networks
- **Source**: Research paper dataset

## Data Preprocessing

All datasets undergo the following preprocessing steps:
1. Remove redundant columns
2. Encode categorical variables
3. Handle missing values
4. Detect and handle outliers
5. Standardize features
6. Handle class imbalance
7. Split into train/test sets

## Usage

Place your raw datasets in the `raw/` directory. The preprocessing pipeline will:
- Load data from `raw/`
- Apply transformations
- Save processed data to `processed/`

Example:
```python
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor

loader = DataLoader()
df = loader.load_dataset('data/raw/cybersecurity_attacks.csv')

preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.run_pipeline(df, target_column='attack_type')
```

## Notes

- Raw data files are excluded from version control (see `.gitignore`)
- Processed data is also excluded from version control
- Download datasets from the research paper or use your own cybersecurity datasets
