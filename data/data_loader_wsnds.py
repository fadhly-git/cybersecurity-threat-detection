"""
WSN-DS Dataset Loader
=====================
Loader untuk dataset Wireless Sensor Network Dataset for Intrusion Detection
Dataset: WSN-DS.csv

Attack Types:
- Normal
- Flooding
- Blackhole
- Grayhole
- TDMA
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class WSNDSLoader:
    """Loader untuk WSN-DS (Wireless Sensor Network) dataset"""
    
    # Mapping attack types
    ATTACK_MAPPING = {
        'Normal': 'Normal',
        'Flooding': 'Flooding',
        'Blackhole': 'Blackhole',
        'Grayhole': 'Grayhole',
        'TDMA': 'TDMA',
        'Scheduling': 'TDMA'  # alias
    }
    
    # Binary mapping (Normal vs Attack)
    BINARY_MAPPING = {
        'Normal': 0,
        'Flooding': 1,
        'Blackhole': 1,
        'Grayhole': 1,
        'TDMA': 1
    }
    
    # Feature columns (excluding id and attack type)
    FEATURE_COLUMNS = [
        'Time', 'Is_CH', 'who_CH', 'Dist_To_CH', 'ADV_S', 'ADV_R',
        'JOIN_S', 'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 'DATA_S',
        'DATA_R', 'Data_Sent_To_BS', 'dist_CH_To_BS', 'send_code', 'Expaned_Energy'
    ]
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """Load WSN-DS dataset"""
        print("=" * 60)
        print("LOADING WSN-DS DATASET")
        print("=" * 60)
        
        filepath = self.data_path / "WSN-DS.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        print(f"  Loading: {filepath.name}")
        self.df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        print(f"    → {len(self.df):,} rows loaded")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        self.df.columns = self.df.columns.str.replace(' ', '_')
        
        # Rename specific columns for consistency
        column_mapping = {
            'who_CH': 'who_CH',
            'Attack_type': 'Attack_type',
            'Expaned_Energy': 'Expaned_Energy'
        }
        
        # Handle variations in column names
        if 'who CH' in self.df.columns:
            self.df.rename(columns={'who CH': 'who_CH'}, inplace=True)
        if 'Attack type' in self.df.columns:
            self.df.rename(columns={'Attack type': 'Attack_type'}, inplace=True)
        if 'Expaned Energy' in self.df.columns:
            self.df.rename(columns={'Expaned Energy': 'Expaned_Energy'}, inplace=True)
        
        print(f"\nColumns: {list(self.df.columns)}")
        
        return self.df
    
    def preprocess(self, binary: bool = True, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Preprocess WSN-DS dataset
        
        Args:
            binary: If True, create binary labels (Normal vs Attack)
            sample_frac: Fraction untuk sampling (0.0-1.0), None untuk load semua
        """
        if self.df is None:
            self.load_data()
        
        print("\nPreprocessing...")
        
        # Sample if specified
        if sample_frac and sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42)
            print(f"  Sampled to {len(self.df):,} rows ({sample_frac*100:.0f}%)")
        
        # Clean attack type column
        self.df['Attack_type'] = self.df['Attack_type'].astype(str).str.strip()
        
        # Show original distribution
        print("\nOriginal label distribution:")
        print(self.df['Attack_type'].value_counts())
        
        # Map attack types
        self.df['attack_category'] = self.df['Attack_type'].map(self.ATTACK_MAPPING)
        self.df['attack_category'].fillna('Normal', inplace=True)  # Handle unmapped
        
        if binary:
            # Binary classification (Normal=0, Attack=1)
            self.df['target'] = self.df['attack_category'].map(self.BINARY_MAPPING)
            
            print("\nBinary label distribution:")
            print(f"  Normal (0): {(self.df['target']==0).sum():,}")
            print(f"  Attack (1): {(self.df['target']==1).sum():,}")
        else:
            # Multi-class classification
            self.label_encoder.fit(self.df['attack_category'])
            self.df['target'] = self.label_encoder.transform(self.df['attack_category'])
            
            print("\nMulti-class label distribution:")
            print(self.df['attack_category'].value_counts())
        
        # Handle missing values and infinity
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.df[col].fillna(median_val, inplace=True)
        
        return self.df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns available in dataset"""
        if self.df is None:
            self.load_data()
        
        # Columns to exclude
        exclude_cols = ['id', 'Attack_type', 'attack_category', 'target']
        
        # Get available feature columns
        available_features = [col for col in self.df.columns 
                           if col not in exclude_cols]
        
        self.feature_columns = available_features
        return available_features
    
    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get features (X) and labels (y)"""
        if self.df is None or 'target' not in self.df.columns:
            raise ValueError("Must call preprocess() before get_X_y()")
        
        feature_cols = self.get_feature_columns()
        
        # Select only numeric features
        numeric_features = []
        for col in feature_cols:
            if self.df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                numeric_features.append(col)
        
        print(f"\nUsing {len(numeric_features)} numeric features")
        
        X = self.df[numeric_features].values.astype(np.float32)
        y = self.df['target'].values.astype(np.int32)
        
        return X, y
    
    def get_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Get train/test split"""
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_X_y()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test


def load_wsnds(data_path: str = "data/raw", binary: bool = True, 
               sample_frac: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to load and preprocess WSN-DS dataset
    
    Args:
        data_path: Path to data directory
        binary: If True, binary classification
        sample_frac: Fraction for sampling
    
    Returns:
        X, y: Features and labels
    """
    loader = WSNDSLoader(data_path)
    loader.preprocess(binary=binary, sample_frac=sample_frac)
    return loader.get_X_y()


if __name__ == "__main__":
    # Test loader
    print("Testing WSN-DS Loader...")
    
    loader = WSNDSLoader("data/raw")
    loader.preprocess(binary=True)
    X, y = loader.get_X_y()
    
    print(f"\nFinal shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
