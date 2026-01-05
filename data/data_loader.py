"""
data_loader.py - Data loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class DataLoader:
    """Load cybersecurity datasets"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Config.RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_nsl_kdd(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load NSL-KDD dataset
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = self.data_dir / "KDDTrain+.txt"
        test_path = self.data_dir / "KDDTest+.txt"
        
        # Download if not exists
        if not train_path.exists() or force_download:
            print("📥 Downloading NSL-KDD training data...")
            self._download_file(Config.NSL_KDD_TRAIN_URL, train_path)
            
        if not test_path.exists() or force_download:
            print("📥 Downloading NSL-KDD test data...")
            self._download_file(Config.NSL_KDD_TEST_URL, test_path)
        
        # Load data
        print("📂 Loading NSL-KDD dataset...")
        train_df = pd.read_csv(train_path, header=None, names=Config.NSL_KDD_COLUMNS)
        test_df = pd.read_csv(test_path, header=None, names=Config.NSL_KDD_COLUMNS)
        
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Test samples: {len(test_df):,}")
        
        return train_df, test_df
    
    def _download_file(self, url: str, filepath: Path):
        """Download file from URL"""
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"   ✅ Downloaded: {filepath.name}")
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            raise
    
    def load_cicids(self, filepath: str) -> pd.DataFrame:
        """Load CICIDS dataset (if you have it locally)"""
        print(f"📂 Loading CICIDS dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"   Samples: {len(df):,}")
        return df
    
    def get_attack_category(self, label: str) -> str:
        """Map attack label to category"""
        label_lower = label.lower().strip()
        return Config.ATTACK_CATEGORIES.get(label_lower, 'Unknown')
    
    def add_attack_categories(self, df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
        """Add attack category column to dataframe"""
        df = df.copy()
        df['attack_category'] = df[label_col].apply(self.get_attack_category)
        return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """Get dataset information"""
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 2,  # Exclude label and difficulty
            'n_classes': df['label'].nunique(),
            'class_distribution': df['label'].value_counts().to_dict(),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        return info


class DataSplitter:
    """Split data into train/val/test sets"""
    
    def __init__(self, 
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 stratified: bool = True):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratified = stratified
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/val/test sets
        
        Returns:
            Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify = y if self.stratified else None
        
        # First split: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Second split: train / val
        val_ratio = self.val_size / (1 - self.test_size)
        stratify_temp = y_temp if self.stratified else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        print(f"📊 Data Split:")
        print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def get_cv_folds(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5):
        """Get cross-validation fold indices"""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        return list(cv.split(X, y))