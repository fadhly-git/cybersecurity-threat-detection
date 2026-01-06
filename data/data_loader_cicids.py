"""
CICIDS Dataset Loader
=====================
Loader khusus untuk dataset CICIDS2017
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class CICIDSLoader:
    """Loader untuk dataset CICIDS2017"""
    
    # Mapping label attacks
    ATTACK_MAPPING = {
        'BENIGN': 'Benign',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'Heartbleed': 'Heartbleed',
        'Web Attack – Brute Force': 'Web Attack',
        'Web Attack – XSS': 'Web Attack',
        'Web Attack – Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Bot': 'Botnet',
        'PortScan': 'PortScan',
        'DDoS': 'DDoS'
    }
    
    # Binary mapping
    BINARY_MAPPING = {
        'Benign': 0,
        'Brute Force': 1,
        'DoS': 1,
        'Heartbleed': 1,
        'Web Attack': 1,
        'Infiltration': 1,
        'Botnet': 1,
        'PortScan': 1,
        'DDoS': 1
    }
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_columns = None
        
    def load_all_files(self, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Load semua file CICIDS2017
        
        Args:
            sample_frac: Fraction untuk sampling (0.0-1.0), None untuk load semua
        """
        print("=" * 60)
        print("LOADING CICIDS2017 DATASET")
        print("=" * 60)
        
        # Find all CSV files
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for file in csv_files:
            print(f"  Loading: {file.name}...")
            try:
                df = pd.read_csv(file, encoding='utf-8', low_memory=False)
                
                # Sample if specified
                if sample_frac and sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42)
                
                dfs.append(df)
                print(f"    → {len(df):,} rows loaded")
            except Exception as e:
                print(f"    → Error: {e}")
        
        # Combine all dataframes
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows: {len(self.df):,}")
        
        return self.df
    
    def clean_column_names(self) -> None:
        """Bersihkan nama kolom"""
        # Strip whitespace from column names
        self.df.columns = self.df.columns.str.strip()
        
        # Standardize column names
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df.columns = self.df.columns.str.replace('/', '_')
        
        print(f"Cleaned {len(self.df.columns)} column names")
    
    def handle_missing_values(self) -> None:
        """Handle missing values dan infinity"""
        print("\nHandling missing values...")
        
        # Remove duplicate columns first
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        
        # Replace infinity with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count missing before
        missing_before = self.df.isnull().sum().sum()
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Fill NaN with median for numeric columns
        for col in numeric_cols:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                median_val = self.df[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback if entire column is NaN
                self.df[col] = self.df[col].fillna(median_val)
        
        # Instead of dropping rows with NaN, fill object columns with 'Unknown'
        object_cols = self.df.select_dtypes(include=['object']).columns
        for col in object_cols:
            self.df[col] = self.df[col].fillna('Unknown')
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  Missing values: {missing_before:,} → {missing_after:,}")
        print(f"  Rows remaining: {len(self.df):,}")
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows"""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        print(f"Removed {before - after:,} duplicate rows")
    
    def get_label_column(self) -> str:
        """Find label column name"""
        possible_names = ['Label', 'label', ' Label', 'LABEL']
        for name in possible_names:
            if name in self.df.columns:
                return name
        raise ValueError("Label column not found!")
    
    def create_labels(self, binary: bool = True) -> None:
        """
        Create label columns
        
        Args:
            binary: If True, create binary labels (0=benign, 1=attack)
        """
        label_col = self.get_label_column()
        
        # Clean label values
        self.df[label_col] = self.df[label_col].str.strip()
        
        # Show original distribution
        print("\nOriginal label distribution:")
        print(self.df[label_col].value_counts())
        
        # Map to attack categories
        self.df['attack_category'] = self.df[label_col].map(self.ATTACK_MAPPING)
        
        # Handle unmapped labels
        unmapped = self.df['attack_category'].isnull()
        if unmapped.any():
            print(f"\nUnmapped labels found:")
            print(self.df.loc[unmapped, label_col].unique())
            # Default unmapped to original value
            self.df.loc[unmapped, 'attack_category'] = self.df.loc[unmapped, label_col]
        
        if binary:
            # Create binary label
            self.df['label'] = self.df['attack_category'].map(self.BINARY_MAPPING)
            # Handle unmapped (assume attack)
            self.df['label'].fillna(1, inplace=True)
            self.df['label'] = self.df['label'].astype(int)
            
            print("\nBinary label distribution:")
            print(self.df['label'].value_counts())
            print(f"  Benign: {(self.df['label']==0).sum():,}")
            print(f"  Attack: {(self.df['label']==1).sum():,}")
        else:
            # Use attack category as label
            self.df['label'] = self.df['attack_category']
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns (exclude labels and non-features)"""
        # Exclude patterns (both original and cleaned versions)
        exclude_cols = [
            'Label', 'label', ' Label', '_Label',
            'attack_category',
            'Flow_ID', 'Flow ID', 'Source_IP', 'Source IP', 
            'Destination_IP', 'Destination IP', 'Src_IP', 'Dst_IP',
            'Timestamp', 'Source_Port', 'Source Port', 
            'Destination_Port', 'Destination Port', 'Src_Port', 'Dst_Port'
        ]
        
        # Get all numeric columns except excluded
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols 
                       and self.df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int', 'float']]
        
        if len(feature_cols) == 0:
            print(f"WARNING: No numeric feature columns found!")
            print(f"Available columns: {list(self.df.columns[:20])}...")
            print(f"Column dtypes: {self.df.dtypes.value_counts().to_dict()}")
        
        self.feature_columns = feature_cols
        return feature_cols
    
    def remove_constant_features(self) -> None:
        """Remove features with zero variance"""
        feature_cols = self.get_feature_columns()
        
        # Find constant columns
        constant_cols = []
        for col in feature_cols:
            if self.df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"\nRemoving {len(constant_cols)} constant features:")
            for col in constant_cols[:5]:  # Show first 5
                print(f"  - {col}")
            if len(constant_cols) > 5:
                print(f"  ... and {len(constant_cols)-5} more")
            
            self.df.drop(columns=constant_cols, inplace=True)
    
    def preprocess(self, binary: bool = True, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            binary: If True, create binary labels
            sample_frac: Fraction for sampling
        """
        # Load data
        self.load_all_files(sample_frac=sample_frac)
        
        # Clean column names
        self.clean_column_names()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Remove duplicates
        self.remove_duplicates()
        
        # Create labels
        self.create_labels(binary=binary)
        
        # Remove constant features
        self.remove_constant_features()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Number of features: {len(self.get_feature_columns())}")
        
        return self.df
    
    def get_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get features (X) and labels (y)"""
        if self.df is None:
            raise ValueError("Data not loaded. Call preprocess() first.")
        
        feature_cols = self.get_feature_columns()
        X = self.df[feature_cols]
        y = self.df['label']
        
        return X, y
    
    def save_processed(self, output_path: str = "data/processed/cicids_processed.csv") -> None:
        """Save processed dataset"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved processed data to: {output_path}")


# ============================================================
# QUICK USAGE FUNCTION
# ============================================================

def load_cicids(data_path: str = "data/raw", 
                binary: bool = True,
                sample_frac: Optional[float] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Quick function to load CICIDS dataset
    
    Args:
        data_path: Path to raw CSV files
        binary: Binary classification (True) or multiclass (False)
        sample_frac: Sample fraction for large datasets
        
    Returns:
        X: Feature dataframe
        y: Label series
    """
    loader = CICIDSLoader(data_path)
    loader.preprocess(binary=binary, sample_frac=sample_frac)
    return loader.get_X_y()


# ============================================================
# MAIN - Test loading
# ============================================================

if __name__ == "__main__":
    # Test the loader
    print("Testing CICIDS Loader...")
    
    try:
        # Load with sampling (10% of data for testing)
        X, y = load_cicids(
            data_path="data/raw",
            binary=True,
            sample_frac=0.1  # 10% sample for testing
        )
        
        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"\nFeature columns:")
        print(X.columns.tolist()[:10], "...")
        
        print(f"\nLabel distribution:")
        print(y.value_counts())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download CICIDS2017 dataset first!")
        print("1. Go to: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("2. Download and extract to: data/raw/")