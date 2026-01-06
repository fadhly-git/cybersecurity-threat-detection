"""
NSL-KDD Dataset Loader
======================
Loader for NSL-KDD dataset (Second dataset from paper)
Based on paper: arxiv.org/abs/2407.06014
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class NSLKDDLoader:
    """Loader for NSL-KDD dataset"""
    
    # Column names for NSL-KDD
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Attack type mapping (5-class)
    ATTACK_MAPPING = {
        'normal': 'Normal',
        # DoS
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
        'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
        'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
        # Probe
        'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'mscan': 'Probe', 'saint': 'Probe',
        # R2L (Remote to Local)
        'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
        'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
        'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
        'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
        # U2R (User to Root)
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
        'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
    }
    
    # Binary mapping
    BINARY_MAPPING = {
        'Normal': 0,
        'DoS': 1, 'Probe': 1, 'R2L': 1, 'U2R': 1
    }
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.train_df = None
        self.test_df = None
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NSL-KDD train and test data"""
        print("=" * 60)
        print("LOADING NSL-KDD DATASET")
        print("=" * 60)
        
        train_path = self.data_path / "KDDTrain+.txt"
        test_path = self.data_path / "KDDTest+.txt"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Load data
        print(f"  Loading training data: {train_path.name}")
        self.train_df = pd.read_csv(train_path, header=None, names=self.COLUMN_NAMES)
        print(f"    → {len(self.train_df):,} rows")
        
        print(f"  Loading test data: {test_path.name}")
        self.test_df = pd.read_csv(test_path, header=None, names=self.COLUMN_NAMES)
        print(f"    → {len(self.test_df):,} rows")
        
        return self.train_df, self.test_df
    
    def preprocess(self, binary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess NSL-KDD dataset
        
        Args:
            binary: If True, create binary labels (Normal vs Attack)
        """
        if self.train_df is None:
            self.load_data()
        
        print("\nPreprocessing...")
        
        # Clean labels
        self.train_df['label'] = self.train_df['label'].str.strip().str.lower()
        self.test_df['label'] = self.test_df['label'].str.strip().str.lower()
        
        # Show original distribution
        print("\nOriginal training label distribution:")
        print(self.train_df['label'].value_counts().head(10))
        
        # Map to attack categories
        self.train_df['attack_category'] = self.train_df['label'].map(self.ATTACK_MAPPING)
        self.test_df['attack_category'] = self.test_df['label'].map(self.ATTACK_MAPPING)
        
        # Handle unmapped labels (default to attack)
        self.train_df['attack_category'].fillna('DoS', inplace=True)
        self.test_df['attack_category'].fillna('DoS', inplace=True)
        
        if binary:
            # Binary classification
            self.train_df['target'] = self.train_df['attack_category'].map(self.BINARY_MAPPING)
            self.test_df['target'] = self.test_df['attack_category'].map(self.BINARY_MAPPING)
            
            print("\nBinary label distribution (Training):")
            print(f"  Normal (0): {(self.train_df['target']==0).sum():,}")
            print(f"  Attack (1): {(self.train_df['target']==1).sum():,}")
        else:
            # Multi-class classification
            all_categories = pd.concat([
                self.train_df['attack_category'], 
                self.test_df['attack_category']
            ])
            self.label_encoder.fit(all_categories)
            
            self.train_df['target'] = self.label_encoder.transform(self.train_df['attack_category'])
            self.test_df['target'] = self.label_encoder.transform(self.test_df['attack_category'])
            
            print("\nMulti-class label distribution (Training):")
            print(self.train_df['attack_category'].value_counts())
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        for col in categorical_cols:
            # Combine train and test for fitting encoder
            all_values = pd.concat([self.train_df[col], self.test_df[col]]).unique()
            encoder = LabelEncoder()
            encoder.fit(all_values)
            
            self.categorical_encoders[col] = encoder
            self.train_df[col] = encoder.transform(self.train_df[col])
            self.test_df[col] = encoder.transform(self.test_df[col])
        
        print(f"\nEncoded {len(categorical_cols)} categorical features")
        
        # Remove non-feature columns
        drop_cols = ['label', 'difficulty', 'attack_category']
        
        return self.train_df, self.test_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        exclude_cols = ['label', 'difficulty', 'attack_category', 'target']
        return [col for col in self.train_df.columns if col not in exclude_cols]
    
    def get_X_y_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training features and labels"""
        feature_cols = self.get_feature_columns()
        X = self.train_df[feature_cols].values.astype(np.float32)
        y = self.train_df['target'].values.astype(np.int32)
        return X, y
    
    def get_X_y_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test features and labels"""
        feature_cols = self.get_feature_columns()
        X = self.test_df[feature_cols].values.astype(np.float32)
        y = self.test_df['target'].values.astype(np.int32)
        return X, y


def load_nslkdd(data_path: str = "data/raw", 
                binary: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick function to load NSL-KDD dataset
    
    Args:
        data_path: Path to data directory containing KDDTrain+.txt and KDDTest+.txt
        binary: Binary classification (True) or multiclass (False)
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    loader = NSLKDDLoader(data_path)
    loader.preprocess(binary=binary)
    
    X_train, y_train = loader.get_X_y_train()
    X_test, y_test = loader.get_X_y_test()
    
    print(f"\nNSL-KDD Dataset loaded:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Features: {loader.get_feature_columns()[:5]}...")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test the loader
    print("Testing NSL-KDD Loader...")
    
    try:
        X_train, y_train, X_test, y_test = load_nslkdd(
            data_path="data/raw",
            binary=True
        )
        
        print(f"\nX_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        print(f"\nLabel distribution (train):")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c:,}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure KDDTrain+.txt and KDDTest+.txt are in data/raw/")
