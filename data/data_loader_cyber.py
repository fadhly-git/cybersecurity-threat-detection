"""
Cyber Security Attacks Dataset Loader
=====================================
Loader untuk dataset Cyber Security Attacks
Dataset: Cyber Security Attacks.csv

Features:
- Timestamp, Source/Destination IP, Ports, Protocol
- Packet Length, Packet Type, Traffic Type
- Payload Data, Malware Indicators, Anomaly Scores
- Alerts/Warnings, Attack Type, Attack Signature
- Action Taken, Severity Level, etc.

Attack Types:
- DDoS
- Malware
- Intrusion
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class CyberSecurityLoader:
    """Loader untuk Cyber Security Attacks dataset - Optimized Version"""
    
    # Attack type mapping
    ATTACK_MAPPING = {
        'DDoS': 'DDoS',
        'Malware': 'Malware',
        'Intrusion': 'Intrusion',
        'Normal': 'Normal',
        'Benign': 'Normal'
    }
    
    # Binary mapping (Normal=0, Attack=1)
    BINARY_MAPPING = {
        'Normal': 0,
        'DDoS': 1,
        'Malware': 1,
        'Intrusion': 1
    }
    
    # Numeric feature columns
    NUMERIC_FEATURES = [
        'Source Port', 'Destination Port', 'Packet Length',
        'Anomaly Scores'
    ]
    
    # Categorical feature columns - Extended
    CATEGORICAL_FEATURES = [
        'Protocol', 'Packet Type', 'Traffic Type',
        'Action Taken', 'Severity Level', 'Network Segment',
        'Malware Indicators', 'Alerts/Warnings', 'Attack Signature',
        'Log Source', 'IDS/IPS Alerts'
    ]
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        self.feature_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """Load Cyber Security Attacks dataset"""
        print("=" * 60)
        print("LOADING CYBER SECURITY ATTACKS DATASET")
        print("=" * 60)
        
        filepath = self.data_path / "Cyber Security Attacks.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        print(f"  Loading: {filepath.name}")
        self.df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        print(f"    → {len(self.df):,} rows loaded")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        print(f"\nColumns: {list(self.df.columns)}")
        
        return self.df
    
    def preprocess(self, binary: bool = True, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Preprocess Cyber Security Attacks dataset
        
        Args:
            binary: If True, create binary labels (Normal vs Attack)
            sample_frac: Fraction untuk sampling (0.0-1.0), None untuk load semua
        """
        if self.df is None:
            self.load_data()
        
        print("\nPreprocessing (Optimized)...")
        
        # Sample if specified
        if sample_frac and sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42)
            print(f"  Sampled to {len(self.df):,} rows ({sample_frac*100:.0f}%)")
        
        # Clean attack type column
        self.df['Attack Type'] = self.df['Attack Type'].astype(str).str.strip()
        
        # Show original distribution
        print("\nOriginal label distribution:")
        print(self.df['Attack Type'].value_counts())
        
        # Map attack types
        self.df['attack_category'] = self.df['Attack Type'].map(self.ATTACK_MAPPING)
        self.df['attack_category'].fillna('Intrusion', inplace=True)  # Default to attack
        
        if binary:
            # Binary classification (Normal=0, Attack=1)
            self.df['target'] = self.df['attack_category'].map(self.BINARY_MAPPING)
            
            # If all are attacks (no normal), adjust
            if self.df['target'].isna().any():
                self.df['target'].fillna(1, inplace=True)
            
            # Check if there's class imbalance
            normal_count = (self.df['target'] == 0).sum()
            attack_count = (self.df['target'] == 1).sum()
            
            print("\nBinary label distribution:")
            print(f"  Normal (0): {normal_count:,}")
            print(f"  Attack (1): {attack_count:,}")
            
            # If no normal traffic, use multi-class instead
            if normal_count == 0:
                print("\n⚠️  No normal traffic found - using multi-class for attack types")
                binary = False
        
        if not binary:
            # Multi-class classification
            self.label_encoder.fit(self.df['attack_category'])
            self.df['target'] = self.label_encoder.transform(self.df['attack_category'])
            
            print("\nMulti-class label distribution:")
            print(self.df['attack_category'].value_counts())
        
        # Process all features
        self._process_numeric_features()
        self._extract_timestamp_features()
        self._extract_ip_features()
        self._encode_categorical_features()
        self._extract_text_features()
        
        return self.df

    def get_catboost_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
        """Return DataFrame features and categorical feature indices for CatBoost.
        Assumes preprocess() has been called to create derived numeric features and target.
        """
        if self.df is None or 'target' not in self.df.columns:
            raise ValueError("Must call preprocess() before get_catboost_data()")

        feature_cols: List[str] = []

        # Numeric base features
        for col in self.NUMERIC_FEATURES:
            if col in self.df.columns:
                feature_cols.append(col)

        # Derived numeric features
        for col in [
            'port_diff', 'port_sum',
            'is_well_known_src', 'is_well_known_dst',
            'is_high_port_src', 'is_high_port_dst',
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours', 'is_night',
            'src_ip_oct1', 'src_ip_oct2', 'src_ip_oct3', 'src_ip_oct4', 'src_ip_class', 'src_ip_is_private',
            'dst_ip_oct1', 'dst_ip_oct2', 'dst_ip_oct3', 'dst_ip_oct4', 'dst_ip_class', 'dst_ip_is_private',
            'same_subnet', 'payload_length', 'payload_word_count', 'is_chrome', 'is_firefox', 'is_msie', 'is_mobile',
            'firewall_log_length'
        ]:
            if col in self.df.columns:
                feature_cols.append(col)

        # Raw categorical features (use native CatBoost handling)
        raw_cat_cols: List[str] = []
        for col in self.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                raw_cat_cols.append(col)
                feature_cols.append(col)

        X_df = self.df[feature_cols].copy()

        # Ensure categorical columns are of type string/object for CatBoost
        for col in raw_cat_cols:
            X_df[col] = X_df[col].astype(str).fillna('Unknown')

        # CatBoost categorical feature indices
        cat_features_idx = [X_df.columns.get_loc(col) for col in raw_cat_cols]

        y = self.df['target'].astype(np.int32).values

        return X_df, y, cat_features_idx
    
    def _process_numeric_features(self):
        """Process numeric features"""
        print("\nProcessing numeric features...")
        
        for col in self.NUMERIC_FEATURES:
            if col in self.df.columns:
                # Convert to numeric, coerce errors to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Fill NaN with median
                median_val = self.df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.df[col].fillna(median_val, inplace=True)
                
                # Handle infinity
                self.df[col].replace([np.inf, -np.inf], median_val, inplace=True)
        
        # Add derived features
        if 'Source Port' in self.df.columns and 'Destination Port' in self.df.columns:
            self.df['port_diff'] = abs(self.df['Source Port'] - self.df['Destination Port'])
            self.df['port_sum'] = self.df['Source Port'] + self.df['Destination Port']
            self.df['is_well_known_src'] = (self.df['Source Port'] < 1024).astype(int)
            self.df['is_well_known_dst'] = (self.df['Destination Port'] < 1024).astype(int)
            self.df['is_high_port_src'] = (self.df['Source Port'] > 49152).astype(int)
            self.df['is_high_port_dst'] = (self.df['Destination Port'] > 49152).astype(int)
    
    def _extract_timestamp_features(self):
        """Extract features from timestamp"""
        print("Extracting timestamp features...")
        
        if 'Timestamp' in self.df.columns:
            try:
                self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')
                self.df['hour'] = self.df['Timestamp'].dt.hour.fillna(0).astype(int)
                self.df['day_of_week'] = self.df['Timestamp'].dt.dayofweek.fillna(0).astype(int)
                self.df['month'] = self.df['Timestamp'].dt.month.fillna(0).astype(int)
                self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
                self.df['is_business_hours'] = ((self.df['hour'] >= 9) & (self.df['hour'] <= 17)).astype(int)
                self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 6)).astype(int)
            except Exception as e:
                print(f"  Warning: Could not extract timestamp features: {e}")
    
    def _extract_ip_features(self):
        """Extract numeric features from IP addresses"""
        print("Extracting IP features...")
        
        for col in ['Source IP Address', 'Destination IP Address']:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].fillna('0.0.0.0').astype(str)
                    ip_split = self.df[col].str.split('.', expand=True)
                    prefix = 'src_ip' if 'Source' in col else 'dst_ip'
                    for i in range(4):
                        self.df[f'{prefix}_oct{i+1}'] = pd.to_numeric(ip_split[i], errors='coerce').fillna(0).astype(int)
                    
                    # IP class (A, B, C based on first octet)
                    self.df[f'{prefix}_class'] = pd.cut(
                        self.df[f'{prefix}_oct1'],
                        bins=[0, 127, 191, 223, 255],
                        labels=[0, 1, 2, 3]
                    ).fillna(0).astype(int)
                    
                    # Is private IP
                    oct1 = self.df[f'{prefix}_oct1']
                    oct2 = self.df[f'{prefix}_oct2']
                    self.df[f'{prefix}_is_private'] = (
                        (oct1 == 10) | 
                        ((oct1 == 172) & (oct2 >= 16) & (oct2 <= 31)) |
                        ((oct1 == 192) & (oct2 == 168))
                    ).astype(int)
                except Exception as e:
                    print(f"  Warning: Could not extract IP features from {col}: {e}")
        
        # Same subnet check
        if 'src_ip_oct1' in self.df.columns and 'dst_ip_oct1' in self.df.columns:
            self.df['same_subnet'] = (
                (self.df['src_ip_oct1'] == self.df['dst_ip_oct1']) &
                (self.df['src_ip_oct2'] == self.df['dst_ip_oct2']) &
                (self.df['src_ip_oct3'] == self.df['dst_ip_oct3'])
            ).astype(int)
    
    def _encode_categorical_features(self):
        """Encode categorical features using LabelEncoder"""
        print("Encoding categorical features...")
        
        for col in self.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                # Fill missing values
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Encode
                encoder = LabelEncoder()
                self.df[f'{col}_encoded'] = encoder.fit_transform(self.df[col])
                self.categorical_encoders[col] = encoder
    
    def _extract_text_features(self):
        """Extract features from text columns"""
        print("Extracting text features...")
        
        # Payload Data length
        if 'Payload Data' in self.df.columns:
            self.df['Payload Data'] = self.df['Payload Data'].fillna('').astype(str)
            self.df['payload_length'] = self.df['Payload Data'].str.len()
            self.df['payload_word_count'] = self.df['Payload Data'].str.split().str.len().fillna(0)
        
        # Device Information
        if 'Device Information' in self.df.columns:
            self.df['Device Information'] = self.df['Device Information'].fillna('').astype(str)
            # Browser type indicators
            self.df['is_chrome'] = self.df['Device Information'].str.contains('Chrome', case=False, na=False).astype(int)
            self.df['is_firefox'] = self.df['Device Information'].str.contains('Firefox', case=False, na=False).astype(int)
            self.df['is_msie'] = self.df['Device Information'].str.contains('MSIE|Trident', case=False, na=False).astype(int)
            self.df['is_mobile'] = self.df['Device Information'].str.contains('Mobile|Android|iPhone', case=False, na=False).astype(int)
        
        # Firewall Logs
        if 'Firewall Logs' in self.df.columns:
            self.df['Firewall Logs'] = self.df['Firewall Logs'].fillna('').astype(str)
            self.df['firewall_log_length'] = self.df['Firewall Logs'].str.len()
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        if self.df is None:
            self.load_data()
        
        feature_cols = []
        
        # Add numeric features
        for col in self.NUMERIC_FEATURES:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # Add derived port features
        derived_features = [
            'port_diff', 'port_sum', 
            'is_well_known_src', 'is_well_known_dst',
            'is_high_port_src', 'is_high_port_dst'
        ]
        for col in derived_features:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # Add timestamp features
        timestamp_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours', 'is_night']
        for col in timestamp_features:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # Add IP features
        for prefix in ['src_ip', 'dst_ip']:
            for i in range(1, 5):
                col = f'{prefix}_oct{i}'
                if col in self.df.columns:
                    feature_cols.append(col)
            for suffix in ['class', 'is_private']:
                col = f'{prefix}_{suffix}'
                if col in self.df.columns:
                    feature_cols.append(col)
        
        if 'same_subnet' in self.df.columns:
            feature_cols.append('same_subnet')
        
        # Add encoded categorical features
        for col in self.CATEGORICAL_FEATURES:
            encoded_col = f'{col}_encoded'
            if encoded_col in self.df.columns:
                feature_cols.append(encoded_col)
        
        # Add text features
        text_features = [
            'payload_length', 'payload_word_count',
            'is_chrome', 'is_firefox', 'is_msie', 'is_mobile',
            'firewall_log_length'
        ]
        for col in text_features:
            if col in self.df.columns:
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        print(f"\nTotal features: {len(feature_cols)}")
        return feature_cols
    
    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get features (X) and labels (y)"""
        if self.df is None or 'target' not in self.df.columns:
            raise ValueError("Must call preprocess() before get_X_y()")
        
        feature_cols = self.get_feature_columns()
        
        print(f"Using {len(feature_cols)} features")
        
        X = self.df[feature_cols].values.astype(np.float32)
        y = self.df['target'].values.astype(np.int32)
        
        # Replace any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
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


def load_cyber_security(data_path: str = "data/raw", binary: bool = True,
                       sample_frac: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to load and preprocess Cyber Security Attacks dataset
    
    Args:
        data_path: Path to data directory
        binary: If True, binary classification
        sample_frac: Fraction for sampling
    
    Returns:
        X, y: Features and labels
    """
    loader = CyberSecurityLoader(data_path)
    loader.preprocess(binary=binary, sample_frac=sample_frac)
    return loader.get_X_y()


if __name__ == "__main__":
    # Test loader
    print("Testing Cyber Security Attacks Loader...")
    
    loader = CyberSecurityLoader("data/raw")
    loader.preprocess(binary=True)
    X, y = loader.get_X_y()
    
    print(f"\nFinal shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Classes: {np.unique(y, return_counts=True)}")
