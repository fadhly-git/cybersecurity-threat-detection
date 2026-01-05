"""
preprocessing.py - Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class DataPreprocessor:
    """Preprocess cybersecurity data"""
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 handle_categorical: str = 'onehot'):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: 'standard', 'minmax', or None
            handle_categorical: 'onehot' or 'label'
        """
        self.scaling_method = scaling_method
        self.handle_categorical = handle_categorical
        self.scaler = None
        self.encoder = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_names = []
        self._fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessor on data"""
        X = X.copy()
        
        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"📊 Preprocessing:")
        print(f"   Categorical features: {len(self.categorical_columns)}")
        print(f"   Numerical features: {len(self.numerical_columns)}")
        
        # Fit encoder for categorical columns
        if self.categorical_columns:
            if self.handle_categorical == 'onehot':
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.encoder.fit(X[self.categorical_columns])
            else:
                self.encoders = {}
                for col in self.categorical_columns:
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    self.encoders[col] = le
        
        # Fit scaler for numerical columns
        if self.numerical_columns and self.scaling_method:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            
            self.scaler.fit(X[self.numerical_columns])
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data"""
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X = X.copy()
        transformed_parts = []
        
        # Transform categorical columns
        if self.categorical_columns:
            if self.handle_categorical == 'onehot':
                cat_encoded = self.encoder.transform(X[self.categorical_columns])
                transformed_parts.append(cat_encoded)
            else:
                cat_encoded = np.zeros((len(X), len(self.categorical_columns)))
                for i, col in enumerate(self.categorical_columns):
                    # Handle unknown categories
                    X[col] = X[col].astype(str)
                    known_mask = X[col].isin(self.encoders[col].classes_)
                    cat_encoded[known_mask, i] = self.encoders[col].transform(X.loc[known_mask, col])
                    cat_encoded[~known_mask, i] = -1  # Unknown category
                transformed_parts.append(cat_encoded)
        
        # Transform numerical columns
        if self.numerical_columns:
            if self.scaler:
                num_scaled = self.scaler.transform(X[self.numerical_columns])
            else:
                num_scaled = X[self.numerical_columns].values
            transformed_parts.append(num_scaled)
        
        # Combine
        X_transformed = np.hstack(transformed_parts)
        
        # Handle missing values
        X_transformed = np.nan_to_num(X_transformed, nan=0, posinf=0, neginf=0)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get transformed feature names"""
        names = []
        
        if self.categorical_columns:
            if self.handle_categorical == 'onehot':
                names.extend(self.encoder.get_feature_names_out(self.categorical_columns))
            else:
                names.extend(self.categorical_columns)
        
        if self.numerical_columns:
            names.extend(self.numerical_columns)
        
        return names


class LabelProcessor:
    """Process labels for classification"""
    
    def __init__(self, binary: bool = False, target_mapping: Optional[dict] = None):
        """
        Initialize label processor
        
        Args:
            binary: If True, convert to binary (normal vs attack)
            target_mapping: Custom mapping for labels
        """
        self.binary = binary
        self.target_mapping = target_mapping
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        self._fitted = False
        
    def fit(self, y: pd.Series) -> 'LabelProcessor':
        """Fit label processor"""
        y = y.copy()
        
        # Apply custom mapping if provided
        if self.target_mapping:
            y = y.map(self.target_mapping)
        
        # Convert to binary if requested
        if self.binary:
            y = y.apply(lambda x: 'normal' if x.lower() == 'normal' else 'attack')
        
        # Fit encoder
        self.label_encoder.fit(y.astype(str))
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        
        print(f"🏷️  Label Processing:")
        print(f"   Binary mode: {self.binary}")
        print(f"   Number of classes: {self.n_classes_}")
        print(f"   Classes: {list(self.classes_)}")
        
        self._fitted = True
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """Transform labels"""
        if not self._fitted:
            raise ValueError("LabelProcessor not fitted. Call fit() first.")
        
        y = y.copy()
        
        if self.target_mapping:
            y = y.map(self.target_mapping)
        
        if self.binary:
            y = y.apply(lambda x: 'normal' if str(x).lower() == 'normal' else 'attack')
        
        return self.label_encoder.transform(y.astype(str))
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fit and transform labels"""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original"""
        return self.label_encoder.inverse_transform(y)
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return list(self.classes_)


class ImbalancedDataHandler:
    """Handle imbalanced datasets"""
    
    def __init__(self, 
                 method: str = 'smote',
                 sampling_strategy: Union[str, dict] = 'auto',
                 random_state: int = 42):
        """
        Initialize handler
        
        Args:
            method: 'smote', 'adasyn', 'undersample', 'smote_tomek'
            sampling_strategy: Sampling strategy
            random_state: Random seed
        """
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.sampler = None
        
        self._init_sampler()
        
    def _init_sampler(self):
        """Initialize sampler based on method"""
        if self.method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=Config.SMOTE_K_NEIGHBORS
            )
        elif self.method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'undersample':
            self.sampler = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'smote_tomek':
            self.sampler = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample data"""
        print(f"⚖️  Handling Imbalanced Data ({self.method}):")
        print(f"   Before: {len(X):,} samples")
        
        # Get class distribution before
        unique, counts = np.unique(y, return_counts=True)
        print(f"   Class distribution: {dict(zip(unique, counts))}")
        
        # Resample
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        print(f"   After: {len(X_resampled):,} samples")
        
        # Get class distribution after
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"   New distribution: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled


class OutlierHandler:
    """Handle outliers in data"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """
        Initialize outlier handler
        
        Args:
            method: 'iqr', 'zscore', or 'isolation_forest'
            threshold: Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X: np.ndarray) -> 'OutlierHandler':
        """Fit outlier detector"""
        if self.method == 'iqr':
            for i in range(X.shape[1]):
                Q1 = np.percentile(X[:, i], 25)
                Q3 = np.percentile(X[:, i], 75)
                IQR = Q3 - Q1
                self.bounds_[i] = (Q1 - self.threshold * IQR, Q3 + self.threshold * IQR)
        elif self.method == 'zscore':
            for i in range(X.shape[1]):
                mean = np.mean(X[:, i])
                std = np.std(X[:, i])
                self.bounds_[i] = (mean - self.threshold * std, mean + self.threshold * std)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers"""
        X = X.copy()
        for i, (lower, upper) in self.bounds_.items():
            X[:, i] = np.clip(X[:, i], lower, upper)
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)