"""Feature engineering for cybersecurity datasets.

This module provides advanced feature engineering capabilities including
temporal, statistical, and network-specific feature creation, as well as
feature selection and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    RFE,
    f_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """Feature engineering for cybersecurity datasets.
    
    Provides methods for creating domain-specific features, feature selection,
    and dimensionality reduction.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        super().__init__()
        self.feature_importances_ = None
        self.selected_features_ = None
        self.pca_ = None
        
        self.logger.info("FeatureEngineer initialized")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional temporal features
        """
        self.logger.info("Creating temporal features...")
        df = df.copy()
        
        # Find timestamp columns
        timestamp_cols = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    timestamp_cols.append(col)
                except:
                    continue
        
        if not timestamp_cols:
            self.logger.info("No timestamp columns found")
            return df
        
        for col in timestamp_cols:
            prefix = col.replace('timestamp', '').replace('time', '').replace('date', '').strip('_')
            if not prefix:
                prefix = 'time'
            
            # Extract time components
            df[f'{prefix}_hour'] = df[col].dt.hour
            df[f'{prefix}_day_of_week'] = df[col].dt.dayofweek
            df[f'{prefix}_day_of_month'] = df[col].dt.day
            df[f'{prefix}_month'] = df[col].dt.month
            df[f'{prefix}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            
            # Hour bins
            df[f'{prefix}_hour_bin'] = pd.cut(
                df[col].dt.hour,
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            # Drop original timestamp
            df = df.drop(columns=[col])
        
        # One-hot encode hour bins
        bin_cols = [col for col in df.columns if 'hour_bin' in col]
        if bin_cols:
            df = pd.get_dummies(df, columns=bin_cols, prefix=bin_cols)
        
        self.logger.info(f"Created temporal features from {len(timestamp_cols)} timestamp columns")
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from numerical columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional statistical features
        """
        self.logger.info("Creating statistical features...")
        df = df.copy()
        
        # Find groups of related numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            self.logger.info("Not enough numeric columns for statistical features")
            return df
        
        # Group columns by common prefixes
        column_groups = {}
        for col in numeric_cols:
            # Extract prefix (words before numbers or common separators)
            parts = col.replace('_', ' ').split()
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in column_groups:
                    column_groups[prefix] = []
                column_groups[prefix].append(col)
        
        # Create statistical features for each group
        for prefix, cols in column_groups.items():
            if len(cols) >= 2:
                subset = df[cols]
                
                # Mean, std, min, max across the group
                df[f'{prefix}_mean'] = subset.mean(axis=1)
                df[f'{prefix}_std'] = subset.std(axis=1)
                df[f'{prefix}_min'] = subset.min(axis=1)
                df[f'{prefix}_max'] = subset.max(axis=1)
                df[f'{prefix}_range'] = df[f'{prefix}_max'] - df[f'{prefix}_min']
                
                # Coefficient of variation
                df[f'{prefix}_cv'] = df[f'{prefix}_std'] / (df[f'{prefix}_mean'] + 1e-10)
        
        # Entropy for columns with patterns
        for col in numeric_cols[:10]:  # Limit to avoid too many features
            try:
                # Discretize and calculate entropy
                discretized = pd.cut(df[col], bins=10, labels=False)
                value_counts = discretized.value_counts(normalize=True)
                df[f'{col}_entropy'] = entropy(value_counts)
            except:
                continue
        
        new_features = len(df.columns) - len(numeric_cols)
        self.logger.info(f"Created {new_features} statistical features")
        return df
    
    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create network-specific features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional network features
        """
        self.logger.info("Creating network-specific features...")
        df = df.copy()
        
        # IP address features
        ip_cols = [col for col in df.columns if 'ip' in col.lower() or 'addr' in col.lower()]
        for col in ip_cols:
            if df[col].dtype == 'object':
                # IP frequency
                ip_freq = df[col].value_counts()
                df[f'{col}_frequency'] = df[col].map(ip_freq)
                
                # Unique IP count (cumulative)
                df[f'{col}_cumulative_unique'] = df.groupby(col).cumcount()
        
        # Port features
        port_cols = [col for col in df.columns if 'port' in col.lower()]
        for col in port_cols:
            if df[col].dtype in [np.int64, np.int32, np.float64]:
                # Well-known ports (0-1023)
                df[f'{col}_is_wellknown'] = (df[col] < 1024).astype(int)
                
                # Registered ports (1024-49151)
                df[f'{col}_is_registered'] = ((df[col] >= 1024) & (df[col] < 49152)).astype(int)
                
                # Dynamic ports (49152-65535)
                df[f'{col}_is_dynamic'] = (df[col] >= 49152).astype(int)
        
        # Protocol features
        protocol_cols = [col for col in df.columns if 'protocol' in col.lower()]
        for col in protocol_cols:
            if df[col].dtype == 'object':
                # Protocol frequency
                protocol_freq = df[col].value_counts()
                df[f'{col}_frequency'] = df[col].map(protocol_freq)
        
        # Packet size features
        size_cols = [col for col in df.columns if 'size' in col.lower() or 'length' in col.lower() or 'bytes' in col.lower()]
        for col in size_cols:
            if df[col].dtype in [np.int64, np.int32, np.float64]:
                # Log transform for skewed distributions
                df[f'{col}_log'] = np.log1p(df[col])
                
                # Standardized size
                mean_size = df[col].mean()
                std_size = df[col].std()
                df[f'{col}_normalized'] = (df[col] - mean_size) / (std_size + 1e-10)
        
        # Connection pattern features
        if 'duration' in df.columns or any('duration' in col.lower() for col in df.columns):
            duration_cols = [col for col in df.columns if 'duration' in col.lower()]
            for col in duration_cols:
                if df[col].dtype in [np.int64, np.int32, np.float64]:
                    # Short, medium, long duration
                    df[f'{col}_is_short'] = (df[col] < df[col].quantile(0.25)).astype(int)
                    df[f'{col}_is_long'] = (df[col] > df[col].quantile(0.75)).astype(int)
        
        new_features = len(df.columns) - len(df.columns)
        self.logger.info(f"Created network-specific features")
        return df
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'importance',
        n_features: int = 50,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Select most important features.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Selection method ('importance', 'rfe', 'mutual_info', 'anova')
            n_features: Number of features to select
            feature_names: List of feature names (optional)
        
        Returns:
            Tuple of (selected features matrix, selected feature indices)
        """
        self.logger.info(f"Selecting {n_features} features using {method} method...")
        
        n_features = min(n_features, X.shape[1])
        
        if method == 'importance':
            # Random Forest feature importance
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
            clf.fit(X, y)
            
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            
            self.feature_importances_ = importances
            
            if feature_names is not None:
                top_features = [feature_names[i] for i in indices[:10]]
                self.logger.info(f"Top 10 features: {top_features}")
        
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
            selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
            selector.fit(X, y)
            
            indices = np.where(selector.support_)[0]
        
        elif method == 'mutual_info':
            # Mutual Information
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            selector.fit(X, y)
            
            indices = np.argsort(selector.scores_)[::-1][:n_features]
        
        elif method == 'anova':
            # ANOVA F-statistic
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X, y)
            
            indices = np.argsort(selector.scores_)[::-1][:n_features]
        
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
        
        X_selected = X[:, indices]
        self.selected_features_ = indices
        
        self.logger.info(f"Selected {len(indices)} features ({X.shape[1]} -> {X_selected.shape[1]})")
        
        return X_selected, indices.tolist()
    
    def reduce_dimensions(
        self,
        X: np.ndarray,
        method: str = 'pca',
        n_components: int = 50
    ) -> np.ndarray:
        """Reduce dimensionality of features.
        
        Args:
            X: Feature matrix
            method: Reduction method ('pca', 'none')
            n_components: Number of components to keep
        
        Returns:
            Reduced feature matrix
        """
        if method == 'none':
            self.logger.info("Dimensionality reduction skipped")
            return X
        
        self.logger.info(f"Reducing dimensions using {method} to {n_components} components...")
        
        n_components = min(n_components, X.shape[1], X.shape[0])
        
        if method == 'pca':
            self.pca_ = PCA(n_components=n_components, random_state=42)
            X_reduced = self.pca_.fit_transform(X)
            
            explained_var = self.pca_.explained_variance_ratio_.sum()
            self.logger.info(f"PCA explained variance: {explained_var*100:.2f}%")
        
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        self.logger.info(f"Reduced dimensions: {X.shape[1]} -> {X_reduced.shape[1]}")
        
        return X_reduced
    
    def create_polynomial_features(
        self,
        X: np.ndarray,
        degree: int = 2,
        interaction_only: bool = True
    ) -> np.ndarray:
        """Create polynomial features.
        
        Args:
            X: Feature matrix
            degree: Polynomial degree
            interaction_only: If True, only create interaction features
        
        Returns:
            Feature matrix with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        self.logger.info(f"Creating polynomial features (degree={degree})...")
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = poly.fit_transform(X)
        
        self.logger.info(f"Created polynomial features: {X.shape[1]} -> {X_poly.shape[1]}")
        
        return X_poly
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Run complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary
        
        Returns:
            DataFrame with engineered features
        """
        config = config or {}
        
        self.logger.info("Starting feature engineering pipeline...")
        
        # Create temporal features
        if config.get('create_temporal', False):
            df = self.create_temporal_features(df)
        
        # Create statistical features
        if config.get('create_statistical', False):
            df = self.create_statistical_features(df)
        
        # Create network features
        if config.get('create_network', False):
            df = self.create_network_features(df)
        
        self.logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df
