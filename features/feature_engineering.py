"""
feature_engineering.py - Feature engineering and selection
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class FeatureEngineer:
    """Create new features from existing ones"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_interaction_features(self, X: np.ndarray, 
                                    feature_pairs: Optional[List[Tuple[int, int]]] = None,
                                    max_features: int = 10) -> np.ndarray:
        """Create interaction features between pairs"""
        n_features = X.shape[1]
        
        if feature_pairs is None:
            # Auto-select top feature pairs based on correlation
            feature_pairs = []
            for i in range(min(max_features, n_features)):
                for j in range(i+1, min(max_features, n_features)):
                    feature_pairs.append((i, j))
        
        interactions = []
        for i, j in feature_pairs[:max_features]:
            if i < n_features and j < n_features:
                # Multiplication
                interactions.append(X[:, i] * X[:, j])
                # Division (with small epsilon to avoid division by zero)
                interactions.append(X[:, i] / (X[:, j] + 1e-8))
        
        if interactions:
            return np.column_stack([X] + interactions)
        return X
    
    def create_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """Create statistical summary features"""
        stats = []
        
        # Row-wise statistics
        stats.append(np.mean(X, axis=1, keepdims=True))
        stats.append(np.std(X, axis=1, keepdims=True))
        stats.append(np.min(X, axis=1, keepdims=True))
        stats.append(np.max(X, axis=1, keepdims=True))
        stats.append(np.median(X, axis=1, keepdims=True))
        
        return np.hstack([X] + stats)
    
    def create_ratio_features(self, X: np.ndarray, 
                              ratio_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """Create ratio features"""
        ratios = []
        n_features = X.shape[1]
        
        for i, j in ratio_pairs:
            if i < n_features and j < n_features:
                ratio = X[:, i] / (X[:, j] + 1e-8)
                ratios.append(ratio)
        
        if ratios:
            return np.column_stack([X] + ratios)
        return X
    
    def create_polynomial_features(self, X: np.ndarray, 
                                   degree: int = 2,
                                   selected_features: Optional[List[int]] = None) -> np.ndarray:
        """Create polynomial features"""
        if selected_features is None:
            selected_features = list(range(min(10, X.shape[1])))
        
        poly_features = []
        for i in selected_features:
            if i < X.shape[1]:
                for d in range(2, degree + 1):
                    poly_features.append(X[:, i] ** d)
        
        if poly_features:
            return np.column_stack([X] + poly_features)
        return X


class FeatureSelector:
    """Select most important features"""
    
    def __init__(self, 
                 method: str = 'mutual_info',
                 n_features: int = 40,
                 random_state: int = 42):
        """
        Initialize feature selector
        
        Args:
            method: 'mutual_info', 'chi2', 'f_classif', 'rfe', 'model_based', 'variance', 'pca'
            n_features: Number of features to select
            random_state: Random seed
        """
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.selector = None
        self.selected_indices_ = None
        self.feature_scores_ = None
        self._fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit selector"""
        print(f"🔍 Feature Selection ({self.method}):")
        print(f"   Original features: {X.shape[1]}")
        
        # Ensure non-negative values for chi2
        if self.method == 'chi2':
            X = np.abs(X)
        
        n_features_to_select = min(self.n_features, X.shape[1])
        
        if self.method == 'mutual_info':
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=n_features_to_select
            )
        elif self.method == 'chi2':
            self.selector = SelectKBest(
                score_func=chi2,
                k=n_features_to_select
            )
        elif self.method == 'f_classif':
            self.selector = SelectKBest(
                score_func=f_classif,
                k=n_features_to_select
            )
        elif self.method == 'rfe':
            base_model = RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=self.random_state
            )
            self.selector = RFE(
                estimator=base_model,
                n_features_to_select=n_features_to_select,
                step=0.1
            )
        elif self.method == 'model_based':
            base_model = RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=self.random_state
            )
            self.selector = SelectFromModel(
                estimator=base_model,
                max_features=n_features_to_select,
                threshold=-np.inf
            )
        elif self.method == 'variance':
            self.selector = VarianceThreshold(threshold=0.01)
        elif self.method == 'pca':
            self.selector = PCA(n_components=n_features_to_select, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit selector
        self.selector.fit(X, y)
        
        # Get selected indices
        if hasattr(self.selector, 'get_support'):
            self.selected_indices_ = np.where(self.selector.get_support())[0]
        elif self.method == 'pca':
            self.selected_indices_ = np.arange(n_features_to_select)
        
        # Get feature scores if available
        if hasattr(self.selector, 'scores_'):
            self.feature_scores_ = self.selector.scores_
        elif hasattr(self.selector, 'feature_importances_'):
            self.feature_scores_ = self.selector.feature_importances_
        
        self._fitted = True
        
        n_selected = len(self.selected_indices_) if self.selected_indices_ is not None else n_features_to_select
        print(f"   Selected features: {n_selected}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting features"""
        if not self._fitted:
            raise ValueError("FeatureSelector not fitted. Call fit() first.")
        return self.selector.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_top_features(self, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
        """Get top features by importance score"""
        if self.feature_scores_ is None:
            return None
        
        # Create dataframe
        df = pd.DataFrame({
            'feature': feature_names[:len(self.feature_scores_)],
            'score': self.feature_scores_
        })
        
        # Sort by score
        df = df.sort_values('score', ascending=False).head(top_k)
        
        return df
    
    def get_selected_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get names of selected features"""
        if self.selected_indices_ is None:
            return feature_names[:self.n_features]
        return [feature_names[i] for i in self.selected_indices_ if i < len(feature_names)]