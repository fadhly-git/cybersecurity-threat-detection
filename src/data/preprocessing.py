"""Enhanced data preprocessing for cybersecurity datasets.

This module implements a 7-stage preprocessing pipeline based on the research paper
with additional enhancements for production-grade data preparation.

Preprocessing Stages:
1. Remove Redundant Columns
2. Encode Categorical Variables
3. Handle Missing Values
4. Detect & Handle Outliers (ENHANCEMENT)
5. Standardize Features
6. Handle Class Imbalance (ENHANCEMENT)
7. Split Data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import IsolationForest
from scipy import stats
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import joblib
from pathlib import Path
import re
import ipaddress

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir, save_pickle


class DataPreprocessor(LoggerMixin):
    """Enhanced preprocessing for cybersecurity datasets.
    
    Based on paper Section III.A with additional improvements including
    outlier detection and class imbalance handling.
    
    Attributes:
        config: Configuration dictionary
        scaler: Fitted scaler object
        encoder: Fitted encoder object
        imputer: Fitted imputer object
        label_encoder: Fitted label encoder for target variable
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize DataPreprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        super().__init__()
        self.config = config or {}
        self.scaler = None
        self.encoder = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.removed_columns = []
        
        self.logger.info("DataPreprocessor initialized")
    
    def extract_ip_features(self, df: pd.DataFrame, ip_column: str, prefix: str) -> pd.DataFrame:
        """Extract features from IP addresses.
        
        Args:
            df: Input DataFrame
            ip_column: Name of column containing IP addresses
            prefix: Prefix for new feature names (e.g., 'src', 'dst')
        
        Returns:
            DataFrame with extracted IP features
        """
        def parse_ip(ip_str):
            try:
                ip = ipaddress.ip_address(str(ip_str).strip())
                octets = str(ip).split('.')
                return {
                    f'{prefix}_is_private': int(ip.is_private),
                    f'{prefix}_is_multicast': int(ip.is_multicast),
                    f'{prefix}_is_loopback': int(ip.is_loopback),
                    f'{prefix}_first_octet': int(octets[0]),
                    f'{prefix}_second_octet': int(octets[1]) if len(octets) > 1 else 0,
                    f'{prefix}_class': int(octets[0]) // 64  # A=0, B=1, C=2-3
                }
            except:
                return {
                    f'{prefix}_is_private': 0,
                    f'{prefix}_is_multicast': 0,
                    f'{prefix}_is_loopback': 0,
                    f'{prefix}_first_octet': 0,
                    f'{prefix}_second_octet': 0,
                    f'{prefix}_class': 0
                }
        
        ip_features = df[ip_column].apply(parse_ip).apply(pd.Series)
        return pd.concat([df, ip_features], axis=1)
    
    def extract_text_features(self, df: pd.DataFrame, text_column: str, prefix: str) -> pd.DataFrame:
        """Extract features from text/payload data.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text data
            prefix: Prefix for new feature names
        
        Returns:
            DataFrame with extracted text features
        """
        def calc_entropy(text):
            """Calculate Shannon entropy of text."""
            try:
                from collections import Counter
                text = str(text)
                if len(text) == 0:
                    return 0
                counts = Counter(text)
                probs = [count/len(text) for count in counts.values()]
                return -sum(p * np.log2(p) for p in probs if p > 0)
            except:
                return 0
        
        df[f'{prefix}_length'] = df[text_column].fillna('').astype(str).str.len()
        df[f'{prefix}_word_count'] = df[text_column].fillna('').astype(str).str.split().str.len()
        df[f'{prefix}_entropy'] = df[text_column].fillna('').apply(calc_entropy)
        df[f'{prefix}_digit_ratio'] = df[text_column].fillna('').astype(str).apply(
            lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        df[f'{prefix}_special_char_ratio'] = df[text_column].fillna('').astype(str).apply(
            lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1))
        
        # Check for common attack patterns
        df[f'{prefix}_has_sql'] = df[text_column].fillna('').astype(str).str.contains(
            r'(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|;--|\/\*)', case=False, regex=True).astype(int)
        df[f'{prefix}_has_xss'] = df[text_column].fillna('').astype(str).str.contains(
            r'(<script|javascript:|onerror=|onload=|<iframe)', case=False, regex=True).astype(int)
        df[f'{prefix}_has_traversal'] = df[text_column].fillna('').astype(str).str.contains(
            r'(\.\./|\.\.\\|/etc/|c:\\)', case=False, regex=True).astype(int)
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame, ua_column: str, prefix: str) -> pd.DataFrame:
        """Extract features from user agent strings.
        
        Args:
            df: Input DataFrame
            ua_column: Name of column containing user agent data
            prefix: Prefix for new feature names
        
        Returns:
            DataFrame with extracted UA features
        """
        df[f'{prefix}_is_mobile'] = df[ua_column].fillna('').astype(str).str.contains(
            r'(Mobile|Android|iPhone|iPad)', case=False, regex=True).astype(int)
        df[f'{prefix}_is_bot'] = df[ua_column].fillna('').astype(str).str.contains(
            r'(bot|crawler|spider|scraper)', case=False, regex=True).astype(int)
        df[f'{prefix}_browser'] = df[ua_column].fillna('').astype(str).str.extract(
            r'(Firefox|Chrome|Safari|MSIE|Edge|Opera|Trident)', expand=False).fillna('Other')
        df[f'{prefix}_os'] = df[ua_column].fillna('').astype(str).str.extract(
            r'(Windows NT [0-9.]+|Windows|Linux|Mac OS X|Mac|Android|iOS)', expand=False).fillna('Other')
        
        return df
    
    def extract_geo_features(self, df: pd.DataFrame, geo_column: str) -> pd.DataFrame:
        """Extract features from geo-location data.
        
        Args:
            df: Input DataFrame
            geo_column: Name of column containing geo data (format: "City, State")
        
        Returns:
            DataFrame with extracted geo features
        """
        # Split city and state
        geo_split = df[geo_column].fillna('Unknown, Unknown').astype(str).str.split(',', n=1, expand=True)
        df['geo_city'] = geo_split[0].str.strip()
        df['geo_state'] = geo_split[1].str.strip() if len(geo_split.columns) > 1 else 'Unknown'
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all intelligent features from high-cardinality columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all extracted features
        """
        self.logger.info("=" * 80)
        self.logger.info("FEATURE EXTRACTION: Extracting intelligent features")
        self.logger.info("=" * 80)
        
        original_shape = df.shape
        
        # Extract IP features
        if 'Source IP Address' in df.columns:
            self.logger.info("Extracting features from Source IP Address...")
            df = self.extract_ip_features(df, 'Source IP Address', 'src_ip')
        
        if 'Destination IP Address' in df.columns:
            self.logger.info("Extracting features from Destination IP Address...")
            df = self.extract_ip_features(df, 'Destination IP Address', 'dst_ip')
        
        # Extract payload features
        if 'Payload Data' in df.columns:
            self.logger.info("Extracting features from Payload Data...")
            df = self.extract_text_features(df, 'Payload Data', 'payload')
        
        # Extract user agent features
        if 'User Information' in df.columns:
            self.logger.info("Extracting features from User Information...")
            df = self.extract_user_agent_features(df, 'User Information', 'user')
        
        if 'Device Information' in df.columns:
            self.logger.info("Extracting features from Device Information...")
            df = self.extract_user_agent_features(df, 'Device Information', 'device')
        
        # Extract geo features
        if 'Geo-location Data' in df.columns:
            self.logger.info("Extracting features from Geo-location Data...")
            df = self.extract_geo_features(df, 'Geo-location Data')
        
        # Now drop the original high-cardinality columns
        high_card_cols = ['Source IP Address', 'Destination IP Address', 'Payload Data', 
                          'User Information', 'Device Information', 'Geo-location Data']
        cols_to_drop = [col for col in high_card_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"Dropped original high-cardinality columns: {cols_to_drop}")
        
        new_features = df.shape[1] - original_shape[1] + len(cols_to_drop)
        self.logger.info(f"Feature extraction complete: {original_shape[1]} -> {df.shape[1]} columns (+{new_features} engineered features)")
        self.logger.info("")
        
        return df
    
    def remove_redundant_columns(
        self,
        df: pd.DataFrame,
        columns_to_remove: Optional[List[str]] = None,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove redundant columns from DataFrame.
        
        Stage 1: Remove duplicate columns, low-variance features, and highly
        correlated features.
        
        Args:
            df: Input DataFrame
            columns_to_remove: List of specific columns to remove
            correlation_threshold: Threshold for removing correlated features
            variance_threshold: Threshold for removing low-variance features
        
        Returns:
            DataFrame with redundant columns removed
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: Removing Redundant Columns")
        self.logger.info("=" * 80)
        
        df = df.copy()
        initial_columns = len(df.columns)
        self.removed_columns = []
        
        # Remove specified columns
        if columns_to_remove:
            cols_to_drop = [col for col in columns_to_remove if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.removed_columns.extend(cols_to_drop)
                self.logger.info(f"Removed {len(cols_to_drop)} specified columns: {cols_to_drop}")
        
        # Remove duplicate columns
        duplicate_cols = []
        cols = df.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_cols.append(cols[j])
        
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
            self.removed_columns.extend(duplicate_cols)
            self.logger.info(f"Removed {len(duplicate_cols)} duplicate columns")
        
        # Remove low-variance features (numeric only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variances = df[numeric_cols].var()
            low_variance_cols = variances[variances < variance_threshold].index.tolist()
            
            if low_variance_cols:
                df = df.drop(columns=low_variance_cols)
                self.removed_columns.extend(low_variance_cols)
                self.logger.info(f"Removed {len(low_variance_cols)} low-variance columns")
        
        # Remove highly correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            correlated_cols = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > correlation_threshold)
            ]
            
            if correlated_cols:
                df = df.drop(columns=correlated_cols)
                self.removed_columns.extend(correlated_cols)
                self.logger.info(f"Removed {len(correlated_cols)} highly correlated columns")
        
        final_columns = len(df.columns)
        total_removed = initial_columns - final_columns
        
        self.logger.info(f"Summary: Removed {total_removed} columns ({initial_columns} -> {final_columns})")
        self.logger.info("")
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        method: str = 'onehot',
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Encode categorical variables.
        
        Stage 2: Convert categorical variables to numerical format using
        one-hot encoding or label encoding.
        
        Args:
            df: Input DataFrame
            method: Encoding method ('onehot' or 'label')
            categorical_columns: List of categorical columns. If None, auto-detect
        
        Returns:
            DataFrame with encoded categorical variables
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: Encoding Categorical Variables")
        self.logger.info("=" * 80)
        
        df = df.copy()
        
        # Auto-detect categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        else:
            categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        if not categorical_columns:
            self.logger.info("No categorical columns found")
            self.logger.info("")
            return df
        
        self.logger.info(f"Encoding {len(categorical_columns)} categorical columns using {method}")
        
        if method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(
                df,
                columns=categorical_columns,
                drop_first=False,
                dtype=int
            )
            
            new_columns = len(df_encoded.columns) - len(df.columns)
            self.logger.info(f"Created {new_columns} new columns from one-hot encoding")
            
            return df_encoded
        
        elif method == 'label':
            # Label encoding
            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.logger.info(f"Label encoded '{col}': {len(le.classes_)} unique values")
            
            self.logger.info("")
            return df
        
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'auto'
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.
        
        Stage 3: Detect and impute missing values using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('auto', 'mean', 'median', 'mode', 'knn')
        
        Returns:
            DataFrame with missing values handled
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: Handling Missing Values")
        self.logger.info("=" * 80)
        
        df = df.copy()
        
        # Count missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) == 0:
            self.logger.info("No missing values found")
            self.logger.info("")
            return df
        
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
        
        self.logger.info(f"Found {total_missing} missing values ({missing_pct:.2f}% of data)")
        self.logger.info(f"Columns with missing values: {len(missing_cols)}")
        
        # Log columns with most missing values
        top_missing = missing_cols.nlargest(5)
        for col, count in top_missing.items():
            pct = (count / len(df)) * 100
            self.logger.info(f"  {col}: {count} ({pct:.2f}%)")
        
        # Determine strategy
        if strategy == 'auto':
            # Use median for numeric, mode for categorical
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0:
                imputer_numeric = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
            
            if len(categorical_cols) > 0:
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
            
            self.logger.info("Applied auto imputation (median for numeric, mode for categorical)")
        
        elif strategy == 'knn':
            # KNN imputation (numeric only)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.logger.info(f"Applied KNN imputation on {len(numeric_cols)} numeric columns")
            
            # Mode for categorical
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
        
        else:
            # Simple imputation with specified strategy
            if strategy not in ['mean', 'median', 'most_frequent']:
                strategy = 'median'
            
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.logger.info(f"Applied {strategy} imputation")
        
        # Verify no missing values remain
        remaining_missing = df.isnull().sum().sum()
        self.logger.info(f"Remaining missing values: {remaining_missing}")
        self.logger.info("")
        
        return df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'isolation_forest',
        contamination: float = 0.1
    ) -> np.ndarray:
        """Detect outliers in DataFrame.
        
        Stage 4a: Detect outliers using various methods.
        
        Args:
            df: Input DataFrame
            method: Detection method ('isolation_forest', 'zscore', 'iqr')
            contamination: Expected proportion of outliers (for isolation_forest)
        
        Returns:
            Boolean array indicating outliers (True = outlier)
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: Detecting Outliers")
        self.logger.info("=" * 80)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            self.logger.info("No numeric columns for outlier detection")
            return np.zeros(len(df), dtype=bool)
        
        self.logger.info(f"Using {method} method on {len(numeric_df.columns)} numeric columns")
        
        if method == 'isolation_forest':
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            predictions = clf.fit_predict(numeric_df.fillna(0))
            outliers = predictions == -1
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df.fillna(0), axis=0))
            outliers = (z_scores > 3).any(axis=1)
        
        elif method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | 
                       (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        self.logger.info(f"Detected {n_outliers} outliers ({pct_outliers:.2f}% of data)")
        self.logger.info("")
        
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        strategy: str = 'clip',
        method: str = 'isolation_forest',
        contamination: float = 0.1
    ) -> pd.DataFrame:
        """Handle outliers in DataFrame.
        
        Stage 4b: Handle detected outliers using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: Handling strategy ('clip', 'remove', 'ignore')
            method: Detection method
            contamination: Expected proportion of outliers
        
        Returns:
            DataFrame with outliers handled
        """
        if strategy == 'ignore':
            self.logger.info("Outlier handling skipped (strategy='ignore')")
            self.logger.info("")
            return df
        
        df = df.copy()
        outliers = self.detect_outliers(df, method=method, contamination=contamination)
        
        if strategy == 'remove':
            initial_rows = len(df)
            df = df[~outliers].reset_index(drop=True)
            removed_rows = initial_rows - len(df)
            self.logger.info(f"Removed {removed_rows} outlier rows")
            self.logger.info("")
        
        elif strategy == 'clip':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.logger.info(f"Clipped outliers in {len(numeric_cols)} numeric columns")
            self.logger.info("")
        
        return df
    
    def standardize_features(
        self,
        df: pd.DataFrame,
        method: str = 'standard',
        feature_range: Tuple[int, int] = (0, 1)
    ) -> pd.DataFrame:
        """Standardize numerical features.
        
        Stage 5: Scale features to have consistent ranges.
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard' or 'minmax')
            feature_range: Range for MinMaxScaler
        
        Returns:
            DataFrame with standardized features
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 5: Standardizing Features")
        self.logger.info("=" * 80)
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.logger.info("No numeric columns to standardize")
            self.logger.info("")
            return df
        
        self.logger.info(f"Standardizing {len(numeric_cols)} numeric columns using {method}")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        self.logger.info(f"Features scaled using {method} scaler")
        self.logger.info("")
        
        return df
    
    def handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'smote',
        sampling_strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using resampling techniques.
        
        Stage 6: Balance class distribution using SMOTE, ADASYN, or undersampling.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Resampling method ('smote', 'adasyn', 'undersample', 'none')
            sampling_strategy: Sampling strategy ('auto', 'minority', 'majority', etc.)
        
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 6: Handling Class Imbalance")
        self.logger.info("=" * 80)
        
        # Log original class distribution
        unique, counts = np.unique(y, return_counts=True)
        self.logger.info("Original class distribution:")
        for label, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            self.logger.info(f"  Class {label}: {count} ({pct:.2f}%)")
        
        if method == 'none':
            self.logger.info("Class imbalance handling skipped")
            self.logger.info("")
            return X, y
        
        try:
            if method == 'smote':
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    k_neighbors=min(5, counts.min() - 1) if counts.min() > 1 else 1
                )
            elif method == 'adasyn':
                sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                    n_neighbors=min(5, counts.min() - 1) if counts.min() > 1 else 1
                )
            elif method == 'undersample':
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported resampling method: {method}")
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Log new class distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            self.logger.info(f"\nResampled class distribution using {method}:")
            for label, count in zip(unique, counts):
                pct = (count / len(y_resampled)) * 100
                self.logger.info(f"  Class {label}: {count} ({pct:.2f}%)")
            
            self.logger.info(f"Dataset size: {len(y)} -> {len(y_resampled)}")
            self.logger.info("")
            
            return X_resampled, y_resampled
        
        except Exception as e:
            self.logger.warning(f"Resampling failed: {str(e)}. Returning original data.")
            self.logger.info("")
            return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        cv_folds: int = 5,
        return_cv: bool = False,
        random_state: int = 42
    ) -> Union[Tuple, StratifiedKFold]:
        """Split data into train and test sets.
        
        Stage 7: Create stratified train-test split or K-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            return_cv: Whether to return CV splitter instead of train/test split
            random_state: Random seed for reproducibility
        
        Returns:
            If return_cv=False: Tuple of (X_train, X_test, y_train, y_test)
            If return_cv=True: StratifiedKFold object
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 7: Splitting Data")
        self.logger.info("=" * 80)
        
        if return_cv:
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state
            )
            self.logger.info(f"Created {cv_folds}-fold stratified cross-validation")
            self.logger.info("")
            return cv
        
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            self.logger.info(f"Train-test split: {test_size*100:.0f}% test size")
            self.logger.info(f"Train set: {len(X_train)} samples")
            self.logger.info(f"Test set: {len(X_test)} samples")
            
            # Log class distribution in splits
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            
            self.logger.info("\nTrain set class distribution:")
            for label, count in zip(train_unique, train_counts):
                pct = (count / len(y_train)) * 100
                self.logger.info(f"  Class {label}: {count} ({pct:.2f}%)")
            
            self.logger.info("\nTest set class distribution:")
            for label, count in zip(test_unique, test_counts):
                pct = (count / len(y_test)) * 100
                self.logger.info(f"  Class {label}: {count} ({pct:.2f}%)")
            
            self.logger.info("")
            return X_train, X_test, y_train, y_test
    
    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        config: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run complete preprocessing pipeline.
        
        Executes all 7 stages of preprocessing in sequence.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            config: Configuration dictionary (uses self.config if None)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if config is None:
            config = self.config
        
        preproc_config = config.get('data', {}).get('preprocessing', {})
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING COMPLETE PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Input shape: {df.shape}")
        self.logger.info(f"Target column: {target_column}\n")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        y = df[target_column].values
        X = df.drop(columns=[target_column])
        
        # Stage 0: Feature Extraction (NEW - before removing columns)
        if preproc_config.get('extract_ip_features', False) or \
           preproc_config.get('extract_text_features', False) or \
           preproc_config.get('extract_geo_features', False):
            X = self.extract_all_features(X)
        
        # Stage 1: Remove redundant columns
        X = self.remove_redundant_columns(
            X,
            columns_to_remove=preproc_config.get('remove_columns'),
            correlation_threshold=preproc_config.get('correlation_threshold', 0.95),
            variance_threshold=preproc_config.get('variance_threshold', 0.01)
        )
        
        # Stage 2: Encode categorical variables
        X = self.encode_categorical(
            X,
            method=preproc_config.get('encoding_method', 'onehot')
        )
        
        # Stage 3: Handle missing values
        X = self.handle_missing_values(
            X,
            strategy=preproc_config.get('missing_value_strategy', 'auto')
        )
        
        # Stage 4: Handle outliers
        if preproc_config.get('handle_outliers', True):
            X = self.handle_outliers(
                X,
                strategy=preproc_config.get('outlier_strategy', 'clip'),
                method=preproc_config.get('outlier_method', 'isolation_forest'),
                contamination=preproc_config.get('outlier_contamination', 0.1)
            )
        
        # Stage 5: Standardize features
        X = self.standardize_features(
            X,
            method=preproc_config.get('scaling_method', 'standard')
        )
        
        # Convert to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Encode target labels if they are strings
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.logger.info(f"Encoded target labels: {len(self.label_encoder.classes_)} classes")
        else:
            y_encoded = y
        
        # Stage 6: Handle class imbalance
        if preproc_config.get('balance_classes', True):
            X_array, y_encoded = self.handle_class_imbalance(
                X_array,
                y_encoded,
                method=preproc_config.get('balance_method', 'smote')
            )
        
        # Stage 7: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X_array,
            y_encoded,
            test_size=preproc_config.get('test_size', 0.2),
            random_state=preproc_config.get('random_state', 42)
        )
        
        self.logger.info("=" * 80)
        self.logger.info("PREPROCESSING PIPELINE COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Final train shape: {X_train.shape}")
        self.logger.info(f"Final test shape: {X_test.shape}")
        self.logger.info("=" * 80 + "\n")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, output_dir: str) -> None:
        """Save preprocessor artifacts (scaler, encoders, etc.).
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_dir = ensure_dir(output_dir)
        
        artifacts = {
            'scaler': self.scaler,
            'encoder': self.encoder,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'removed_columns': self.removed_columns,
            'config': self.config
        }
        
        for name, obj in artifacts.items():
            if obj is not None:
                filepath = output_dir / f'{name}.pkl'
                save_pickle(obj, str(filepath))
                self.logger.info(f"Saved {name} to {filepath}")
    
    def load_preprocessor(self, input_dir: str) -> None:
        """Load preprocessor artifacts.
        
        Args:
            input_dir: Directory containing artifacts
        """
        input_dir = Path(input_dir)
        
        artifacts = ['scaler', 'encoder', 'imputer', 'label_encoder', 'config']
        
        for name in artifacts:
            filepath = input_dir / f'{name}.pkl'
            if filepath.exists():
                setattr(self, name, joblib.load(filepath))
                self.logger.info(f"Loaded {name} from {filepath}")
