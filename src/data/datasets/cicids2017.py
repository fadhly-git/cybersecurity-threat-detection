"""
CICIDS2017 Dataset Loader and Preprocessor

Complete preprocessing pipeline for CICIDS2017 dataset (2.8M samples, 79 features).

Preprocessing steps (updated):
1. Remove duplicates (308,381 rows)
2. Handle missing values (353 entries - mean imputation)
3. Remove infinity values
4. Label consolidation (group attack types into 7 classes)
5. Correlation-based feature filtering
6. Stratified train-test split
7. Mutual-information top-k feature selection (default k=60)
8. Min-Max scaling (robust for skewed network flows)
9. SMOTE-ENN balancing (oversample + clean noise)
10. Reshape for deep models
"""

import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTEENN


class CICIDS2017Loader:
    """
    Loader and preprocessor for CICIDS2017 dataset.
    
    Dataset: 2,830,743 rows, 79 columns
    """
    
    def __init__(self, data_path):
        """
        Initialize loader.
        
        Args:
            data_path: Path to CICIDS2017 CSV files
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.feature_selector = None
    
    def load_raw_data(self):
        """
        Load CICIDS2017 CSV files.
        
        CICIDS2017 consists of multiple CSV files (Monday-Friday).
        
        Returns:
            Combined DataFrame
        """
        print("\n" + "="*60)
        print("  LOADING CICIDS2017 DATASET")
        print("="*60)
        
        # List all CSV files
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        print(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {f}")
        
        # Load and combine all files
        dfs = []
        print(f"\nLoading CSV files...")
        for csv_file in tqdm(csv_files, desc="Loading Data", ncols=80, leave=True):
            file_path = os.path.join(self.data_path, csv_file)
            df = pd.read_csv(file_path)
            dfs.append(df)
        
        # Combine all dataframes
        print(f"\nCombining dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        print(f"\n✅ Total dataset shape: {combined_df.shape}")
        print(f"   Rows: {combined_df.shape[0]:,}")
        print(f"   Columns: {combined_df.shape[1]}")
        
        return combined_df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows.
        
        Paper reports: 308,381 duplicates
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        print(f"\n[1/10] Removed {removed:,} duplicate rows")
        print(f"      Remaining: {len(df):,} rows")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values with mean imputation.
        
        Paper reports: 353 missing values
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with imputed values
        """
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Count missing values
        missing_count = df[numeric_cols].isnull().sum().sum()
        
        if missing_count > 0:
            print(f"\n[2/10] Found {missing_count} missing values")
            print("      Applying mean imputation...")
            
            # Mean imputation
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            print("      ✅ Missing values imputed")
        else:
            print("\n[2/10] No missing values found")
        
        return df
    
    def remove_infinity_values(self, df):
        """
        Remove rows with infinity values.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame without infinity values
        """
        initial_rows = len(df)
        
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN (former inf values)
        df = df.dropna()
        
        removed = initial_rows - len(df)
        
        print(f"\n[3/10] Removed {removed:,} rows with infinity values")
        print(f"      Remaining: {len(df):,} rows")
        
        return df
    
    def consolidate_labels(self, df):
        """
        Consolidate attack types into 7 main classes (sequential 0-6).
        
        Classes:
        0: BENIGN
        1: DoS/DDoS (DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, DDoS)
        2: PortScan
        3: Bot
        4: Infiltration
        5: Web Attack (Brute Force, XSS, SQL Injection) + Brute Force (FTP-Patator, SSH-Patator)
        6: Heartbleed
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with consolidated labels
        """
        print("\n[4/10] Consolidating labels...")
        
        # Label mapping - SEQUENTIAL 0-6 (no gaps!)
        label_mapping = {
            'BENIGN': 0,
            'DoS Hulk': 1,
            'DoS GoldenEye': 1,
            'DoS slowloris': 1,
            'DoS Slowhttptest': 1,
            'DDoS': 1,
            'PortScan': 2,
            'Bot': 3,
            'Infiltration': 4,
            'Web Attack Brute Force': 5,
            'Web Attack XSS': 5,
            'Web Attack Sql Injection': 5,
            'FTP-Patator': 5,
            'SSH-Patator': 5,
            'Heartbleed': 6
        }
        
        # Handle different column names
        label_col = 'Label' if 'Label' in df.columns else ' Label'
        
        # Show original distribution
        print("\n      Original label distribution:")
        print(df[label_col].value_counts())
        
        # Map labels
        df['Label'] = df[label_col].map(label_mapping)
        
        # Handle unmapped labels (use default 0 = BENIGN)
        df['Label'] = df['Label'].fillna(0).astype(int)
        
        # Drop old label column if different
        if label_col != 'Label':
            df = df.drop(columns=[label_col])
        
        print("\n      Consolidated label distribution:")
        print(df['Label'].value_counts().sort_index())
        
        return df
    
    def correlation_feature_selection(self, df, threshold=0.95):
        """
        Feature selection based on correlation.
        
        Remove highly correlated features (correlation > 0.95).
        Reduces from 79 → ~40 features.
        
        Args:
            df: DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with selected features
        """
        print(f"\n[5/10] Correlation-based feature selection (threshold={threshold})...")
        
        # Separate features and target
        target_col = 'Label'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"      Initial features: {len(X.columns)}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Remove features with correlation > threshold
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > threshold)]
        
        print(f"      Removing {len(to_drop)} highly correlated features")
        
        X = X.drop(columns=to_drop)
        
        print(f"      Final features: {len(X.columns)}")
        
        # Store selected features
        self.selected_features = X.columns.tolist()
        
        # Combine back with target
        result = pd.concat([X, y], axis=1)
        
        return result

    def select_top_features(self, X_train, y_train, X_test, k=60):
        """
        Select top-k informative features using mutual information.
        Reduces dimensionality while keeping predictive signals.
        """
        k = min(k, X_train.shape[1])
        print(f"\n[7/10] Selecting top {k} features via mutual information...")

        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        with tqdm(total=2, desc="Feature Selection", ncols=80, leave=True) as pbar:
            pbar.set_description("Feature Selection [Training]")
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            pbar.update(1)
            
            pbar.set_description("Feature Selection [Test]")
            X_test_selected = self.feature_selector.transform(X_test)
            pbar.update(1)

        support_mask = self.feature_selector.get_support()
        selected = X_train.columns[support_mask]
        self.selected_features = selected.tolist()

        print(f"      Selected features: {len(self.selected_features)}")
        return X_train_selected, X_test_selected
    
    def apply_smoteenn(self, X_train, y_train):
        """
        SMOTE-ENN for oversampling + noise cleaning.
        Combines SMOTE with Edited Nearest Neighbors to reduce overlap.
        """
        print("\n[9/10] Applying SMOTE-ENN (oversample + clean)...")

        unique, counts = np.unique(y_train, return_counts=True)
        print("\n      Class distribution BEFORE SMOTE-ENN:")
        for cls, count in zip(unique, counts):
            print(f"        Class {cls}: {count:,} samples")

        # Create SMOTE-ENN with verbose callback for progress
        smote_enn = SMOTEENN(
            random_state=42, 
            smote_kwargs={'k_neighbors': 5},
            n_jobs=-1  # Use all CPU cores for faster processing
        )

        print("\n      Processing... (this may take a few minutes)")
        print("      ⏳ Running SMOTE oversampling and ENN noise cleaning...")
        
        if len(X_train.shape) > 2:
            original_shape = X_train.shape
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            print(f"      Flattening data: {original_shape} → {X_train_flat.shape}")
            
            # Apply SMOTE-ENN with progress indicator
            with tqdm(total=2, desc="SMOTE-ENN", ncols=80, leave=True) as pbar:
                pbar.set_description("SMOTE-ENN [SMOTE oversampling]")
                pbar.update(0.5)
                
                X_resampled, y_resampled = smote_enn.fit_resample(X_train_flat, y_train)
                pbar.update(1)
                
                pbar.set_description("SMOTE-ENN [Reshaping]")
                X_resampled = X_resampled.reshape(-1, *original_shape[1:])
                pbar.update(0.5)
        else:
            with tqdm(total=1, desc="SMOTE-ENN", ncols=80, leave=True) as pbar:
                X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
                pbar.update(1)

        unique, counts = np.unique(y_resampled, return_counts=True)
        print("\n      Class distribution AFTER SMOTE-ENN:")
        for cls, count in zip(unique, counts):
            print(f"        Class {cls}: {count:,} samples")

        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, test_size=0.2, apply_smote=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            test_size: Test set ratio
            apply_smote: Whether to apply SMOTE oversampling
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("  CICIDS2017 PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load data
        df = self.load_raw_data()
        
        # 2. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 3. Handle missing values
        df = self.handle_missing_values(df)
        
        # 4. Remove infinity values
        df = self.remove_infinity_values(df)
        
        # 5. Consolidate labels
        df = self.consolidate_labels(df)
        
        # 6. Correlation-based feature selection
        df = self.correlation_feature_selection(df, threshold=0.95)
        
        # 7. Separate features and target
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        print(f"\n[6/10] Stratified train-test split ({int((1-test_size)*100)}/{int(test_size*100)})...")

        # 8. Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"      Train set: {X_train.shape[0]:,} samples")
        print(f"      Test set: {X_test.shape[0]:,} samples")
        
        # 7. Mutual information feature selection
        X_train, X_test = self.select_top_features(X_train, y_train, X_test, k=60)
        
        # 8. Min-Max scaling
        print(f"\n[8/10] Applying Min-Max scaling...")
        with tqdm(total=2, desc="Scaling", ncols=80, leave=True) as pbar:
            pbar.set_description("Scaling [Training set]")
            X_train = self.scaler.fit_transform(X_train)
            pbar.update(1)
            
            pbar.set_description("Scaling [Test set]")
            X_test = self.scaler.transform(X_test)
            pbar.update(1)
        
        # 11. SMOTE-ENN (optional)
        if apply_smote:
            X_train, y_train = self.apply_smoteenn(X_train, y_train)
        
        # 12. Reshape for DL models (samples, features, 1)
        print(f"\n[10/10] Reshaping for deep learning models...")
        with tqdm(total=2, desc="Reshaping", ncols=80, leave=True) as pbar:
            pbar.set_description("Reshaping [Training set]")
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            pbar.update(1)
            
            pbar.set_description("Reshaping [Test set]")
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            pbar.update(1)
        
        
        print(f"\n{'='*60}")
        print("  PREPROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"\nFinal shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test:  {y_test.shape}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"{'='*60}\n")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """
        Save preprocessed data and metadata.
        
        Args:
            X_train, X_test, y_train, y_test: Preprocessed data
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        np.save(f'{output_dir}/X_train.npy', X_train)
        np.save(f'{output_dir}/X_test.npy', X_test)
        np.save(f'{output_dir}/y_train.npy', y_train)
        np.save(f'{output_dir}/y_test.npy', y_test)
        
        # Save scaler and features
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.selected_features, f'{output_dir}/selected_features.pkl')
        
        print(f"\n✅ Preprocessed data saved to: {output_dir}/")
        print(f"   - X_train.npy, X_test.npy")
        print(f"   - y_train.npy, y_test.npy")
        print(f"   - scaler.pkl, selected_features.pkl")
