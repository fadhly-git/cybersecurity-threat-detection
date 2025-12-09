import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import logging

logging.basicConfig(level=logging.INFO)

class CICIDS2017Loader:
    """
    Loader and preprocessor for CICIDS2017 dataset.
    
    Dataset: 2,830,743 rows, 79 columns
    """
    
    def __init__(self, data_path, subsample_ratio=None):
        """
        Initialize loader.
        
        Args:
            data_path: Path to CICIDS2017 CSV files
            subsample_ratio: Optional fraction to subsample the data for faster testing
        """
        self.data_path = data_path
        self.subsample_ratio = subsample_ratio
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.feature_selector = None
    
    def load_raw_data(self):
        """
        Load CICIDS2017 CSV files with chunking for memory efficiency.
        
        Returns:
            Combined DataFrame
        """
        logging.info("\n" + "="*60)
        logging.info("  LOADING CICIDS2017 DATASET")
        logging.info("="*60)
        
        # List all CSV files
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        logging.info(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files:
            logging.info(f"  - {f}")
        
        # Load and combine all files with chunking
        dfs = []
        logging.info(f"\nLoading CSV files with chunks...")
        for csv_file in tqdm(csv_files, desc="Loading Files", ncols=80, leave=True):
            file_path = os.path.join(self.data_path, csv_file)
            dtypes = {col: 'float32' for col in pd.read_csv(file_path, nrows=1).columns if col != 'Label' and col != ' Label'}
            for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False, dtype=dtypes):
                dfs.append(chunk)
        
        # Combine all dataframes
        logging.info(f"\nCombining dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Subsample if requested
        if self.subsample_ratio is not None:
            combined_df = combined_df.sample(frac=self.subsample_ratio, random_state=42)
            logging.info(f"Subsampled to {self.subsample_ratio*100}%: {len(combined_df):,} rows")
        
        logging.info(f"\n✅ Total dataset shape: {combined_df.shape}")
        logging.info(f"   Rows: {combined_df.shape[0]:,}")
        logging.info(f"   Columns: {combined_df.shape[1]}")
        
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
        
        logging.info(f"\n[1/10] Removed {removed:,} duplicate rows")
        logging.info(f"      Remaining: {len(df):,} rows")
        
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
            logging.info(f"\n[2/10] Found {missing_count} missing values")
            logging.info("      Applying mean imputation...")
            
            # Mean imputation
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            logging.info("      ✅ Missing values imputed")
        else:
            logging.info("\n[2/10] No missing values found")
        
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
        
        logging.info(f"\n[3/10] Removed {removed:,} rows with infinity values")
        logging.info(f"      Remaining: {len(df):,} rows")
        
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
        logging.info("\n[4/10] Consolidating labels...")
        
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
        logging.info("\n      Original label distribution:")
        logging.info(df[label_col].value_counts().to_string())
        
        # Map labels
        df['Label'] = df[label_col].map(label_mapping)
        
        # Handle unmapped labels (use default 0 = BENIGN)
        df['Label'] = df['Label'].fillna(0).astype(int)
        
        # Drop old label column if different
        if label_col != 'Label':
            df = df.drop(columns=[label_col])
        
        logging.info("\n      Consolidated label distribution:")
        logging.info(df['Label'].value_counts().sort_index().to_string())
        
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
        logging.info(f"\n[5/10] Correlation-based feature selection (threshold={threshold})...")
        
        # Separate features and target
        target_col = 'Label'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        logging.info(f"      Initial features: {len(X.columns)}")
        
        # Calculate correlation matrix with tqdm
        with tqdm(total=1, desc="Computing Correlation", ncols=80, leave=True) as pbar:
            corr_matrix = X.corr().abs()
            pbar.update(1)
        
        # Find highly correlated pairs
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Remove features with correlation > threshold
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > threshold)]
        
        logging.info(f"      Removing {len(to_drop)} highly correlated features")
        
        X = X.drop(columns=to_drop)
        
        logging.info(f"      Final features: {len(X.columns)}")
        
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
        logging.info(f"\n[7/10] Selecting top {k} features via mutual information...")

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

        logging.info(f"      Selected features: {len(self.selected_features)}")
        return X_train_selected, X_test_selected
    
    def apply_smoteenn(self, X_train, y_train):
        """
        SMOTE-ENN for oversampling + noise cleaning.
        Combines SMOTE with Edited Nearest Neighbors to reduce overlap.
        """
        logging.info("\n[9/10] Applying SMOTE-ENN (oversample + clean)...")

        unique, counts = np.unique(y_train, return_counts=True)
        logging.info("\n      Class distribution BEFORE SMOTE-ENN:")
        for cls, count in zip(unique, counts):
            logging.info(f"        Class {cls}: {count:,} samples")

        # Check minimum class size to determine k_neighbors
        min_samples = counts.min()
        
        # k_neighbors must be < min_samples (SMOTE needs k_neighbors + 1 samples)
        # Use adaptive k_neighbors based on smallest class
        if min_samples <= 2:
            logging.warning(f"      ⚠️  Smallest class has only {min_samples} samples - skipping SMOTE-ENN")
            logging.warning("      Consider using a larger dataset or different sampling strategy")
            return X_train, y_train
        elif min_samples <= 5:
            k_neighbors = 1  # Minimum safe value
            logging.warning(f"      ⚠️  Small class detected ({min_samples} samples), using k_neighbors={k_neighbors}")
        elif min_samples <= 10:
            k_neighbors = min(3, min_samples - 2)
            logging.info(f"      Using adaptive k_neighbors={k_neighbors} (min_samples={min_samples})")
        else:
            k_neighbors = 5  # Default value
            logging.info(f"      Using k_neighbors={k_neighbors}")

        # Create SMOTE and ENN instances with adaptive parameters
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy='not majority')
        enn = EditedNearestNeighbours(n_neighbors=min(3, min_samples - 1), n_jobs=-1)
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)

        logging.info("\n      Processing... (this may take a few minutes)")
        logging.info("      ⏳ Running SMOTE oversampling and ENN noise cleaning...")
        
        try:
            if len(X_train.shape) > 2:
                original_shape = X_train.shape
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                logging.info(f"      Flattening data: {original_shape} → {X_train_flat.shape}")
                
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
        
        except ValueError as e:
            logging.error(f"\n      ❌ SMOTE-ENN failed: {str(e)}")
            logging.warning(f"      Falling back to SMOTE only (without ENN cleaning)...")
            
            # Fallback to SMOTE only
            try:
                if len(X_train.shape) > 2:
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    with tqdm(total=1, desc="SMOTE (fallback)", ncols=80, leave=True) as pbar:
                        X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)
                        X_resampled = X_resampled.reshape(-1, *X_train.shape[1:])
                        pbar.update(1)
                else:
                    with tqdm(total=1, desc="SMOTE (fallback)", ncols=80, leave=True) as pbar:
                        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                        pbar.update(1)
            except Exception as e2:
                logging.error(f"      ❌ SMOTE also failed: {str(e2)}")
                logging.warning(f"      Returning original data without resampling")
                return X_train, y_train

        unique, counts = np.unique(y_resampled, return_counts=True)
        logging.info("\n      Class distribution AFTER SMOTE-ENN:")
        for cls, count in zip(unique, counts):
            logging.info(f"        Class {cls}: {count:,} samples")

        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, test_size=0.2, apply_smote=True):
        """
        Complete preprocessing pipeline with overall progress bar.
        
        Args:
            test_size: Test set ratio
            apply_smote: Whether to apply SMOTE oversampling
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logging.info("\n" + "="*60)
        logging.info("  CICIDS2017 PREPROCESSING PIPELINE")
        logging.info("="*60)
        
        with tqdm(total=10, desc="Pipeline Progress", ncols=80, leave=True) as pbar:
            # 1. Load data
            df = self.load_raw_data()
            pbar.update(1)
            
            # 2. Remove duplicates
            df = self.remove_duplicates(df)
            pbar.update(1)
            
            # 3. Handle missing values
            df = self.handle_missing_values(df)
            pbar.update(1)
            
            # 4. Remove infinity values
            df = self.remove_infinity_values(df)
            pbar.update(1)
            
            # 5. Consolidate labels
            df = self.consolidate_labels(df)
            pbar.update(1)
            
            # 6. Correlation-based feature selection
            df = self.correlation_feature_selection(df, threshold=0.95)
            pbar.update(1)
            
            # 7. Separate features and target
            X = df.drop('Label', axis=1)
            y = df['Label']
            
            logging.info(f"\n[6/10] Stratified train-test split ({int((1-test_size)*100)}/{int(test_size*100)})...")

            # 8. Stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logging.info(f"      Train set: {X_train.shape[0]:,} samples")
            logging.info(f"      Test set: {X_test.shape[0]:,} samples")
            
            # Check class distribution in training set
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            logging.info(f"\n      Training set class distribution:")
            for cls, count in zip(unique_train, counts_train):
                logging.info(f"        Class {cls}: {count:,} samples")
            
            min_class_samples = counts_train.min()
            if min_class_samples < 6:
                logging.warning(f"\n      ⚠️  WARNING: Smallest class has only {min_class_samples} samples!")
                logging.warning(f"      This may cause issues with SMOTE (requires at least 6 samples)")
                if min_class_samples < 3:
                    logging.warning(f"      Consider increasing dataset size or removing rare classes")
            
            pbar.update(1)
            
            # 7. Mutual information feature selection
            X_train, X_test = self.select_top_features(X_train, y_train, X_test, k=60)
            pbar.update(1)
            
            # 8. Min-Max scaling
            logging.info(f"\n[8/10] Applying Min-Max scaling...")
            with tqdm(total=2, desc="Scaling", ncols=80, leave=True) as scale_bar:
                scale_bar.set_description("Scaling [Training set]")
                X_train = self.scaler.fit_transform(X_train)
                scale_bar.update(1)
                
                scale_bar.set_description("Scaling [Test set]")
                X_test = self.scaler.transform(X_test)
                scale_bar.update(1)
            pbar.update(1)
            
            # 11. SMOTE-ENN (optional)
            if apply_smote:
                X_train, y_train = self.apply_smoteenn(X_train, y_train)
            pbar.update(1)
            
            # 12. Reshape for DL models (samples, features, 1)
            logging.info(f"\n[10/10] Reshaping for deep learning models...")
            with tqdm(total=2, desc="Reshaping", ncols=80, leave=True) as reshape_bar:
                reshape_bar.set_description("Reshaping [Training set]")
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                reshape_bar.update(1)
                
                reshape_bar.set_description("Reshaping [Test set]")
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                reshape_bar.update(1)
        
        
        logging.info(f"\n{'='*60}")
        logging.info("  PREPROCESSING COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"\nFinal shapes:")
        logging.info(f"  X_train: {X_train.shape}")
        logging.info(f"  X_test:  {X_test.shape}")
        logging.info(f"  y_train: {y_train.shape}")
        logging.info(f"  y_test:  {y_test.shape}")
        logging.info(f"  Features: {X_train.shape[1]}")
        logging.info(f"  Classes: {len(np.unique(y_train))}")
        logging.info(f"{'='*60}\n")
        
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
        
        logging.info(f"\n✅ Preprocessed data saved to: {output_dir}/")
        logging.info(f"   - X_train.npy, X_test.npy")
        logging.info(f"   - y_train.npy, y_test.npy")
        logging.info(f"   - scaler.pkl, selected_features.pkl")