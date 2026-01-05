"""
Main Pipeline dengan CICIDS Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import CICIDS loader
from data.data_loader_cicids import load_cicids, CICIDSLoader

def main():
    print("=" * 60)
    print("CYBERSECURITY ML - CICIDS2017 DATASET")
    print("=" * 60)
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n[1/5] Loading CICIDS2017 Dataset...")
    
    # Load dengan sampling 10% untuk testing cepat
    # Hapus sample_frac untuk full dataset
    X, y = load_cicids(
        data_path="data/raw",
        binary=True,
        sample_frac=0.1  # Gunakan None untuk full data
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # ============================================================
    # 2. SPLIT DATA
    # ============================================================
    print("\n[2/5] Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # ============================================================
    # 3. SCALE FEATURES
    # ============================================================
    print("\n[3/5] Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 4. TRAIN MODEL
    # ============================================================
    print("\n[4/5] Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    model.fit(X_train_scaled, y_train)
    
    # ============================================================
    # 5. EVALUATE
    # ============================================================
    print("\n[5/5] Evaluating model...")
    
    y_pred = model.predict(X_test_scaled)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Benign', 'Attack']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"               Predicted")
    print(f"              Benign  Attack")
    print(f"Actual Benign  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Feature importance
    print("\nTop 10 Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()