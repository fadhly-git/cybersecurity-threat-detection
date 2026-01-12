"""
Training Script - WSN-DS & Cyber Security Datasets Only
========================================================
Script khusus untuk menjalankan training pada 2 dataset baru:
1. WSN-DS (Wireless Sensor Network)
2. Cyber Security Attacks
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

# Force CPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import models
from models.ml_models import get_sklearn_models
from models.dl_models import get_dl_models, get_callbacks

# Import data loaders
from data.data_loader_wsnds import load_wsnds, WSNDSLoader
from data.data_loader_cyber import load_cyber_security, CyberSecurityLoader
from catboost import CatBoostClassifier, Pool


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    """Setup logging to both file and console"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"new_datasets_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    
    return logger, log_file

logger, LOG_FILE = setup_logging()

def log_print(message="", level="info"):
    """Print to console and log to file"""
    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.info(message)


# ============================================================================
# MODEL EVALUATOR
# ============================================================================
class ModelEvaluator:
    """Comprehensive model evaluation with best model tracking"""
    
    def __init__(self, dataset_name: str = ""):
        self.results = []
        self.dataset_name = dataset_name
        self.best_model = None
        self.best_model_name = None
        self.best_f1 = 0
        self.trained_models = {}
        
    def evaluate(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray = None, train_time: float = 0, model=None) -> dict:
        """Evaluate model performance"""
        
        metrics = {
            'Dataset': self.dataset_name,
            'Model': name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'Training Time (s)': round(train_time, 2)
        }
        
        # ROC-AUC for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                if y_proba.ndim == 2:
                    metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['ROC-AUC'] = 'N/A'
        else:
            metrics['ROC-AUC'] = 'N/A'
        
        self.results.append(metrics)
        
        # Track best model
        if model is not None:
            self.trained_models[name] = model
            if metrics['F1-Score'] > self.best_f1:
                self.best_f1 = metrics['F1-Score']
                self.best_model = model
                self.best_model_name = name
        
        return metrics
    
    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
    
    def print_results(self):
        """Print formatted results"""
        df = self.get_results_df()
        log_print("\n" + "=" * 120)
        log_print(f"📊 MODEL COMPARISON RESULTS - {self.dataset_name}")
        log_print("=" * 120)
        log_print(df.to_string(index=False))
        log_print("=" * 120)
        
        best_idx = df['F1-Score'].idxmax()
        best_model = df.loc[best_idx]
        log_print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        log_print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        log_print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    
    def save_best_model(self, save_dir="saved_models"):
        """Save the best performing model"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.best_model is None:
            log_print("No best model to save!", level="warning")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if it's a Keras model or sklearn model
        if hasattr(self.best_model, 'save'):
            # Keras model
            model_path = os.path.join(save_dir, f"{self.dataset_name}_{self.best_model_name}_{timestamp}.keras")
            self.best_model.save(model_path)
        else:
            # Sklearn model
            model_path = os.path.join(save_dir, f"{self.dataset_name}_{self.best_model_name}_{timestamp}.joblib")
            joblib.dump(self.best_model, model_path)
        
        log_print(f"\n💾 Best model saved: {model_path}")
        log_print(f"   Model: {self.best_model_name}")
        log_print(f"   F1-Score: {self.best_f1:.4f}")
        
        return model_path


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_ml_models(X_train, y_train, X_test, y_test, evaluator):
    """Train ML models: NB, DT, RF, KNN, SVM, Extra Trees"""
    
    log_print("\n" + "=" * 60)
    log_print("🤖 TRAINING MACHINE LEARNING MODELS")
    log_print("=" * 60)
    
    ml_models = get_sklearn_models(random_state=42)
    
    for name, model in ml_models.items():
        log_print(f"\n🔄 Training {name}...")
        
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            # Handle models without predict_proba
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    pass
            
            metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time, model)
            
            log_print(f"   ✅ Accuracy:  {metrics['Accuracy']:.6f}")
            log_print(f"   ✅ Precision: {metrics['Precision']:.6f}")
            log_print(f"   ✅ Recall:    {metrics['Recall']:.6f}")
            log_print(f"   ✅ F1-Score:  {metrics['F1-Score']:.6f}")
            log_print(f"   ⏱️  Time: {train_time:.2f}s")
            
        except Exception as e:
            log_print(f"   ❌ Error training {name}: {e}", level="error")


def train_dl_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                    evaluator, n_classes, input_dim, epochs=30):
    """Train DL models: VGG16, VGG19, ResNet18, ResNet50, Inception"""
    
    log_print("\n" + "=" * 60)
    log_print("🧠 TRAINING DEEP LEARNING MODELS")
    log_print("=" * 60)
    
    dl_models = get_dl_models(input_dim, n_classes)
    
    for name, model in dl_models.items():
        log_print(f"\n🔄 Training {name}...")
        
        try:
            callbacks = get_callbacks(patience=10)
            
            start_time = time.time()
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=128,
                callbacks=callbacks,
                verbose=0
            )
            train_time = time.time() - start_time
            
            y_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            
            metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time, model)
            
            log_print(f"   ✅ Accuracy:  {metrics['Accuracy']:.6f}")
            log_print(f"   ✅ Precision: {metrics['Precision']:.6f}")
            log_print(f"   ✅ Recall:    {metrics['Recall']:.6f}")
            log_print(f"   ✅ F1-Score:  {metrics['F1-Score']:.6f}")
            log_print(f"   ⏱️  Time: {train_time:.2f}s")
            
        except Exception as e:
            log_print(f"   ❌ Error training {name}: {e}", level="error")


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================
def run_experiment_wsnds(sample_frac=None, dl_epochs=30):
    """Run experiment on WSN-DS dataset"""
    
    log_print("\n" + "=" * 80)
    log_print("📂 DATASET 1: WSN-DS (Wireless Sensor Network)")
    log_print("=" * 80)
    
    evaluator = ModelEvaluator(dataset_name="WSN-DS")
    
    log_print("\n📥 Loading WSN-DS dataset...")
    X, y = load_wsnds(data_path="data/raw", binary=True, sample_frac=sample_frac)
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    log_print(f"\nFeatures shape: {X.shape}")
    log_print(f"Labels shape: {y.shape}")
    log_print(f"Class distribution: Normal={np.sum(y==0):,}, Attack={np.sum(y==1):,}")
    
    log_print("\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    log_print(f"Training set: {X_train.shape[0]:,} samples")
    log_print(f"Validation set: {X_val.shape[0]:,} samples")
    log_print(f"Test set: {X_test.shape[0]:,} samples")
    
    log_print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, f"saved_models/scaler_WSN-DS.joblib")
    
    n_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]
    
    # Train models
    train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test, evaluator)
    train_dl_models(X_train_scaled, y_train, X_val_scaled, y_val, 
                    X_test_scaled, y_test, evaluator, n_classes, input_dim, dl_epochs)
    
    return evaluator


def run_experiment_cyber_security(sample_frac=None, dl_epochs=30):
    """Run experiment on Cyber Security Attacks dataset"""
    
    log_print("\n" + "=" * 80)
    log_print("📂 DATASET 2: CYBER SECURITY ATTACKS")
    log_print("=" * 80)
    
    evaluator = ModelEvaluator(dataset_name="Cyber-Security")
    
    log_print("\n📥 Loading Cyber Security Attacks dataset...")
    X, y = load_cyber_security(data_path="data/raw", binary=True, sample_frac=sample_frac)
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    log_print(f"\nFeatures shape: {X.shape}")
    log_print(f"Labels shape: {y.shape}")
    
    # Handle case where all samples are attacks (no normal)
    unique_classes = np.unique(y)
    if len(unique_classes) == 1:
        log_print("⚠️  Only one class found - using multi-class attack types")
        # Reload with multi-class
        loader = CyberSecurityLoader("data/raw")
        loader.preprocess(binary=False, sample_frac=sample_frac)
        X, y = loader.get_X_y()
    
    log_print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    log_print("\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    log_print(f"Training set: {X_train.shape[0]:,} samples")
    log_print(f"Validation set: {X_val.shape[0]:,} samples")
    log_print(f"Test set: {X_test.shape[0]:,} samples")
    
    log_print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, f"saved_models/scaler_Cyber-Security.joblib")
    
    n_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]
    
    # Train models
    train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test, evaluator)

    # CatBoost with native categorical handling (use raw DataFrame)
    try:
        log_print("\n🔄 Training CatBoost (native categorical)...")
        loader_cb = CyberSecurityLoader("data/raw")
        # Keep label scheme consistent with earlier split
        loader_cb.preprocess(binary=(len(unique_classes) == 2), sample_frac=sample_frac)
        X_df, y_df, cat_idx = loader_cb.get_catboost_data()

        # Match splits with same seeds and stratification
        X_temp_df, X_test_df, y_temp_df, y_test_df = train_test_split(
            X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
        )
        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            X_temp_df, y_temp_df, test_size=0.125, random_state=42, stratify=y_temp_df
        )

        loss_fn = 'MultiClass' if n_classes > 2 else 'Logloss'
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=8,
            loss_function=loss_fn,
            eval_metric='TotalF1' if n_classes > 2 else 'AUC',
            random_seed=42,
            verbose=False
        )

        train_pool = Pool(X_train_df, y_train_df, cat_features=cat_idx)
        val_pool = Pool(X_val_df, y_val_df, cat_features=cat_idx)
        start_time = time.time()
        cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        train_time = time.time() - start_time

        test_pool = Pool(X_test_df, y_test_df, cat_features=cat_idx)
        # Quick sanity check on train
        y_proba_train_cb = cb_model.predict_proba(train_pool)
        y_pred_train_cb = np.argmax(y_proba_train_cb, axis=1)
        train_acc = accuracy_score(y_train_df, y_pred_train_cb)
        log_print(f"   ℹ️  CatBoost train accuracy: {train_acc:.4f}")

        y_proba_cb = cb_model.predict_proba(test_pool)
        y_pred_cb = np.argmax(y_proba_cb, axis=1)

        metrics = evaluator.evaluate('CatBoost', y_test_df, y_pred_cb, y_proba_cb, train_time, cb_model)
        log_print(f"   ✅ Accuracy:  {metrics['Accuracy']:.6f}")
        log_print(f"   ✅ Precision: {metrics['Precision']:.6f}")
        log_print(f"   ✅ Recall:    {metrics['Recall']:.6f}")
        log_print(f"   ✅ F1-Score:  {metrics['F1-Score']:.6f}")
        log_print(f"   ⏱️  Time: {train_time:.2f}s")
    except Exception as e:
        log_print(f"   ❌ Error training CatBoost: {e}", level="error")

    train_dl_models(X_train_scaled, y_train, X_val_scaled, y_val, 
                    X_test_scaled, y_test, evaluator, n_classes, input_dim, dl_epochs)
    
    return evaluator


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main execution function for WSN-DS and Cyber Security datasets"""
    
    log_print("=" * 90)
    log_print("🔒 CYBERSECURITY THREAT DETECTION - NEW DATASETS")
    log_print("   WSN-DS & Cyber Security Attacks")
    log_print("=" * 90)
    log_print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Log file: {LOG_FILE}")
    
    log_print("\n📋 Models:")
    log_print("   ML: Naive Bayes, Decision Tree, Random Forest, KNN, SVM, Extra Trees")
    log_print("   DL: VGG16, VGG19, ResNet18, ResNet50, Inception")
    
    log_print("\n📂 Datasets:")
    log_print("   1. WSN-DS (Wireless Sensor Network) - 374,661 samples")
    log_print("   2. Cyber Security Attacks - 40,000 samples")
    
    # ================================================================
    # CONFIGURATION
    # ================================================================
    # Use 0.1 (10%) for faster testing, None for full dataset
    SAMPLE_FRAC = 0.1     # 10% sampling for reasonable speed
    DL_EPOCHS = 30         # Epochs for deep learning models
    
    all_results = []
    best_models = {}
    
    # ================================================================
    # EXPERIMENT 1: WSN-DS
    # ================================================================
    try:
        evaluator_wsnds = run_experiment_wsnds(sample_frac=SAMPLE_FRAC, dl_epochs=DL_EPOCHS)
        evaluator_wsnds.print_results()
        all_results.append(evaluator_wsnds.get_results_df())
        
        # Save best model
        model_path = evaluator_wsnds.save_best_model()
        if model_path:
            best_models['WSN-DS'] = {
                'path': model_path,
                'name': evaluator_wsnds.best_model_name,
                'f1': evaluator_wsnds.best_f1
            }
    except Exception as e:
        log_print(f"\n❌ Error in WSN-DS experiment: {e}", level="error")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # EXPERIMENT 2: CYBER SECURITY ATTACKS
    # ================================================================
    try:
        evaluator_cyber = run_experiment_cyber_security(sample_frac=SAMPLE_FRAC, dl_epochs=DL_EPOCHS)
        evaluator_cyber.print_results()
        all_results.append(evaluator_cyber.get_results_df())
        
        # Save best model
        model_path = evaluator_cyber.save_best_model()
        if model_path:
            best_models['Cyber-Security'] = {
                'path': model_path,
                'name': evaluator_cyber.best_model_name,
                'f1': evaluator_cyber.best_f1
            }
    except Exception as e:
        log_print(f"\n❌ Error in Cyber Security experiment: {e}", level="error")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"results/new_datasets_results_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        combined_results.to_csv(results_path, index=False)
        log_print(f"\n📁 Results saved to: {results_path}")
        
        # Summary
        log_print("\n" + "=" * 90)
        log_print("📊 FINAL SUMMARY - NEW DATASETS")
        log_print("=" * 90)
        log_print(combined_results.to_string(index=False))
        
        # Best models summary
        log_print("\n" + "=" * 90)
        log_print("💾 SAVED BEST MODELS")
        log_print("=" * 90)
        for dataset, info in best_models.items():
            log_print(f"   {dataset}: {info['name']} (F1={info['f1']:.4f})")
            log_print(f"      Path: {info['path']}")
    
    # ================================================================
    # COMPLETE
    # ================================================================
    log_print("\n" + "=" * 90)
    log_print("✅ EXPERIMENT COMPLETED!")
    log_print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Log file: {LOG_FILE}")
    log_print("=" * 90)
    
    return combined_results if all_results else None


if __name__ == "__main__":
    results = main()
