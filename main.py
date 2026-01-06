"""
Main Pipeline - Cybersecurity Threat Detection
Comparing ML and DL models with SOTA from paper:
"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of 
Machine and Deep Learning Techniques for Threat Detection"
https://arxiv.org/abs/2407.06014

Models from Paper:
- ML: Naive Bayes, Decision Tree, Random Forest, KNN, SVM, Extra Trees
- DL: VGG16, VGG19, ResNet18, ResNet50, Inception (adapted for tabular)
- Two datasets: CICIDS2017 and NSL-KDD
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

# Force CPU for TensorFlow (must be before tensorflow import)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import models from models/ directory
from models.ml_models import get_sklearn_models
from models.dl_models import (
    create_vgg16_tabular, create_vgg19_tabular,
    create_resnet18_tabular, create_resnet50_tabular,
    create_inception_tabular, get_dl_models, get_callbacks
)

# Import data loaders
from data.data_loader_cicids import load_cicids, CICIDSLoader
from data.data_loader_nslkdd import load_nslkdd, NSLKDDLoader


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    """Setup logging to both file and console"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
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
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
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
    """Train ML models from paper: NB, DT, RF, KNN, SVM, Extra Trees"""
    
    log_print("\n" + "=" * 60)
    log_print("🤖 TRAINING MACHINE LEARNING MODELS (From Paper)")
    log_print("=" * 60)
    
    # Get ML models from models/ml_models.py
    ml_models = get_sklearn_models(random_state=42)
    
    for name, model in ml_models.items():
        log_print(f"\n🔄 Training {name}...")
        
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            # Handle models without predict_proba (like SGDClassifier/SVM)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    pass
            
            metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time, model)
            
            # Display with 6 decimals to show differences between models
            log_print(f"   ✅ Accuracy:  {metrics['Accuracy']:.6f}")
            log_print(f"   ✅ Precision: {metrics['Precision']:.6f}")
            log_print(f"   ✅ Recall:    {metrics['Recall']:.6f}")
            log_print(f"   ✅ F1-Score:  {metrics['F1-Score']:.6f}")
            log_print(f"   ⏱️  Time: {train_time:.2f}s")
            
        except Exception as e:
            log_print(f"   ❌ Error training {name}: {e}", level="error")


def train_dl_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                    evaluator, n_classes, input_dim, epochs=30):
    """Train DL models from paper: VGG16, VGG19, ResNet18, ResNet50, Inception"""
    
    log_print("\n" + "=" * 60)
    log_print("🧠 TRAINING DEEP LEARNING MODELS (From Paper)")
    log_print("=" * 60)
    
    # Get DL models from models/dl_models.py
    dl_models = get_dl_models(input_dim, n_classes)
    
    for name, model in dl_models.items():
        log_print(f"\n🔄 Training {name}...")
        
        try:
            # Create fresh callbacks for each model
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
            
            # Display with 6 decimals to show differences between models
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
def run_experiment_cicids(sample_frac=0.1, dl_epochs=30):
    """Run experiment on CICIDS2017 dataset"""
    
    log_print("\n" + "=" * 80)
    log_print("📂 DATASET 1: CICIDS2017")
    log_print("=" * 80)
    
    evaluator = ModelEvaluator(dataset_name="CICIDS2017")
    
    log_print("\n📥 Loading CICIDS2017 dataset...")
    X, y = load_cicids(data_path="data/raw", binary=True, sample_frac=sample_frac)
    
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
    joblib.dump(scaler, f"saved_models/scaler_CICIDS2017.joblib")
    
    n_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]
    
    # Train models
    train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test, evaluator)
    train_dl_models(X_train_scaled, y_train, X_val_scaled, y_val, 
                    X_test_scaled, y_test, evaluator, n_classes, input_dim, dl_epochs)
    
    return evaluator


def run_experiment_nslkdd(dl_epochs=30):
    """Run experiment on NSL-KDD dataset"""
    
    log_print("\n" + "=" * 80)
    log_print("📂 DATASET 2: NSL-KDD")
    log_print("=" * 80)
    
    evaluator = ModelEvaluator(dataset_name="NSL-KDD")
    
    log_print("\n📥 Loading NSL-KDD dataset...")
    X_train, y_train, X_test, y_test = load_nslkdd(data_path="data/raw", binary=True)
    
    log_print(f"\nTrain shape: {X_train.shape}")
    log_print(f"Test shape: {X_test.shape}")
    log_print(f"Train distribution: Normal={np.sum(y_train==0):,}, Attack={np.sum(y_train==1):,}")
    log_print(f"Test distribution: Normal={np.sum(y_test==0):,}, Attack={np.sum(y_test==1):,}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    log_print(f"\nTraining set: {X_train.shape[0]:,} samples")
    log_print(f"Validation set: {X_val.shape[0]:,} samples")
    log_print(f"Test set: {X_test.shape[0]:,} samples")
    
    log_print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, f"saved_models/scaler_NSL-KDD.joblib")
    
    n_classes = len(np.unique(y_train))
    input_dim = X_train_scaled.shape[1]
    
    # Train models
    train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test, evaluator)
    train_dl_models(X_train_scaled, y_train, X_val_scaled, y_val, 
                    X_test_scaled, y_test, evaluator, n_classes, input_dim, dl_epochs)
    
    return evaluator


# ============================================================================
# SOTA COMPARISON FROM PAPER
# ============================================================================
def print_sota_comparison(results_df):
    """Print comparison with exact paper SOTA results"""
    
    log_print("\n" + "=" * 90)
    log_print("📚 COMPARISON WITH PAPER SOTA (arxiv.org/abs/2407.06014)")
    log_print("=" * 90)
    
    # Exact values from paper Table 2 & Table 3
    paper_sota = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                         PAPER SOTA - CICIDS2017 DATASET                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  Model          │ Accuracy  │ Precision │ Recall    │ F1-Score  │ Type              ║
    ╠═════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════════════╣
    ║  Naive Bayes    │ 0.8817    │ 0.8776    │ 0.8817    │ 0.8639    │ ML                ║
    ║  Decision Tree  │ 0.9969    │ 0.9969    │ 0.9969    │ 0.9969    │ ML                ║
    ║  Random Forest  │ 0.9976    │ 0.9976    │ 0.9976    │ 0.9976    │ ML                ║
    ║  KNN            │ 0.9967    │ 0.9967    │ 0.9967    │ 0.9966    │ ML                ║
    ║  SVM            │ 0.9779    │ 0.9780    │ 0.9779    │ 0.9774    │ ML                ║
    ║  Extra Trees    │ 0.9980    │ 0.9980    │ 0.9980    │ 0.9980    │ ML                ║
    ║  VGG16          │ 0.9906    │ 0.9906    │ 0.9906    │ 0.9904    │ DL                ║
    ║  VGG19          │ 0.9900    │ 0.9899    │ 0.9900    │ 0.9898    │ DL                ║
    ║  ResNet18       │ 0.9843    │ 0.9843    │ 0.9843    │ 0.9839    │ DL                ║
    ║  ResNet50       │ 0.9831    │ 0.9830    │ 0.9831    │ 0.9826    │ DL                ║
    ║  Inception      │ 0.9886    │ 0.9885    │ 0.9886    │ 0.9883    │ DL                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                          PAPER SOTA - NSL-KDD DATASET                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  Model          │ Accuracy  │ Precision │ Recall    │ F1-Score  │ Type              ║
    ╠═════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════════════╣
    ║  Naive Bayes    │ 0.7953    │ 0.8155    │ 0.7953    │ 0.7847    │ ML                ║
    ║  Decision Tree  │ 0.9936    │ 0.9937    │ 0.9936    │ 0.9936    │ ML                ║
    ║  Random Forest  │ 0.9949    │ 0.9949    │ 0.9949    │ 0.9949    │ ML                ║
    ║  KNN            │ 0.9936    │ 0.9936    │ 0.9936    │ 0.9936    │ ML                ║
    ║  SVM            │ 0.9726    │ 0.9729    │ 0.9726    │ 0.9724    │ ML                ║
    ║  Extra Trees    │ 0.9950    │ 0.9951    │ 0.9950    │ 0.9950    │ ML                ║
    ║  VGG16          │ 0.9870    │ 0.9870    │ 0.9870    │ 0.9869    │ DL                ║
    ║  VGG19          │ 0.9862    │ 0.9862    │ 0.9862    │ 0.9860    │ DL                ║
    ║  ResNet18       │ 0.9764    │ 0.9766    │ 0.9764    │ 0.9763    │ DL                ║
    ║  ResNet50       │ 0.9758    │ 0.9759    │ 0.9758    │ 0.9756    │ DL                ║
    ║  Inception      │ 0.9842    │ 0.9843    │ 0.9842    │ 0.9841    │ DL                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    log_print(paper_sota)
    
    # Paper best scores from paper
    paper_best = {
        'CICIDS2017': {'model': 'Extra Trees', 'f1': 0.9980, 'acc': 0.9980},
        'NSL-KDD': {'model': 'Extra Trees', 'f1': 0.9950, 'acc': 0.9950}
    }
    
    log_print("\n🎯 OUR RESULTS vs PAPER SOTA:")
    log_print("-" * 90)
    
    for dataset in results_df['Dataset'].unique():
        dataset_results = results_df[results_df['Dataset'] == dataset]
        
        if len(dataset_results) == 0:
            continue
            
        best_idx = dataset_results['F1-Score'].idxmax()
        our_best = dataset_results.loc[best_idx]
        
        paper_info = paper_best.get(dataset, {'f1': 0.99, 'acc': 0.99, 'model': 'Unknown'})
        
        log_print(f"\n📊 {dataset}:")
        log_print(f"   Our Best Model: {our_best['Model']}")
        log_print(f"   Our F1-Score:   {our_best['F1-Score']:.4f}")
        log_print(f"   Our Accuracy:   {our_best['Accuracy']:.4f}")
        log_print(f"   ")
        log_print(f"   Paper Best:     {paper_info['model']}")
        log_print(f"   Paper F1-Score: {paper_info['f1']:.4f}")
        log_print(f"   ")
        
        f1_diff = our_best['F1-Score'] - paper_info['f1']
        
        if f1_diff >= 0:
            log_print(f"   ✅ 🎉 BEATS SOTA by {f1_diff:+.4f}")
        else:
            log_print(f"   📌 Gap to SOTA: {f1_diff:.4f}")
            log_print(f"   💡 Tips: Use full dataset (sample_frac=None), increase epochs")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main execution function"""
    
    log_print("=" * 90)
    log_print("🔒 CYBERSECURITY THREAT DETECTION SYSTEM")
    log_print("   Comparing ML/DL Models with Paper SOTA")
    log_print("   Paper: arxiv.org/abs/2407.06014")
    log_print("=" * 90)
    log_print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Log file: {LOG_FILE}")
    
    log_print("\n📋 Models from Paper:")
    log_print("   ML: Naive Bayes, Decision Tree, Random Forest, KNN, SVM, Extra Trees")
    log_print("   DL: VGG16, VGG19, ResNet18, ResNet50, Inception")
    
    # ================================================================
    # CONFIGURATION
    # ================================================================
    # Use 0.1 (10%) for faster testing, None for full dataset (VERY SLOW for KNN/SVM)
    # Recommended: 0.1-0.2 for testing, None only for final results
    SAMPLE_FRAC = None      # 10% sampling for reasonable speed
    DL_EPOCHS = 30         # Increase for better DL performance
    
    all_results = []
    best_models = {}
    
    # ================================================================
    # EXPERIMENT 1: CICIDS2017
    # ================================================================
    try:
        evaluator_cicids = run_experiment_cicids(sample_frac=SAMPLE_FRAC, dl_epochs=DL_EPOCHS)
        evaluator_cicids.print_results()
        all_results.append(evaluator_cicids.get_results_df())
        
        # Save best model
        model_path = evaluator_cicids.save_best_model()
        if model_path:
            best_models['CICIDS2017'] = {
                'path': model_path,
                'name': evaluator_cicids.best_model_name,
                'f1': evaluator_cicids.best_f1
            }
    except Exception as e:
        log_print(f"\n❌ Error in CICIDS2017 experiment: {e}", level="error")
    
    # ================================================================
    # EXPERIMENT 2: NSL-KDD
    # ================================================================
    try:
        evaluator_nslkdd = run_experiment_nslkdd(dl_epochs=DL_EPOCHS)
        evaluator_nslkdd.print_results()
        all_results.append(evaluator_nslkdd.get_results_df())
        
        # Save best model
        model_path = evaluator_nslkdd.save_best_model()
        if model_path:
            best_models['NSL-KDD'] = {
                'path': model_path,
                'name': evaluator_nslkdd.best_model_name,
                'f1': evaluator_nslkdd.best_f1
            }
    except Exception as e:
        log_print(f"\n❌ Error in NSL-KDD experiment: {e}", level="error")
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"results/comparison_results_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        combined_results.to_csv(results_path, index=False)
        log_print(f"\n📁 Results saved to: {results_path}")
        
        # Print SOTA comparison
        print_sota_comparison(combined_results)
        
        # Summary
        log_print("\n" + "=" * 90)
        log_print("📊 FINAL SUMMARY - ALL DATASETS")
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
