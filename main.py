"""
Main Pipeline - Cybersecurity Threat Detection
Comparing ML and DL models with SOTA from paper:
"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of 
Machine and Deep Learning Techniques for Threat Detection"
https://arxiv.org/abs/2407.06014
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Force CPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import CICIDS loader
from data.data_loader_cicids import load_cicids, CICIDSLoader

# Import ML models
from models.ml_models import (
    OptimizedRandomForest, 
    OptimizedXGBoost, 
    OptimizedLightGBM, 
    OptimizedCatBoost
)

# Import DL models
from models.dl_models import (
    DeepNeuralNetwork, 
    LightCNN, 
    LightLSTM,
    HybridCNNLSTM
)

# Import Ensemble models
from models.ensemble_models import (
    VotingEnsembleClassifier, 
    StackingEnsemble
)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.results = []
        
    def evaluate(self, name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray = None, train_time: float = 0) -> dict:
        """Evaluate model performance"""
        
        metrics = {
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
        return metrics
    
    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        return pd.DataFrame(self.results)
    
    def print_results(self):
        """Print formatted results"""
        df = self.get_results_df()
        print("\n" + "=" * 100)
        print("📊 MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        # Best model
        best_idx = df['F1-Score'].idxmax()
        best_model = df.loc[best_idx]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("🔒 CYBERSECURITY THREAT DETECTION SYSTEM")
    print("   Comparing ML/DL Models with Paper SOTA")
    print("   Paper: arxiv.org/abs/2407.06014")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    evaluator = ModelEvaluator()
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("📂 STEP 1: LOADING CICIDS2017 DATASET")
    print("=" * 60)
    
    X, y = load_cicids(
        data_path="data/raw",
        binary=True,
        sample_frac=0.1  # Use None for full dataset (slower but more accurate)
    )
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # ============================================================
    # 2. SPLIT DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("✂️  STEP 2: SPLITTING DATA")
    print("=" * 60)
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # ============================================================
    # 3. SCALE FEATURES
    # ============================================================
    print("\n" + "=" * 60)
    print("⚙️  STEP 3: SCALING FEATURES")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    
    n_classes = len(np.unique(y))
    input_dim = X_train_scaled.shape[1]
    
    print(f"Number of classes: {n_classes}")
    print(f"Input dimension: {input_dim}")
    
    # ============================================================
    # 4. TRAIN MACHINE LEARNING MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("🤖 STEP 4: TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    ml_models = {
        'Random Forest': OptimizedRandomForest(random_state=42),
        'XGBoost': OptimizedXGBoost(random_state=42),
        'LightGBM': OptimizedLightGBM(random_state=42),
        'CatBoost': OptimizedCatBoost(random_state=42),
    }
    
    trained_ml_models = {}
    
    for name, model in ml_models.items():
        print(f"\n🔄 Training {name}...")
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time)
        trained_ml_models[name] = model
        
        print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
        print(f"   ⏱️  Time: {train_time:.2f}s")
    
    # ============================================================
    # 5. TRAIN DEEP LEARNING MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("🧠 STEP 5: TRAINING DEEP LEARNING MODELS (CPU)")
    print("=" * 60)
    
    dl_configs = {
        'DNN': {
            'class': DeepNeuralNetwork,
            'params': {
                'input_dim': input_dim,
                'n_classes': n_classes,
                'hidden_layers': (256, 128, 64),
                'dropout_rate': 0.4,
                'random_state': 42
            }
        },
        'CNN-1D': {
            'class': LightCNN,
            'params': {
                'input_dim': input_dim,
                'n_classes': n_classes,
                'filters': (32, 64),
                'dropout_rate': 0.4,
                'random_state': 42
            }
        },
        'LSTM': {
            'class': LightLSTM,
            'params': {
                'input_dim': input_dim,
                'n_classes': n_classes,
                'lstm_units': (64, 32),
                'dropout_rate': 0.4,
                'random_state': 42
            }
        },
        'Hybrid CNN-LSTM': {
            'class': HybridCNNLSTM,
            'params': {
                'input_dim': input_dim,
                'n_classes': n_classes,
                'cnn_filters': (32,),
                'lstm_units': (32,),
                'dropout_rate': 0.4,
                'random_state': 42
            }
        }
    }
    
    trained_dl_models = {}
    
    for name, config in dl_configs.items():
        print(f"\n🔄 Training {name}...")
        
        try:
            model = config['class'](**config['params'])
            
            # Reshape for CNN/LSTM if needed
            if name in ['CNN-1D', 'LSTM', 'Hybrid CNN-LSTM']:
                X_train_dl = X_train_scaled.reshape(-1, input_dim, 1)
                X_val_dl = X_val_scaled.reshape(-1, input_dim, 1)
                X_test_dl = X_test_scaled.reshape(-1, input_dim, 1)
            else:
                X_train_dl = X_train_scaled
                X_val_dl = X_val_scaled
                X_test_dl = X_test_scaled
            
            start_time = time.time()
            model.fit(
                X_train_dl, y_train,
                X_val=X_val_dl, y_val=y_val,
                epochs=30,  # Reduced for CPU
                batch_size=128,
                verbose=0
            )
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test_dl)
            y_proba = model.predict_proba(X_test_dl)
            
            metrics = evaluator.evaluate(name, y_test, y_pred, y_proba, train_time)
            trained_dl_models[name] = model
            
            print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
            print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
            print(f"   ⏱️  Time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {e}")
    
    # ============================================================
    # 6. TRAIN ENSEMBLE MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("🎯 STEP 6: TRAINING ENSEMBLE MODELS")
    print("=" * 60)
    
    # Voting Ensemble
    print("\n🔄 Training Voting Ensemble...")
    try:
        start_time = time.time()
        voting = VotingEnsembleClassifier(voting='soft', random_state=42)
        voting.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = voting.predict(X_test_scaled)
        y_proba = voting.predict_proba(X_test_scaled)
        
        metrics = evaluator.evaluate('Voting Ensemble', y_test, y_pred, y_proba, train_time)
        print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
        print(f"   ⏱️  Time: {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Stacking Ensemble
    print("\n🔄 Training Stacking Ensemble...")
    try:
        start_time = time.time()
        stacking = StackingEnsemble(cv=3, random_state=42)
        stacking.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = stacking.predict(X_test_scaled)
        y_proba = stacking.predict_proba(X_test_scaled)
        
        metrics = evaluator.evaluate('Stacking Ensemble', y_test, y_pred, y_proba, train_time)
        print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   ✅ F1-Score: {metrics['F1-Score']:.4f}")
        print(f"   ⏱️  Time: {train_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # ============================================================
    # 7. FINAL RESULTS & COMPARISON WITH SOTA
    # ============================================================
    print("\n" + "=" * 60)
    print("📈 STEP 7: FINAL RESULTS & SOTA COMPARISON")
    print("=" * 60)
    
    evaluator.print_results()
    
    # Save results
    results_df = evaluator.get_results_df()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"results/comparison_results_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n📁 Results saved to: {results_path}")
    
    # Print SOTA comparison
    print("\n" + "=" * 60)
    print("📚 COMPARISON WITH PAPER SOTA (arxiv.org/abs/2407.06014)")
    print("=" * 60)
    print("""
    Paper SOTA Results on CICIDS2017 (Binary Classification):
    ┌─────────────────────┬──────────┬───────────┬──────────┬──────────┐
    │ Model               │ Accuracy │ Precision │ Recall   │ F1-Score │
    ├─────────────────────┼──────────┼───────────┼──────────┼──────────┤
    │ Random Forest       │ 0.9976   │ 0.9976    │ 0.9976   │ 0.9976   │
    │ Decision Tree       │ 0.9969   │ 0.9969    │ 0.9969   │ 0.9969   │
    │ XGBoost             │ 0.9986   │ 0.9986    │ 0.9986   │ 0.9986   │
    │ LightGBM            │ 0.9977   │ 0.9977    │ 0.9977   │ 0.9977   │
    │ CNN                 │ 0.9960   │ 0.9960    │ 0.9960   │ 0.9960   │
    │ LSTM                │ 0.9942   │ 0.9942    │ 0.9942   │ 0.9942   │
    │ DNN                 │ 0.9958   │ 0.9958    │ 0.9958   │ 0.9958   │
    └─────────────────────┴──────────┴───────────┴──────────┴──────────┘
    
    Note: Results may vary based on:
    - Data sampling (full dataset vs sampled)
    - Preprocessing pipeline
    - Hyperparameter tuning
    - Random seed
    """)
    
    # Compare with our results
    print("\n🎯 OUR RESULTS vs PAPER SOTA:")
    print("-" * 60)
    
    best_model = results_df.loc[results_df['F1-Score'].idxmax()]
    paper_best = 0.9986  # XGBoost from paper
    
    our_f1 = best_model['F1-Score']
    diff = our_f1 - paper_best
    
    print(f"Our Best Model: {best_model['Model']}")
    print(f"Our F1-Score: {our_f1:.4f}")
    print(f"Paper Best F1: {paper_best:.4f}")
    print(f"Difference: {diff:+.4f}")
    
    if diff >= 0:
        print("\n✅ 🎉 CONGRATULATIONS! Our model BEATS the paper SOTA!")
    else:
        print(f"\n📌 Gap to beat: {abs(diff):.4f}")
        print("   Suggestions to improve:")
        print("   1. Use full dataset (remove sample_frac)")
        print("   2. Tune hyperparameters")
        print("   3. Try different ensemble combinations")
        print("   4. Feature engineering")
    
    # ============================================================
    # COMPLETE
    # ============================================================
    print("\n" + "=" * 80)
    print("✅ CYBERSECURITY THREAT DETECTION COMPLETED!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    results = main()