"""
metrics.py - Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class ComprehensiveMetrics:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self):
        self.results = {}
        self.confusion_matrix_ = None
        self.classification_report_ = None
        self.roc_curve_ = None
        self.pr_curve_ = None
        
    def calculate_all_metrics(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate all metrics"""
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Advanced metrics
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        results['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                n_classes = y_proba.shape[1]
                if n_classes == 2:
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    results['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    )
                    results['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovo', average='weighted'
                    )
            except Exception as e:
                results['roc_auc_ovr'] = None
                results['roc_auc_ovo'] = None
        
        # Confusion matrix
        self.confusion_matrix_ = confusion_matrix(y_true, y_pred)
        
        # Classification report
        self.classification_report_ = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        self.results = results
        return results
    
    def calculate_per_class_metrics(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate per-class metrics"""
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Convert to dataframe
        df = pd.DataFrame(report).transpose()
        
        # Rename index if class names provided
        if class_names is not None:
            index_map = {str(i): name for i, name in enumerate(class_names)}
            df = df.rename(index=index_map)
        
        return df
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return self.confusion_matrix_
    
    def get_classification_report(self, as_dict: bool = False):
        """Get classification report"""
        if as_dict:
            return self.classification_report_
        return pd.DataFrame(self.classification_report_).transpose()
    
    def calculate_detection_metrics(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    attack_label: int = 1) -> Dict:
        """Calculate detection-specific metrics"""
        
        # Binary classification metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # For multi-class, aggregate attack classes
        if len(cm) > 2:
            # Assume label 0 is normal, others are attacks
            tn = cm[0, 0]
            fp = cm[0, 1:].sum()
            fn = cm[1:, 0].sum()
            tp = cm[1:, 1:].sum()
        else:
            tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': (tp + tn) / total if total > 0 else 0
        }
        
        return metrics
    
    def print_summary(self):
        """Print metrics summary"""
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        
        for metric, value in self.results.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        print("="*60)


class ModelComparator:
    """Compare multiple models"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def add_model(self,
                  name: str,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None):
        """Add model results"""
        
        evaluator = ComprehensiveMetrics()
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_proba)
        
        self.models[name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'evaluator': evaluator
        }
        self.metrics[name] = metrics
        
    def get_comparison_df(self) -> pd.DataFrame:
        """Get comparison dataframe"""
        return pd.DataFrame(self.metrics).transpose()
    
    def get_best_model(self, metric: str = 'f1_weighted') -> str:
        """Get best model by metric"""
        df = self.get_comparison_df()
        if metric in df.columns:
            return df[metric].idxmax()
        return list(self.models.keys())[0]
    
    def print_comparison(self):
        """Print comparison table"""
        df = self.get_comparison_df()
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Select key metrics
        key_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 
                       'f1_weighted', 'cohen_kappa', 'mcc']
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        print(df[available_metrics].round(4).to_string())
        print("="*80)
        
        # Best model
        best = self.get_best_model('f1_weighted')
        print(f"\n🏆 Best Model (by F1 Weighted): {best}")
        print(f"   F1 Score: {self.metrics[best]['f1_weighted']:.4f}")
    
    def get_ranking(self, metric: str = 'f1_weighted') -> pd.DataFrame:
        """Get model ranking by metric"""
        df = self.get_comparison_df()
        if metric in df.columns:
            return df.sort_values(metric, ascending=False)[[metric]]
        return df