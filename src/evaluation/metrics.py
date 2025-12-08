"""Comprehensive evaluation metrics for model assessment.

This module provides extensive evaluation capabilities including accuracy,
precision, recall, F1-score, ROC-AUC, and detailed classification reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from pathlib import Path

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir


class ModelEvaluator(LoggerMixin):
    """Comprehensive model evaluation.
    
    Provides methods for calculating various performance metrics,
    generating confusion matrices, ROC curves, and comparison reports.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        super().__init__()
        self.logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            average: Averaging method for multiclass metrics
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-Score
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                
                if n_classes == 2:
                    # Binary classification
                    if len(y_pred_proba.shape) > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multiclass - one-vs-rest
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovr',
                        average='weighted'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovo',
                        average='weighted'
                    )
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
        
        return metrics
    
    def confusion_matrix_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Generate confusion matrix and per-class statistics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Tuple of (confusion_matrix, per_class_stats)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class statistics
        per_class_stats = {}
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_stats[f'class_{i}'] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return cm, per_class_stats
    
    def roc_curve_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        num_classes: int
    ) -> Dict:
        """Calculate ROC curve data for multiclass classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            num_classes: Number of classes
        
        Returns:
            Dictionary with ROC curve data for each class
        """
        roc_data = {}
        
        # Binarize labels for multiclass
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Calculate ROC curve for each class
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_data[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        # Micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        roc_data['micro_average'] = {
            'fpr': fpr_micro.tolist(),
            'tpr': tpr_micro.tolist(),
            'auc': roc_auc_micro
        }
        
        return roc_data
    
    def classification_report_detailed(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> str:
        """Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        
        Returns:
            Classification report string
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4
        )
        
        return report
    
    def compare_models(
        self,
        results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Compare multiple models side-by-side.
        
        Args:
            results: Dictionary mapping model names to their metrics
        
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy (descending)
        if 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)
        
        return df
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str = 'results/evaluation_report.md'
    ) -> None:
        """Generate markdown evaluation report.
        
        Args:
            results: Dictionary with all evaluation results
            output_path: Path to save report
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Model comparison table
            if 'comparison' in results:
                f.write("## Model Comparison\n\n")
                comparison_df = results['comparison']
                f.write(comparison_df.to_markdown(index=False))
                f.write("\n\n")
            
            # Detailed metrics for each model
            f.write("## Detailed Metrics\n\n")
            for model_name, metrics in results.items():
                if model_name == 'comparison':
                    continue
                
                f.write(f"### {model_name}\n\n")
                
                if isinstance(metrics, dict):
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"| {metric} | {value:.4f} |\n")
                        else:
                            f.write(f"| {metric} | {value} |\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a single model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: List of class names
        
        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info(f"\nEvaluating {model_name}...")
        
        results = {}
        
        # Basic metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        results['metrics'] = metrics
        
        # Confusion matrix
        cm, per_class = self.confusion_matrix_analysis(y_true, y_pred)
        results['confusion_matrix'] = cm
        results['per_class_stats'] = per_class
        
        # ROC curve data
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
            num_classes = y_pred_proba.shape[1]
            roc_data = self.roc_curve_analysis(y_true, y_pred_proba, num_classes)
            results['roc_data'] = roc_data
        
        # Classification report
        report = self.classification_report_detailed(y_true, y_pred, class_names)
        results['classification_report'] = report
        
        # Log key metrics
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        self.logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        self.logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_ovr' in metrics:
            self.logger.info(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
        return results
