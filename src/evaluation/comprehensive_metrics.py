"""
Comprehensive Evaluation Metrics

Focus on minority classes performance and detailed metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns


class ComprehensiveEvaluator:
    """
    Comprehensive model evaluation with focus on minority classes.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Calculate comprehensive metrics with focus on minority classes.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            if len(y_pred.shape) > 1:
                y_pred_proba = y_pred
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred_proba = None
        else:
            raise ValueError("Model must have predict or predict_proba method")
        
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        results['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        results['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        results['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        results['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        results['per_class_metrics'] = {
            f'class_{i}': {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i])
            }
            for i in range(len(per_class_precision))
        }
        
        # Minority class focus (classes 3 and 4 from paper)
        minority_classes = [3, 4]
        minority_f1_scores = [per_class_f1[i] for i in minority_classes if i < len(per_class_f1)]
        results['minority_f1_avg'] = float(np.mean(minority_f1_scores)) if minority_f1_scores else 0.0
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        results['classification_report'] = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                results['roc_auc_ovr'] = roc_auc_score(
                    y_test, y_pred_proba,
                    multi_class='ovr',
                    average='weighted'
                )
                results['roc_auc_ovo'] = roc_auc_score(
                    y_test, y_pred_proba,
                    multi_class='ovo',
                    average='weighted'
                )
            except Exception as e:
                print(f"  Warning: Could not calculate ROC-AUC: {e}")
        
        # Cohen's Kappa
        results['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
        
        # Matthews Correlation Coefficient
        try:
            results['matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred)
        except Exception as e:
            print(f"  Warning: Could not calculate MCC: {e}")
            results['matthews_corrcoef'] = 0.0
        
        # Store results
        self.results[model_name] = results
        
        # Print summary
        self._print_summary(model_name, results)
        
        return results
    
    def _print_summary(self, model_name, results):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"  {model_name} - EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:          {results['accuracy']:.4f}")
        print(f"Precision (Macro): {results['precision_macro']:.4f}")
        print(f"Recall (Macro):    {results['recall_macro']:.4f}")
        print(f"F1 (Macro):        {results['f1_macro']:.4f}")
        print(f"F1 (Weighted):     {results['f1_weighted']:.4f}")
        print(f"Minority F1 Avg:   {results['minority_f1_avg']:.4f}")
        if 'roc_auc_ovr' in results:
            print(f"ROC-AUC (OvR):     {results['roc_auc_ovr']:.4f}")
        print(f"Cohen's Kappa:     {results['cohen_kappa']:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"  {class_name}: {metrics['f1']:.4f}")
        print(f"{'='*60}\n")
    
    def compare_models(self, metric='f1_macro'):
        """
        Compare all evaluated models based on metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Sorted list of (model_name, score)
        """
        if not self.results:
            print("No models evaluated yet!")
            return []
        
        comparison = {
            name: results.get(metric, 0.0)
            for name, results in self.results.items()
        }
        
        # Sort by metric
        sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON ({metric})")
        print(f"{'='*60}")
        for i, (name, score) in enumerate(sorted_models, 1):
            print(f"{i}. {name:30s} {score:.4f}")
        print(f"{'='*60}\n")
        
        return sorted_models
    
    def plot_comparison(self, metrics=['accuracy', 'f1_macro', 'f1_weighted', 'minority_f1_avg']):
        """
        Plot comparison bar chart for multiple metrics.
        
        Args:
            metrics: List of metrics to plot
        """
        if not self.results:
            print("No models evaluated yet!")
            return
        
        model_names = list(self.results.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            scores = [self.results[name].get(metric, 0) for name in model_names]
            
            axes[idx].barh(model_names, scores, color='skyblue')
            axes[idx].set_xlabel(metric.replace('_', ' ').title())
            axes[idx].set_xlim([0, 1])
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, score in enumerate(scores):
                axes[idx].text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name, class_names=None):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
            class_names: List of class names
        """
        if model_name not in self.results:
            print(f"Model '{model_name}' not found!")
            return
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm))
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
