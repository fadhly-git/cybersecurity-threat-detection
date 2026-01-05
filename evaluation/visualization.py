"""
visualization.py - Result visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ResultVisualizer:
    """Visualize model results"""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else Config.RESULTS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = "Confusion Matrix",
                              figsize: tuple = (10, 8),
                              save_name: Optional[str] = None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_roc_curves(self,
                        models_data: Dict[str, Dict],
                        figsize: tuple = (10, 8),
                        save_name: Optional[str] = None):
        """Plot ROC curves for multiple models"""
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
        
        for (name, data), color in zip(models_data.items(), colors):
            y_true = data['y_true']
            y_proba = data['y_proba']
            
            if y_proba is not None and y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_model_comparison(self,
                              comparison_df: pd.DataFrame,
                              metrics: Optional[List[str]] = None,
                              figsize: tuple = (12, 6),
                              save_name: Optional[str] = None):
        """Plot model comparison bar chart"""
        
        if metrics is None:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        df_plot = comparison_df[available_metrics]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(df_plot.index))
        width = 0.8 / len(available_metrics)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(available_metrics)))
        
        for i, (metric, color) in enumerate(zip(available_metrics, colors)):
            offset = (i - len(available_metrics)/2 + 0.5) * width
            ax.bar(x + offset, df_plot[metric], width, label=metric, color=color)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot.index, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        
        # Add value labels
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2 + 0.5) * width
            for j, v in enumerate(df_plot[metric]):
                ax.text(j + offset, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_feature_importance(self,
                                feature_names: List[str],
                                importances: np.ndarray,
                                top_k: int = 20,
                                title: str = "Feature Importance",
                                figsize: tuple = (10, 8),
                                save_name: Optional[str] = None):
        """Plot feature importance"""
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=figsize)
        
        y_pos = np.arange(top_k)
        plt.barh(y_pos, importances[indices][::-1], color='steelblue')
        plt.yticks(y_pos, [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_training_history(self,
                              history: Dict,
                              metrics: List[str] = ['loss', 'accuracy'],
                              figsize: tuple = (12, 4),
                              save_name: Optional[str] = None):
        """Plot training history"""
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}')
                if f'val_{metric}' in history:
                    ax.plot(history[f'val_{metric}'], label=f'Val {metric}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} over Epochs')
                ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_class_distribution(self,
                                y: np.ndarray,
                                class_names: Optional[List[str]] = None,
                                title: str = "Class Distribution",
                                figsize: tuple = (10, 6),
                                save_name: Optional[str] = None):
        """Plot class distribution"""
        
        unique, counts = np.unique(y, return_counts=True)
        
        if class_names is None:
            class_names = [str(c) for c in unique]
        
        plt.figure(figsize=figsize)
        
        bars = plt.bar(class_names, counts, color='steelblue')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()