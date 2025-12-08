"""Visualization utilities for model results and analysis.

This module provides comprehensive visualization capabilities including
confusion matrices, ROC curves, feature importance plots, and interactive dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir


class Visualizer(LoggerMixin):
    """Visualization utilities for cybersecurity threat detection.
    
    Provides methods for creating various plots and interactive visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize Visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        super().__init__()
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set seaborn style
        sns.set_palette("husl")
        
        self.logger.info("Visualizer initialized")
    
    def plot_preprocessing_stats(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot data distribution before/after preprocessing.
        
        Args:
            df_before: DataFrame before preprocessing
            df_after: DataFrame after preprocessing
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Preprocessing Comparison', fontsize=16, fontweight='bold')
        
        # Shape comparison
        ax = axes[0, 0]
        shapes = [df_before.shape[0], df_after.shape[0]]
        ax.bar(['Before', 'After'], shapes, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Number of Rows')
        ax.set_title('Dataset Size')
        for i, v in enumerate(shapes):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        # Missing values comparison
        ax = axes[0, 1]
        missing_before = df_before.isnull().sum().sum()
        missing_after = df_after.isnull().sum().sum()
        ax.bar(['Before', 'After'], [missing_before, missing_after], color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Missing Values')
        ax.set_title('Missing Values')
        for i, v in enumerate([missing_before, missing_after]):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        # Data types comparison
        ax = axes[1, 0]
        dtypes_before = df_before.dtypes.value_counts()
        dtypes_after = df_after.dtypes.value_counts()
        x = np.arange(len(dtypes_before))
        width = 0.35
        ax.bar(x - width/2, dtypes_before.values, width, label='Before', color='#FF6B6B')
        ax.bar(x + width/2, dtypes_after.values, width, label='After', color='#4ECDC4')
        ax.set_ylabel('Count')
        ax.set_title('Data Types Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels([str(dt) for dt in dtypes_before.index], rotation=45)
        ax.legend()
        
        # Memory usage comparison
        ax = axes[1, 1]
        mem_before = df_before.memory_usage(deep=True).sum() / 1024**2  # MB
        mem_after = df_after.memory_usage(deep=True).sum() / 1024**2  # MB
        ax.bar(['Before', 'After'], [mem_before, mem_after], color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage')
        for i, v in enumerate([mem_before, mem_after]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Preprocessing stats plot saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance.
        
        Args:
            importances: Feature importance values
            feature_names: List of feature names
            top_n: Number of top features to show
            save_path: Path to save the figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_importances, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        normalize: bool = False
    ) -> None:
        """Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the figure
            normalize: Whether to normalize the matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(
        self,
        models_roc_data: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curves for multiple models.
        
        Args:
            models_roc_data: Dictionary mapping model names to ROC data
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_roc_data)))
        
        for (model_name, roc_data), color in zip(models_roc_data.items(), colors):
            if 'micro_average' in roc_data:
                fpr = roc_data['micro_average']['fpr']
                tpr = roc_data['micro_average']['tpr']
                auc_score = roc_data['micro_average']['auc']
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curves plot saved to {save_path}")
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Any,
        save_path: Optional[str] = None
    ) -> None:
        """Plot training/validation loss and accuracy.
        
        Args:
            history: Keras training history object
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison
            metrics: List of metrics to plot
            save_path: Path to save the figure
        """
        # Filter metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            self.logger.warning("No valid metrics found in comparison dataframe")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.8 / len(available_metrics)
        
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2) * width + width/2
            ax.bar(x + offset, comparison_df[metric], width, 
                   label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.close()
    
    def create_dashboard(
        self,
        all_results: Dict[str, Any],
        save_path: str = 'results/dashboard.html'
    ) -> None:
        """Create interactive Plotly dashboard.
        
        Args:
            all_results: Dictionary with all results
            save_path: Path to save HTML dashboard
        """
        self.logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'F1-Score Comparison',
                          'Precision vs Recall', 'ROC-AUC Scores'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Extract model names and metrics
        if 'comparison' in all_results:
            df = all_results['comparison']
            models = df['model'].tolist()
            
            # Accuracy comparison
            if 'accuracy' in df.columns:
                fig.add_trace(
                    go.Bar(x=models, y=df['accuracy'], name='Accuracy',
                          marker_color='lightblue'),
                    row=1, col=1
                )
            
            # F1-Score comparison
            if 'f1_weighted' in df.columns:
                fig.add_trace(
                    go.Bar(x=models, y=df['f1_weighted'], name='F1-Score',
                          marker_color='lightgreen'),
                    row=1, col=2
                )
            
            # Precision vs Recall scatter
            if 'precision_weighted' in df.columns and 'recall_weighted' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['precision_weighted'], y=df['recall_weighted'],
                             mode='markers+text', text=models,
                             textposition='top center',
                             marker=dict(size=12, color=range(len(models)),
                                       colorscale='Viridis'),
                             name='Models'),
                    row=2, col=1
                )
            
            # ROC-AUC scores
            roc_col = 'roc_auc_ovr' if 'roc_auc_ovr' in df.columns else 'roc_auc' if 'roc_auc' in df.columns else None
            if roc_col:
                fig.add_trace(
                    go.Bar(x=models, y=df[roc_col], name='ROC-AUC',
                          marker_color='coral'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Cybersecurity Threat Detection - Model Performance Dashboard",
            title_font_size=20,
            showlegend=False,
            height=800
        )
        
        # Save dashboard
        ensure_dir(Path(save_path).parent)
        fig.write_html(save_path)
        
        self.logger.info(f"Interactive dashboard saved to {save_path}")
