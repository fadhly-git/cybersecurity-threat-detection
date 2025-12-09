"""
SHAP (SHapley Additive exPlanations) Explainer

Model interpretability using SHAP values.
Explains contribution of each feature to predictions.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:
    """
    SHAP Explainer for model interpretability.
    
    Supports deep learning, tree-based, and general models.
    """
    
    def __init__(self, model, X_background, model_type='deep'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (subset of training data)
            model_type: 'deep' (DL), 'tree' (RF/XGBoost), or 'kernel' (general)
        """
        self.model = model
        self.model_type = model_type
        
        # Initialize appropriate explainer
        if model_type == 'deep':
            self.explainer = shap.DeepExplainer(model, X_background)
        elif model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:
            # Kernel explainer for general models
            self.explainer = shap.KernelExplainer(
                lambda x: model.predict(x),
                X_background
            )
    
    def explain_predictions(self, X_test):
        """
        Calculate SHAP values for test samples.
        
        Args:
            X_test: Test samples
            
        Returns:
            SHAP values for each feature and sample
        """
        print("Calculating SHAP values...")
        shap_values = self.explainer.shap_values(X_test)
        print(f"✅ SHAP values calculated: {np.array(shap_values).shape}")
        
        return shap_values
    
    def plot_feature_importance(self, shap_values, feature_names=None, top_n=20):
        """
        Bar plot showing top N important features (global).
        
        Args:
            shap_values: SHAP values
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            plot_type='bar',
            max_display=top_n,
            show=False
        )
        plt.title('Top Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_summary(self, shap_values, X_test, feature_names=None):
        """
        Summary plot showing feature importance and distribution.
        
        Args:
            shap_values: SHAP values
            X_test: Test data
            feature_names: List of feature names
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall(self, shap_values, X_test, sample_idx, feature_names=None):
        """
        Waterfall plot for single prediction.
        Shows how each feature contributes.
        
        Args:
            shap_values: SHAP values
            X_test: Test data
            sample_idx: Index of sample to explain
            feature_names: List of feature names
        """
        # Convert to SHAP Explanation object
        if isinstance(shap_values, list):
            # Multi-class: use first class
            values = shap_values[0][sample_idx]
        else:
            values = shap_values[sample_idx]
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=self.explainer.expected_value,
                data=X_test[sample_idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title(f'Waterfall Plot - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_force(self, shap_values, X_test, sample_idx, feature_names=None):
        """
        Force plot for visualizing feature contributions.
        
        Args:
            shap_values: SHAP values
            X_test: Test data
            sample_idx: Index of sample to explain
            feature_names: List of feature names
        """
        if isinstance(shap_values, list):
            values = shap_values[0][sample_idx]
        else:
            values = shap_values[sample_idx]
        
        shap.force_plot(
            self.explainer.expected_value,
            values,
            X_test[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'Force Plot - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_dependence(self, shap_values, X_test, feature_idx, feature_names=None):
        """
        Dependence plot: relationship between feature value and SHAP value.
        
        Args:
            shap_values: SHAP values
            X_test: Test data
            feature_idx: Index of feature to analyze
            feature_names: List of feature names
        """
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        feature_name = feature_names[feature_idx] if feature_names else f'Feature {feature_idx}'
        plt.title(f'Dependence Plot - {feature_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
