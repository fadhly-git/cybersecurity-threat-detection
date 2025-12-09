"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer

Explain individual predictions with local linear approximation.
"""

import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt


class LIMEExplainer:
    """
    LIME Explainer for local interpretability.
    
    Explains individual predictions using interpretable local models.
    """
    
    def __init__(self, model, X_train, feature_names, class_names):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            X_train: Training data (for feature distributions)
            feature_names: List of feature names
            class_names: List of class names
        """
        self.model = model
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
    
    def explain_instance(self, instance, num_features=10):
        """
        Explain a single prediction.
        
        Args:
            instance: Single sample to explain
            num_features: Number of top features to show
            
        Returns:
            LIME explanation object
        """
        # Create prediction function
        def predict_fn(x):
            preds = self.model.predict(x)
            if len(preds.shape) == 1:
                # Binary classification
                return np.column_stack([1 - preds, preds])
            return preds
        
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, output_path=None):
        """
        Visualize LIME explanation as bar chart.
        
        Args:
            explanation: LIME explanation object
            output_path: Optional path to save figure
        """
        fig = explanation.as_pyplot_figure()
        plt.title('LIME Explanation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Explanation saved to: {output_path}")
        
        plt.show()
    
    def generate_html_report(self, explanation, output_path):
        """
        Generate interactive HTML report.
        
        Args:
            explanation: LIME explanation object
            output_path: Path to save HTML file
        """
        html = explanation.as_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML report saved to: {output_path}")
    
    def explain_batch(self, X_test, y_test, num_samples=10, num_features=10):
        """
        Explain multiple predictions and aggregate results.
        
        Args:
            X_test: Test data
            y_test: Test labels
            num_samples: Number of samples to explain
            num_features: Number of features per explanation
            
        Returns:
            List of explanations
        """
        explanations = []
        
        print(f"\nGenerating LIME explanations for {num_samples} samples...")
        
        for i in range(min(num_samples, len(X_test))):
            print(f"  Explaining sample {i+1}/{num_samples}...", end='\r')
            exp = self.explain_instance(X_test[i], num_features)
            explanations.append(exp)
        
        print(f"\n✅ Generated {len(explanations)} explanations")
        
        return explanations
    
    def get_feature_importance_stats(self, explanations):
        """
        Aggregate feature importance across multiple explanations.
        
        Args:
            explanations: List of LIME explanations
            
        Returns:
            Dictionary with aggregated feature importance
        """
        feature_importance = {}
        
        for exp in explanations:
            for feature, weight in exp.as_list():
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(abs(weight))
        
        # Calculate statistics
        stats = {
            feature: {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(weights),
                'count': len(weights)
            }
            for feature, weights in feature_importance.items()
        }
        
        # Sort by mean importance
        stats = dict(sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True))
        
        return stats
    
    def plot_aggregated_importance(self, explanations, top_n=20):
        """
        Plot aggregated feature importance across multiple explanations.
        
        Args:
            explanations: List of LIME explanations
            top_n: Number of top features to display
        """
        stats = self.get_feature_importance_stats(explanations)
        
        # Get top N features
        top_features = list(stats.keys())[:top_n]
        means = [stats[f]['mean'] for f in top_features]
        stds = [stats[f]['std'] for f in top_features]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, means, xerr=stds, color='skyblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Absolute Importance')
        ax.set_title('Aggregated Feature Importance (LIME)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
