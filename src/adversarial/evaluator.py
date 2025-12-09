"""
Robustness Evaluator

Comprehensive evaluation of model robustness against adversarial attacks.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .attacks import AdversarialAttacks


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation framework.
    """
    
    def __init__(self, model):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
        """
        self.model = model
        self.attacks = AdversarialAttacks()
    
    def evaluate_all_attacks(self, X_test, y_test, epsilon=0.1):
        """
        Test model against all 5 attacks.
        
        Args:
            X_test: Test data
            y_test: Test labels
            epsilon: Perturbation magnitude
            
        Returns:
            Dictionary with metrics for each attack
        """
        print("="*60)
        print("  ADVERSARIAL ROBUSTNESS EVALUATION")
        print("="*60)
        
        results = {}
        
        # Clean accuracy (baseline)
        print("\n[Baseline] Evaluating on clean data...")
        results['clean'] = self._evaluate_clean(X_test, y_test)
        
        # Sample subset for faster evaluation
        n_samples = min(1000, len(X_test))
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        # FGSM
        print("\n[1/5] Evaluating against FGSM attack...")
        try:
            X_fgsm = self.attacks.fgsm_attack(self.model, X_sample, y_sample, epsilon)
            results['fgsm'] = self._evaluate_adversarial(X_fgsm, y_sample)
        except Exception as e:
            print(f"  Error: {e}")
            results['fgsm'] = {'accuracy': 0.0}
        
        # PGD
        print("\n[2/5] Evaluating against PGD attack...")
        try:
            X_pgd = self.attacks.pgd_attack(self.model, X_sample, y_sample, epsilon)
            results['pgd'] = self._evaluate_adversarial(X_pgd, y_sample)
        except Exception as e:
            print(f"  Error: {e}")
            results['pgd'] = {'accuracy': 0.0}
        
        # C&W
        print("\n[3/5] Evaluating against C&W attack...")
        try:
            X_cw = self.attacks.carlini_wagner_attack(self.model, X_sample[:100], y_sample[:100])
            results['cw'] = self._evaluate_adversarial(X_cw, y_sample[:100])
        except Exception as e:
            print(f"  Error: {e}")
            results['cw'] = {'accuracy': 0.0}
        
        # DeepFool
        print("\n[4/5] Evaluating against DeepFool attack...")
        try:
            X_deepfool = self.attacks.deepfool_attack(self.model, X_sample[:100])
            results['deepfool'] = self._evaluate_adversarial(X_deepfool, y_sample[:100])
        except Exception as e:
            print(f"  Error: {e}")
            results['deepfool'] = {'accuracy': 0.0}
        
        # JSMA
        print("\n[5/5] Evaluating against JSMA attack...")
        try:
            X_jsma = self.attacks.jsma_attack(self.model, X_sample[:100], y_sample[:100])
            results['jsma'] = self._evaluate_adversarial(X_jsma, y_sample[:100])
        except Exception as e:
            print(f"  Error: {e}")
            results['jsma'] = {'accuracy': 0.0}
        
        # Calculate overall robustness score
        results['robustness_score'] = self._calculate_robustness_score(results)
        
        return results
    
    def _evaluate_clean(self, X_test, y_test):
        """Evaluate on clean data."""
        predictions = self.model.predict(X_test, verbose=0)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def _evaluate_adversarial(self, X_adv, y_test):
        """Evaluate on adversarial examples."""
        predictions = self.model.predict(X_adv, verbose=0)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def _calculate_robustness_score(self, results):
        """
        Calculate overall robustness score.
        
        Robustness = average accuracy across all attacks
        """
        attack_accuracies = [
            results.get('fgsm', {}).get('accuracy', 0),
            results.get('pgd', {}).get('accuracy', 0),
            results.get('cw', {}).get('accuracy', 0),
            results.get('deepfool', {}).get('accuracy', 0),
            results.get('jsma', {}).get('accuracy', 0)
        ]
        
        robustness = np.mean(attack_accuracies)
        
        print(f"\n{'='*60}")
        print(f"  Overall Robustness Score: {robustness:.4f}")
        print(f"{'='*60}")
        
        return robustness
    
    def visualize_perturbations(self, X_original, X_adversarial, n_samples=5):
        """
        Visualize original vs adversarial examples.
        
        Args:
            X_original: Original samples
            X_adversarial: Adversarial samples
            n_samples: Number of samples to visualize
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
        
        for i in range(min(n_samples, len(X_original))):
            # Original
            axes[i, 0].plot(X_original[i].flatten())
            axes[i, 0].set_title(f'Sample {i+1}: Original')
            axes[i, 0].set_ylabel('Value')
            
            # Adversarial
            axes[i, 1].plot(X_adversarial[i].flatten())
            axes[i, 1].set_title(f'Sample {i+1}: Adversarial')
            
            # Perturbation
            perturbation = X_adversarial[i] - X_original[i]
            axes[i, 2].plot(perturbation.flatten())
            axes[i, 2].set_title(f'Sample {i+1}: Perturbation')
            axes[i, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
