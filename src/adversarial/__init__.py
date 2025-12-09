"""
Adversarial Attacks Module

Implementation of 5 adversarial attack methods to test model robustness:
1. FGSM (Fast Gradient Sign Method)
2. PGD (Projected Gradient Descent)
3. C&W (Carlini & Wagner)
4. DeepFool
5. JSMA (Jacobian-based Saliency Map Attack)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class AdversarialAttacks:
    """
    Collection of adversarial attack methods for testing model robustness.
    """
    
    @staticmethod
    def fgsm_attack(model, X, y, epsilon=0.1):
        """
        Fast Gradient Sign Method (FGSM).
        
        Args:
            model: Trained model
            X: Input samples
            y: True labels
            epsilon: Perturbation magnitude
            
        Returns:
            X_adv: Adversarial examples
        """
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.int64)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
        
        # Get gradient
        gradient = tape.gradient(loss, X_tensor)
        
        # Create perturbation
        perturbation = epsilon * tf.sign(gradient)
        
        # Add perturbation
        X_adv = X_tensor + perturbation
        X_adv = tf.clip_by_value(X_adv, 0, 1)  # Clip to valid range
        
        return X_adv.numpy()
    
    @staticmethod
    def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, iterations=40):
        """
        Projected Gradient Descent (PGD).
        Iterative version of FGSM - stronger attack.
        
        Args:
            model: Trained model
            X: Input samples
            y: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            iterations: Number of iterations
            
        Returns:
            X_adv: Adversarial examples
        """
        X_adv = tf.Variable(X, dtype=tf.float32)
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(X_adv)
                predictions = model(X_adv)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            
            gradient = tape.gradient(loss, X_adv)
            
            # Update adversarial example
            X_adv.assign(X_adv + alpha * tf.sign(gradient))
            
            # Project back to epsilon ball
            perturbation = tf.clip_by_value(X_adv - X, -epsilon, epsilon)
            X_adv.assign(X + perturbation)
            X_adv.assign(tf.clip_by_value(X_adv, 0, 1))
        
        return X_adv.numpy()
    
    @staticmethod
    def carlini_wagner_attack(model, X, y, c=1.0, kappa=0, iterations=100, learning_rate=0.01):
        """
        Carlini & Wagner (C&W) Attack.
        Optimization-based attack.
        
        Args:
            model: Trained model
            X: Input samples
            y: True labels
            c: Balance parameter
            kappa: Confidence parameter
            iterations: Number of optimization steps
            learning_rate: Learning rate
            
        Returns:
            X_adv: Adversarial examples
        """
        # Initialize perturbation
        delta = tf.Variable(tf.zeros_like(X), dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                X_adv = X + delta
                X_adv = tf.clip_by_value(X_adv, 0, 1)
                
                predictions = model(X_adv)
                
                # Get top 2 predictions
                top2 = tf.nn.top_k(predictions, k=2)
                
                # Loss: minimize perturbation + maximize misclassification
                l2_loss = tf.reduce_sum(tf.square(delta))
                
                # Classification loss
                real_class = tf.reduce_max(predictions * tf.one_hot(y, predictions.shape[-1]), axis=1)
                other_class = tf.reduce_max(predictions * (1 - tf.one_hot(y, predictions.shape[-1])), axis=1)
                class_loss = tf.maximum(0.0, real_class - other_class + kappa)
                
                total_loss = l2_loss + c * tf.reduce_sum(class_loss)
            
            gradients = tape.gradient(total_loss, delta)
            optimizer.apply_gradients([(gradients, delta)])
        
        X_adv = tf.clip_by_value(X + delta, 0, 1)
        return X_adv.numpy()
    
    @staticmethod
    def deepfool_attack(model, X, overshoot=0.02, max_iterations=50):
        """
        DeepFool Attack.
        Minimal perturbation for misclassification.
        
        Args:
            model: Trained model
            X: Input samples
            overshoot: Overshoot parameter
            max_iterations: Maximum iterations
            
        Returns:
            X_adv: Adversarial examples
        """
        X_adv = X.copy()
        
        for sample_idx in range(len(X)):
            x = X[sample_idx:sample_idx+1]
            x_adv = x.copy()
            
            for iteration in range(max_iterations):
                x_tensor = tf.Variable(x_adv, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    predictions = model(x_tensor)
                    pred_class = tf.argmax(predictions, axis=1)[0]
                    pred_value = predictions[0, pred_class]
                
                gradient = tape.gradient(pred_value, x_tensor)
                
                # Find minimal perturbation
                pert = overshoot * gradient / (tf.norm(gradient) + 1e-10)
                x_adv = x_adv + pert.numpy()
                
                # Check if misclassified
                new_pred = tf.argmax(model(x_adv), axis=1)[0]
                if new_pred != pred_class:
                    break
            
            X_adv[sample_idx] = x_adv[0]
        
        return X_adv
    
    @staticmethod
    def jsma_attack(model, X, y, theta=1.0, gamma=0.1):
        """
        Jacobian-based Saliency Map Attack (JSMA).
        Targeted attack using saliency map.
        
        Args:
            model: Trained model
            X: Input samples
            y: True labels
            theta: Perturbation magnitude
            gamma: Maximum distortion
            
        Returns:
            X_adv: Adversarial examples
        """
        X_adv = X.copy()
        num_features = X.shape[1]
        
        for sample_idx in range(len(X)):
            x = X[sample_idx:sample_idx+1]
            x_adv = x.copy()
            
            # Target class (different from true class)
            target = (y[sample_idx] + 1) % model.output_shape[-1]
            
            modified_features = 0
            max_modifications = int(gamma * num_features)
            
            while modified_features < max_modifications:
                x_tensor = tf.Variable(x_adv, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    predictions = model(x_tensor)
                    target_score = predictions[0, target]
                
                # Calculate saliency
                gradient = tape.gradient(target_score, x_tensor)
                saliency = tf.abs(gradient[0])
                
                # Find most salient feature
                feature_idx = tf.argmax(saliency)
                
                # Perturb feature
                x_adv[0, feature_idx] += theta
                x_adv = np.clip(x_adv, 0, 1)
                
                modified_features += 1
                
                # Check if target achieved
                if tf.argmax(model(x_adv), axis=1)[0] == target:
                    break
            
            X_adv[sample_idx] = x_adv[0]
        
        return X_adv
