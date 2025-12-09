"""
Adversarial Defense Mechanisms

Defense methods against adversarial attacks:
1. Adversarial Training
2. Defensive Distillation
3. Input Transformation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model
from .attacks import AdversarialAttacks


class AdversarialDefenses:
    """
    Defense mechanisms against adversarial attacks.
    """
    
    @staticmethod
    def adversarial_training(model, X_train, y_train, attack_method='fgsm',
                           epsilon=0.1, epochs=50, batch_size=256):
        """
        Train model with adversarial examples.
        Improves robustness through adversarial augmentation.
        
        Args:
            model: Model to train
            X_train: Training data
            y_train: Training labels
            attack_method: Attack to use ('fgsm' or 'pgd')
            epsilon: Perturbation magnitude
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Trained hardened model
        """
        print("="*60)
        print(f"  ADVERSARIAL TRAINING ({attack_method.upper()})")
        print("="*60)
        
        attacks = AdversarialAttacks()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Generate adversarial examples
                if attack_method == 'fgsm':
                    X_adv = attacks.fgsm_attack(model, X_batch, y_batch, epsilon)
                elif attack_method == 'pgd':
                    X_adv = attacks.pgd_attack(model, X_batch, y_batch, epsilon)
                else:
                    raise ValueError(f"Unknown attack method: {attack_method}")
                
                # Mix clean and adversarial (50-50)
                X_mixed = np.vstack([X_batch, X_adv])
                y_mixed = np.hstack([y_batch, y_batch])
                
                # Train on mixed data
                model.train_on_batch(X_mixed, y_mixed)
            
            # Evaluate on clean data
            if epoch % 10 == 0:
                loss, acc = model.evaluate(X_train[:1000], y_train[:1000], verbose=0)
                print(f"  Clean accuracy: {acc:.4f}")
        
        print("\n✅ Adversarial training completed!")
        return model
    
    @staticmethod
    def defensive_distillation(teacher_model, X_train, y_train,
                              temperature=10, epochs=50, batch_size=256):
        """
        Defensive Distillation.
        Train student model with soft labels from teacher.
        
        Args:
            teacher_model: Pretrained teacher model
            X_train: Training data
            y_train: Training labels
            temperature: Temperature for softening predictions
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Hardened student model
        """
        print("="*60)
        print("  DEFENSIVE DISTILLATION")
        print("="*60)
        
        # Clone teacher architecture
        student_model = clone_model(teacher_model)
        student_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get soft labels from teacher
        print("\nGenerating soft labels from teacher...")
        soft_labels = teacher_model.predict(X_train) / temperature
        soft_labels = tf.nn.softmax(soft_labels).numpy()
        
        # Train student
        print("\nTraining student model...")
        student_model.fit(
            X_train, soft_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\n✅ Defensive distillation completed!")
        return student_model
    
    @staticmethod
    def input_transformation(X, method='median_filter'):
        """
        Input transformation to remove adversarial perturbations.
        
        Methods: median_filter, bit_depth_reduction
        
        Args:
            X: Input data
            method: Transformation method
            
        Returns:
            Transformed input
        """
        if method == 'median_filter':
            from scipy.ndimage import median_filter
            X_transformed = np.array([median_filter(x, size=3) for x in X])
            
        elif method == 'bit_depth_reduction':
            # Reduce to 4-bit depth then restore
            X_transformed = np.round(X * 15) / 15
            
        elif method == 'gaussian_noise':
            noise = np.random.normal(0, 0.01, X.shape)
            X_transformed = X + noise
            X_transformed = np.clip(X_transformed, 0, 1)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return X_transformed
