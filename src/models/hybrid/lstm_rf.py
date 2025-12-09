"""
LSTM-RandomForest Hybrid Model

Bidirectional LSTM for temporal feature extraction + Random Forest for classification.
Target: 98.5%+ accuracy

Architecture:
- Bidirectional LSTM layers (128, 64 units)
- Dropout for regularization
- Random Forest classifier (500 estimators)
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LSTMRandomForestHybrid:
    """
    LSTM-RandomForest Hybrid Model for cybersecurity threat detection.
    
    Combines temporal feature extraction (LSTM) with ensemble classification (RF).
    """
    
    def __init__(self, input_shape, num_classes, class_weight=None):
        """
        Initialize LSTM-RF Hybrid model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_feature_extractor = self._build_lstm(input_shape)
        rf_class_weight = class_weight if class_weight is not None else 'balanced_subsample'
        self.rf_classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=1,
            class_weight=rf_class_weight
        )
        self.history = None
    
    def _build_lstm(self, input_shape):
        """
        Build Bidirectional LSTM feature extractor.
        
        Architecture:
        - Bidirectional LSTM: 128 units (return_sequences=True)
        - Dropout: 0.3
        - Bidirectional LSTM: 64 units
        - Dropout: 0.3
        
        Returns:
            LSTM model for feature extraction
        """
        model = Sequential([
            # First LSTM layer
            Bidirectional(
                LSTM(128, return_sequences=True, activation='tanh'),
                input_shape=input_shape
            ),
            Dropout(0.3),
            
            # Second LSTM layer
            Bidirectional(
                LSTM(64, activation='tanh')
            ),
            Dropout(0.3)
        ], name='LSTM_Feature_Extractor')
        
        return model
    
    def fit(self, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=2, class_weight=None):
        """
        Train the hybrid model.
        
        Process:
        1. Train LSTM for temporal feature extraction
        2. Extract features from training data
        3. Train Random Forest on extracted features
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of epochs for LSTM training
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print("="*60)
        print("  TRAINING LSTM-RANDOMFOREST HYBRID MODEL")
        print("="*60)
        
        # Step 1: Train LSTM feature extractor
        print("\n[1/3] Training LSTM feature extractor...")
        
        # Add temporary classification head for training
        temp_model = Sequential([
            self.lstm_feature_extractor,
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])
        
        temp_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]
        
        self.history = temp_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight
        )
        
        # Step 2: Extract temporal features
        print("\n[2/3] Extracting temporal features from training data...")
        X_train_features = self.lstm_feature_extractor.predict(X_train, verbose=1)
        
        print(f"Extracted features shape: {X_train_features.shape}")
        
        # Step 3: Train Random Forest
        print("\n[3/3] Training Random Forest classifier...")
        sample_weight = None
        if class_weight is not None:
            sample_weight = np.array([class_weight.get(label, 1.0) for label in y_train])

        self.rf_classifier.fit(X_train_features, y_train, sample_weight=sample_weight)
        
        print("\n✅ LSTM-RandomForest Hybrid training completed!")
        
        return self.history
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test data
            
        Returns:
            Predicted class labels
        """
        # Extract temporal features with LSTM
        X_test_features = self.lstm_feature_extractor.predict(X_test, verbose=0)
        
        # Predict with Random Forest
        predictions = self.rf_classifier.predict(X_test_features)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test: Test data
            
        Returns:
            Class probabilities
        """
        # Extract temporal features with LSTM
        X_test_features = self.lstm_feature_extractor.predict(X_test, verbose=0)
        
        # Predict probabilities with Random Forest
        probabilities = self.rf_classifier.predict_proba(X_test_features)
        
        return probabilities
    
    def get_feature_importance(self):
        """
        Get Random Forest feature importance for interpretability.
        
        Returns:
            Feature importance scores
        """
        if not hasattr(self.rf_classifier, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.rf_classifier.feature_importances_
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot top N important features.
        
        Args:
            top_n: Number of top features to display
        """
        import matplotlib.pyplot as plt
        
        importances = self.get_feature_importance()
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Random Forest Feature Importance')
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path):
        """
        Save LSTM and Random Forest separately.
        
        Args:
            path: Base path for saving (without extension)
        """
        # Save LSTM (H5 format)
        self.lstm_feature_extractor.save(f'{path}_lstm.h5')
        
        # Save Random Forest (pickle format)
        joblib.dump(self.rf_classifier, f'{path}_rf.pkl')
        
        print(f"✅ Model saved:")
        print(f"  - LSTM: {path}_lstm.h5")
        print(f"  - RF: {path}_rf.pkl")
    
    def load_model(self, path):
        """
        Load LSTM and Random Forest from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        from tensorflow.keras.models import load_model
        
        # Load LSTM
        self.lstm_feature_extractor = load_model(f'{path}_lstm.h5')
        
        # Load Random Forest
        self.rf_classifier = joblib.load(f'{path}_rf.pkl')
        
        print(f"✅ Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        print("="*60)
        print("  LSTM-RANDOMFOREST HYBRID MODEL SUMMARY")
        print("="*60)
        print("\n[LSTM Feature Extractor]")
        self.lstm_feature_extractor.summary()
        print("\n[Random Forest Classifier]")
        print(f"  Number of estimators: {self.rf_classifier.n_estimators}")
        print(f"  Max depth: {self.rf_classifier.max_depth}")
        print(f"  Min samples split: {self.rf_classifier.min_samples_split}")
        print("="*60)
