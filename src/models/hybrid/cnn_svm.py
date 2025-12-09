"""
CNN-SVM Hybrid Model

CNN for feature extraction + SVM for classification.
Target: 98.5%+ accuracy

Architecture:
- CNN Feature Extractor: Conv1D layers (64, 128, 256 filters) with BatchNormalization
- SVM Classifier: RBF kernel with probability estimates
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D, 
    GlobalAveragePooling1D, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CNNSVMHybrid:
    """
    CNN-SVM Hybrid Model for cybersecurity threat detection.
    
    Combines deep feature extraction (CNN) with classical classification (SVM).
    """
    
    def __init__(self, input_shape, num_classes, class_weight=None):
        """
        Initialize CNN-SVM Hybrid model.
        
        Args:
            input_shape: Shape of input data (features, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_feature_extractor = self._build_cnn(input_shape)
        self.svm_classifier = SVC(
            kernel='rbf', 
            probability=True,
            C=1.0,
            gamma='scale',
            random_state=42,
            verbose=True,
            class_weight=class_weight
        )
        self.history = None
    
    def _build_cnn(self, input_shape):
        """
        Build CNN feature extractor.
        
        Architecture:
        - Conv1D: 64 filters, kernel=3
        - BatchNormalization + MaxPooling
        - Conv1D: 128 filters, kernel=3
        - BatchNormalization + MaxPooling
        - Conv1D: 256 filters, kernel=3
        - BatchNormalization
        - GlobalAveragePooling1D
        
        Returns:
            CNN model without classification head
        """
        model = Sequential([
            # Block 1
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Block 2
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Block 3
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D()
        ], name='CNN_Feature_Extractor')
        
        return model
    
    def fit(self, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=2, class_weight=None):
        """
        Train the hybrid model.
        
        Process:
        1. Train CNN for feature extraction
        2. Extract features from training data
        3. Train SVM on extracted features
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of epochs for CNN training
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print("="*60)
        print("  TRAINING CNN-SVM HYBRID MODEL")
        print("="*60)
        
        # Step 1: Train CNN feature extractor
        print("\n[1/3] Training CNN feature extractor...")
        
        # Add temporary classification head for training
        temp_model = Sequential([
            self.cnn_feature_extractor,
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
        
        # Step 2: Extract features
        print("\n[2/3] Extracting features from training data...")
        X_train_features = self.cnn_feature_extractor.predict(X_train, verbose=1)
        
        print(f"Extracted features shape: {X_train_features.shape}")
        
        # Step 3: Train SVM
        print("\n[3/3] Training SVM classifier...")
        self.svm_classifier.fit(X_train_features, y_train)
        
        print("\n✅ CNN-SVM Hybrid training completed!")
        
        return self.history
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test data
            
        Returns:
            Predicted class labels
        """
        # Extract features with CNN
        X_test_features = self.cnn_feature_extractor.predict(X_test, verbose=0)
        
        # Predict with SVM
        predictions = self.svm_classifier.predict(X_test_features)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test: Test data
            
        Returns:
            Class probabilities
        """
        # Extract features with CNN
        X_test_features = self.cnn_feature_extractor.predict(X_test, verbose=0)
        
        # Predict probabilities with SVM
        probabilities = self.svm_classifier.predict_proba(X_test_features)
        
        return probabilities
    
    def save_model(self, path):
        """
        Save CNN and SVM separately.
        
        Args:
            path: Base path for saving (without extension)
        """
        # Save CNN (H5 format)
        self.cnn_feature_extractor.save(f'{path}_cnn.h5')
        
        # Save SVM (pickle format)
        joblib.dump(self.svm_classifier, f'{path}_svm.pkl')
        
        print(f"✅ Model saved:")
        print(f"  - CNN: {path}_cnn.h5")
        print(f"  - SVM: {path}_svm.pkl")
    
    def load_model(self, path):
        """
        Load CNN and SVM from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        from tensorflow.keras.models import load_model
        
        # Load CNN
        self.cnn_feature_extractor = load_model(f'{path}_cnn.h5')
        
        # Load SVM
        self.svm_classifier = joblib.load(f'{path}_svm.pkl')
        
        print(f"✅ Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        print("="*60)
        print("  CNN-SVM HYBRID MODEL SUMMARY")
        print("="*60)
        print("\n[CNN Feature Extractor]")
        self.cnn_feature_extractor.summary()
        print("\n[SVM Classifier]")
        print(f"  Kernel: {self.svm_classifier.kernel}")
        print(f"  C: {self.svm_classifier.C}")
        print(f"  Gamma: {self.svm_classifier.gamma}")
        print("="*60)
