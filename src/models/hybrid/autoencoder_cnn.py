"""
Autoencoder-CNN Hybrid Model

Autoencoder for unsupervised feature learning + CNN classifier.
Supports semi-supervised learning and anomaly detection.
"""

import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class AutoencoderCNNHybrid:
    """
    Autoencoder-CNN Hybrid Model for cybersecurity threat detection.
    
    Two-stage approach:
    1. Unsupervised pretraining with autoencoder on normal traffic
    2. Supervised training with CNN classifier on labeled data
    """
    
    def __init__(self, input_shape, num_classes, encoding_dim=32):
        """
        Initialize Autoencoder-CNN Hybrid.
        
        Args:
            input_shape: Shape of input data (features, channels)
            num_classes: Number of output classes
            encoding_dim: Dimension of encoded representation
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        
        # Flatten input shape for autoencoder
        self.input_dim = input_shape[0] * input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        self.autoencoder = self._build_autoencoder(self.input_dim, encoding_dim)
        self.encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('encoding').output
        )
        self.cnn_classifier = self._build_cnn_classifier(encoding_dim, num_classes)
    
    def _build_autoencoder(self, input_dim, encoding_dim):
        """
        Build autoencoder for unsupervised feature learning.
        
        Architecture:
        Encoder: input_dim → 256 → 128 → 64 → encoding_dim
        Decoder: encoding_dim → 64 → 128 → 256 → input_dim
        
        Returns:
            Autoencoder model
        """
        # Encoder
        encoder_input = Input(shape=(input_dim,), name='autoencoder_input')
        
        x = Dense(256, activation='relu')(encoder_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        encoded = Dense(encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = Dense(64, activation='relu')(encoded)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        
        decoded = Dense(input_dim, activation='sigmoid')(x)
        
        # Autoencoder
        autoencoder = Model(
            inputs=encoder_input,
            outputs=decoded,
            name='Autoencoder'
        )
        
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def _build_cnn_classifier(self, encoding_dim, num_classes):
        """
        Build CNN classifier on encoded features.
        
        Args:
            encoding_dim: Dimension of encoded input
            num_classes: Number of output classes
            
        Returns:
            CNN classifier model
        """
        classifier_input = Input(shape=(encoding_dim,), name='classifier_input')
        
        # Reshape for Conv1D
        x = Lambda(lambda z: tf.expand_dims(z, axis=-1))(classifier_input)
        
        # CNN layers
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        
        # Classification head
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        output = Dense(num_classes, activation='softmax')(x)
        
        classifier = Model(
            inputs=classifier_input,
            outputs=output,
            name='CNN_Classifier'
        )
        
        classifier.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return classifier
    
    def pretrain_autoencoder(self, X_unlabeled, epochs=100, batch_size=256, verbose=2):
        """
        Unsupervised pretraining on normal traffic.
        
        Args:
            X_unlabeled: Unlabeled normal traffic data
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        print("="*60)
        print("  PRETRAINING AUTOENCODER (UNSUPERVISED)")
        print("="*60)
        
        # Flatten input if needed
        if len(X_unlabeled.shape) > 2:
            X_unlabeled_flat = X_unlabeled.reshape(X_unlabeled.shape[0], -1)
        else:
            X_unlabeled_flat = X_unlabeled
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                verbose=1
            )
        ]
        
        history = self.autoencoder.fit(
            X_unlabeled_flat, X_unlabeled_flat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✅ Autoencoder pretraining completed!")
        
        return history
    
    def train_classifier(self, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=2, class_weight=None):
        """
        Train CNN classifier on encoded features.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        print("="*60)
        print("  TRAINING CNN CLASSIFIER (SUPERVISED)")
        print("="*60)
        
        # Flatten input if needed
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Encode features
        print("Encoding features...")
        X_train_encoded = self.encoder.predict(X_train_flat, verbose=1)
        
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
        
        history = self.cnn_classifier.fit(
            X_train_encoded, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight
        )
        
        print("\n✅ CNN classifier training completed!")
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test data
            
        Returns:
            Predicted class labels
        """
        # Flatten and encode
        if len(X_test.shape) > 2:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        X_test_encoded = self.encoder.predict(X_test_flat, verbose=0)
        
        # Predict
        predictions_proba = self.cnn_classifier.predict(X_test_encoded, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test: Test data
            
        Returns:
            Class probabilities
        """
        # Flatten and encode
        if len(X_test.shape) > 2:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        X_test_encoded = self.encoder.predict(X_test_flat, verbose=0)
        
        return self.cnn_classifier.predict(X_test_encoded, verbose=0)
    
    def detect_anomaly(self, X):
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            X: Input data
            
        Returns:
            Reconstruction errors (higher = more anomalous)
        """
        # Flatten
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Reconstruct
        X_reconstructed = self.autoencoder.predict(X_flat, verbose=0)
        
        # Calculate reconstruction error (MSE)
        errors = np.mean(np.square(X_flat - X_reconstructed), axis=1)
        
        return errors
    
    def save_model(self, path):
        """
        Save autoencoder and classifier separately.
        
        Args:
            path: Base path for saving (without extension)
        """
        self.autoencoder.save(f'{path}_autoencoder.h5')
        self.cnn_classifier.save(f'{path}_classifier.h5')
        
        print(f"✅ Model saved:")
        print(f"  - Autoencoder: {path}_autoencoder.h5")
        print(f"  - Classifier: {path}_classifier.h5")
    
    def load_model(self, path):
        """
        Load autoencoder and classifier from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        from tensorflow.keras.models import load_model
        
        self.autoencoder = load_model(f'{path}_autoencoder.h5')
        self.encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('encoding').output
        )
        self.cnn_classifier = load_model(f'{path}_classifier.h5')
        
        print(f"✅ Model loaded from {path}")


# Import tensorflow for Lambda layer
import tensorflow as tf
