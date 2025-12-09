"""
Attention-LSTM-DNN Model

Multi-head attention + Bidirectional LSTM + Deep Neural Network.
Focus on temporal patterns with attention mechanism.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, MultiHeadAttention, LayerNormalization,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class AttentionLSTMDNN:
    """
    Attention-LSTM-DNN Model for cybersecurity threat detection.
    
    Combines:
    - Multi-head attention for feature importance
    - Bidirectional LSTM for temporal modeling
    - Deep neural network for classification
    """
    
    def __init__(self, input_shape, num_classes, num_heads=4):
        """
        Initialize Attention-LSTM-DNN.
        
        Args:
            input_shape: Shape of input data (features, channels)
            num_classes: Number of output classes
            num_heads: Number of attention heads
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.model = self._build_model(input_shape, num_classes, num_heads)
    
    def _build_model(self, input_shape, num_classes, num_heads):
        """
        Build Attention-LSTM-DNN architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input
        input_layer = Input(shape=input_shape, name='input')
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=64,
            name='multi_head_attention'
        )(input_layer, input_layer)
        attention_output = LayerNormalization(name='attention_norm')(attention_output)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh'),
            name='bilstm1'
        )(attention_output)
        lstm_out = Dropout(0.3, name='lstm_dropout1')(lstm_out)
        
        lstm_out = Bidirectional(
            LSTM(64, activation='tanh'),
            name='bilstm2'
        )(lstm_out)
        lstm_out = Dropout(0.3, name='lstm_dropout2')(lstm_out)
        
        # Deep Neural Network
        x = Dense(256, activation='relu', name='dnn1')(lstm_out)
        x = LayerNormalization(name='dnn_norm1')(x)
        x = Dropout(0.4, name='dnn_dropout1')(x)
        
        x = Dense(128, activation='relu', name='dnn2')(x)
        x = Dropout(0.3, name='dnn_dropout2')(x)
        
        # Output
        output = Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='Attention_LSTM_DNN')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully!")
    
    def fit(self, X_train, y_train, validation_data=None, epochs=50, batch_size=256, verbose=2, class_weight=None):
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        print("="*60)
        print("  TRAINING ATTENTION-LSTM-DNN MODEL")
        print("="*60)
        
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
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight
        )
        
        print("\n✅ Attention-LSTM-DNN training completed!")
        
        return history
    
    def predict(self, X_test):
        """Make predictions."""
        predictions_proba = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions_proba, axis=1)
    
    def predict_proba(self, X_test):
        """Predict class probabilities."""
        return self.model.predict(X_test, verbose=0)
    
    def get_attention_weights(self, X):
        """
        Extract attention weights for visualization.
        
        Args:
            X: Input data
            
        Returns:
            Attention weights
        """
        # Create model that outputs attention weights
        attention_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('multi_head_attention').output
        )
        
        attention_weights = attention_model.predict(X, verbose=0)
        
        return attention_weights
    
    def save_model(self, path):
        """Save model to disk."""
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk."""
        from tensorflow.keras.models import load_model
        
        self.model = load_model(path)
        print(f"✅ Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        print("="*60)
        print("  ATTENTION-LSTM-DNN MODEL SUMMARY")
        print("="*60)
        self.model.summary()
        print("="*60)
