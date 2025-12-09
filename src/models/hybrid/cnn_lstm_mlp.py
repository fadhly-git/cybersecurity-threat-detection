"""
CNN-LSTM-MLP Ensemble Model

Ensemble with 3 parallel branches:
- CNN branch for spatial features
- LSTM branch for temporal features  
- MLP branch for tabular features

Paper target: 98%+ accuracy
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D,
    Bidirectional, LSTM, Dense, Dropout, Flatten, Concatenate, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CNNLSTMMLPEnsemble:
    """
    CNN-LSTM-MLP Ensemble Model for cybersecurity threat detection.
    
    Combines three parallel branches:
    1. CNN: Spatial/local pattern extraction
    2. LSTM: Temporal sequence modeling
    3. MLP: Tabular feature processing
    
    Target: 98%+ accuracy on CICIDS2017 dataset
    """
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize CNN-LSTM-MLP Ensemble.
        
        Args:
            input_shape: Shape of input data (features, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_ensemble(input_shape, num_classes)
    
    def _build_ensemble(self, input_shape, num_classes):
        """
        Build the ensemble architecture with 3 parallel branches.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = Input(shape=input_shape, name='input')
        
        # ============ CNN Branch ============
        cnn_branch = Conv1D(64, 3, activation='relu', padding='same', name='cnn_conv1')(input_layer)
        cnn_branch = BatchNormalization(name='cnn_bn1')(cnn_branch)
        cnn_branch = MaxPooling1D(2, name='cnn_pool1')(cnn_branch)
        cnn_branch = Dropout(0.2, name='cnn_dropout1')(cnn_branch)
        
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same', name='cnn_conv2')(cnn_branch)
        cnn_branch = BatchNormalization(name='cnn_bn2')(cnn_branch)
        cnn_branch = MaxPooling1D(2, name='cnn_pool2')(cnn_branch)
        cnn_branch = Dropout(0.3, name='cnn_dropout2')(cnn_branch)
        
        cnn_branch = Conv1D(256, 3, activation='relu', padding='same', name='cnn_conv3')(cnn_branch)
        cnn_branch = BatchNormalization(name='cnn_bn3')(cnn_branch)
        cnn_branch = GlobalAveragePooling1D(name='cnn_gap')(cnn_branch)
        
        # ============ LSTM Branch ============
        lstm_branch = Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh'),
            name='lstm_bilstm1'
        )(input_layer)
        lstm_branch = Dropout(0.3, name='lstm_dropout1')(lstm_branch)

        # Attention highlights informative timesteps before final aggregation
        lstm_branch = Attention(name='lstm_attention')([lstm_branch, lstm_branch])
        
        lstm_branch = Bidirectional(
            LSTM(64, activation='tanh'),
            name='lstm_bilstm2'
        )(lstm_branch)
        lstm_branch = Dropout(0.3, name='lstm_dropout2')(lstm_branch)
        
        # ============ MLP Branch ============
        mlp_branch = Flatten(name='mlp_flatten')(input_layer)
        mlp_branch = Dense(256, activation='relu', name='mlp_dense1')(mlp_branch)
        mlp_branch = BatchNormalization(name='mlp_bn1')(mlp_branch)
        mlp_branch = Dropout(0.4, name='mlp_dropout1')(mlp_branch)
        mlp_branch = Dense(128, activation='relu', name='mlp_dense2')(mlp_branch)
        mlp_branch = Dropout(0.3, name='mlp_dropout2')(mlp_branch)
        
        # ============ Concatenate All Branches ============
        concatenated = Concatenate(name='concatenate')([cnn_branch, lstm_branch, mlp_branch])
        
        # ============ Final Classification Layers ============
        x = Dense(256, activation='relu', name='fc1')(concatenated)
        x = BatchNormalization(name='fc_bn1')(x)
        x = Dropout(0.5, name='fc_dropout1')(x)
        
        x = Dense(128, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='fc_bn2')(x)
        x = Dropout(0.3, name='fc_dropout2')(x)
        
        output = Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='CNN_LSTM_MLP_Ensemble')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully!")
    
    def fit(self, X_train, y_train, validation_data=None, epochs=50, batch_size=256, class_weight=None, verbose=2):
        """
        Train the ensemble model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Dictionary of class weights to handle imbalanced data
            
        Returns:
            Training history
        """
        print("="*60)
        print("  TRAINING CNN-LSTM-MLP ENSEMBLE MODEL")
        print("="*60)
        
        # Callbacks
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
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✅ CNN-LSTM-MLP Ensemble training completed!")
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test data
            
        Returns:
            Predicted class labels
        """
        predictions_proba = self.model.predict(X_test, verbose=0)
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
        return self.model.predict(X_test, verbose=0)
    
    def save_model(self, path):
        """
        Save the entire model.
        
        Args:
            path: Path to save model (with .h5 extension)
        """
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        """
        from tensorflow.keras.models import load_model
        
        self.model = load_model(path)
        print(f"✅ Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        print("="*60)
        print("  CNN-LSTM-MLP ENSEMBLE MODEL SUMMARY")
        print("="*60)
        self.model.summary()
        print("="*60)
        
        # Count parameters per branch
        cnn_params = sum([np.prod(layer.get_weights()[0].shape) 
                         for layer in self.model.layers 
                         if 'cnn' in layer.name and len(layer.get_weights()) > 0])
        lstm_params = sum([np.prod(layer.get_weights()[0].shape) 
                          for layer in self.model.layers 
                          if 'lstm' in layer.name and len(layer.get_weights()) > 0])
        mlp_params = sum([np.prod(layer.get_weights()[0].shape) 
                         for layer in self.model.layers 
                         if 'mlp' in layer.name and len(layer.get_weights()) > 0])
        
        print(f"\nParameters per branch:")
        print(f"  CNN Branch: {cnn_params:,}")
        print(f"  LSTM Branch: {lstm_params:,}")
        print(f"  MLP Branch: {mlp_params:,}")
        print("="*60)
