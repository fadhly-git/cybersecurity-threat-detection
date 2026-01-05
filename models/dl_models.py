"""
dl_models.py - Deep Learning Models
Optimized for CPU (macOS)
"""

import numpy as np
import os

# Force CPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config, ModelConfig


def check_device():
    """Check available devices"""
    print("="*50)
    print("DEVICE CONFIGURATION")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Running on: CPU (GPU disabled for macOS)")
    print("="*50)


class BaseDeepModel:
    """Base class for deep learning models"""
    
    def __init__(self, 
                 input_dim: int, 
                 n_classes: int, 
                 random_state: int = 42):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def _get_callbacks(self, patience: int = 10, 
                       model_path: Optional[str] = None) -> List:
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
            )
        
        return callbacks
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None,
            epochs: int = 50, 
            batch_size: int = 128,
            callbacks: Optional[List] = None,
            verbose: int = 1) -> 'BaseDeepModel':
        """Train model"""
        
        if callbacks is None:
            callbacks = self._get_callbacks()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        proba = self.model.predict(X, verbose=0, batch_size=256)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict(X, verbose=0, batch_size=256)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model"""
        results = self.model.evaluate(X, y, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}
    
    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)
        
    def load(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        return self
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


class DeepNeuralNetwork(BaseDeepModel):
    """Deep Neural Network for classification"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 hidden_layers: Tuple[int, ...] = (256, 128, 64),
                 dropout_rate: float = 0.4,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 l2_reg: float = 0.001,
                 random_state: int = 42):
        
        super().__init__(input_dim, n_classes, random_state)
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build DNN architecture"""
        
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # Initial batch normalization
        x = layers.BatchNormalization(name='bn_input')(inputs)
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'dense_{i}'
            )(x)
            
            if self.batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
            
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.n_classes,
            activation='softmax',
            name='output'
        )(x)
        
        model = Model(inputs, outputs, name='DNN')
        
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class LightCNN(BaseDeepModel):
    """Lightweight 1D CNN for classification"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 filters: Tuple[int, ...] = (32, 64),
                 kernel_size: int = 3,
                 dropout_rate: float = 0.4,
                 random_state: int = 42):
        
        super().__init__(input_dim, n_classes, random_state)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build CNN architecture"""
        
        inputs = layers.Input(shape=(self.input_dim, 1), name='input')
        
        x = inputs
        
        # Convolutional layers
        for i, f in enumerate(self.filters):
            x = layers.Conv1D(
                f, 
                self.kernel_size, 
                padding='same', 
                activation='relu',
                name=f'conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling1D(2, padding='same', name=f'pool_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs, outputs, name='CNN')
        
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for CNN"""
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        X_train = self._reshape_input(X_train)
        if X_val is not None:
            X_val = self._reshape_input(X_val)
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = self._reshape_input(X)
        return super().predict(X)
    
    def predict_proba(self, X):
        X = self._reshape_input(X)
        return super().predict_proba(X)


class LightLSTM(BaseDeepModel):
    """Lightweight LSTM for classification"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 lstm_units: Tuple[int, ...] = (64, 32),
                 dropout_rate: float = 0.4,
                 recurrent_dropout: float = 0.1,
                 random_state: int = 42):
        
        super().__init__(input_dim, n_classes, random_state)
        
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build LSTM architecture"""
        
        inputs = layers.Input(shape=(self.input_dim, 1), name='input')
        
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units[:-1]):
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                name=f'lstm_{i}'
            )(x)
        
        # Last LSTM layer
        x = layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout,
            name=f'lstm_{len(self.lstm_units)-1}'
        )(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Output
        outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs, outputs, name='LSTM')
        
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for LSTM"""
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        X_train = self._reshape_input(X_train)
        if X_val is not None:
            X_val = self._reshape_input(X_val)
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = self._reshape_input(X)
        return super().predict(X)
    
    def predict_proba(self, X):
        X = self._reshape_input(X)
        return super().predict_proba(X)


class HybridCNNLSTM(BaseDeepModel):
    """Hybrid CNN-LSTM model"""
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 cnn_filters: Tuple[int, ...] = (32, 64),
                 lstm_units: int = 64,
                 dropout_rate: float = 0.4,
                 random_state: int = 42):
        
        super().__init__(input_dim, n_classes, random_state)
        
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build hybrid CNN-LSTM architecture"""
        
        inputs = layers.Input(shape=(self.input_dim, 1), name='input')
        
        x = inputs
        
        # CNN layers
        for i, f in enumerate(self.cnn_filters):
            x = layers.Conv1D(f, 3, padding='same', activation='relu', name=f'conv_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling1D(2, padding='same', name=f'pool_{i}')(x)
        
        # LSTM layer
        x = layers.LSTM(self.lstm_units, dropout=self.dropout_rate, name='lstm')(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout')(x)
        
        # Output
        outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs, outputs, name='CNN_LSTM')
        
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        X_train = self._reshape_input(X_train)
        if X_val is not None:
            X_val = self._reshape_input(X_val)
        return super().fit(X_train, y_train, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = self._reshape_input(X)
        return super().predict(X)
    
    def predict_proba(self, X):
        X = self._reshape_input(X)
        return super().predict_proba(X)