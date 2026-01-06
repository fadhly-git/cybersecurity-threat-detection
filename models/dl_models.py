"""
dl_models.py - Deep Learning Models from Paper
Based on paper: arxiv.org/abs/2407.06014
"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of 
Machine and Deep Learning Techniques for Threat Detection"

Models: VGG16, VGG19, ResNet18, ResNet50, Inception (adapted for tabular data)
"""

import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Any

# Force CPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURES FROM PAPER (Adapted for Tabular Data)
# ============================================================================

def create_vgg16_tabular(input_dim: int, n_classes: int) -> Model:
    """
    VGG16-style architecture adapted for tabular data.
    Original VGG16 has 16 weight layers - adapted here with dense layers.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Block 1
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='VGG16_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_vgg19_tabular(input_dim: int, n_classes: int) -> Model:
    """
    VGG19-style architecture adapted for tabular data.
    Original VGG19 has 19 weight layers - adapted here with dense layers.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Block 1 (3 layers)
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 2 (3 layers)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3 (3 layers)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='VGG19_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_resnet18_tabular(input_dim: int, n_classes: int) -> Model:
    """
    ResNet18-style architecture with skip connections for tabular data.
    Uses residual blocks to enable gradient flow.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial block
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks
    for units in [256, 128, 64]:
        # Shortcut connection
        shortcut = layers.Dense(units)(x)
        
        # Main path
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Add shortcut (skip connection)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet18_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_resnet50_tabular(input_dim: int, n_classes: int) -> Model:
    """
    ResNet50-style architecture with deeper residual blocks.
    More residual blocks than ResNet18 for increased depth.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial block
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Deeper residual blocks (6 blocks)
    for units in [512, 256, 256, 128, 128, 64]:
        # Shortcut connection
        shortcut = layers.Dense(units)(x)
        
        # Main path (bottleneck-style)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units)(x)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet50_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_inception_tabular(input_dim: int, n_classes: int) -> Model:
    """
    Inception-style architecture with parallel paths for tabular data.
    Multiple parallel paths with different receptive fields concatenated together.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # First Inception module - multiple parallel paths
    # Path 1: Direct transformation
    path1 = layers.Dense(128, activation='relu')(inputs)
    path1 = layers.BatchNormalization()(path1)
    
    # Path 2: Two-layer transformation
    path2 = layers.Dense(64, activation='relu')(inputs)
    path2 = layers.Dense(128, activation='relu')(path2)
    path2 = layers.BatchNormalization()(path2)
    
    # Path 3: Three-layer transformation (deeper features)
    path3 = layers.Dense(32, activation='relu')(inputs)
    path3 = layers.Dense(64, activation='relu')(path3)
    path3 = layers.Dense(128, activation='relu')(path3)
    path3 = layers.BatchNormalization()(path3)
    
    # Concatenate all paths
    x = layers.Concatenate()([path1, path2, path3])
    x = layers.Dropout(0.3)(x)
    
    # Second Inception module
    path1 = layers.Dense(64, activation='relu')(x)
    path2 = layers.Dense(32, activation='relu')(x)
    path2 = layers.Dense(64, activation='relu')(path2)
    
    x = layers.Concatenate()([path1, path2])
    x = layers.Dropout(0.3)(x)
    
    # Final dense layer
    x = layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='Inception_Tabular')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_dl_models(input_dim: int, n_classes: int) -> Dict[str, Model]:
    """
    Get all DL models from paper as a dictionary.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of output classes
    
    Returns:
        Dict with model names as keys and Keras model instances as values
    """
    return {
        'VGG16': create_vgg16_tabular(input_dim, n_classes),
        'VGG19': create_vgg19_tabular(input_dim, n_classes),
        'ResNet18': create_resnet18_tabular(input_dim, n_classes),
        'ResNet50': create_resnet50_tabular(input_dim, n_classes),
        'Inception': create_inception_tabular(input_dim, n_classes),
    }


def get_dl_model_creators() -> Dict[str, callable]:
    """
    Get model creator functions (for lazy instantiation).
    Useful when you want to create models later with specific dimensions.
    
    Returns:
        Dict with model names as keys and creator functions as values
    """
    return {
        'VGG16': create_vgg16_tabular,
        'VGG19': create_vgg19_tabular,
        'ResNet18': create_resnet18_tabular,
        'ResNet50': create_resnet50_tabular,
        'Inception': create_inception_tabular,
    }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_callbacks(patience: int = 10, model_path: Optional[str] = None) -> List:
    """
    Get standard training callbacks.
    
    Args:
        patience: Patience for early stopping
        model_path: Path to save best model (optional)
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
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


def train_dl_model(model: Model, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 30, 
                   batch_size: int = 128,
                   patience: int = 10,
                   verbose: int = 0) -> Tuple[Model, Any]:
    """
    Train a deep learning model with standard configuration.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
    
    Returns:
        Tuple of (trained model, training history)
    """
    callbacks = get_callbacks(patience=patience)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history
