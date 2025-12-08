"""Deep Learning models for cybersecurity threat detection.

This module implements deep learning architectures including CNN, LSTM,
VGG-like, and ResNet-like models for threat classification.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tqdm.keras import TqdmCallback

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir, Timer


class DLModels(LoggerMixin):
    """Deep Learning models for threat detection.
    
    Implements 4 DL architectures from the research paper:
    - CNN (Convolutional Neural Network)
    - LSTM (Long Short-Term Memory)
    - VGG-like architecture
    - ResNet-like architecture with residual connections
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize DLModels.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config or {}
        self.models = {}
        self.histories = {}
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.logger.info("DLModels initialized")
        self.logger.info(f"TensorFlow version: {tf.__version__}")
        self.logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    def build_cnn(
        self,
        input_shape: Tuple,
        num_classes: int,
        **params
    ) -> keras.Model:
        """Build Convolutional Neural Network.
        
        Architecture:
        - Conv1D(64) -> BatchNorm -> MaxPool -> Dropout
        - Conv1D(128) -> BatchNorm -> MaxPool -> Dropout
        - Conv1D(256) -> BatchNorm -> GlobalAvgPool
        - Dense(128) -> Dropout -> Dense(num_classes)
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            **params: Additional parameters
        
        Returns:
            Keras Model
        """
        self.logger.info(f"Building CNN model...")
        self.logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")
        
        # Get parameters from config or use defaults
        filters = params.get('filters', [64, 128, 256])
        kernel_size = params.get('kernel_size', 3)
        dropout_conv = params.get('dropout_conv', 0.3)
        dropout_dense = params.get('dropout', 0.5)
        dense_units = params.get('dense_units', 128)
        
        model = models.Sequential(name='CNN')
        
        # First Conv Block
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout_conv))
        
        # Second Conv Block
        model.add(layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout_conv))
        
        # Third Conv Block
        model.add(layers.Conv1D(filters[2], kernel_size, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling1D())
        
        # Dense Layers
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(dropout_dense))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.logger.info(f"CNN model built with {model.count_params():,} parameters")
        
        return model
    
    def build_lstm(
        self,
        input_shape: Tuple,
        num_classes: int,
        **params
    ) -> keras.Model:
        """Build LSTM Network.
        
        Architecture:
        - Bidirectional LSTM(128) with return_sequences
        - Bidirectional LSTM(64)
        - Dense(64) -> Dropout -> Dense(num_classes)
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            **params: Additional parameters
        
        Returns:
            Keras Model
        """
        self.logger.info(f"Building LSTM model...")
        self.logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")
        
        # Get parameters from config or use defaults
        units = params.get('units', [128, 64])
        dropout = params.get('dropout', 0.3)
        recurrent_dropout = params.get('recurrent_dropout', 0.3)
        bidirectional = params.get('bidirectional', True)
        dense_units = params.get('dense_units', 64)
        dropout_dense = params.get('dropout_dense', 0.5)
        
        model = models.Sequential(name='LSTM')
        
        model.add(layers.Input(shape=input_shape))
        
        # First LSTM layer
        lstm1 = layers.LSTM(
            units[0],
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
        if bidirectional:
            model.add(layers.Bidirectional(lstm1))
        else:
            model.add(lstm1)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            units[1],
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
        if bidirectional:
            model.add(layers.Bidirectional(lstm2))
        else:
            model.add(lstm2)
        
        # Dense layers
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(dropout_dense))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.logger.info(f"LSTM model built with {model.count_params():,} parameters")
        
        return model
    
    def build_vgg(
        self,
        input_shape: Tuple,
        num_classes: int,
        **params
    ) -> keras.Model:
        """Build VGG-like architecture.
        
        Architecture:
        - Conv1D Block 1: Conv x2 -> MaxPool
        - Conv1D Block 2: Conv x2 -> MaxPool
        - Conv1D Block 3: Conv x2 -> GlobalAvgPool
        - Dense(256) -> Dropout -> Dense(num_classes)
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            **params: Additional parameters
        
        Returns:
            Keras Model
        """
        self.logger.info(f"Building VGG model...")
        self.logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")
        
        # Get parameters from config or use defaults
        filters = params.get('filters', [64, 128, 256])
        kernel_size = params.get('kernel_size', 3)
        dropout = params.get('dropout', 0.5)
        dense_units = params.get('dense_units', 256)
        
        model = models.Sequential(name='VGG')
        
        model.add(layers.Input(shape=input_shape))
        
        # Block 1
        model.add(layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'))
        model.add(layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Block 2
        model.add(layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'))
        model.add(layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Block 3
        model.add(layers.Conv1D(filters[2], kernel_size, activation='relu', padding='same'))
        model.add(layers.Conv1D(filters[2], kernel_size, activation='relu', padding='same'))
        model.add(layers.GlobalAveragePooling1D())
        
        # Dense layers
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.logger.info(f"VGG model built with {model.count_params():,} parameters")
        
        return model
    
    def _residual_block(
        self,
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1
    ) -> tf.Tensor:
        """Create a residual block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride size
        
        Returns:
            Output tensor
        """
        # Shortcut connection
        shortcut = x
        
        # First conv layer
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv layer
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if dimensions changed
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def build_resnet(
        self,
        input_shape: Tuple,
        num_classes: int,
        **params
    ) -> keras.Model:
        """Build ResNet-like architecture with residual connections.
        
        Architecture:
        - Initial Conv1D(64) -> BatchNorm -> MaxPool
        - Residual Block 1 (64 filters) x 2
        - Residual Block 2 (128 filters) x 2
        - Residual Block 3 (256 filters) x 2
        - GlobalAvgPool -> Dense(num_classes)
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            **params: Additional parameters
        
        Returns:
            Keras Model
        """
        self.logger.info(f"Building ResNet model...")
        self.logger.info(f"Input shape: {input_shape}, Num classes: {num_classes}")
        
        # Get parameters from config or use defaults
        filters = params.get('filters', [64, 128, 256])
        blocks_per_stage = params.get('blocks_per_stage', 2)
        initial_kernel = params.get('initial_kernel_size', 7)
        initial_strides = params.get('initial_strides', 2)
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Initial conv layer
        x = layers.Conv1D(64, initial_kernel, strides=initial_strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        
        # Stage 1
        for _ in range(blocks_per_stage):
            x = self._residual_block(x, filters[0])
        
        # Stage 2
        x = self._residual_block(x, filters[1], strides=2)  # Downsample
        for _ in range(blocks_per_stage - 1):
            x = self._residual_block(x, filters[1])
        
        # Stage 3
        x = self._residual_block(x, filters[2], strides=2)  # Downsample
        for _ in range(blocks_per_stage - 1):
            x = self._residual_block(x, filters[2])
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet')
        
        self.logger.info(f"ResNet model built with {model.count_params():,} parameters")
        
        return model
    
    def compile_model(
        self,
        model: keras.Model,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'sparse_categorical_crossentropy',
        metrics: List[str] = None
    ) -> keras.Model:
        """Compile model with optimizer and loss function.
        
        Args:
            model: Keras model to compile
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
        
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
        
        return model
    
    def create_callbacks(
        self,
        model_name: str,
        patience: int = 10,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        save_best: bool = True,
        use_progress_bar: bool = True
    ) -> List[callbacks.Callback]:
        """Create training callbacks.
        
        Args:
            model_name: Name of the model
            patience: Early stopping patience
            reduce_lr_patience: ReduceLR patience
            reduce_lr_factor: ReduceLR factor
            save_best: Whether to save best model
            use_progress_bar: Whether to use tqdm progress bar
        
        Returns:
            List of callbacks
        """
        callback_list = []
        
        # Progress bar (tqdm)
        if use_progress_bar:
            tqdm_callback = TqdmCallback(verbose=1)
            callback_list.append(tqdm_callback)
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        if save_best:
            checkpoint_dir = ensure_dir('models/checkpoints')
            checkpoint_path = checkpoint_dir / f'{model_name}_best.h5'
            
            checkpoint = callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # TensorBoard
        log_dir = ensure_dir('logs/tensorboard') / model_name
        tensorboard = callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True
        )
        callback_list.append(tensorboard)
        
        self.logger.info(f"Created {len(callback_list)} callbacks for {model_name}")
        
        return callback_list
    
    def train_model(
        self,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks_list: Optional[List] = None,
        model_name: Optional[str] = None
    ) -> keras.callbacks.History:
        """Train deep learning model.
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks_list: List of callbacks
            model_name: Name of the model
        
        Returns:
            Training history
        """
        if model_name is None:
            model_name = model.name
        
        self.logger.info(f"\nTraining {model_name}...")
        self.logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        self.logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Reshape data if needed (add timestep dimension for 1D conv/LSTM)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            self.logger.info(f"Reshaped data to: {X_train.shape}")
        
        with Timer(f"{model_name} training", verbose=False) as timer:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
        
        self.logger.info(f"Training completed in {timer.elapsed:.2f}s")
        
        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        self.logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        self.logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        # Store model and history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return history
    
    def evaluate_model(
        self,
        model: keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Reshape data if needed
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        self.logger.info(f"Evaluating {model.name}...")
        
        results = model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {}
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = results[i]
            self.logger.info(f"{metric_name}: {results[i]:.4f}")
        
        return metrics
    
    def predict(
        self,
        model: keras.Model,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions.
        
        Args:
            model: Trained model
            X: Input features
        
        Returns:
            Tuple of (predicted labels, prediction probabilities)
        """
        # Reshape data if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
    
    def save_model(self, model: keras.Model, filepath: str) -> None:
        """Save model to file.
        
        Args:
            model: Model to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        
        model.save(str(filepath))
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> keras.Model:
        """Load model from file.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Loaded model
        """
        model = keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Tuple[keras.Model, keras.callbacks.History]]:
        """Train all DL models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_classes: Number of classes
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Dictionary mapping model names to (model, history) tuples
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING ALL DL MODELS")
        self.logger.info("=" * 80)
        
        # Determine input shape
        if len(X_train.shape) == 2:
            input_shape = (X_train.shape[1], 1)
        else:
            input_shape = X_train.shape[1:]
        
        results = {}
        
        # CNN
        cnn = self.build_cnn(input_shape, num_classes)
        cnn = self.compile_model(cnn)
        cnn_callbacks = self.create_callbacks('CNN')
        cnn_history = self.train_model(
            cnn, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks_list=cnn_callbacks,
            model_name='CNN'
        )
        results['CNN'] = (cnn, cnn_history)
        
        # LSTM
        lstm = self.build_lstm(input_shape, num_classes)
        lstm = self.compile_model(lstm)
        lstm_callbacks = self.create_callbacks('LSTM')
        lstm_history = self.train_model(
            lstm, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks_list=lstm_callbacks,
            model_name='LSTM'
        )
        results['LSTM'] = (lstm, lstm_history)
        
        # VGG
        vgg = self.build_vgg(input_shape, num_classes)
        vgg = self.compile_model(vgg)
        vgg_callbacks = self.create_callbacks('VGG')
        vgg_history = self.train_model(
            vgg, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks_list=vgg_callbacks,
            model_name='VGG'
        )
        results['VGG'] = (vgg, vgg_history)
        
        # ResNet
        resnet = self.build_resnet(input_shape, num_classes)
        resnet = self.compile_model(resnet)
        resnet_callbacks = self.create_callbacks('ResNet')
        resnet_history = self.train_model(
            resnet, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            callbacks_list=resnet_callbacks,
            model_name='ResNet'
        )
        results['ResNet'] = (resnet, resnet_history)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ALL DL MODELS TRAINED")
        self.logger.info("=" * 80 + "\n")
        
        return results
