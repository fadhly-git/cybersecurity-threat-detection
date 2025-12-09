"""
Stacking Ensemble Model

Stacking ensemble with multiple base models and meta-learner.
Base models: CNN, LSTM, Random Forest, XGBoost
Meta-learner: Logistic Regression or LightGBM
"""

import numpy as np
import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam


class StackingEnsemble:
    """
    Stacking Ensemble for cybersecurity threat detection.
    
    Combines predictions from multiple base models using a meta-learner.
    """
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize Stacking Ensemble.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_models = self._create_base_models(input_shape, num_classes)
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42,
            verbose=1
        )
    
    def _create_base_models(self, input_shape, num_classes):
        """
        Create base models for stacking.
        
        Returns:
            Dictionary of base models
        """
        # CNN model
        cnn = Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            Dropout(0.3),
            Conv1D(128, 3, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ], name='CNN_Base')
        
        cnn.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # LSTM model
        lstm = Sequential([
            Bidirectional(LSTM(128, activation='tanh'), input_shape=input_shape),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ], name='LSTM_Base')
        
        lstm.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced_subsample'
        )
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42
        )
        
        return {
            'cnn': cnn,
            'lstm': lstm,
            'rf': rf,
            'xgb': xgb
        }
    
    def fit_base_models(self, X_train, y_train, epochs=50, batch_size=256, verbose=2, class_weight=None):
        """
        Train all base models.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Epochs for deep learning models
            batch_size: Batch size for deep learning models
        """
        print("="*60)
        print("  TRAINING BASE MODELS")
        print("="*60)
        
        # Train CNN
        print("\n[1/4] Training CNN...")
        self.base_models['cnn'].fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            class_weight=class_weight
        )
        
        # Train LSTM
        print("\n[2/4] Training LSTM...")
        self.base_models['lstm'].fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            class_weight=class_weight
        )
        
        # Flatten for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train Random Forest
        print("\n[3/4] Training Random Forest...")
        self.base_models['rf'].fit(X_train_flat, y_train)
        
        # Train XGBoost
        print("\n[4/4] Training XGBoost...")
        self.base_models['xgb'].fit(X_train_flat, y_train)
        
        print("\n✅ All base models trained!")
    
    def fit_meta_learner(self, X_train, y_train, cv=5):
        """
        Train meta-learner using cross-validation predictions.
        
        Args:
            X_train: Training data
            y_train: Training labels
            cv: Number of cross-validation folds
        """
        print("="*60)
        print("  TRAINING META-LEARNER")
        print("="*60)
        
        # Generate meta-features using cross-validation
        meta_features = []
        
        # CNN predictions
        print("\nGenerating CNN meta-features...")
        cnn_preds = cross_val_predict(
            self.base_models['cnn'],
            X_train, y_train,
            cv=cv,
            method='predict_proba'
        )
        meta_features.append(cnn_preds)
        
        # LSTM predictions
        print("Generating LSTM meta-features...")
        lstm_preds = cross_val_predict(
            self.base_models['lstm'],
            X_train, y_train,
            cv=cv,
            method='predict_proba'
        )
        meta_features.append(lstm_preds)
        
        # Flatten for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # RF predictions
        print("Generating Random Forest meta-features...")
        rf_preds = cross_val_predict(
            self.base_models['rf'],
            X_train_flat, y_train,
            cv=cv,
            method='predict_proba'
        )
        meta_features.append(rf_preds)
        
        # XGBoost predictions
        print("Generating XGBoost meta-features...")
        xgb_preds = cross_val_predict(
            self.base_models['xgb'],
            X_train_flat, y_train,
            cv=cv,
            method='predict_proba'
        )
        meta_features.append(xgb_preds)
        
        # Stack meta-features
        X_meta = np.hstack(meta_features)
        
        print(f"\nMeta-features shape: {X_meta.shape}")
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_learner.fit(X_meta, y_train)
        
        print("\n✅ Meta-learner trained!")
    
    def predict(self, X_test):
        """
        Make predictions using stacking ensemble.
        
        Args:
            X_test: Test data
            
        Returns:
            Predicted class labels
        """
        # Get predictions from base models
        meta_features = []
        
        # CNN
        cnn_preds = self.base_models['cnn'].predict(X_test, verbose=0)
        meta_features.append(cnn_preds)
        
        # LSTM
        lstm_preds = self.base_models['lstm'].predict(X_test, verbose=0)
        meta_features.append(lstm_preds)
        
        # Flatten for traditional ML
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # RF
        rf_preds = self.base_models['rf'].predict_proba(X_test_flat)
        meta_features.append(rf_preds)
        
        # XGBoost
        xgb_preds = self.base_models['xgb'].predict_proba(X_test_flat)
        meta_features.append(xgb_preds)
        
        # Stack and predict with meta-learner
        X_meta = np.hstack(meta_features)
        predictions = self.meta_learner.predict(X_meta)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test: Test data
            
        Returns:
            Class probabilities
        """
        # Get predictions from base models
        meta_features = []
        
        meta_features.append(self.base_models['cnn'].predict(X_test, verbose=0))
        meta_features.append(self.base_models['lstm'].predict(X_test, verbose=0))
        
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        meta_features.append(self.base_models['rf'].predict_proba(X_test_flat))
        meta_features.append(self.base_models['xgb'].predict_proba(X_test_flat))
        
        X_meta = np.hstack(meta_features)
        probabilities = self.meta_learner.predict_proba(X_meta)
        
        return probabilities
    
    def save_model(self, path):
        """
        Save all models.
        
        Args:
            path: Base path for saving
        """
        # Save DL models
        self.base_models['cnn'].save(f'{path}_cnn.h5')
        self.base_models['lstm'].save(f'{path}_lstm.h5')
        
        # Save ML models
        joblib.dump(self.base_models['rf'], f'{path}_rf.pkl')
        joblib.dump(self.base_models['xgb'], f'{path}_xgb.pkl')
        
        # Save meta-learner
        joblib.dump(self.meta_learner, f'{path}_meta.pkl')
        
        print(f"✅ Stacking ensemble saved to {path}_*")
    
    def load_model(self, path):
        """Load all models from disk."""
        from tensorflow.keras.models import load_model
        
        self.base_models['cnn'] = load_model(f'{path}_cnn.h5')
        self.base_models['lstm'] = load_model(f'{path}_lstm.h5')
        self.base_models['rf'] = joblib.load(f'{path}_rf.pkl')
        self.base_models['xgb'] = joblib.load(f'{path}_xgb.pkl')
        self.meta_learner = joblib.load(f'{path}_meta.pkl')
        
        print(f"✅ Stacking ensemble loaded from {path}")
