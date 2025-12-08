"""Machine Learning models for cybersecurity threat detection.

This module implements traditional ML models including Random Forest, SVM,
XGBoost, and Gradient Boosting for threat classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
import optuna
from tqdm import tqdm

from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_dir, Timer


class MLModels(LoggerMixin):
    """Machine Learning models for threat detection.
    
    Implements 4 ML models from the research paper:
    - Random Forest
    - Support Vector Machine (SVM)
    - XGBoost
    - Gradient Boosting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize MLModels.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config or {}
        self.models = {}
        self.best_params = {}
        
        self.logger.info("MLModels initialized")
    
    def build_random_forest(self, **params) -> RandomForestClassifier:
        """Build Random Forest Classifier.
        
        Args:
            **params: Model parameters (overrides defaults)
        
        Returns:
            RandomForestClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1  # Show progress
        }
        
        # Override with config
        if 'random_forest' in self.config.get('models', {}).get('ml', {}):
            default_params.update(self.config['models']['ml']['random_forest'])
        
        # Override with params
        default_params.update(params)
        
        self.logger.info(f"Building Random Forest with params: {default_params}")
        
        return RandomForestClassifier(**default_params)
    
    def build_svm(self, **params) -> SVC:
        """Build Support Vector Machine.
        
        Args:
            **params: Model parameters (overrides defaults)
        
        Returns:
            SVC instance
        """
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'verbose': False
        }
        
        # Override with config
        if 'svm' in self.config.get('models', {}).get('ml', {}):
            default_params.update(self.config['models']['ml']['svm'])
        
        # Override with params
        default_params.update(params)
        
        self.logger.info(f"Building SVM with params: {default_params}")
        
        return SVC(**default_params)
    
    def build_xgboost(self, **params) -> XGBClassifier:
        """Build XGBoost Classifier.
        
        Args:
            **params: Model parameters (overrides defaults)
        
        Returns:
            XGBClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1  # Show progress
        }
        
        # Override with config
        if 'xgboost' in self.config.get('models', {}).get('ml', {}):
            default_params.update(self.config['models']['ml']['xgboost'])
        
        # Override with params
        default_params.update(params)
        
        self.logger.info(f"Building XGBoost with params: {default_params}")
        
        return XGBClassifier(**default_params)
    
    def build_gradient_boosting(self, **params) -> GradientBoostingClassifier:
        """Build Gradient Boosting Classifier.
        
        Args:
            **params: Model parameters (overrides defaults)
        
        Returns:
            GradientBoostingClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42,
            'verbose': 1  # Show progress
        }
        
        # Override with config
        if 'gradient_boosting' in self.config.get('models', {}).get('ml', {}):
            default_params.update(self.config['models']['ml']['gradient_boosting'])
        
        # Override with params
        default_params.update(params)
        
        self.logger.info(f"Building Gradient Boosting with params: {default_params}")
        
        return GradientBoostingClassifier(**default_params)
    
    def train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_cv: bool = True,
        cv_folds: int = 5,
        model_name: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """Train model with optional cross-validation.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
            model_name: Name of the model (for logging)
        
        Returns:
            Tuple of (trained model, training metrics)
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        self.logger.info(f"\nTraining {model_name}...")
        self.logger.info(f"Training set shape: {X_train.shape}")
        
        metrics = {}
        
        # Cross-validation
        if use_cv:
            self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
            self.logger.info(f"⏳ This may take a while for {model_name}...")
            
            # Manual progress tracking for CV
            cv_scores = []
            for fold_idx in tqdm(range(cv_folds), desc=f"CV Folds ({model_name})", leave=False):
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    if i == fold_idx:
                        X_train_fold = X_train[train_idx]
                        y_train_fold = y_train[train_idx]
                        X_val_fold = X_train[val_idx]
                        y_val_fold = y_train[val_idx]
                        
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train_fold, y_train_fold)
                        score = model_copy.score(X_val_fold, y_val_fold)
                        cv_scores.append(score)
                        break
            
            cv_scores = np.array(cv_scores)
            
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            self.logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        self.logger.info(f"Training on full training set...")
        self.logger.info(f"⏳ Training {model_name}... Please wait...")
        
        with Timer(f"{model_name} training", verbose=False) as timer:
            model.fit(X_train, y_train)
        
        metrics['training_time'] = timer.elapsed
        self.logger.info(f"✅ Training completed in {timer.elapsed:.2f}s")
        
        # Store model
        self.models[model_name] = model
        
        return model, metrics
    
    def predict(
        self,
        model: Any,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and probabilities.
        
        Args:
            model: Trained model
            X_test: Test features
        
        Returns:
            Tuple of (predicted labels, prediction probabilities)
        """
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            # For SVM without probability=True
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = None
        
        return y_pred, y_pred_proba
    
    def optimize_hyperparameters(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'grid',
        param_grid: Optional[Dict] = None,
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict:
        """Hyperparameter optimization.
        
        Args:
            model_name: Name of model ('rf', 'svm', 'xgb', 'gb')
            X: Feature matrix
            y: Target labels
            method: Optimization method ('grid', 'random', 'optuna')
            param_grid: Parameter grid for search
            n_iter: Number of iterations for random search
            cv: Number of CV folds
        
        Returns:
            Dictionary with best parameters and scores
        """
        self.logger.info(f"Optimizing {model_name} hyperparameters using {method}...")
        
        # Build base model
        if model_name == 'rf':
            model = self.build_random_forest()
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2']
                }
        
        elif model_name == 'svm':
            model = self.build_svm()
            if param_grid is None:
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'poly']
                }
        
        elif model_name == 'xgb':
            model = self.build_xgboost()
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
        
        elif model_name == 'gb':
            model = self.build_gradient_boosting()
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Perform search
        if method == 'grid':
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2
            )
        
        elif method == 'random':
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        with Timer(f"{model_name} hyperparameter optimization"):
            search.fit(X, y)
        
        self.best_params[model_name] = search.best_params_
        
        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Get feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning(f"{model.__class__.__name__} does not have feature_importances_")
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save_model(self, model: Any, filepath: str) -> None:
        """Save model to file.
        
        Args:
            model: Model to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        
        joblib.dump(model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load model from file.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_cv: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Tuple[Any, Dict]]:
        """Train all ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
        
        Returns:
            Dictionary mapping model names to (model, metrics) tuples
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING ALL ML MODELS")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Define all models to train
        models_to_train = [
            ('RandomForest', self.build_random_forest),
            ('SVM', self.build_svm),
            ('XGBoost', self.build_xgboost),
            ('GradientBoosting', self.build_gradient_boosting)
        ]
        
        # Train with progress bar
        print("\n🚀 Starting ML Model Training...\n")
        for idx, (model_name, build_func) in enumerate(tqdm(models_to_train, desc="🔄 Training ML Models", unit="model"), 1):
            print(f"\n{'='*80}")
            print(f"📊 Model {idx}/{len(models_to_train)}: {model_name}")
            print(f"{'='*80}")
            
            model = build_func()
            trained_model, metrics = self.train_model(
                model, X_train, y_train,
                use_cv=use_cv,
                cv_folds=cv_folds,
                model_name=model_name
            )
            results[model_name] = (trained_model, metrics)
            
            print(f"✅ {model_name} completed!")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ALL ML MODELS TRAINED")
        self.logger.info("=" * 80 + "\n")
        
        return results
