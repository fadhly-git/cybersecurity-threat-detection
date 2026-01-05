"""
ml_models.py - Machine Learning Models
Optimized for CPU
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config, ModelConfig


class BaseMLModel:
    """Base class for ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        self.cv_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5, scoring: str = 'f1_weighted') -> Dict:
        """Cross validate model"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        self.cv_scores_ = scores
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df


class OptimizedRandomForest(BaseMLModel):
    """Optimized Random Forest Classifier"""
    
    def __init__(self, 
                 params: Optional[Dict] = None,
                 random_state: int = 42):
        super().__init__(random_state)
        
        if params is None:
            params = ModelConfig.RF_DEFAULT.copy()
        
        params['random_state'] = random_state
        self.model = RandomForestClassifier(**params)
        
    def tune(self, X: np.ndarray, y: np.ndarray, 
             n_trials: int = 30, timeout: int = 600) -> Dict:
        """Tune hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'n_jobs': -1,
                'random_state': self.random_state
            }
            
            model = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        self.best_params_['n_jobs'] = -1
        self.best_params_['random_state'] = self.random_state
        self.model = RandomForestClassifier(**self.best_params_)
        
        return {'best_params': self.best_params_, 'best_score': study.best_value}


class OptimizedXGBoost(BaseMLModel):
    """Optimized XGBoost Classifier"""
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 random_state: int = 42):
        super().__init__(random_state)
        
        if params is None:
            params = ModelConfig.XGB_DEFAULT.copy()
        
        params['random_state'] = random_state
        self.model = XGBClassifier(**params)
        
    def tune(self, X: np.ndarray, y: np.ndarray,
             n_trials: int = 30, timeout: int = 600) -> Dict:
        """Tune hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            model = XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        self.best_params_['n_jobs'] = -1
        self.best_params_['random_state'] = self.random_state
        self.best_params_['verbosity'] = 0
        self.model = XGBClassifier(**self.best_params_)
        
        return {'best_params': self.best_params_, 'best_score': study.best_value}


class OptimizedLightGBM(BaseMLModel):
    """Optimized LightGBM Classifier"""
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 random_state: int = 42):
        super().__init__(random_state)
        
        if params is None:
            params = ModelConfig.LGBM_DEFAULT.copy()
        
        params['random_state'] = random_state
        self.model = LGBMClassifier(**params)
        
    def tune(self, X: np.ndarray, y: np.ndarray,
             n_trials: int = 30, timeout: int = 600) -> Dict:
        """Tune hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        self.best_params_['n_jobs'] = -1
        self.best_params_['random_state'] = self.random_state
        self.best_params_['verbose'] = -1
        self.model = LGBMClassifier(**self.best_params_)
        
        return {'best_params': self.best_params_, 'best_score': study.best_value}


class OptimizedCatBoost(BaseMLModel):
    """Optimized CatBoost Classifier"""
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 random_state: int = 42):
        super().__init__(random_state)
        
        if params is None:
            params = ModelConfig.CATBOOST_DEFAULT.copy()
        
        params['random_state'] = random_state
        self.model = CatBoostClassifier(**params)
        
    def tune(self, X: np.ndarray, y: np.ndarray,
             n_trials: int = 30, timeout: int = 600) -> Dict:
        """Tune hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = CatBoostClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        self.best_params_['random_state'] = self.random_state
        self.best_params_['verbose'] = False
        self.model = CatBoostClassifier(**self.best_params_)
        
        return {'best_params': self.best_params_, 'best_score': study.best_value}