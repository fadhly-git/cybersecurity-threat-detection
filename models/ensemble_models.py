"""
ensemble_models.py - Ensemble Models
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.ensemble import (
    VotingClassifier, 
    StackingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class VotingEnsembleClassifier:
    """Voting Ensemble Classifier"""
    
    def __init__(self,
                 estimators: Optional[List[Tuple[str, Any]]] = None,
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 random_state: int = 42):
        
        self.voting = voting
        self.weights = weights
        self.random_state = random_state
        
        if estimators is None:
            estimators = self._get_default_estimators()
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
    def _get_default_estimators(self) -> List[Tuple[str, Any]]:
        """Get default base estimators"""
        return [
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=20, n_jobs=-1, random_state=self.random_state
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=7, n_jobs=-1, verbosity=0, random_state=self.random_state
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=200, max_depth=7, n_jobs=-1, verbose=-1, random_state=self.random_state
            ))
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble"""
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
        """Cross validate ensemble"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}


class StackingEnsemble:
    """Stacking Ensemble Classifier"""
    
    def __init__(self,
                 base_models: Optional[List[Tuple[str, Any]]] = None,
                 meta_model: Optional[Any] = None,
                 cv: int = 5,
                 random_state: int = 42):
        
        self.cv = cv
        self.random_state = random_state
        
        if base_models is None:
            base_models = self._get_default_base_models()
        
        if meta_model is None:
            meta_model = LogisticRegression(
                max_iter=1000, 
                random_state=random_state,
                n_jobs=-1
            )
        
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=-1,
            passthrough=False
        )
        
    def _get_default_base_models(self) -> List[Tuple[str, Any]]:
        """Get default base models"""
        return [
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=15, n_jobs=-1, random_state=self.random_state
            )),
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, verbosity=0, random_state=self.random_state
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=100, max_depth=5, n_jobs=-1, verbose=-1, random_state=self.random_state
            ))
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble"""
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
        """Cross validate ensemble"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}


class WeightedEnsemble:
    """Weighted Average Ensemble"""
    
    def __init__(self,
                 models: List[Any],
                 weights: Optional[List[float]] = None):
        
        self.models = models
        self.weights = weights
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self._fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models"""
        for model in self.models:
            model.fit(X, y)
        self._fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of probabilities"""
        if not self._fitted:
            raise ValueError("Models not fitted")
        
        probas = []
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            probas.append(proba * weight)
        
        return np.sum(probas, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 100) -> List[float]:
        """Optimize weights using validation data"""
        from sklearn.metrics import f1_score
        
        # Get predictions from all models
        all_probas = [model.predict_proba(X_val) for model in self.models]
        
        best_score = 0
        best_weights = self.weights.copy()
        
        # Random search for best weights
        for _ in range(n_trials):
            # Random weights
            weights = np.random.dirichlet(np.ones(len(self.models)))
            
            # Weighted prediction
            weighted_proba = np.zeros_like(all_probas[0])
            for proba, w in zip(all_probas, weights):
                weighted_proba += proba * w
            
            y_pred = np.argmax(weighted_proba, axis=1)
            score = f1_score(y_val, y_pred, average='weighted')
            
            if score > best_score:
                best_score = score
                best_weights = weights.tolist()
        
        self.weights = best_weights
        print(f"Optimized weights: {best_weights}")
        print(f"Best F1 score: {best_score:.4f}")
        
        return best_weights