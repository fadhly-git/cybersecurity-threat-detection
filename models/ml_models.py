"""
ml_models.py - Machine Learning Models from Paper
Based on paper: arxiv.org/abs/2407.06014
"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of 
Machine and Deep Learning Techniques for Threat Detection"

Models: Naive Bayes, Decision Tree, Random Forest, KNN, SVM, Extra Trees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class BaseMLModel:
    """Base class for ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
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
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
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
        """Get feature importance (for tree-based models)"""
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


# ============================================================================
# MODELS FROM PAPER
# ============================================================================

class NaiveBayesModel(BaseMLModel):
    """Naive Bayes Classifier - Model from Paper"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model = GaussianNB()
        self.name = "Naive Bayes"


class DecisionTreeModel(BaseMLModel):
    """Decision Tree Classifier - Model from Paper"""
    
    def __init__(self, max_depth: int = 20, random_state: int = 42):
        super().__init__(random_state)
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        self.name = "Decision Tree"


class RandomForestModel(BaseMLModel):
    """Random Forest Classifier - Model from Paper"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, 
                 random_state: int = 42):
        super().__init__(random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.name = "Random Forest"


class KNNModel(BaseMLModel):
    """K-Nearest Neighbors Classifier - Model from Paper"""
    
    def __init__(self, n_neighbors: int = 5, random_state: int = 42):
        super().__init__(random_state)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            n_jobs=-1
        )
        self.name = "KNN"


class SVMModel(BaseMLModel):
    """Support Vector Machine Classifier - Model from Paper"""
    
    def __init__(self, kernel: str = 'rbf', random_state: int = 42):
        super().__init__(random_state)
        self.model = SVC(
            kernel=kernel,
            probability=True,
            random_state=random_state
        )
        self.name = "SVM"


class ExtraTreesModel(BaseMLModel):
    """Extra Trees Classifier - Model from Paper (Best performing!)"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, 
                 random_state: int = 42):
        super().__init__(random_state)
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.name = "Extra Trees"


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_ml_models(random_state: int = 42) -> Dict[str, BaseMLModel]:
    """
    Get all ML models from paper as a dictionary (wrapped in class).
    
    Returns:
        Dict with model names as keys and model instances as values
    """
    return {
        'Naive Bayes': NaiveBayesModel(random_state=random_state),
        'Decision Tree': DecisionTreeModel(random_state=random_state),
        'Random Forest': RandomForestModel(random_state=random_state),
        'KNN': KNNModel(random_state=random_state),
        'SVM': SVMModel(random_state=random_state),
        'Extra Trees': ExtraTreesModel(random_state=random_state),
    }


def get_sklearn_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Get raw sklearn models (not wrapped in class).
    Useful for direct training in main.py.
    
    Returns:
        Dict with model names as keys and sklearn model instances as values
    """
    from sklearn.linear_model import SGDClassifier
    
    return {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # Use SGDClassifier with hinge loss (equivalent to linear SVM) - MUCH faster for large datasets
        'SVM': SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_state),
    }


# For backward compatibility
OptimizedRandomForest = RandomForestModel
OptimizedDecisionTree = DecisionTreeModel
