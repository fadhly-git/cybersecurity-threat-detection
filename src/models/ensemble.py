"""Ensemble methods for combining multiple models.

This module provides ensemble techniques for combining predictions
from multiple models to improve accuracy.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.logger import LoggerMixin


class EnsembleModels(LoggerMixin):
    """Ensemble methods for combining multiple models."""
    
    def __init__(self):
        """Initialize EnsembleModels."""
        super().__init__()
        self.ensemble_model = None
        
        self.logger.info("EnsembleModels initialized")
    
    def voting_ensemble(
        self,
        models: List[Tuple[str, Any]],
        voting: str = 'soft',
        weights: List[float] = None
    ) -> VotingClassifier:
        """Create voting ensemble.
        
        Args:
            models: List of (name, model) tuples
            voting: 'hard' or 'soft' voting
            weights: Optional weights for each model
        
        Returns:
            VotingClassifier
        """
        self.logger.info(f"Creating {voting} voting ensemble with {len(models)} models")
        
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        self.ensemble_model = ensemble
        return ensemble
    
    def stacking_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model: Any = None
    ) -> StackingClassifier:
        """Create stacking ensemble.
        
        Args:
            base_models: List of (name, base_model) tuples
            meta_model: Meta-learner model (default: LogisticRegression)
        
        Returns:
            StackingClassifier
        """
        if meta_model is None:
            meta_model = LogisticRegression(max_iter=1000)
        
        self.logger.info(f"Creating stacking ensemble with {len(base_models)} base models")
        
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        self.ensemble_model = ensemble
        return ensemble
    
    def weighted_average_prediction(
        self,
        predictions: List[np.ndarray],
        weights: List[float] = None
    ) -> np.ndarray:
        """Weighted average of predictions.
        
        Args:
            predictions: List of prediction arrays
            weights: Optional weights for each prediction
        
        Returns:
            Averaged predictions
        """
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        weighted_preds = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, weights):
            weighted_preds += pred * weight
        
        return weighted_preds
