"""
models module - Machine Learning and Deep Learning models
Based on paper: arxiv.org/abs/2407.06014
"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of 
Machine and Deep Learning Techniques for Threat Detection"

ML Models: Naive Bayes, Decision Tree, Random Forest, KNN, SVM, Extra Trees
DL Models: VGG16, VGG19, ResNet18, ResNet50, Inception (adapted for tabular)
"""

# ML Models
from .ml_models import (
    BaseMLModel,
    NaiveBayesModel,
    DecisionTreeModel,
    RandomForestModel,
    KNNModel,
    SVMModel,
    ExtraTreesModel,
    get_ml_models,
    get_sklearn_models,
    # Backward compatibility
    OptimizedRandomForest,
    OptimizedDecisionTree
)

# DL Models
from .dl_models import (
    create_vgg16_tabular,
    create_vgg19_tabular,
    create_resnet18_tabular,
    create_resnet50_tabular,
    create_inception_tabular,
    get_dl_models,
    get_dl_model_creators,
    get_callbacks,
    train_dl_model
)

__all__ = [
    # ML Models
    'BaseMLModel',
    'NaiveBayesModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'KNNModel',
    'SVMModel',
    'ExtraTreesModel',
    'get_ml_models',
    'get_sklearn_models',
    'OptimizedRandomForest',
    'OptimizedDecisionTree',
    # DL Models
    'create_vgg16_tabular',
    'create_vgg19_tabular',
    'create_resnet18_tabular',
    'create_resnet50_tabular',
    'create_inception_tabular',
    'get_dl_models',
    'get_dl_model_creators',
    'get_callbacks',
    'train_dl_model'
]