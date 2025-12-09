"""Train Machine Learning models.

Usage:
    python scripts/train_ml_models.py --data data/processed --models rf,xgb --cv 5
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.helpers import load_config, Timer
from src.utils.logger import setup_logger, DualOutput
from src.models.ml_models import MLModels
from src.evaluation.metrics import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--data', type=str, default='results/data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models (rf,svm,xgb,gb) or "all"')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/models/ml',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Setup logging with timestamp
    Path('logs/training').mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training/train_ml_models_{timestamp}.log'
    
    # Redirect all output to log file
    with DualOutput(log_file):
        _run_ml_training(args)


def _run_ml_training(args):
    """Execute ML training workflow."""
    logger = setup_logger('train_ml')
    
    logger.info("Loading preprocessed data...")
    data_dir = Path(args.data)
    
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load config
    config = load_config(args.config)
    
    # Initialize models
    ml = MLModels(config)
    evaluator = ModelEvaluator()
    
    # Determine which models to train
    if args.models == 'all':
        model_list = ['rf', 'svm', 'xgb', 'gb']
    else:
        model_list = [m.strip() for m in args.models.split(',')]
    
    logger.info(f"Training models: {model_list}")
    
    # Train selected models
    results = {}
    
    for model_name in model_list:
        if model_name == 'rf':
            model = ml.build_random_forest()
            name = 'RandomForest'
        elif model_name == 'svm':
            model = ml.build_svm()
            name = 'SVM'
        elif model_name == 'xgb':
            model = ml.build_xgboost()
            name = 'XGBoost'
        elif model_name == 'gb':
            model = ml.build_gradient_boosting()
            name = 'GradientBoosting'
        else:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        # Train
        trained_model, metrics = ml.train_model(
            model, X_train, y_train,
            use_cv=True, cv_folds=args.cv,
            model_name=name
        )
        
        # Evaluate
        y_pred, y_pred_proba = ml.predict(trained_model, X_test)
        eval_results = evaluator.evaluate_model(name, y_test, y_pred, y_pred_proba)
        
        results[name] = eval_results
        
        # Save model
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f'{model_name}.pkl'
        ml.save_model(trained_model, str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    # Print comparison
    comparison_df = evaluator.compare_models(
        {name: res['metrics'] for name, res in results.items()}
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    logger.info("Training complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
