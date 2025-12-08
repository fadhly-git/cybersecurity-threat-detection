"""Evaluate trained models.

Usage:
    python scripts/evaluate_models.py --models-dir models/ --data data/processed
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.logger import setup_logger
from src.models.ml_models import MLModels
from src.models.dl_models import DLModels
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--models-dir', type=str, default='results/models',
                       help='Directory containing trained models')
    parser.add_argument('--data', type=str, default='results/data',
                       help='Directory containing test data')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    logger = setup_logger('evaluate')
    
    # Load test data
    logger.info("Loading test data...")
    data_dir = Path(args.data)
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Initialize
    ml = MLModels()
    dl = DLModels()
    evaluator = ModelEvaluator()
    visualizer = Visualizer()
    
    all_results = {}
    
    # Evaluate ML models
    ml_dir = Path(args.models_dir) / 'ml'
    if ml_dir.exists():
        logger.info("\nEvaluating ML models...")
        
        for model_path in ml_dir.glob('*.pkl'):
            model_name = model_path.stem.upper()
            
            logger.info(f"Loading {model_name}...")
            model = ml.load_model(str(model_path))
            
            y_pred, y_pred_proba = ml.predict(model, X_test)
            results = evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)
            
            all_results[model_name] = results
    
    # Evaluate DL models
    dl_dir = Path(args.models_dir) / 'dl'
    if dl_dir.exists():
        logger.info("\nEvaluating DL models...")
        
        for model_path in dl_dir.glob('*.h5'):
            model_name = model_path.stem.upper()
            
            logger.info(f"Loading {model_name}...")
            model = dl.load_model(str(model_path))
            
            y_pred, y_pred_proba = dl.predict(model, X_test)
            results = evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)
            
            all_results[model_name] = results
    
    # Generate comparison
    comparison_df = evaluator.compare_models(
        {name: res['metrics'] for name, res in all_results.items()}
    )
    
    all_results['comparison'] = comparison_df
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Save report
    output_dir = Path(args.output)
    evaluator.generate_report(all_results, output_dir / 'evaluation_report.md')
    
    # Create visualizations
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer.plot_model_comparison(
        comparison_df,
        save_path=viz_dir / 'model_comparison.png'
    )
    
    visualizer.create_dashboard(
        all_results,
        save_path=output_dir / 'dashboard.html'
    )
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
