"""Complete end-to-end pipeline for cybersecurity threat detection.

Usage:
    python scripts/run_pipeline.py --config config/config.yaml --dataset cybersecurity_attacks
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.helpers import load_config, Timer, ensure_dir
from src.utils.logger import setup_logger
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.ml_models import MLModels
from src.models.dl_models import DLModels
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import Visualizer


def main():
    """Run complete pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run complete threat detection pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='realistic_attacks',
                       help='Dataset name from config')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML model training')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip DL model training')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger('pipeline', level=20)  # INFO level
    
    logger.info("=" * 80)
    logger.info("CYBERSECURITY THREAT DETECTION PIPELINE")
    logger.info("=" * 80)
    
    try:
        # 1. Load configuration
        logger.info("\n[1/9] Loading configuration...")
        config = load_config(args.config)
        
        # 2. Load dataset
        logger.info("\n[2/9] Loading dataset...")
        loader = DataLoader()
        df, target_column = loader.load_from_config(config, args.dataset)
        
        logger.info(f"Dataset loaded: {df.shape}")
        logger.info(f"Target column: {target_column}")
        
        # 3. Run preprocessing pipeline
        logger.info("\n[3/9] Running preprocessing pipeline...")
        preprocessor = DataPreprocessor(config)
        
        with Timer("Preprocessing"):
            X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
                df, target_column, config
            )
        
        # Save preprocessed data
        output_dir = ensure_dir(args.output_dir)
        data_dir = ensure_dir(output_dir / 'data')
        
        np.save(data_dir / 'X_train.npy', X_train)
        np.save(data_dir / 'X_test.npy', X_test)
        np.save(data_dir / 'y_train.npy', y_train)
        np.save(data_dir / 'y_test.npy', y_test)
        
        logger.info(f"Preprocessed data saved to {data_dir}")
        
        # Save preprocessor artifacts
        preprocessor.save_preprocessor(data_dir / 'preprocessor')
        
        # 4. Feature engineering (optional)
        logger.info("\n[4/9] Feature engineering...")
        fe_config = config.get('feature_engineering', {})
        
        if fe_config.get('feature_selection', {}).get('enabled', False):
            engineer = FeatureEngineer()
            method = fe_config['feature_selection'].get('method', 'importance')
            n_features = fe_config['feature_selection'].get('n_features', 50)
            
            X_train_selected, selected_indices = engineer.select_features(
                X_train, y_train, method=method, n_features=n_features
            )
            X_test_selected = X_test[:, selected_indices]
            
            # Use selected features for training
            X_train = X_train_selected
            X_test = X_test_selected
            
            logger.info(f"Selected {len(selected_indices)} features")
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        logger.info(f"Number of classes: {num_classes}")
        
        # Initialize results dictionary
        all_results = {}
        
        # 5. Train ML models
        if not args.skip_ml:
            logger.info("\n[5/9] Training ML models...")
            
            ml_models = MLModels(config)
            ml_results = ml_models.train_all_models(
                X_train, y_train,
                use_cv=config.get('training', {}).get('use_cross_validation', True),
                cv_folds=config.get('training', {}).get('cv_folds', 5)
            )
            
            # Save ML models
            models_dir = ensure_dir(output_dir / 'models' / 'ml')
            
            for model_name, (model, metrics) in ml_results.items():
                model_path = models_dir / f'{model_name.lower()}.pkl'
                ml_models.save_model(model, str(model_path))
            
            logger.info(f"ML models saved to {models_dir}")
        else:
            ml_results = {}
            logger.info("\n[5/9] Skipping ML model training")
        
        # 6. Train DL models
        if not args.skip_dl:
            logger.info("\n[6/9] Training DL models...")
            
            # Split training data for validation
            val_split = 0.2
            val_size = int(len(X_train) * val_split)
            
            X_val = X_train[:val_size]
            y_val = y_train[:val_size]
            X_train_dl = X_train[val_size:]
            y_train_dl = y_train[val_size:]
            
            dl_models = DLModels(config)
            dl_config = config.get('models', {}).get('dl', {}).get('cnn', {})
            
            epochs = dl_config.get('epochs', 50)
            batch_size = dl_config.get('batch_size', 32)
            
            dl_results = dl_models.train_all_models(
                X_train_dl, y_train_dl,
                X_val, y_val,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Save DL models
            models_dir = ensure_dir(output_dir / 'models' / 'dl')
            
            for model_name, (model, history) in dl_results.items():
                model_path = models_dir / f'{model_name.lower()}.h5'
                dl_models.save_model(model, str(model_path))
            
            logger.info(f"DL models saved to {models_dir}")
        else:
            dl_results = {}
            logger.info("\n[6/9] Skipping DL model training")
        
        # 7. Evaluate all models
        logger.info("\n[7/9] Evaluating all models...")
        
        evaluator = ModelEvaluator()
        
        # Evaluate ML models
        for model_name, (model, train_metrics) in ml_results.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            y_pred, y_pred_proba = ml_models.predict(model, X_test)
            
            results = evaluator.evaluate_model(
                model_name, y_test, y_pred, y_pred_proba
            )
            
            # Add training metrics
            results['training_metrics'] = train_metrics
            
            all_results[model_name] = results
        
        # Evaluate DL models
        for model_name, (model, history) in dl_results.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            y_pred, y_pred_proba = dl_models.predict(model, X_test)
            
            results = evaluator.evaluate_model(
                model_name, y_test, y_pred, y_pred_proba
            )
            
            all_results[model_name] = results
        
        # 8. Generate comparison report
        logger.info("\n[8/9] Generating comparison report...")
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, results in all_results.items():
            row = {'model': model_name}
            row.update(results['metrics'])
            comparison_data.append(row)
        
        comparison_df = evaluator.compare_models(
            {name: res['metrics'] for name, res in all_results.items()}
        )
        
        all_results['comparison'] = comparison_df
        
        # Print comparison
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        print(comparison_df.to_string(index=False))
        logger.info("=" * 80)
        
        # Generate detailed report
        evaluator.generate_report(all_results, output_dir / 'evaluation_report.md')
        
        # 9. Create visualizations
        logger.info("\n[9/9] Creating visualizations...")
        
        visualizer = Visualizer()
        viz_dir = ensure_dir(output_dir / 'visualizations')
        
        # Model comparison plot
        visualizer.plot_model_comparison(
            comparison_df,
            save_path=viz_dir / 'model_comparison.png'
        )
        
        # Confusion matrices
        cm_dir = ensure_dir(viz_dir / 'confusion_matrices')
        
        for model_name, results in all_results.items():
            if model_name == 'comparison':
                continue
            
            if 'confusion_matrix' in results:
                class_names = [f'Class {i}' for i in range(num_classes)]
                visualizer.plot_confusion_matrix(
                    results['confusion_matrix'],
                    class_names,
                    save_path=cm_dir / f'{model_name}_confusion_matrix.png'
                )
        
        # ROC curves
        roc_data_all = {}
        for model_name, results in all_results.items():
            if model_name != 'comparison' and 'roc_data' in results:
                roc_data_all[model_name] = results['roc_data']
        
        if roc_data_all:
            visualizer.plot_roc_curves(
                roc_data_all,
                save_path=viz_dir / 'roc_curves.png'
            )
        
        # Training history for DL models
        for model_name, (model, history) in dl_results.items():
            visualizer.plot_training_history(
                history,
                save_path=viz_dir / f'{model_name}_training_history.png'
            )
        
        # Create interactive dashboard
        visualizer.create_dashboard(
            all_results,
            save_path=output_dir / 'dashboard.html'
        )
        
        logger.info(f"\nAll visualizations saved to {viz_dir}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Models saved to: {output_dir / 'models'}")
        logger.info(f"Visualizations saved to: {viz_dir}")
        logger.info(f"Dashboard: {output_dir / 'dashboard.html'}")
        logger.info("=" * 80)
        
        # Best model
        best_model = comparison_df.iloc[0]
        logger.info(f"\nBest Model: {best_model['model']}")
        logger.info(f"Accuracy: {best_model['accuracy']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
