"""Train Deep Learning models.

Usage:
    python scripts/train_dl_models.py --data data/processed --models cnn,lstm --epochs 50
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.helpers import load_config
from src.utils.logger import setup_logger
from src.models.dl_models import DLModels
from src.evaluation.metrics import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Train DL models')
    parser.add_argument('--data', type=str, default='results/data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list (cnn,lstm,vgg,resnet) or "all"')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/models/dl',
                       help='Output directory')
    
    args = parser.parse_args()
    
    logger = setup_logger('train_dl')
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path(args.data)
    
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Split for validation
    val_split = 0.2
    val_size = int(len(X_train) * val_split)
    
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Determine number of classes
    num_classes = len(np.unique(y_train))
    input_shape = (X_train.shape[1], 1)
    
    # Load config
    config = load_config(args.config)
    
    # Initialize
    dl = DLModels(config)
    evaluator = ModelEvaluator()
    
    # Determine models
    if args.models == 'all':
        model_list = ['cnn', 'lstm', 'vgg', 'resnet']
    else:
        model_list = [m.strip() for m in args.models.split(',')]
    
    logger.info(f"Training models: {model_list}")
    
    # Train models
    results = {}
    
    for model_name in model_list:
        logger.info(f"\nTraining {model_name.upper()}...")
        
        if model_name == 'cnn':
            model = dl.build_cnn(input_shape, num_classes)
            name = 'CNN'
        elif model_name == 'lstm':
            model = dl.build_lstm(input_shape, num_classes)
            name = 'LSTM'
        elif model_name == 'vgg':
            model = dl.build_vgg(input_shape, num_classes)
            name = 'VGG'
        elif model_name == 'resnet':
            model = dl.build_resnet(input_shape, num_classes)
            name = 'ResNet'
        else:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        model = dl.compile_model(model)
        callbacks = dl.create_callbacks(name)
        
        history = dl.train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks_list=callbacks,
            model_name=name
        )
        
        # Evaluate
        y_pred, y_pred_proba = dl.predict(model, X_test)
        eval_results = evaluator.evaluate_model(name, y_test, y_pred, y_pred_proba)
        
        results[name] = eval_results
        
        # Save
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f'{model_name}.h5'
        dl.save_model(model, str(model_path))
    
    # Print results
    comparison_df = evaluator.compare_models(
        {name: res['metrics'] for name, res in results.items()}
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
