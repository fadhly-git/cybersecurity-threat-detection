"""
Unified Training Script for All Hybrid Models

Train CNN-SVM, LSTM-RF, CNN-LSTM-MLP, Autoencoder-CNN, Attention-LSTM, and Stacking models.

Usage:
    python scripts/train_all_models.py --dataset cicids2017 --models all --epochs 50
    python scripts/train_all_models.py --models cnn_lstm_mlp,lstm_rf --epochs 30 --apply-smote
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight

# Import hybrid models
from src.models.hybrid.cnn_svm import CNNSVMHybrid
from src.models.hybrid.lstm_rf import LSTMRandomForestHybrid
from src.models.hybrid.cnn_lstm_mlp import CNNLSTMMLPEnsemble
from src.models.hybrid.autoencoder_cnn import AutoencoderCNNHybrid
from src.models.hybrid.attention_lstm import AttentionLSTMDNN
from src.models.hybrid.stacking import StackingEnsemble

# Import data loader and evaluator
from src.data.datasets.cicids2017 import CICIDS2017Loader
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator
from src.utils.logger import DualOutput


def format_model_label(model_name: str) -> str:
    """Convert internal model key to human-readable training label."""
    return model_name.upper().replace('_', '-')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train hybrid models for cybersecurity threat detection'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cicids2017',
        help='Dataset name (default: cicids2017)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/CICIDS2017',
        help='Path to raw dataset'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated model names or "all" (default: all)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)'
    )
    
    parser.add_argument(
        '--apply-smote',
        action='store_true',
        help='Apply SMOTE oversampling'
    )
    
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=None,
        help='Sample ratio (0.0-1.0) for faster training (e.g., 0.1 = 10%% of data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--shutdown',
        action='store_true',
        help='Shutdown computer after training completes'
    )
    
    parser.add_argument(
        '--shutdown-delay',
        type=int,
        default=60,
        help='Delay before shutdown in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--load-preprocessed',
        action='store_true',
        help='Load preprocessed data instead of reprocessing'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume by skipping models with existing checkpoints and metrics'
    )

    parser.add_argument(
        '--fit-verbose',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Keras verbosity: 0=silent, 1=progress bar, 2=per-epoch (default: 2)'
    )
    
    return parser.parse_args()


def load_or_preprocess_data(args):
    """Load or preprocess dataset."""
    print("\n" + "="*60)
    print("  DATA LOADING")
    print("="*60)
    
    data_dir = f'{args.output_dir}/data'
    os.makedirs(data_dir, exist_ok=True)
    
    if args.load_preprocessed and os.path.exists(f'{data_dir}/X_train.npy'):
        # Load preprocessed data
        print("\nLoading preprocessed data...")
        X_train = np.load(f'{data_dir}/X_train.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
        
        print(f"\n✅ Data loaded:")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   y_test: {y_test.shape}")
    
    else:
        # Preprocess data
        if args.dataset == 'cicids2017':
            loader = CICIDS2017Loader(args.data_path, args.sample_ratio)
            X_train, X_test, y_train, y_test = loader.preprocess_pipeline(
                apply_smote=args.apply_smote
            )
            
            # Save preprocessed data
            loader.save_preprocessed_data(
                X_train, X_test, y_train, y_test,
                data_dir
            )
        else:
            raise ValueError(f"Dataset '{args.dataset}' not supported")
    
    # Apply sampling if requested
    if args.sample_ratio is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        print(f"\n{'='*60}")
        print(f"  📊 SAMPLING DATA: {args.sample_ratio*100}%")
        print(f"{'='*60}")

        original_size = len(X_train)

        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=args.sample_ratio,
            stratify=y_train,
            random_state=42
        )
        
        sampled_size = len(X_train)
        print(f"\n✅ Sampling completed:")
        print(f"   Original: {original_size:,} rows")
        print(f"   Sampled:  {sampled_size:,} rows ({sampled_size/original_size*100:.1f}%)")
        print(f"   Ratio:    {args.sample_ratio}")
        print(f"   Method:   Stratified (class distribution preserved)")
        
        # Show class distribution after sampling
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n   Class distribution after sampling:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"     Class {cls}: {count:,} ({pct:.2f}%)")
        
        # Re-map labels to be sequential (0, 1, 2, ..., n-1)
        # This is critical after sampling as some classes may be missing
        label_encoder = LabelEncoder()
        y_train_original = y_train.copy()
        y_train = label_encoder.fit_transform(y_train)
        y_test_original = y_test.copy()
        y_test = label_encoder.transform(y_test)
        
        print(f"\n   Label remapping (to ensure sequential 0-{len(unique)-1}):")
        for old_label, new_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
            print(f"     {old_label} → {new_label}")
        
        # Update class distribution after remapping
        unique_new, counts_new = np.unique(y_train, return_counts=True)
        print(f"\n   Final class distribution (after remapping):")
        for cls, count in zip(unique_new, counts_new):
            pct = count / len(y_train) * 100
            print(f"     Class {cls}: {count:,} ({pct:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def train_model(model_name, X_train, X_test, y_train, y_test, args):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"  TRAINING {format_model_label(model_name)}")
    print(f"{'='*60}")
    
    input_shape = (X_train.shape[1], 1)
    num_classes = len(np.unique(y_train))

    # Compute balanced class weights for imbalanced data handling
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    
    # Create output directory
    model_dir = f'{args.output_dir}/models/hybrid'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        if model_name == 'cnn_svm':
            model = CNNSVMHybrid(input_shape, num_classes, class_weight=class_weight_dict)
            model.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=args.fit_verbose,
                class_weight=class_weight_dict
            )
            model.save_model(f'{model_dir}/cnn_svm')
            
        elif model_name == 'lstm_rf':
            model = LSTMRandomForestHybrid(input_shape, num_classes, class_weight=class_weight_dict)
            model.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=args.fit_verbose,
                class_weight=class_weight_dict
            )
            model.save_model(f'{model_dir}/lstm_rf')
            
        elif model_name == 'cnn_lstm_mlp':
            model = CNNLSTMMLPEnsemble(input_shape, num_classes)
            model.compile_model(learning_rate=0.001)
            
            # Validation split
            val_size = int(0.2 * len(X_train))
            X_val = X_train[:val_size]
            y_val = y_train[:val_size]
            X_train_fit = X_train[val_size:]
            y_train_fit = y_train[val_size:]
            
            # Compute class weights to handle imbalance
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_fit),
                y=y_train_fit
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            print(f"\n📊 Class Distribution:")
            unique, counts = np.unique(y_train_fit, return_counts=True)
            for cls, count in zip(unique, counts):
                percentage = (count / len(y_train_fit)) * 100
                weight = class_weight_dict.get(cls, 1.0)
                print(f"   Class {cls}: {count:>8,} samples ({percentage:>5.2f}%) - Weight: {weight:.2f}")
            
            model.fit(
                X_train_fit, y_train_fit,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                class_weight=class_weight_dict,
                verbose=args.fit_verbose
            )
            model.save_model(f'{model_dir}/cnn_lstm_mlp.h5')
            
        elif model_name == 'autoencoder_cnn':
            model = AutoencoderCNNHybrid(input_shape, num_classes)
            
            # Pretrain on normal traffic
            X_normal = X_train[y_train == 0]
            model.pretrain_autoencoder(X_normal, epochs=100, verbose=args.fit_verbose)
            
            # Train classifier
            model.train_classifier(
                X_train, y_train,
                epochs=args.epochs,
                verbose=args.fit_verbose,
                class_weight=class_weight_dict
            )
            model.save_model(f'{model_dir}/autoencoder_cnn')
            
        elif model_name == 'attention_lstm':
            model = AttentionLSTMDNN(input_shape, num_classes, num_heads=4)
            model.compile_model(learning_rate=0.001)
            
            val_size = int(0.2 * len(X_train))
            X_val = X_train[:val_size]
            y_val = y_train[:val_size]
            X_train_fit = X_train[val_size:]
            y_train_fit = y_train[val_size:]
            
            model.fit(
                X_train_fit, y_train_fit,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=args.fit_verbose,
                class_weight=class_weight_dict
            )
            model.save_model(f'{model_dir}/attention_lstm.h5')
            
        elif model_name == 'stacking':
            model = StackingEnsemble(input_shape, num_classes)
            
            # Train base models
            model.fit_base_models(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=args.fit_verbose,
                class_weight=class_weight_dict
            )
            
            # Train meta-learner (use subset for faster training)
            subset_size = min(10000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            model.fit_meta_learner(X_train[indices], y_train[indices], cv=3)
            
            model.save_model(f'{model_dir}/stacking')
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(f'{args.output_dir}/models/hybrid', exist_ok=True)
    os.makedirs(f'{args.output_dir}/metrics', exist_ok=True)
    os.makedirs('logs/training', exist_ok=True)
    
    # Setup logging file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training/train_all_models_{timestamp}.log'
    
    # Redirect all output to log file
    with DualOutput(log_file):
        _run_training(args)


def _run_training(args):
    """Execute training workflow (called with output redirection)."""
    print(f"\n{'='*70}")
    print(f"  CYBERSECURITY THREAT DETECTION - MODEL TRAINING")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  SMOTE: {args.apply_smote}")
    print(f"  Output: {args.output_dir}")
    
    # Load/preprocess data
    X_train, X_test, y_train, y_test = load_or_preprocess_data(args)
    
    # Determine models to train
    if args.models == 'all':
        model_list = [
            'cnn_lstm_mlp',
            'lstm_rf',
            'cnn_svm',
            'attention_lstm',
            'autoencoder_cnn',
            'stacking'
        ]
    else:
        model_list = [m.strip() for m in args.models.split(',')]
    
    print(f"\n{'='*60}")
    print(f"  TRAINING {len(model_list)} MODEL(S)")
    print(f"{'='*60}")
    for i, model_name in enumerate(model_list, 1):
        print(f"  {i}. {format_model_label(model_name)}")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()

    trained_models = {}
    evaluation_results = {}
    skipped_models = set()

    if args.resume:
        print(f"\nResume mode enabled. Searching for existing metrics...")
        for model_name in model_list:
            results_file = f"{args.output_dir}/metrics/{model_name}_results.pkl"
            if not os.path.exists(results_file):
                continue

            try:
                existing_results = joblib.load(results_file)
            except Exception as exc:
                print(
                    f"  ⚠️  Could not load previous metrics for {format_model_label(model_name)}: {exc}"
                )
                continue

            evaluation_results[model_name] = existing_results
            evaluator.results[format_model_label(model_name)] = existing_results
            skipped_models.add(model_name)
            print(
                f"  ↪ Resuming without retraining {format_model_label(model_name)} (found {results_file})."
            )

        if not skipped_models:
            print("  ℹ️  No existing metrics found; full training will run.")

    for model_name in model_list:
        if model_name in skipped_models:
            print(f"\n{'='*60}")
            print(f"  SKIPPING {format_model_label(model_name)} (resume mode)")
            print(f"{'='*60}")
            continue

        model = train_model(model_name, X_train, X_test, y_train, y_test, args)
        
        if model is not None:
            trained_models[model_name] = model
            
            # Evaluate
            print(f"\n{'='*60}")
            print(f"  EVALUATING {format_model_label(model_name)}")
            print(f"{'='*60}")
            
            results = evaluator.evaluate_model(
                model, X_test, y_test,
                format_model_label(model_name)
            )
            
            # Save results
            results_file = f'{args.output_dir}/metrics/{model_name}_results.pkl'
            joblib.dump(results, results_file)
            print(f"\n✅ Results saved to: {results_file}")

            evaluation_results[model_name] = results
    
    # Final comparison
    if len(evaluator.results) > 1:
        print(f"\n{'='*60}")
        print("  FINAL MODEL COMPARISON")
        print(f"{'='*60}")
        
        evaluator.compare_models(metric='accuracy')
        evaluator.compare_models(metric='f1_macro')
        evaluator.compare_models(metric='minority_f1_avg')

    if evaluation_results:
        print(f"\n{'='*70}")
        print("  METRIC SUMMARY (TOP-LINE)")
        print(f"{'='*70}")
        header = (
            f"{'Model':25s}"
            f"{'Accuracy':>10s}"
            f"{'Precision':>12s}"
            f"{'Recall':>10s}"
            f"{'F1 Macro':>12s}"
            f"{'F1 Weighted':>14s}"
            f"{'Minority F1':>14s}"
        )
        print(header)
        print('-' * len(header))

        for model_key in model_list:
            metrics = evaluation_results.get(model_key)
            if not metrics:
                continue

            model_label = format_model_label(model_key)
            line = (
                f"{model_label:25s}"
                f"{metrics.get('accuracy', 0.0):>10.4f}"
                f"{metrics.get('precision_macro', 0.0):>12.4f}"
                f"{metrics.get('recall_macro', 0.0):>10.4f}"
                f"{metrics.get('f1_macro', 0.0):>12.4f}"
                f"{metrics.get('f1_weighted', 0.0):>14.4f}"
                f"{metrics.get('minority_f1_avg', 0.0):>14.4f}"
            )
            print(line)

        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("  TRAINING COMPLETED")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    total_models = len(model_list)
    completed_models = len(evaluation_results)
    trained_count = len(trained_models)
    resumed_count = len(skipped_models)

    print(f"\n✅ {completed_models}/{total_models} model(s) completed (trained or resumed).")
    if trained_count:
        print(f"   Trained in this run: {trained_count}")
    if resumed_count:
        print(f"   Resumed from disk:  {resumed_count}")
    if completed_models < total_models:
        remaining = total_models - completed_models
        print(f"   ⚠️  Pending/failed models: {remaining}")
    print(f"   Models saved to: {args.output_dir}/models/hybrid/")
    print(f"   Metrics saved to: {args.output_dir}/metrics/")
    print(f"{'='*70}\n")
    
    # Auto-shutdown if requested
    if args.shutdown:
        import subprocess
        import time
        
        print(f"\n{'='*70}")
        print(f"  ⚠️  AUTO-SHUTDOWN IN {args.shutdown_delay} SECONDS")
        print(f"{'='*70}")
        print(f"\n🔌 Computer akan shutdown dalam {args.shutdown_delay} detik...")
        print(f"   Press Ctrl+C sekarang untuk membatalkan!\n")
        
        try:
            for remaining in range(args.shutdown_delay, 0, -1):
                print(f"   Shutdown in {remaining} seconds...", end='\r')
                time.sleep(1)
            
            print(f"\n\n🔴 Shutting down NOW...")
            subprocess.run(['shutdown', '/s', '/t', '0'], check=True)
            
        except KeyboardInterrupt:
            print(f"\n\n✅ Shutdown cancelled by user.")
            print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
