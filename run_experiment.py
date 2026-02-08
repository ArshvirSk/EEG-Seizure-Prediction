"""
Run Experiment Script

Main entry point for training and evaluating the EEG Seizure Prediction System.

Usage:
    python run_experiment.py                    # Full experiment
    python run_experiment.py --test-mode        # Quick test with synthetic data
    python run_experiment.py --model cnn        # Train specific model type

This script:
1. Downloads/loads the CHB-MIT dataset
2. Preprocesses and segments EEG data
3. Trains the CNN+Transformer model
4. Trains baseline models for comparison
5. Evaluates all models and generates reports
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    eeg_config,
    cnn_config,
    transformer_config,
    classifier_config,
    training_config,
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
)
from data.preprocessing import EEGPreprocessor
from data.dataset import create_data_loaders, create_synthetic_dataset
from models.seizure_predictor import create_model
from models.baselines import create_baseline
from training.trainer import Trainer
from training.metrics import (
    compute_seizure_prediction_metrics,
    print_classification_report,
    compare_models,
    get_roc_curve_data,
)
from utils.visualization import (
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_multiple_roc_curves,
)
from utils.gpu_utils import setup_gpu, print_gpu_status


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EEG Seizure Prediction System - Training and Evaluation"
    )

    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode with synthetic data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["cnn_transformer", "cnn", "lstm", "cnn_lstm", "all"],
        help="Model type to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Data directory path"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save models or plots"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (auto-detected by default)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only mode")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training for faster GPU execution (default: True)",
    )
    parser.add_argument(
        "--gpu-memory-limit",
        type=int,
        default=None,
        help="GPU memory limit in MB (default: no limit)",
    )

    return parser.parse_args()


def load_or_create_data(args):
    """Load real data or create synthetic data for testing."""

    if args.test_mode:
        print("\n" + "=" * 60)
        print("RUNNING IN TEST MODE WITH SYNTHETIC DATA")
        print("=" * 60)

        # Create synthetic dataset
        X, y = create_synthetic_dataset(
            n_samples=500,
            n_channels=22,
            n_timesteps=2560,
            preictal_ratio=0.3,
            random_seed=training_config.random_seed,
        )

    else:
        print("\n" + "=" * 60)
        print("LOADING CHB-MIT EEG DATASET")
        print("=" * 60)

        # Check if data exists
        data_dir = args.data_dir or DATA_DIR

        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print("\nNo data found. Downloading CHB-MIT sample...")
            from data.download import download_chb_mit_sample

            patients = download_chb_mit_sample(
                patient_ids=["chb01", "chb02", "chb03", "chb04", "chb05"],
                data_dir=data_dir,
                max_files_per_patient=10,
            )
            print("Download complete!")

        # Try to load real EDF files
        try:
            from data.dataset import EEGDataset
            from data.download import parse_seizure_summary

            print("\nLoading real EDF files...")
            dataset = EEGDataset(data_dir=data_dir)

            total_samples = 0
            # Find patient directories
            for patient_id in os.listdir(data_dir):
                patient_dir = os.path.join(data_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue

                # Try to get seizure info from summary file
                summary_file = os.path.join(patient_dir, f"{patient_id}-summary.txt")
                if os.path.exists(summary_file):
                    seizure_info, file_list = parse_seizure_summary(summary_file)
                    print(
                        f"  Loading {patient_id}: {len(seizure_info)} seizures annotated"
                    )
                    samples = dataset.load_patient_data(patient_id, seizure_info)
                    total_samples += samples
                    print(f"    Loaded {samples} windows")

            if total_samples > 0:
                print(f"\nTotal samples loaded from EDF: {total_samples}")
                X, y = dataset.build_arrays()
            else:
                raise ValueError("No samples loaded from EDF files")

        except Exception as e:
            print(f"\nError loading EDF files: {e}")
            print("Falling back to synthetic data for demonstration.")
            X, y = create_synthetic_dataset(
                n_samples=1000,
                n_channels=22,
                n_timesteps=2560,
                preictal_ratio=0.3,
                random_seed=training_config.random_seed,
            )

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split data
    train_data, val_data, test_data = create_data_loaders(
        X,
        y,
        train_ratio=training_config.train_ratio,
        val_ratio=training_config.val_ratio,
        random_seed=training_config.random_seed,
    )

    return train_data, val_data, test_data


def train_model(model, model_name, train_data, val_data, args):
    """Train a single model."""
    X_train, y_train = train_data
    X_val, y_val = val_data

    epochs = args.epochs or training_config.epochs
    batch_size = args.batch_size or training_config.batch_size
    lr = args.lr or training_config.learning_rate

    if args.test_mode:
        epochs = min(epochs, 5)  # Limit epochs in test mode

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")

    trainer = Trainer(
        model=model,
        learning_rate=lr,
        checkpoint_dir=os.path.join(MODEL_DIR, model_name),
    )

    history = trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_class_weights=training_config.use_class_weights,
        verbose=1,
    )

    return trainer, history


def evaluate_model(trainer, model_name, test_data, save_dir=None):
    """Evaluate a trained model."""
    X_test, y_test = test_data

    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' * 60}")

    # Get predictions
    y_prob, y_pred = trainer.predict(X_test)
    y_pred = y_pred.flatten()
    y_prob = y_prob.flatten()

    # Compute metrics
    metrics = compute_seizure_prediction_metrics(y_test, y_pred, y_prob)

    # Print report
    print_classification_report(y_test, y_pred, y_prob)

    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            y_test,
            y_pred,
            save_path=os.path.join(save_dir, f"{model_name}_confusion_matrix.png"),
        )

        # ROC curve
        fpr, tpr, _, auc = get_roc_curve_data(y_test, y_prob)
        plot_roc_curve(
            fpr,
            tpr,
            auc,
            model_name,
            save_path=os.path.join(save_dir, f"{model_name}_roc_curve.png"),
        )

    return metrics, (fpr, tpr, auc)


def run_experiment(args):
    """Run the full experiment."""

    # Configure GPU
    if args.no_gpu:
        # Force CPU mode
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        print("\nGPU disabled - running on CPU only")
    else:
        # Setup GPU with optional settings
        gpu_available, num_gpus = setup_gpu(
            memory_growth=True,
            memory_limit=args.gpu_memory_limit,
            mixed_precision=args.mixed_precision and not args.test_mode,
            verbose=True,
        )
        if gpu_available:
            print(f"\n✓ GPU acceleration enabled ({num_gpus} GPU(s))")
        else:
            print("\nNo GPU available - running on CPU")

    # Set seeds
    set_seed(training_config.random_seed)

    # Load data
    train_data, val_data, test_data = load_or_create_data(args)

    # Get data shape
    X_train, y_train = train_data
    n_channels = X_train.shape[1]
    n_timesteps = X_train.shape[2]

    # Results storage
    all_results = {}
    all_histories = {}
    all_roc_curves = {}

    # Determine which models to train
    models_to_train = []
    if args.model == "all":
        models_to_train = ["cnn_transformer", "cnn", "lstm", "cnn_lstm"]
    else:
        models_to_train = [args.model]

    # Train and evaluate each model
    for model_type in models_to_train:
        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model_type.upper()}")
        print(f"{'#' * 60}")

        # Create model
        if model_type == "cnn_transformer":
            model = create_model(
                n_channels=n_channels,
                n_timesteps=n_timesteps,
                conv_filters=cnn_config.conv_filters,
                conv_kernel_sizes=cnn_config.conv_kernel_sizes,
                pool_sizes=cnn_config.pool_sizes,
                cnn_dropout=cnn_config.dropout_rate,
                n_heads=transformer_config.n_heads,
                n_transformer_layers=transformer_config.n_layers,
                d_ff=transformer_config.d_ff,
                transformer_dropout=transformer_config.dropout_rate,
                classifier_hidden=classifier_config.hidden_units,
                classifier_dropout=classifier_config.dropout_rate,
            )
        else:
            model = create_baseline(
                model_type, n_channels=n_channels, n_timesteps=n_timesteps
            )

        # Build and print model summary
        model.build((None, n_channels, n_timesteps))
        print(f"\nModel parameters: {model.count_params():,}")

        # Train
        trainer, history = train_model(model, model_type, train_data, val_data, args)
        all_histories[model_type] = history

        # Evaluate
        save_dir = None if args.no_save else os.path.join(RESULTS_DIR, model_type)
        metrics, roc_data = evaluate_model(trainer, model_type, test_data, save_dir)
        all_results[model_type] = metrics
        all_roc_curves[model_type] = roc_data

        # Save training history plot
        if not args.no_save and save_dir:
            plot_training_history(
                history, save_path=os.path.join(save_dir, "training_history.png")
            )

        # Save model
        if not args.no_save:
            model_path = os.path.join(MODEL_DIR, f"{model_type}_final.keras")
            trainer.save_model(model_path)

    # Compare models
    print("\n")
    compare_models(all_results)

    # Save comparison plots
    if not args.no_save and len(models_to_train) > 1:
        save_dir = RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)

        # Model comparison bar chart
        plot_model_comparison(
            all_results,
            metrics=["accuracy", "sensitivity", "specificity", "f1_score", "auc_roc"],
            save_path=os.path.join(save_dir, "model_comparison.png"),
        )

        # Multiple ROC curves
        plot_multiple_roc_curves(
            all_roc_curves, save_path=os.path.join(save_dir, "roc_comparison.png")
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)

    if not args.no_save:
        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"Models saved to: {MODEL_DIR}")

    return all_results, all_histories


if __name__ == "__main__":
    args = parse_args()
    results, histories = run_experiment(args)
