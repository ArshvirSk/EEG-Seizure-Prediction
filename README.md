# EEG-Based Pediatric Seizure Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning framework for predicting epileptic seizures from pediatric EEG recordings using a CNN + Transformer architecture.

## 📚 Documentation

| Document                                   | Description                          |
| ------------------------------------------ | ------------------------------------ |
| [Installation Guide](docs/INSTALLATION.md) | Setup instructions for all platforms |
| [Usage Guide](docs/USAGE.md)               | How to train and use the models      |
| [API Reference](docs/API.md)               | Complete module documentation        |
| [Architecture](docs/ARCHITECTURE.md)       | Model design and components          |
| [Configuration](docs/CONFIGURATION.md)     | All hyperparameters explained        |
| [Contributing](CONTRIBUTING.md)            | How to contribute to the project     |

## Features

- **CNN + Transformer Architecture**: Combines convolutional neural networks for local feature extraction with Transformer encoders for capturing long-range temporal dependencies
- **Comprehensive Preprocessing**: Bandpass filtering, notch filtering, z-score normalization, and overlapping window segmentation
- **Multiple Baselines**: CNN, LSTM, and CNN-LSTM hybrid models for benchmarking
- **Clinical Metrics**: Sensitivity, specificity, AUC-ROC, and false prediction rate per hour
- **Interpretability**: Attention weight visualization for understanding model decisions

## Installation

```bash
# Install dependencies
pip install tensorflow numpy scipy scikit-learn matplotlib seaborn mne

# For GPU support (recommended)
pip install tensorflow[and-cuda]
```

## Quick Start

### 1. Test Mode (Synthetic Data)

```bash
python run_experiment.py --test-mode
```

### 2. Full Training

```bash
# Train all models
python run_experiment.py --model all

# Train specific model
python run_experiment.py --model cnn_transformer --epochs 100
```

### 3. Inference

```bash
# Interactive demo
python inference.py --interactive

# Predict on EEG file
python inference.py --model-path saved_models/cnn_transformer_final.keras --input eeg_data.npy
```

## Project Structure

```
Code/
├── config.py                 # Configuration settings
├── run_experiment.py         # Main training script
├── inference.py              # Inference/deployment script
├── data/
│   ├── download.py           # CHB-MIT dataset download
│   ├── preprocessing.py      # EEG preprocessing
│   └── dataset.py            # Dataset and data loading
├── models/
│   ├── cnn_encoder.py        # CNN feature extractor
│   ├── transformer_encoder.py # Transformer encoder
│   ├── seizure_predictor.py  # Main CNN+Transformer model
│   └── baselines.py          # Baseline models for comparison
├── training/
│   ├── trainer.py            # Training loop and callbacks
│   └── metrics.py            # Evaluation metrics
└── utils/
    └── visualization.py      # Plotting functions
```

## Model Architecture

```
Input (22 channels × 2560 samples)
    │
    ▼
CNN Encoder (Conv1D → BatchNorm → ReLU → MaxPool) × 3
    │
    ▼
Positional Encoding
    │
    ▼
Transformer Encoder (Multi-Head Attention → FFN) × 4
    │
    ▼
Classification Head (Global Pool → Dense → Sigmoid)
    │
    ▼
Output: Preictal Probability [0, 1]
```

## Configuration

Edit `config.py` to modify:

- EEG preprocessing parameters (filtering, windowing)
- Model architecture (CNN filters, Transformer heads/layers)
- Training parameters (learning rate, batch size, epochs)

## Dataset

This system is designed for the **CHB-MIT Scalp EEG Database**:

- 23 pediatric subjects with intractable seizures
- 22-channel EEG recordings at 256 Hz
- Source: [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)

## Performance Targets

| Metric                | Target   |
| --------------------- | -------- |
| Sensitivity           | ≥ 90%    |
| False Prediction Rate | ≤ 1/hour |
| AUC-ROC               | ≥ 0.85   |

## License

MIT License
