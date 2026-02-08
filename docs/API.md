# API Reference

## Core Modules

---

## `data.preprocessing` - EEG Signal Preprocessing

### `EEGPreprocessor`

Comprehensive EEG signal preprocessor implementing bandpass filtering, notch filtering, normalization, and windowing.

```python
from data.preprocessing import EEGPreprocessor

preprocessor = EEGPreprocessor(
    sampling_rate=256,       # Hz
    bandpass_low=0.5,        # Hz
    bandpass_high=40.0,      # Hz
    notch_freq=60.0,         # Hz (50 for EU)
    notch_width=2.0,         # Hz
    window_duration=10.0,    # seconds
    window_overlap=0.5       # 50% overlap
)
```

#### Methods

| Method                  | Description                 | Input                 | Output                         |
| ----------------------- | --------------------------- | --------------------- | ------------------------------ |
| `preprocess(data)`      | Full preprocessing pipeline | `(channels, samples)` | `(channels, samples)`          |
| `bandpass_filter(data)` | Apply bandpass filter       | `(channels, samples)` | `(channels, samples)`          |
| `notch_filter(data)`    | Remove powerline noise      | `(channels, samples)` | `(channels, samples)`          |
| `normalize(data)`       | Z-score normalization       | `(channels, samples)` | `(channels, samples)`          |
| `segment(data)`         | Window segmentation         | `(channels, samples)` | `(windows, channels, samples)` |

#### Example

```python
import numpy as np

# Raw EEG: 22 channels × 10 minutes at 256 Hz
raw_eeg = np.random.randn(22, 256 * 600)

# Preprocess
processed = preprocessor.preprocess(raw_eeg)

# Segment into windows
windows = preprocessor.segment(processed)
print(windows.shape)  # (N_windows, 22, 2560)
```

---

## `data.dataset` - Data Loading

### `create_data_loaders()`

Create train/validation/test data loaders from processed EEG data.

```python
from data.dataset import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    data_dir="data/processed",
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    shuffle=True,
    seed=42
)
```

### `create_synthetic_dataset()`

Generate synthetic EEG-like data for testing.

```python
from data.dataset import create_synthetic_dataset

X_train, X_val, X_test, y_train, y_val, y_test = create_synthetic_dataset(
    n_samples=1000,
    n_channels=22,
    n_timesteps=2560,
    preictal_ratio=0.3
)
```

---

## `models.seizure_predictor` - Main Model

### `SeizurePredictorCNNTransformer`

The main CNN + Transformer architecture for seizure prediction.

```python
from models.seizure_predictor import create_model

model = create_model(
    # Input shape
    n_channels=22,
    n_timesteps=2560,

    # CNN Encoder
    conv_filters=(64, 128, 256),
    conv_kernel_sizes=(7, 5, 3),
    pool_sizes=(2, 2, 2),
    cnn_dropout=0.3,

    # Transformer Encoder
    n_heads=8,
    n_transformer_layers=4,
    d_ff=512,
    transformer_dropout=0.1,

    # Classification Head
    classifier_hidden=(128, 64),
    classifier_dropout=0.5
)

model.summary()
```

#### Model Architecture

```
Input: (batch, 22, 2560)
   │
   ├─► CNN Encoder
   │      Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)
   │      Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
   │      Conv1D(256, k=3) → BatchNorm → ReLU → MaxPool(2)
   │
   ├─► Positional Encoding
   │
   ├─► Transformer Encoder × 4
   │      Multi-Head Attention (8 heads)
   │      Feed-Forward Network (512 units)
   │
   ├─► Global Average Pooling
   │
   ├─► Dense(128) → Dropout → Dense(64) → Dropout
   │
   └─► Dense(1, sigmoid)

Output: (batch, 1) - Preictal probability
```

---

## `models.baselines` - Baseline Models

### `create_baseline()`

Create baseline models for comparison.

```python
from models.baselines import create_baseline

# Pure CNN model
cnn_model = create_baseline(
    model_type="cnn",
    n_channels=22,
    n_timesteps=2560
)

# Pure LSTM model
lstm_model = create_baseline(
    model_type="lstm",
    n_channels=22,
    n_timesteps=2560
)

# CNN-LSTM hybrid
cnn_lstm_model = create_baseline(
    model_type="cnn_lstm",
    n_channels=22,
    n_timesteps=2560
)
```

---

## `training.trainer` - Training Pipeline

### `Trainer`

Handles the complete training workflow with callbacks.

```python
from training.trainer import Trainer

trainer = Trainer(
    model=model,
    learning_rate=1e-4,
    use_class_weights=True
)

history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    save_dir="saved_models/cnn_transformer",
    early_stopping_patience=15
)
```

#### Callbacks Included

- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Stops training when validation loss plateaus
- **ReduceLROnPlateau**: Reduces learning rate when stuck
- **TensorBoard**: Logs metrics for visualization

---

## `training.metrics` - Evaluation Metrics

### `compute_metrics()`

Compute comprehensive classification metrics.

```python
from training.metrics import compute_metrics

metrics = compute_metrics(
    y_true=y_test,
    y_pred=predictions,
    y_prob=probabilities,
    threshold=0.5
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Sensitivity: {metrics['sensitivity']:.2%}")
print(f"Specificity: {metrics['specificity']:.2%}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### `compute_false_prediction_rate()`

Calculate clinical metric: false alarms per hour.

```python
from training.metrics import compute_false_prediction_rate

fpr = compute_false_prediction_rate(
    y_true=y_test,
    y_pred=predictions,
    window_duration=10.0,  # seconds per window
    total_duration_hours=24.0  # total recording time
)
print(f"False Prediction Rate: {fpr:.2f}/hour")
```

---

## `inference.SeizurePredictor` - Deployment

Production-ready inference class.

```python
from inference import SeizurePredictor

predictor = SeizurePredictor(
    model_path="saved_models/cnn_transformer_final.keras",
    threshold=0.5
)

# Single prediction
result = predictor.predict(eeg_window)
# Returns: {"probability": 0.85, "prediction": "preictal", "confidence": 0.85}

# Batch prediction
results = predictor.predict_batch(eeg_windows)
```

---

## Configuration Objects

All hyperparameters are defined in `config.py`:

```python
from config import (
    eeg_config,         # EEGConfig dataclass
    cnn_config,         # CNNConfig dataclass
    transformer_config, # TransformerConfig dataclass
    classifier_config,  # ClassifierConfig dataclass
    training_config     # TrainingConfig dataclass
)

# View all settings
from config import print_config
print_config()
```
