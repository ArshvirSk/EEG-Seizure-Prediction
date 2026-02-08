# Configuration Reference

All configuration parameters are defined in `config.py` using Python dataclasses for type safety and documentation.

---

## EEG Processing Configuration

```python
@dataclass
class EEGConfig:
    # Sampling parameters
    original_sampling_rate: int = 256    # CHB-MIT dataset (Hz)
    target_sampling_rate: int = 256      # After resampling (Hz)

    # Channel configuration
    n_channels: int = 22                 # Standard EEG channels

    # Filtering parameters
    bandpass_low: float = 0.5            # High-pass cutoff (Hz)
    bandpass_high: float = 40.0          # Low-pass cutoff (Hz)
    notch_freq: float = 60.0             # Powerline noise (60 US, 50 EU)
    notch_width: float = 2.0             # Notch filter width (Hz)

    # Windowing parameters
    window_duration: float = 10.0        # Window size (seconds)
    window_overlap: float = 0.5          # 50% overlap

    # Seizure labeling
    preictal_duration: int = 300         # 5 minutes before seizure
    seizure_prediction_horizon: int = 30 # Predict within 30 seconds
```

| Parameter                    | Default | Description                                |
| ---------------------------- | ------- | ------------------------------------------ |
| `original_sampling_rate`     | 256 Hz  | CHB-MIT native sampling rate               |
| `target_sampling_rate`       | 256 Hz  | Output sampling rate                       |
| `n_channels`                 | 22      | Number of EEG channels                     |
| `bandpass_low`               | 0.5 Hz  | Remove slow drifts                         |
| `bandpass_high`              | 40 Hz   | Remove high-frequency noise                |
| `notch_freq`                 | 60 Hz   | Powerline interference (use 50 for Europe) |
| `window_duration`            | 10 sec  | Each input window                          |
| `window_overlap`             | 0.5     | 50% overlap between windows                |
| `preictal_duration`          | 300 sec | Preictal period before seizure onset       |
| `seizure_prediction_horizon` | 30 sec  | How far ahead to predict                   |

---

## CNN Encoder Configuration

```python
@dataclass
class CNNConfig:
    conv_filters: Tuple[int, ...] = (64, 128, 256)
    conv_kernel_sizes: Tuple[int, ...] = (7, 5, 3)
    pool_sizes: Tuple[int, ...] = (2, 2, 2)
    activation: str = "relu"
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
```

| Parameter           | Default        | Description                |
| ------------------- | -------------- | -------------------------- |
| `conv_filters`      | (64, 128, 256) | Filters per layer          |
| `conv_kernel_sizes` | (7, 5, 3)      | Kernel sizes per layer     |
| `pool_sizes`        | (2, 2, 2)      | Max pooling sizes          |
| `activation`        | "relu"         | Activation function        |
| `dropout_rate`      | 0.3            | Dropout probability        |
| `use_batch_norm`    | True           | Enable batch normalization |

---

## Transformer Encoder Configuration

```python
@dataclass
class TransformerConfig:
    d_model: int = 256           # Must match CNN output
    n_heads: int = 8             # Attention heads
    n_layers: int = 4            # Encoder layers
    d_ff: int = 512              # FFN hidden dimension
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    max_seq_length: int = 500    # Max positional encoding
```

| Parameter           | Default | Description                   |
| ------------------- | ------- | ----------------------------- |
| `d_model`           | 256     | Model/embedding dimension     |
| `n_heads`           | 8       | Number of attention heads     |
| `n_layers`          | 4       | Number of encoder layers      |
| `d_ff`              | 512     | Feed-forward hidden dimension |
| `dropout_rate`      | 0.1     | General dropout               |
| `attention_dropout` | 0.1     | Attention-specific dropout    |
| `max_seq_length`    | 500     | Maximum sequence length       |

---

## Classification Head Configuration

```python
@dataclass
class ClassifierConfig:
    hidden_units: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.5
    n_classes: int = 1           # Binary classification
```

| Parameter      | Default   | Description                   |
| -------------- | --------- | ----------------------------- |
| `hidden_units` | (128, 64) | Dense layer sizes             |
| `dropout_rate` | 0.5       | Dropout probability           |
| `n_classes`    | 1         | Output classes (1 for binary) |

---

## Training Configuration

```python
@dataclass
class TrainingConfig:
    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = "adam"

    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-7

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Model checkpointing
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    # Class balancing
    use_class_weights: bool = True

    # Reproducibility
    random_seed: int = 42
```

| Parameter                 | Default             | Description                |
| ------------------------- | ------------------- | -------------------------- |
| `train_ratio`             | 0.70                | Training set proportion    |
| `val_ratio`               | 0.15                | Validation set proportion  |
| `test_ratio`              | 0.15                | Test set proportion        |
| `batch_size`              | 32                  | Training batch size        |
| `epochs`                  | 100                 | Maximum training epochs    |
| `learning_rate`           | 1e-4                | Initial learning rate      |
| `weight_decay`            | 1e-5                | L2 regularization          |
| `optimizer`               | "adam"              | Optimizer type             |
| `lr_scheduler`            | "reduce_on_plateau" | LR schedule type           |
| `lr_patience`             | 5                   | Epochs before LR reduction |
| `lr_factor`               | 0.5                 | LR reduction factor        |
| `early_stopping_patience` | 15                  | Epochs before early stop   |
| `use_class_weights`       | True                | Balance class frequencies  |
| `random_seed`             | 42                  | For reproducibility        |

---

## Modifying Configuration

### Option 1: Edit config.py directly

```python
# config.py
eeg_config = EEGConfig(
    bandpass_high=50.0,  # Changed from 40
    window_duration=5.0  # Changed from 10
)
```

### Option 2: Override at runtime

```python
from config import EEGConfig, TrainingConfig

# Create custom configs
my_eeg_config = EEGConfig(
    notch_freq=50.0,  # For European data
    window_overlap=0.75
)

my_training_config = TrainingConfig(
    batch_size=64,
    learning_rate=5e-5,
    epochs=200
)
```

### Option 3: Command line arguments

```bash
python run_experiment.py --epochs 200 --batch-size 64 --lr 0.00005
```

---

## Recommended Configurations

### Low Memory (8GB RAM)

```python
training_config = TrainingConfig(
    batch_size=16,
    epochs=50
)
```

### High Performance (32GB+ RAM, GPU)

```python
training_config = TrainingConfig(
    batch_size=128,
    epochs=200,
    learning_rate=3e-4
)
```

### Quick Testing

```python
training_config = TrainingConfig(
    batch_size=32,
    epochs=10,
    early_stopping_patience=3
)
```
