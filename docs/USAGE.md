# Usage Guide

## Quick Start

### Test Mode (No Dataset Required)

Run with synthetic data to verify everything works:

```bash
python run_experiment.py --test-mode
```

### Full Training

```bash
python run_experiment.py --model all
```

---

## Training Models

### Train All Models

```bash
python run_experiment.py --model all
```

### Train Specific Model

```bash
# CNN + Transformer (Main Model)
python run_experiment.py --model cnn_transformer --epochs 100

# Baseline Models
python run_experiment.py --model cnn --epochs 50
python run_experiment.py --model lstm --epochs 50
python run_experiment.py --model cnn_lstm --epochs 50
```

### Custom Training Parameters

```bash
python run_experiment.py \
    --model cnn_transformer \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --data-dir /path/to/data
```

### Training Without Saving

```bash
python run_experiment.py --test-mode --no-save
```

---

## Inference / Prediction

### Interactive Demo

```bash
python inference.py --interactive
```

### Predict on EEG File

```bash
# Using .npy file (preprocessed)
python inference.py \
    --model-path saved_models/cnn_transformer_final.keras \
    --input data/my_eeg.npy

# Output includes:
# - Seizure probability per window
# - Overall prediction (Preictal/Interictal)
# - Confidence score
```

### Batch Prediction

```python
from inference import SeizurePredictor
import numpy as np

# Load predictor
predictor = SeizurePredictor(
    model_path="saved_models/cnn_transformer_final.keras",
    threshold=0.5
)

# Load your EEG data (22 channels × N samples)
eeg_data = np.load("my_eeg_recording.npy")

# Get predictions
results = predictor.predict(eeg_data)
print(f"Seizure Probability: {results['probability']:.2%}")
print(f"Prediction: {results['prediction']}")
```

---

## Working with Jupyter Notebook

Open `EEG.ipynb` for interactive exploration:

```bash
jupyter notebook EEG.ipynb
```

The notebook includes:

- Data exploration and visualization
- Step-by-step preprocessing demo
- Model training with live plots
- Attention weight visualization

---

## Command Line Reference

```
usage: run_experiment.py [-h] [--test-mode] [--model MODEL] [--epochs EPOCHS]
                         [--batch-size BATCH_SIZE] [--lr LR]
                         [--data-dir DATA_DIR] [--no-save]

Arguments:
  --test-mode           Run with synthetic data (quick test)
  --model MODEL         Model type: cnn_transformer, cnn, lstm, cnn_lstm, all
  --epochs EPOCHS       Number of training epochs
  --batch-size SIZE     Training batch size
  --lr LR               Learning rate
  --data-dir PATH       Custom data directory
  --no-save             Don't save models or plots
```

```
usage: inference.py [-h] [--model-path PATH] [--input FILE] [--interactive]
                    [--threshold FLOAT]

Arguments:
  --model-path PATH     Path to trained .keras model
  --input FILE          Input EEG file (.npy format)
  --interactive         Run interactive demo mode
  --threshold FLOAT     Classification threshold (default: 0.5)
```

---

## Output Files

After training, you'll find:

```
saved_models/
├── cnn_transformer_final.keras    # Best CNN+Transformer model
├── cnn_final.keras                # Best CNN baseline
├── lstm_final.keras               # Best LSTM baseline
├── cnn_lstm_final.keras           # Best CNN-LSTM baseline
└── cnn_transformer/
    └── logs/                      # TensorBoard logs

results/
├── cnn_transformer/
│   ├── training_history.png       # Loss/accuracy curves
│   ├── confusion_matrix.png       # Test confusion matrix
│   ├── roc_curve.png              # ROC curve
│   └── metrics.json               # All evaluation metrics
└── model_comparison.png           # Side-by-side comparison
```

### View TensorBoard Logs

```bash
tensorboard --logdir saved_models/cnn_transformer/logs
```
