# Model Architecture

## Overview

This system implements a **CNN + Transformer** hybrid architecture for epileptic seizure prediction from EEG signals. The model predicts whether a seizure will occur within a defined prediction horizon (default: 30 seconds).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│              EEG Window: (22 channels × 2560 samples)           │
│                    ~10 seconds at 256 Hz                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CNN ENCODER                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)         │   │
│  │ Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)        │   │
│  │ Conv1D(256, k=3) → BatchNorm → ReLU → MaxPool(2)        │   │
│  │ Dropout(0.3)                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Output: (batch, seq_len, 256)                                  │
│  Extracts local spatial-temporal patterns                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  POSITIONAL ENCODING                            │
│                                                                 │
│  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))                    │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                  │
│                                                                 │
│  Adds sequence position information                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER ENCODER                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  × 4 Layers                              │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Multi-Head Self-Attention (8 heads)            │     │   │
│  │  │  Q, K, V projections → Scaled Dot-Product       │     │   │
│  │  │  Attention → Concatenate → Linear               │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  │                        │                                 │   │
│  │                  Add & LayerNorm                         │   │
│  │                        │                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Feed-Forward Network                           │     │   │
│  │  │  Dense(512, ReLU) → Dense(256)                  │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  │                        │                                 │   │
│  │                  Add & LayerNorm                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Output: (batch, seq_len, 256)                                  │
│  Captures long-range temporal dependencies                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CLASSIFICATION HEAD                            │
│                                                                 │
│  GlobalAveragePooling1D()                                       │
│           │                                                     │
│           ▼                                                     │
│  Dense(128, ReLU) → Dropout(0.5)                                │
│           │                                                     │
│           ▼                                                     │
│  Dense(64, ReLU) → Dropout(0.3)                                 │
│           │                                                     │
│           ▼                                                     │
│  Dense(1, Sigmoid)                                              │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
│                                                                 │
│              Preictal Probability: [0, 1]                       │
│                                                                 │
│  > 0.5 → Preictal (seizure likely within 30 seconds)            │
│  ≤ 0.5 → Interictal (normal activity)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. CNN Encoder

**Purpose**: Extract local spatial-temporal features from raw EEG signals.

| Layer                     | Filters | Kernel | Output Shape      |
| ------------------------- | ------- | ------ | ----------------- |
| Conv1D + BN + ReLU + Pool | 64      | 7      | (batch, 1280, 64) |
| Conv1D + BN + ReLU + Pool | 128     | 5      | (batch, 640, 128) |
| Conv1D + BN + ReLU + Pool | 256     | 3      | (batch, 320, 256) |

**Design Choices**:

- Decreasing kernel sizes (7→5→3) capture progressively finer features
- Batch normalization for training stability
- MaxPooling reduces sequence length for transformer efficiency

### 2. Transformer Encoder

**Purpose**: Model long-range temporal dependencies and global patterns.

| Parameter | Value | Description     |
| --------- | ----- | --------------- |
| d_model   | 256   | Model dimension |
| n_heads   | 8     | Attention heads |
| n_layers  | 4     | Encoder layers  |
| d_ff      | 512   | FFN hidden dim  |
| dropout   | 0.1   | Regularization  |

**Attention Mechanism**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The multi-head attention allows the model to focus on different time points simultaneously, crucial for detecting pre-seizure patterns that may span several seconds.

### 3. Classification Head

**Purpose**: Map transformer features to seizure probability.

- **Global Average Pooling**: Aggregates temporal information
- **Dense Layers**: Non-linear classification
- **Dropout**: Prevents overfitting
- **Sigmoid**: Outputs probability [0, 1]

---

## Baseline Models

For comparison, we implement three baseline architectures:

### CNN Only

```
Input → [Conv1D → BN → ReLU → Pool] × 3 → GlobalPool → Dense → Sigmoid
```

### LSTM Only

```
Input → LSTM(128) → LSTM(64) → Dense → Sigmoid
```

### CNN-LSTM Hybrid

```
Input → [Conv1D → BN → ReLU → Pool] × 2 → LSTM(128) → Dense → Sigmoid
```

---

## Model Parameters

| Model             | Parameters | Size (MB) |
| ----------------- | ---------- | --------- |
| CNN + Transformer | ~2.5M      | ~10       |
| CNN Only          | ~0.5M      | ~2        |
| LSTM Only         | ~0.3M      | ~1.2      |
| CNN-LSTM          | ~0.8M      | ~3.2      |

---

## Why CNN + Transformer?

1. **CNNs excel at local patterns**: Epileptic spikes and transient waveforms are localized in time
2. **Transformers capture global context**: Pre-ictal states involve gradual changes across longer time scales
3. **Attention provides interpretability**: We can visualize which time points the model focuses on
4. **Efficient processing**: CNNs reduce sequence length before transformer, enabling faster training

---

## Training Strategy

| Aspect                | Approach                                 |
| --------------------- | ---------------------------------------- |
| **Loss**              | Binary Cross-Entropy                     |
| **Optimizer**         | Adam (lr=1e-4, weight_decay=1e-5)        |
| **LR Schedule**       | ReduceOnPlateau (factor=0.5, patience=5) |
| **Early Stopping**    | Patience=15 on val_loss                  |
| **Class Weights**     | Balanced (preictal is rare)              |
| **Data Augmentation** | Time shifting, noise injection           |

---

## Input/Output Specification

### Input

- **Shape**: `(batch_size, 22, 2560)`
- **Channels**: 22 standard EEG channels
- **Samples**: 2560 (10 seconds × 256 Hz)
- **Preprocessing**: Bandpass filtered, notch filtered, z-normalized

### Output

- **Shape**: `(batch_size, 1)`
- **Range**: [0, 1]
- **Interpretation**: Probability of being in preictal state
