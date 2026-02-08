# Installation Guide

## Prerequisites

- **Python**: 3.9 or higher
- **GPU** (Recommended): NVIDIA GPU with CUDA support for faster training
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk Space**: ~5GB for CHB-MIT dataset + models

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/eeg-seizure-prediction.git
cd eeg-seizure-prediction
```

## Step 2: Create Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## Step 3: Install Dependencies

### CPU Only

```bash
pip install -r requirements.txt

```

### GPU Support (NVIDIA)

```bash
pip install -r requirements.txt

pip install tensorflow[and-cuda]
```

### Manual Installation

```bash
pip install tensorflow>=2.15.0 keras>=3.0.0 numpy scipy pandas mne scikit-learn matplotlib seaborn jupyter
```

## Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

## Step 5: Download Dataset (Optional)

The CHB-MIT dataset is automatically downloaded when you run training. To manually download:

```bash
python data/download.py
```

Or download directly from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

## Troubleshooting

### TensorFlow GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi


# Install CUDA toolkit if needed
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

### MNE Import Errors

```bash
pip install mne --upgrade
```

### Memory Issues

Reduce batch size in `config.py`:

```python
batch_size: int = 16  # Default is 32
```

## Docker Installation (Alternative)

```dockerfile
FROM tensorflow/tensorflow:latest-gpu


WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_experiment.py", "--test-mode"]
```

Build and run:

```bash
docker build -t seizure-predictor .
docker run --gpus all seizure-predictor
```
