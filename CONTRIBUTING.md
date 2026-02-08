# Contributing Guide

Thank you for considering contributing to the EEG Seizure Prediction System!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/eeg-seizure-prediction.git
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code style
- Add docstrings to all functions and classes
- Add type hints where possible
- Write tests for new functionality

### 3. Test Your Changes

```bash
# Run in test mode to verify nothing breaks
python run_experiment.py --test-mode

# Check for syntax errors
python -m py_compile your_file.py
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add new feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Open a PR on GitHub with a clear description of your changes.

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metric names to values
    """
```

### Type Hints

```python
from typing import Tuple, Optional, Dict

def process_eeg(
    data: np.ndarray,
    sampling_rate: int = 256,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    ...
```

---

## Project Structure

```
Code/
├── config.py           # All hyperparameters
├── run_experiment.py   # Training entry point
├── inference.py        # Deployment interface
├── data/               # Data loading and preprocessing
├── models/             # Model architectures
├── training/           # Training loop and metrics
├── utils/              # Visualization and helpers
└── docs/               # Documentation
```

---

## Areas for Contribution

### 🔧 Code Improvements

- [ ] Add more baseline models (Random Forest, XGBoost)
- [ ] Implement cross-validation
- [ ] Add data augmentation techniques
- [ ] Optimize memory usage for large datasets

### 📊 Features

- [ ] Real-time prediction demo
- [ ] Web interface for inference
- [ ] Support for other EEG formats (.set, .bdf)
- [ ] Multi-class prediction (seizure type)

### 📝 Documentation

- [ ] Add more usage examples
- [ ] Create video tutorials
- [ ] Translate to other languages

### 🧪 Testing

- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Set up CI/CD pipeline

---

## Commit Message Format

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:

```
feat: add attention visualization
fix: handle edge case in preprocessing
docs: update installation guide
refactor: simplify CNN encoder
```

---

## Questions?

Open an issue on GitHub or reach out to the maintainers.

Thank you for contributing! 🎉
