# ✍️ MNIST Digit Classifier — PyTorch MLP

A clean, well-documented PyTorch implementation of a Multi-Layer Perceptron (MLP) that classifies handwritten digits (0–9) from the MNIST dataset.


---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Workflow](#workflow)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Known Issue](#known-issue)
- [Future Improvements](#future-improvements)

---

## Overview

This project trains a simple feedforward neural network to recognize handwritten digits from 28×28 grayscale images. It's designed to be easy to read, easy to run, and a solid reference point for anyone learning PyTorch or neural network fundamentals.

**Highlights**
- 3-layer fully connected neural network built with `torch.nn`
- Full evaluation suite: accuracy, precision/recall/F1, confusion matrix
- Clean, minimal codebase — no unnecessary abstraction
- Trains in a few minutes on CPU, no GPU required

---

## Project Structure

```
mnist-mlp-pytorch/
├── Hand_written_digit.ipynb   # Main notebook — data loading, training, evaluation
├── train_images.npy           # Training images (60000, 28, 28)
├── train_labels.npy           # Training labels (60000,)
├── test_images.npy            # Test images (10000, 28, 28)
├── test_labels.npy            # Test labels (10000,)
└── README.md
```

---

## Dataset

MNIST-style handwritten digits, stored as NumPy arrays of raw pixel values (0–255).

| File | Shape | Description |
|---|---|---|
| `train_images.npy` | (60000, 28, 28) | Grayscale training images |
| `train_labels.npy` | (60000,) | Training labels (0–9) |
| `test_images.npy` | (10000, 28, 28) | Grayscale test images |
| `test_labels.npy` | (10000,) | Test labels (0–9) |

---

## Model Architecture

```
Input (784)
   → Linear(784 → 128) → ReLU
   → Linear(128 → 64)  → ReLU
   → Linear(64 → 10)   → Output (digit class 0–9)
```

| Setting | Value |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `SGD` |
| Learning rate | `0.01` |
| Epochs | `30` |
| Batch size | `64` |
| Train / Validation split | `90% / 10%` |

---

## Workflow

1. **Load & inspect data** — read `.npy` arrays, verify shapes and label distribution
2. **Visualize samples** — preview a grid of digits with their true labels
3. **Split & tensorize** — train/validation split, convert to PyTorch tensors, wrap in `DataLoader`s
4. **Define the model** — `DigitClassification` MLP class
5. **Train** — 30 epochs, tracking training and validation loss
6. **Evaluate** — predictions on the test set, scored with accuracy, a classification report, and a confusion matrix

---

## Results

**Test Accuracy: 92.45%**

| Class | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|
| 0 | 0.95 | 0.97 | 0.96 |
| 1 | 0.96 | 0.97 | 0.96 |
| 2 | 0.81 | 0.95 | 0.88 |
| 3 | 0.96 | 0.83 | 0.89 |
| 4 | 0.91 | 0.96 | 0.93 |
| 5 | 0.90 | 0.87 | 0.88 |
| 6 | 0.96 | 0.95 | 0.95 |
| 7 | 0.96 | 0.96 | 0.96 |
| 8 | 0.91 | 0.87 | 0.89 |
| 9 | 0.94 | 0.91 | 0.92 |
| **Accuracy** | | | **0.92** |
| **Macro avg** | 0.93 | 0.92 | 0.92 |
| **Weighted avg** | 0.93 | 0.92 | 0.92 |

The model performs strongly across most classes, with F1-scores of 0.95 or higher for digits 0, 1, 6, and 7. The main confusion happens between visually similar digit pairs — notably **3 ↔ 2**, **5 ↔ 8**, and **9 ↔ 4** — which is expected for a simple MLP with no spatial awareness of pixel structure.

---

## Requirements

- Python 3.x
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

Install with:

```bash
pip install torch numpy matplotlib scikit-learn seaborn
```

---

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/mnist-mlp-pytorch.git
   cd mnist-mlp-pytorch
   ```
2. Ensure the `.npy` data files are in the project directory (or update the paths in the notebook).
3. Open the notebook:
   ```bash
   jupyter notebook Hand_written_digit.ipynb
   ```
4. Run all cells to train the model and reproduce the results above.

---

## Known Issue

The notebook currently loads the **test set from the training files**:

```python
test_img = np.load('/content/train_images.npy')
test_label = np.load('/content/train_labels.npy')
```

This means the reported accuracy is measured against training data rather than the held-out `test_images.npy` / `test_labels.npy`. Fix it by pointing to the actual test files:

```python
test_img = np.load('/content/test_images.npy')
test_label = np.load('/content/test_labels.npy')
```

---

## Future Improvements

- [ ] Fix the train/test data leakage noted above
- [ ] Add a CNN variant for improved accuracy
- [ ] Try Adam optimizer with learning rate scheduling
- [ ] Add early stopping based on validation loss
- [ ] Export trained model weights (`.pt`) for inference
- [ ] Add a simple inference script or demo

---


---

Built with [PyTorch](https://pytorch.org/), evaluated using [scikit-learn](https://scikit-learn.org/) and [Seaborn](https://seaborn.pydata.org/).
