# Handwritten Digit Classification (MNIST) — PyTorch MLP

A simple fully-connected neural network (MLP) built with PyTorch to classify handwritten digits (0–9) from the MNIST dataset.

## Project Structure

```
.
├── Hand_written_digit.ipynb   # Main notebook: data loading, training, evaluation
├── train_images.npy           # Training images (60000, 28, 28), uint8
├── train_labels.npy           # Training labels (60000,), uint8
├── test_images.npy            # Test images (10000, 28, 28), uint8
├── test_labels.npy            # Test labels (10000,), uint8
└── README.md
```

## Dataset

The dataset is MNIST-style handwritten digit images stored as NumPy arrays:

| File | Shape | Description |
|---|---|---|
| `train_images.npy` | (60000, 28, 28) | Grayscale training images |
| `train_labels.npy` | (60000,) | Training labels (0–9) |
| `test_images.npy` | (10000, 28, 28) | Grayscale test images |
| `test_labels.npy` | (10000,) | Test labels (0–9) |

## Model Architecture

A simple Multi-Layer Perceptron (MLP):

```
Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10)
```

- Input images (28×28) are flattened to a 784-length vector.
- Loss function: `CrossEntropyLoss`
- Optimizer: `SGD` (learning rate = 0.01)
- Epochs: 30
- Batch size: 64

## Workflow

1. **Load data** — read `.npy` files with NumPy.
2. **Visualize samples** — preview a grid of digits with their labels.
3. **Split data** — 90% train / 10% validation split via `train_test_split`.
4. **Prepare tensors & DataLoaders** — convert arrays to PyTorch tensors and wrap in `DataLoader`s.
5. **Define model** — the `DigitClassification` MLP class.
6. **Train** — loop over epochs, tracking training and validation loss.
7. **Evaluate** — compute predictions on the test set and report:
   - Accuracy score
   - Classification report (precision, recall, F1-score per class)
   - Confusion matrix (visualized with Seaborn)

## Results

- **Test Accuracy:** 92.45%

**Classification Report (summary):**

| Metric | Score |
|---|---|
| Accuracy | 0.92 |
| Macro avg F1 | 0.92 |
| Weighted avg F1 | 0.92 |

The model performs strongly on most digits (e.g., 0, 1, 6, 7 all ≥ 0.95 F1), with more confusion between visually similar digits such as **3/2**, **5/8**, and **9/4** — visible in the confusion matrix.

## Requirements

- Python 3
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

Install with:

```bash
pip install torch numpy matplotlib scikit-learn seaborn
```

## Usage

1. Place the `.npy` data files in the same directory as the notebook (or update the file paths).
2. Open `Hand_written_digit.ipynb` in Jupyter or Google Colab.
3. Run all cells in order to train the model and view the evaluation results.

## Known Issue

In the current notebook, the test set is loaded from `train_images.npy` / `train_labels.npy` instead of `test_images.npy` / `test_labels.npy`:

```python
test_img = np.load('/content/train_images.npy')
test_label = np.load('/content/train_labels.npy')
```

This means the reported "test accuracy" is actually evaluated on the training data, not the held-out `test_images.npy`/`test_labels.npy` files. To properly evaluate generalization, update these lines to load the actual test files:

```python
test_img = np.load('/content/test_images.npy')
test_label = np.load('/content/test_labels.npy')
```

## License

Add your preferred license here (e.g., MIT).
