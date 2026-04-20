# Self-Pruning Neural Network (CIFAR-10)

This project implements a feed-forward neural network that learns to **prune its own weights during training** using learnable gate parameters.

Instead of removing weights after training, the model identifies and suppresses unnecessary connections as part of the learning process.

---

## Overview

Each weight in the network is paired with a learnable gate value in the range `[0, 1]`.
During training:

* Gates close to `0` effectively remove weights
* Gates close to `1` retain important connections

A sparsity penalty is applied to encourage the model to keep only essential weights, resulting in a **compact and efficient network**.

---

## Key Idea

For every layer:

* A gate parameter is learned alongside each weight
* Gates are obtained using a sigmoid function
* Effective weights are computed as:

```
pruned_weight = weight × sigmoid(gate_score)
```

---

## Loss Function

The model is trained using a combination of classification loss and sparsity regularization:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

* CrossEntropyLoss ensures prediction accuracy
* SparsityLoss (L1 on gates) encourages pruning
* λ controls the trade-off between accuracy and sparsity

---

## Project Structure

```
.
├── config/          # Configuration files
├── src/
│   ├── models/      # Prunable layers and network
│   ├── training/    # Training loop and loss
│   ├── data/        # Dataset loading
│   ├── evaluation/  # Metrics
│   └── utils/       # Helpers and visualization
├── reports/         # Generated results and plots
├── checkpoints/     # Saved models
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (Windows)

pip install -r requirements.txt
```

---

## Usage

Run training:

```bash
python main.py
```

This will:

* Train the model on CIFAR-10
* Run experiments for multiple λ values
* Generate evaluation metrics and plots
* Save results and a report

---

## Evaluation

After training, the following are reported:

* **Test Accuracy**
* **Sparsity Level (%)**
  Percentage of weights where gate value < 0.01

---

## Expected Behavior

As λ increases:

* Sparsity increases (more weights removed)
* Accuracy may decrease

This demonstrates the trade-off between **model size and performance**.

---

## Outputs

* Model checkpoints (`checkpoints/`)
* Gate distribution plots (`reports/`)
* Accuracy vs sparsity comparison
* Markdown report with experiment results

---

## Notes

* Implemented using PyTorch
* Custom linear layer ensures gradients flow through both weights and gates
* Designed for clarity, modularity, and reproducibility

---

## License

MIT
