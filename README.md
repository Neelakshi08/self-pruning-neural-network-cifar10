# Self-Pruning Neural Network for Efficient Representation Learning

## Abstract

We propose a self-pruning feed-forward neural network that learns to identify and suppress redundant connections during training. Unlike traditional pruning methods that operate post hoc, this approach integrates sparsity directly into the optimization process via learnable gating parameters. The resulting model achieves a balance between predictive performance and structural efficiency, producing a sparse architecture without requiring a separate pruning phase.

---

## 1. Introduction

Modern neural networks are often over-parameterized, leading to inefficiencies in memory and computation. Pruning techniques aim to address this by removing less important weights. However, most approaches rely on a two-stage pipeline: train a dense model and prune afterward.

In this work, we explore an alternative paradigm in which the model **learns to prune itself during training**, enabling dynamic adaptation of its structure. This is achieved by introducing learnable gate parameters that control the contribution of each weight.

---

## 2. Methodology

### 2.1 Prunable Linear Layer

Each weight is associated with a learnable gate score. The gate value is obtained using a sigmoid transformation:

```id="eq1"
g = sigmoid(s)
```

The effective weight used in computation is:

```id="eq2"
w' = w × g
```

where:

* ( w ) is the original weight
* ( g \in (0, 1) ) is the gate value

This formulation allows gradients to flow through both weights and gate parameters.

---

### 2.2 Sparsity-Inducing Objective

To encourage sparsity, we augment the standard classification loss with an L1 penalty on the gate values:

```id="eq3"
L = L_classification + λ × Σ g
```

The L1 term promotes smaller gate values, effectively pushing unnecessary connections toward zero.

---

### 2.3 Training Procedure

The model is trained end-to-end using stochastic gradient descent (Adam optimizer). Both weights and gate parameters are updated simultaneously.

Key components:

* Dataset: CIFAR-10
* Loss: Cross-Entropy + sparsity regularization
* Optimization: Backpropagation with joint parameter updates

---

## 3. Experimental Setup

* Dataset: CIFAR-10
* Model: Feed-forward network with custom prunable linear layers
* Evaluation Metrics:

  * Test Accuracy
  * Sparsity Level (% of pruned weights)

Sparsity is computed as the percentage of weights whose corresponding gate value falls below a threshold (e.g., 0.01).

---

## 4. Results

The model demonstrates a clear trade-off between sparsity and accuracy as the regularization strength (λ) varies.

| λ      | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| Low    | High          | Low          |
| Medium | Moderate      | Moderate     |
| High   | Lower         | High         |

Higher values of λ encourage more aggressive pruning at the cost of reduced accuracy.

---

## 5. Analysis

The L1 penalty on sigmoid-activated gates effectively induces sparsity by pushing many gate values toward zero. This results in:

* A bimodal distribution of gate values (near 0 and near 1)
* Automatic identification of unimportant connections
* A compact model representation without explicit pruning steps

---

## 6. Implementation Details

* Framework: PyTorch
* Custom Layer: `PrunableLinear`
* Configuration: YAML-based hyperparameter control
* Features:

  * Modular code structure
  * Reproducible training
  * Automated evaluation and reporting

---

## 7. Reproducibility

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The pipeline:

* Trains models for multiple λ values
* Evaluates performance and sparsity
* Generates plots and a report

---

## 8. Conclusion

This work demonstrates that neural networks can be trained to adapt their own structure through learnable gating mechanisms. By integrating sparsity into the training objective, we eliminate the need for separate pruning stages while maintaining competitive performance.

---

## License

MIT
