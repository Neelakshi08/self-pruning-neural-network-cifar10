🧠 Self-Pruning Neural Network for CIFAR-10

PyTorch • Python • MIT License

A production-grade PyTorch implementation of a self-pruning feed-forward neural network that learns both weights and their importance simultaneously.
Instead of pruning after training, the model suppresses unnecessary connections during training itself, resulting in a compact and efficient architecture.

💡 Core Concept

Each weight in the network is paired with a learnable gate parameter:

Gate scores → passed through a sigmoid → values in [0, 1]

Each weight is scaled by its gate:

W_pruned = W × sigmoid(gate_score)
Gates close to:
1 → Important connection (retained)
0 → Unnecessary connection (effectively pruned)

An L1 regularization term on gate values encourages sparsity, pushing redundant connections toward zero.

🛠️ Project Structure
newproj/
├── config/
│   └── config.yaml              # Hyperparameters & experiment configs
├── src/
│   ├── models/
│   │   └── prunable_net.py      # Prunable layers & network definition
│   ├── training/
│   │   ├── loss.py              # Sparsity-aware loss function
│   │   └── trainer.py           # Training loop + early stopping
│   ├── data/
│   │   └── dataset.py           # CIFAR-10 loader & augmentation
│   ├── evaluation/
│   │   └── metrics.py           # Accuracy & sparsity metrics
│   └── utils/
│       ├── helpers.py           # Seeds, device, configs
│       ├── visualization.py     # Plots & graphs
│       └── report.py            # Auto Markdown report generation
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
🚀 Setup
1. Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
2. Install Dependencies
pip install -r requirements.txt
💻 Usage
▶️ Run Full Training (CIFAR-10)
python main.py

This will:

Download CIFAR-10
Train models across multiple λ values
Generate plots & reports
Save checkpoints
⚡ Quick Pipeline Test (No Dataset Download)
python main.py --synthetic --epochs 3

Useful for verifying:

Training loop
Early stopping
Logging & plotting
🧰 CLI Options
Flag	Description
--config PATH	Custom config file
--experiment NAME	Run specific experiment
--epochs N	Override epochs
--batch-size N	Override batch size
--lr FLOAT	Override learning rate
--seed INT	Set random seed
--no-tensorboard	Disable TensorBoard
--synthetic	Use random data (fast testing)
📊 Monitoring

Run TensorBoard to track training:

tensorboard --logdir runs/
⚙️ Configuration

All settings are defined in:

config/config.yaml

Includes:

Model architecture
Training parameters
Pruning thresholds
Early stopping criteria
Experiment λ sweeps
🔬 Under the Hood
🧩 Prunable Layer
gates = sigmoid(gate_scores)
pruned_weight = weight * gates
output = input @ pruned_weight.T + bias
weight and gate_scores are both learnable parameters
Gradients flow through both → enabling joint optimization
Custom implementation (not using nn.Linear) for full control
📉 Loss Function
Total Loss = CrossEntropy + λ × Σ sigmoid(gate_scores)
CrossEntropy → ensures prediction accuracy
L1 sparsity term → encourages pruning
λ (lambda) → controls sparsity vs accuracy trade-off
📏 Sparsity Metric

A weight is considered pruned if:

sigmoid(gate_score) < 0.01
📦 Outputs

After training, check:

Output	Location
Model checkpoints	checkpoints/<experiment>/
Gate histograms	reports/gate_histogram_*.png
Accuracy vs Sparsity plot	reports/accuracy_sparsity_tradeoff.png
Full report	reports/experiment_report.md
TensorBoard logs	runs/<experiment>/
📈 Expected Trade-offs
λ Value	Accuracy	Sparsity
1e-5 (low)	~52–55%	Low (~5–15%)
1e-4 (medium)	~48–52%	Medium (~30–60%)
1e-3 (high)	~35–45%	High (~70–95%)

As λ increases → sparsity increases → accuracy may drop
This highlights the efficiency vs performance trade-off.

🧠 Why This Matters
Eliminates need for post-training pruning
Produces lighter, faster models
Improves deployment efficiency (edge/low-resource devices)
Demonstrates adaptive model compression
📝 License

MIT License
