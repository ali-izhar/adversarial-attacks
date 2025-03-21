# Optimization Methods for Adversarial Attacks

This repository contains implementations of various optimization methods for generating adversarial examples against convolutional neural networks. It accompanies the research paper "Optimization Methods for Efficient Adversarial Attacks on Neural Networks."

## Overview

Adversarial examples are carefully crafted perturbations that, when added to an input image, cause neural networks to misclassify the image while appearing visually imperceptible to humans. This project compares three classical optimization approaches for creating these perturbations:

1. Projected Gradient Descent (PGD)
2. Conjugate Gradient Method (CG)
3. Limited-memory BFGS (L-BFGS)

## Repository Structure

```
adversarial-attacks/
├── data/                   # Directory for dataset storage
├── experiments/            # Experiment scripts
│   └── compare_optimizers.py
├── results/                # Experimental results
├── src/                    # Source code
│   ├── attacks/            # Attack implementations
│   │   ├── base.py         # Base attack class
│   │   ├── optimization/   # Optimization methods
│   │   │   ├── pgd.py      # Projected Gradient Descent
│   │   │   ├── cg.py       # Conjugate Gradient Method
│   │   │   └── lbfgs.py    # L-BFGS Method
│   ├── models/             # Model wrappers
│   ├── plot/               # Visualization tools
│   └── utils/              # Utility functions
│       ├── data.py         # Data loading utilities
│       ├── evaluation.py   # Evaluation metrics
│       ├── metrics.py      # Performance metrics
│       └── projections.py  # Projection operations
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/adversarial-attacks.git
cd adversarial-attacks

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
# Compare optimization methods
python experiments/compare_optimizers.py
```

### Implementing Your Own Attacks

You can extend the base attack class to implement your own optimization methods:

```python
from src.attacks.base import BaseAttack

class MyAttack(BaseAttack):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        
    def generate(self, images, labels):
        # Implement your attack here
        pass
```

## Evaluation Metrics

We evaluate each optimization method using:

- **Attack Success Rate**: Percentage of inputs successfully misclassified
- **Perturbation Efficiency**: L2 norm, L∞ norm, and SSIM
- **Computational Efficiency**: Iterations, gradient computations, and runtime
- **Transferability**: Cross-model success rate

## Citation

If you use this code in your research, please cite our paper:

```
@article{ali2023optimization,
  title={Optimization Methods for Efficient Adversarial Attacks on Neural Networks},
  author={Ali, Izhar},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.