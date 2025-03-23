# Optimization Methods for Adversarial Attacks

This repository contains implementations of various optimization methods for generating adversarial examples against convolutional neural networks. It accompanies the research paper "Optimization Methods for Efficient Adversarial Attacks on Neural Networks."

## Overview

Adversarial examples are carefully crafted perturbations that, when added to an input image, cause neural networks to misclassify the image while appearing visually imperceptible to humans. This project compares three classical optimization approaches for creating these perturbations:

1. Projected Gradient Descent (PGD)
2. Conjugate Gradient Method (CG)
3. Limited-memory BFGS (L-BFGS)

## Recommended Attack Methods

Based on extensive testing, the following attack methods are recommended:

1. **C&W (Carlini & Wagner)**: Most reliable attack with high success rates. Use with increased confidence parameter (10.0) and learning rate (0.05) for best results.
   ```bash
   python demo.py --method cw --confidence 10 --learning-rate 0.05 --c-init 0.1 --steps 500
   ```

2. **PGD (Projected Gradient Descent)**: Good balance between efficiency and effectiveness. Works well with both L2 and Linf norms.
   ```bash
   python demo.py --method pgd --epsilon 0.05 --norm Linf --steps 100
   ```

3. **L-BFGS**: More effective with L2 norm than Linf. Consider using strong_wolfe line search and higher initial step size.
   ```bash
   python demo.py --method lbfgs --epsilon 0.3 --norm L2 --line-search-fn strong_wolfe
   ```

4. **FGSM/FFGSM**: Fastest but least effective. Good for quick testing.
   ```bash
   python demo.py --method ffgsm --epsilon 0.1
   ```

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