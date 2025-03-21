A framework for generating **adversarial examples against deep neural networks using classical optimization techniques.**

# Project Structure
```
adversarial-attacks/
├── attacks/                 # Attack implementations
│   ├── __init__.py
│   ├── base.py              # Base class for all attacks
│   ├── pgd.py               # Projected Gradient Descent
│   ├── cg.py                # Conjugate Gradient
│   └── lbfgs.py             # L-BFGS
├── models/                  # Model definitions and wrappers
│   ├── __init__.py
│   └── wrappers.py          # Wrapper classes for pre-trained models
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data.py              # Data loading utilities
│   ├── metrics.py           # Evaluation metrics
│   ├── projections.py       # Projection operators
│   └── visualization.py     # Visualization utilities
├── experiments/             # Experimental configurations
│   ├── __init__.py
│   └── configs.py           # Experiment configuration
├── run_experiment.py        # Main script to run experiments
├── requirements.txt         # Package dependencies
├── setup.py                 # Installation script
└── README.md                # Project README
```

# Methodology
The project evaluates optimization methods using:

- Attack Effectiveness: Success rate and confidence gap
- Perturbation Efficiency: L2 norm, L∞ norm, and SSIM
- Computational Efficiency: Iterations, gradient computations, and runtime
- Transferability: Cross-model success rate

All methods are tested on 100 correctly classified images from the ImageNet validation set, spanning 50 different classes.