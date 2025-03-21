# Adversarial Attack Implementations

This module implements adversarial attacks using various optimization methods. Each attack uses a specific optimization algorithm to solve the adversarial example generation problem.

## Attack Framework

All attacks follow a common framework defined in the `BaseAttack` class:

1. Define a loss function that encourages misclassification
2. Iteratively optimize the input to maximize/minimize this loss
3. Project perturbations to satisfy norm constraints
4. Track success rate and metrics

The mathematical objective depends on whether the attack is targeted or untargeted:

- **Untargeted attack**: Maximize loss for the true class
  ```
  maximize L(f(x+δ), y_true)  subject to ||δ||_p ≤ ε
  ```

- **Targeted attack**: Minimize loss for the target class
  ```
  minimize L(f(x+δ), y_target)  subject to ||δ||_p ≤ ε
  ```

Where:
- $f(x)$ is the model
- $\delta$ is the perturbation
- $L$ is the loss function
- $\left||\delta|\right|_p$ is the p-norm of the perturbation
- $\epsilon$ is the perturbation budget

## Attack Implementations

### 1. PGD Attack

The Projected Gradient Descent attack (`PGD`) implements the simplest but most robust attack strategy:

1. **Initialization**: Start with the original image or add small random noise
2. **Iteration**:
   - Compute gradient of loss w.r.t. input
   - Take a step in the gradient direction (opposite for targeted attacks)
   - Project back onto $\epsilon$-ball using the specified norm
   - Clamp to valid image range [0,1]
3. **Early stopping**: Optionally stop when attack succeeds

Key parameters:
- `alpha_init`: Initial step size
- `alpha_type`: Step size schedule ('constant' or 'diminishing')

### 2. Conjugate Gradient Attack

The Conjugate Gradient attack (`ConjugateGradient`) uses conjugate directions rather than simple gradient descent:

1. **Initialization**: Same as PGD
2. **Iteration**:
   - Compute conjugate direction using Fletcher-Reeves or Polak-Ribière formula
   - Perform line search to find optimal step size
   - Update image and project back to constraint set
   - Reset conjugate direction periodically to avoid accumulated numerical errors
3. **Line search**: Use backtracking line search with Armijo condition

Key parameters:
- `fletcher_reeves`: Whether to use Fletcher-Reeves (true) or Polak-Ribière (false)
- `restart_interval`: Frequency of direction resets
- `backtracking_factor`: Rate to reduce step size during line search

### 3. L-BFGS Attack

The Limited-memory BFGS attack (`LBFGS`) approximates second-order information for faster convergence:

1. **Initialization**: Same as other methods
2. **Iteration**:
   - Compute search direction using two-loop recursion to approximate inverse Hessian
   - Perform line search satisfying Wolfe conditions
   - Update image and projection
   - Store position and gradient differences for Hessian approximation
3. **Memory**: Maintain limited history (typically 5-20 iterations) to approximate Hessian

Key parameters:
- `history_size`: Number of past iterations to store
- `line_search_fn`: Search method ('strong_wolfe' or 'armijo')
- `initial_step`: Starting step size for line search

## Usage Example

```python
import torch
from src.attacks import PGD, ConjugateGradient, LBFGS

# Load a model
model = load_model()
model.eval()

# Choose attack method
attack = PGD(
    model=model,
    norm='L2',
    eps=0.5,
    targeted=False,
    loss_fn='cross_entropy'
)

# Generate adversarial examples
adv_examples, metrics = attack.generate(images, labels)

print(f"Success rate: {metrics['success_rate']}%")
```

## Performance Comparison

The three methods offer different trade-offs:

- **PGD**: Most reliable but slowest convergence; works well on any model
- **Conjugate Gradient**: Better convergence than PGD with moderately higher complexity
- **L-BFGS**: Fastest convergence when loss landscape is well-behaved, but may struggle with highly non-convex objectives

In practice, L-BFGS typically requires fewer iterations, but each iteration is more expensive due to line search and two-loop recursion.

## References

For detailed mathematical formulation of the optimization methods, see [Optimization Methods](../attacks/optimization/README.md).
