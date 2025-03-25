"""Projected Gradient Descent (PGD) optimization method implementation.

This file implements the PGD optimizer for generating adversarial examples against
neural networks. PGD is a first-order iterative method that applies a gradient step
followed by a projection operation to ensure the perturbation remains within
a specified constraint set.

Key features:
- Simple yet effective first-order optimization
- Configurable step size schedules (constant or diminishing)
- Constraint handling for both L2 and Linf perturbation norms
- Batch processing for simultaneous optimization of multiple examples
- Optional random initialization within the allowed perturbation space
- Early stopping capability when adversarial criteria are met

Expected inputs:
- Initial images (usually clean images to be perturbed)
- Gradient function that computes gradients of the adversarial loss
- Loss function that returns per-example losses (optional)
- Success function that determines if adversarial criteria are met (optional)
- Original images (for projection constraints)

Expected outputs:
- Optimized adversarial examples
- Optimization metrics (iterations, gradient calls, time, success rate)
"""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable

from ..projections import project_adversarial_example


class PGDOptimizer:
    """
    Projected Gradient Descent (PGD) optimization method.

    PGD is a first-order method that iteratively updates the adversarial examples
    by taking a step in the gradient direction (to increase the loss) and then
    projects the perturbed inputs back onto a feasible set defined by a norm constraint.
    """

    def __init__(
        self,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 100,
        alpha_init: float = 0.1,
        alpha_type: str = "diminishing",
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        maximize: bool = True,  # Default to maximization for adversarial attacks
    ):
        """
        Initialize the PGD optimizer with parameters controlling:
          - the perturbation norm and maximum allowed perturbation (eps)
          - the maximum number of iterations and the step size schedule
          - whether to start with random perturbation and early stopping behavior
          - whether to maximize or minimize the objective function
        """
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations
        self.alpha_init = alpha_init
        self.alpha_type = alpha_type
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.maximize = maximize  # For adversarial attacks, typically True

    def _get_step_size(self, t: int) -> float:
        """
        Determine the step size for iteration t.

        For a constant schedule, the step size remains alpha_init.
        For a diminishing schedule, the step size decays as alpha_init / sqrt(t + 1).

        Args:
            t: Current iteration number (starting from 0)

        Returns:
            Step size (alpha) for this iteration.
        """
        if self.alpha_type == "constant":
            return self.alpha_init
        elif self.alpha_type == "diminishing":
            # Diminishing step size: decreases as iterations increase.
            return self.alpha_init / np.sqrt(t + 1)
        else:
            raise ValueError(f"Unsupported alpha_type: {self.alpha_type}")

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run PGD optimization to generate adversarial examples.

        The algorithm follows these steps:
          1. (Optionally) initialize the adversarial example with random noise.
          2. For each iteration, compute the gradient of the loss with respect to the input.
          3. Update the input by taking a step in the direction of the gradient (or negative gradient).
          4. Project the updated input back onto the allowed perturbation set (eps-ball) and clamp
             values to valid ranges (e.g., [0, 1] for image pixels).
          5. Optionally stop early if the adversarial success condition is met.

        Args:
            x_init: The initial input (e.g., clean images).
            gradient_fn: Function to compute the gradient of the loss with respect to x.
            loss_fn: Function to compute the loss (per-example).
            success_fn: Function that returns a Boolean tensor indicating adversarial success.
            x_original: The original input, used for projection onto the eps-ball.

        Returns:
            A tuple (x_adv, metrics) where:
              - x_adv: The adversarial examples after optimization.
              - metrics: Dictionary with iteration count, gradient call count, total time, and success rate.
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Step 1: Initialize adversarial examples.
        if self.rand_init and self.init_std > 0:
            # Add random noise scaled by init_std.
            noise = torch.randn_like(x_init) * self.init_std
            x_adv = x_init + noise
            if x_original is not None:
                # Project the initial noisy example back into the allowed eps-ball.
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
        else:
            # Use a copy of the original inputs if no random initialization.
            x_adv = x_init.clone()

        # Initialize metrics for tracking the optimization.
        start_time = time.time()
        iterations = 0
        gradient_calls = 0

        # Check initial adversarial success.
        if success_fn is not None:
            success = success_fn(x_adv)
            if self.verbose:
                success_rate = success.float().mean().item() * 100
                print(f"Initial success rate: {success_rate:.2f}%")
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Store initial success rate
        initial_success = success.clone()

        # Main optimization loop.
        for t in range(self.n_iterations):
            iterations += 1

            # Early stopping: if all examples are already adversarial, break.
            if self.early_stopping and success.all():
                if self.verbose:
                    print(f"Early stopping at iteration {t}: all examples successful")
                break

            # Step 2: Compute the gradient for unsolved examples.
            if self.early_stopping:
                # Compute gradient only for examples that haven't succeeded.
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # When only a subset of examples need processing
                if working_examples.sum() < batch_size:
                    # Create a subset of the current batch for examples still being optimized
                    x_working = x_adv[working_examples]

                    # Calculate gradient only for working examples
                    grad_working = gradient_fn(x_working)

                    # Count only the gradient computations for working examples
                    gradient_calls += working_examples.sum().item()

                    # Create full gradient tensor
                    grad = torch.zeros_like(x_adv)

                    # Update only the working examples
                    grad[working_examples] = grad_working
                else:
                    # If all examples are being processed, compute gradient for all
                    grad = gradient_fn(x_adv)
                    # Count gradient calls for all examples in the batch
                    gradient_calls += batch_size
            else:
                grad = gradient_fn(x_adv)
                # Count gradient calls for all examples in the batch
                gradient_calls += batch_size

            # Step 3: Determine the step size for the current iteration.
            alpha = self._get_step_size(t)

            # PGD update: take a step in the gradient direction or opposite direction
            # Maximize: move in gradient direction (increase loss)
            # Minimize: move in negative gradient direction (decrease loss)
            if self.maximize:
                x_adv = x_adv + alpha * grad
            else:
                x_adv = x_adv - alpha * grad

            # Step 4: Project the updated example back onto the eps-ball (and ensure valid pixel range).
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # Step 5: Check for adversarial success.
            if success_fn is not None:
                new_success = success_fn(x_adv)
                # Only update for examples that were not already successful
                if self.early_stopping:
                    success = success | new_success
                else:
                    success = new_success

                if self.verbose and (t + 1) % 10 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

        # Compile final metrics.
        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "gradient_calls": gradient_calls,
            "time": total_time,
            "success_rate": (
                (success & ~initial_success).float().mean().item() * 100
                if success_fn is not None
                else 0.0
            ),
            "initial_success_rate": (
                initial_success.float().mean().item() if success_fn is not None else 0.0
            ),
        }

        return x_adv, metrics
