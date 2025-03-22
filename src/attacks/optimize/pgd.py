"""Projected Gradient Descent (PGD) optimization method implementation."""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable

from src.utils.projections import project_adversarial_example


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
    ):
        """
        Initialize the PGD optimizer with parameters controlling:
          - the perturbation norm and maximum allowed perturbation (eps)
          - the maximum number of iterations and the step size schedule
          - whether to start with random perturbation and early stopping behavior
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
          3. Update the input by taking a step in the direction of the gradient.
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
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Main optimization loop.
        for t in range(self.n_iterations):
            iterations += 1

            # Early stopping: if all examples are already adversarial, break.
            if self.early_stopping and success.all():
                break

            # Step 2: Compute the gradient for unsolved examples.
            if self.early_stopping:
                # Compute gradient only for examples that haven't succeeded.
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                grad_full = torch.zeros_like(x_adv)
                grad_working = gradient_fn(x_adv[working_examples])
                grad_full[working_examples] = grad_working
                grad = grad_full
            else:
                grad = gradient_fn(x_adv)
            gradient_calls += 1

            # Step 3: Determine the step size for the current iteration.
            alpha = self._get_step_size(t)
            # PGD update: take a step in the gradient direction.
            # (Note: Depending on the loss formulation, the gradient direction may be ascent or descent.)
            x_adv = x_adv + alpha * grad

            # Step 4: Project the updated example back onto the eps-ball (and ensure valid pixel range).
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # Step 5: Check for adversarial success.
            if success_fn is not None:
                success = success_fn(x_adv)
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
                success.float().mean().item() if success_fn is not None else 0.0
            ),
        }

        return x_adv, metrics
