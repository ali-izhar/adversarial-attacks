"""Projected Gradient Descent (PGD) optimization method implementation."""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable

from src.utils.projections import project_adversarial_example


class PGDOptimizer:
    """
    Projected Gradient Descent (PGD) optimization method.

    PGD is a first-order method that iteratively takes steps in the direction
    of the gradient, projecting the perturbation back onto the constraint set
    after each step.
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
        Initialize the PGD optimizer.

        Args:
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            n_iterations: Maximum number of iterations
            alpha_init: Initial step size
            alpha_type: Step size schedule ('constant', 'diminishing', or 'adaptive')
            rand_init: Whether to initialize with random perturbation
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop early when objective is achieved
            verbose: Whether to print progress information
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
        Get the step size for the current iteration.

        Args:
            t: Current iteration

        Returns:
            Step size
        """
        if self.alpha_type == "constant":
            return self.alpha_init
        elif self.alpha_type == "diminishing":
            # Diminishing step size schedule: alpha_t = alpha_0 / sqrt(t+1)
            return self.alpha_init / np.sqrt(t + 1)
        else:
            raise ValueError(f"Unsupported alpha_type: {self.alpha_type}")

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run PGD optimization.

        Args:
            x_init: Initial point
            gradient_fn: Function that computes gradient given current x
            success_fn: Function that returns True if optimization goal is achieved
            x_original: Original input (for projection)

        Returns:
            Tuple of (optimized_x, metrics_dict)
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Initialize adversarial example
        if self.rand_init and self.init_std > 0:
            # Random initialization
            noise = torch.randn_like(x_init) * self.init_std
            x_adv = x_init + noise
            if x_original is not None:
                # Project back to eps-ball
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
        else:
            x_adv = x_init.clone()

        # Initialize metrics
        start_time = time.time()
        iterations = 0
        gradient_calls = 0

        # For early stopping
        if success_fn is not None:
            success = success_fn(x_adv)
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Main optimization loop
        for t in range(self.n_iterations):
            iterations += 1

            # Skip already successful examples if early stopping is enabled
            if self.early_stopping and success.all():
                break

            # Compute gradient (skip successful examples)
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Compute gradient only for examples that are still being optimized
                grad_full = torch.zeros_like(x_adv)
                grad_working = gradient_fn(x_adv[working_examples])
                grad_full[working_examples] = grad_working
                grad = grad_full
            else:
                grad = gradient_fn(x_adv)

            gradient_calls += 1

            # Take a step in the gradient direction
            alpha = self._get_step_size(t)
            x_adv = x_adv + alpha * grad

            # Project back to eps-ball
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )

            # Ensure valid image range [0, 1]
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # Check for success
            if success_fn is not None:
                success = success_fn(x_adv)
                if self.verbose and (t + 1) % 10 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

        # Return the adversarial examples and metrics
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
