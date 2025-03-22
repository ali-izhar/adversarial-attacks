"""Projected Gradient Descent (PGD) adversarial attack implementation.

This file implements the PGD attack, which is a powerful first-order iterative method for
generating adversarial examples. The PGD method works by taking steps in the direction of the
gradient and then projecting back onto the constraint set defined by the perturbation norm ball,
making it an effective and widely used attack in adversarial machine learning.

Key features:
- Simple yet powerful first-order optimization method
- Supports both targeted and untargeted attacks
- Configurable step size schedules (constant or diminishing)
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
- Optional random initialization within the perturbation space
- Early stopping capability when adversarial criteria are met

Expected inputs:
- A neural network model to attack
- Input samples (images) to perturb
- Target labels (true labels for untargeted attacks, target labels for targeted attacks)
- Configuration parameters for the attack and optimizer

Expected outputs:
- Adversarial examples that aim to fool the model
- Performance metrics (iterations, gradient calls, time, success rate)
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack
from .optimize import PGDOptimizer


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    This attack uses the PGDOptimizer to generate adversarial examples by iteratively taking
    steps in the gradient direction (to maximize or minimize the loss depending on the attack)
    and then projecting the perturbed input back onto the constraint set (the eps-ball).
    It supports both targeted and untargeted attacks and allows for different step size schedules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "L2",
        eps: float = 0.5,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        n_iterations: int = 100,
        alpha_init: float = 0.1,
        alpha_type: str = "diminishing",
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the PGD attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude.
            targeted: Whether to perform a targeted attack.
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            n_iterations: Maximum number of iterations to run.
            alpha_init: Initial step size for the updates.
            alpha_type: Step size schedule ('constant' or 'diminishing').
            rand_init: Whether to initialize the attack with random noise.
            init_std: Standard deviation for random initialization.
            early_stopping: Stop early if adversarial criteria are met.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Initialize the base attack with model and common parameters.
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Instantiate the PGD optimizer with the specified configuration.
        self.optimizer = PGDOptimizer(
            norm=norm,
            eps=eps,
            n_iterations=n_iterations,
            alpha_init=alpha_init,
            alpha_type=alpha_type,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using PGD.

        The attack performs the following steps:
          1. Moves inputs and targets to the appropriate device.
          2. Ensures that the batch sizes match (expanding targets if needed for targeted attacks).
          3. Defines helper functions for gradient computation, loss evaluation, and success checking.
          4. Calls the PGDOptimizer to update the inputs iteratively.
          5. Returns the final adversarial examples along with metrics.

        Args:
            inputs: Input images (clean samples) to perturb.
            targets: Target labels (for targeted attacks) or true labels (for untargeted attacks).

        Returns:
            A tuple (adversarial_examples, metrics), where metrics include the number of iterations,
            gradient calls, total time, and the attack success rate.
        """
        # Reset any stored metrics from previous runs.
        self.reset_metrics()
        start_time = time.time()

        batch_size = inputs.shape[0]

        # Ensure inputs and targets are on the same device (e.g., GPU).
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions
        self.store_original_predictions(inputs)

        # Check that inputs and targets have the same batch size.
        if inputs.size(0) != targets.size(0):
            # For targeted attacks with a single target class, expand targets.
            if self.targeted and targets.size(0) == 1:
                targets = targets.expand(inputs.size(0))
            else:
                raise ValueError(
                    f"Input batch size {inputs.size(0)} doesn't match target batch size {targets.size(0)}"
                )

        # Store targets to reuse them in helper functions.
        self.original_targets = targets

        # Define a helper function to compute the gradient of the loss with respect to inputs.
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            # If x is a subset, use the corresponding targets (assumed to be a contiguous subset).
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            # Detach and clone x to ensure fresh computation and enable gradient tracking.
            x = x.detach().clone()
            x.requires_grad_(True)

            # Forward pass: compute model outputs.
            outputs = self.model(x)
            # Compute loss with mean reduction to obtain a scalar loss per batch.
            loss = self._compute_loss(outputs, curr_targets, reduction="mean")

            # Zero out previous gradients and backpropagate.
            self.model.zero_grad()
            loss.backward()

            # Clone the computed gradient.
            grad = x.grad.clone()

            # Disable further gradient tracking.
            x.requires_grad_(False)
            return grad

        # Define a loss function that returns per-example losses (without gradient computation).
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            # Use the appropriate targets for the current batch.
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x)
                # Use 'none' reduction to get individual losses per example.
                loss = self._compute_loss(outputs, curr_targets, reduction="none")
            return loss

        # Define a helper function to check whether the adversarial example is successful.
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x)
                # _check_success should return a Boolean tensor for each example.
                return self._check_success(outputs, curr_targets)

        # Run the PGD optimizer using the helper functions defined above.
        x_adv, opt_metrics = self.optimizer.optimize(
            x_init=inputs,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=inputs,  # Use original inputs for projection constraints.
        )

        # Update internal metrics based on optimizer outputs.
        self.total_iterations = opt_metrics["iterations"]
        self.total_gradient_calls = opt_metrics["gradient_calls"]
        self.total_time = time.time() - start_time

        # Compile final metrics and convert the success rate to a percentage.
        metrics = {
            **self.get_metrics(),
            "success_rate": opt_metrics["success_rate"],
        }

        return x_adv, metrics
