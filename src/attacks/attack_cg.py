"""Conjugate Gradient (CG) adversarial attack implementation.

This file implements the Conjugate Gradient attack, which uses non-linear conjugate gradient descent
to create adversarial examples.

Key features:
- Leverages the conjugate gradient optimization algorithm for improved convergence
- Supports both targeted and untargeted attacks
- Configurable convergence criteria and optimization parameters
- Compatible with different loss functions (cross-entropy and margin loss)
- Optional random initialization within the perturbation space
- Early stopping when examples successfully fool the model
- Constraint handling for both L2 and Linf perturbation norms

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
from .optimize import ConjugateGradientOptimizer


class ConjugateGradient(BaseAttack):
    """
    Conjugate Gradient adversarial attack.

    This attack uses non-linear conjugate gradient descent to craft adversarial examples.
    It leverages the ConjugateGradientOptimizer to iteratively update the input by combining
    gradient information with conjugate directions. The attack supports both targeted and
    untargeted scenarios and includes mechanisms for early stopping and projection
    onto the allowed perturbation space.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "L2",
        eps: float = 0.5,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        n_iterations: int = 100,
        fletcher_reeves: bool = True,
        restart_interval: int = 20,
        backtracking_factor: float = 0.7,
        sufficient_decrease: float = 1e-4,
        line_search_max_iter: int = 10,
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Conjugate Gradient attack.

        Args:
            model: The neural network to attack.
            norm: The perturbation norm ('L2' or 'Linf').
            eps: Maximum perturbation allowed.
            targeted: Whether the attack is targeted (aiming for a specific target label).
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            n_iterations: Maximum number of optimization iterations.
            fletcher_reeves: Use Fletcher-Reeves (True) or Polak-Ribière (False) for conjugate update.
            restart_interval: Restart conjugate directions every N iterations.
            backtracking_factor: Factor to decrease step size in line search.
            sufficient_decrease: Armijo condition parameter for sufficient loss decrease.
            line_search_max_iter: Maximum iterations for the line search procedure.
            rand_init: Whether to start with random perturbation.
            init_std: Standard deviation for random initialization.
            early_stopping: Stop early if the adversarial criterion is met.
            verbose: Print progress updates.
            device: Device to run the attack on (CPU or GPU).
        """
        # Initialize the base attack with the provided model and parameters.
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Scale epsilon for pixel space if normalization is being used
        eps_for_optimizer = eps
        if self.std is not None:
            # Store the unscaled epsilon for proper scaling during optimization
            self.unscaled_eps = eps
            # Will be used with proper scaling in the optimization loop
            if self.verbose:
                print(
                    f"Using normalized epsilon: {eps} (will be scaled during optimization)"
                )

        # Instantiate the Conjugate Gradient optimizer with the provided configuration.
        self.optimizer = ConjugateGradientOptimizer(
            norm=norm,
            eps=eps_for_optimizer,
            n_iterations=n_iterations,
            fletcher_reeves=fletcher_reeves,
            restart_interval=restart_interval,
            backtracking_factor=backtracking_factor,
            sufficient_decrease=sufficient_decrease,
            line_search_max_iter=line_search_max_iter,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using the Conjugate Gradient method.

        This method prepares the inputs and targets, defines helper functions for
        computing the gradient, loss, and success condition, and then calls the optimizer.

        Args:
            inputs: The input images (clean samples to perturb).
            targets: Target labels for targeted attacks, or true labels for untargeted attacks.

        Returns:
            A tuple of (adversarial_examples, metrics) where metrics include iterations,
            gradient calls, total time, and success rate.
        """
        # Reset any stored metrics from previous runs.
        self.reset_metrics()
        start_time = time.time()

        batch_size = inputs.shape[0]

        # Ensure inputs and targets are on the correct device.
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions
        self.store_original_predictions(inputs)

        # Make sure the number of targets matches the number of inputs.
        if inputs.size(0) != targets.size(0):
            # If a single target is provided for a targeted attack, expand it.
            if self.targeted and targets.size(0) == 1:
                targets = targets.expand(inputs.size(0))
            else:
                raise ValueError(
                    f"Input batch size {inputs.size(0)} doesn't match target batch size {targets.size(0)}"
                )

        # Store the targets so they can be reused in the helper functions.
        self.original_targets = targets

        # Denormalize inputs to operate in original pixel space
        denormalized_inputs = self._denormalize(inputs)

        # Scale epsilon by std if normalization is used
        if self.std is not None:
            # For each step of the optimization, we'll scale gradient by std
            # This ensures perturbation is applied in pixel space rather than normalized space
            self.optimizer.eps = self.unscaled_eps * self.std

        # Define a gradient function that computes the gradient of the loss with respect to x.
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # If x is a subset of the original inputs, use the corresponding targets.
            if x_normalized.size(0) != self.original_targets.size(0):
                # Here we assume x is a contiguous subset; in practice, tracking indices is preferable.
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            # Ensure that x requires gradients.
            x_normalized = x_normalized.detach().clone()
            x_normalized.requires_grad_(True)

            # Forward pass through the model.
            outputs = self.model(x_normalized)
            # Compute the loss using the chosen loss function.
            loss = self._compute_loss(outputs, curr_targets, reduction="mean")

            # Backpropagate to compute gradients.
            self.model.zero_grad()
            loss.backward()

            # Clone the gradient from x.
            grad = x_normalized.grad.clone()

            # Scale gradient if we're using normalization
            # This makes the gradient meaningful in the original pixel space
            if self.std is not None:
                grad = grad * self.std

            # Turn off gradient tracking for x.
            x_normalized.requires_grad_(False)
            return grad

        # Define a loss function for the optimizer that returns per-example losses.
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # Retrieve the appropriate targets for the given x.
            if x_normalized.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x_normalized)
                # Use 'none' reduction to compute individual losses for each example.
                loss = self._compute_loss(outputs, curr_targets, reduction="none")
            return loss

        # Define a success function to check if the adversarial example fools the model.
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # Retrieve the appropriate targets if x is a subset.
            if x_normalized.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x_normalized)
                # _check_success should return a Boolean tensor indicating success per example.
                return self._check_success(outputs, curr_targets)

        # Run the conjugate gradient optimizer using the helper functions.
        # The optimizer will work in original pixel space
        x_adv_denorm, opt_metrics = self.optimizer.optimize(
            x_init=denormalized_inputs,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=denormalized_inputs,  # Use the original denormalized inputs as reference for projection.
        )

        # Renormalize the adversarial examples before returning
        x_adv = self._renormalize(x_adv_denorm)

        # Update the attack metrics with the optimizer's results.
        self.total_iterations = opt_metrics["iterations"]
        self.total_gradient_calls = opt_metrics["gradient_calls"]
        self.total_time = time.time() - start_time

        # Compile final metrics and convert the success rate to a percentage.
        metrics = {
            **self.get_metrics(),
            "success_rate": opt_metrics["success_rate"],
        }

        return x_adv, metrics
