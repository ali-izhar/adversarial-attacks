"""Limited-memory BFGS (L-BFGS) adversarial attack implementation.

This file implements the L-BFGS attack, which leverages the quasi-Newton L-BFGS optimization
algorithm to create adversarial examples. The L-BFGS method approximates second-order curvature
information without explicitly forming the Hessian matrix, making it efficient for high-dimensional
optimization problems like adversarial attacks against deep neural networks.

Key features:
- Efficient second-order optimization using limited memory approximation
- Supports both targeted and untargeted attacks
- Configurable line search methods (strong Wolfe conditions or Armijo)
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
- Adjustable memory usage for Hessian approximation via history size
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
from .optimize import LBFGSOptimizer


class LBFGS(BaseAttack):
    """
    Limited-memory BFGS (L-BFGS) adversarial attack.

    This attack uses the LBFGSOptimizer to generate adversarial examples. L-BFGS is a
    quasi-Newton method that approximates the BFGS algorithm using limited memory. It
    leverages historical gradient and position differences to estimate curvature information,
    which is then used to craft effective adversarial perturbations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "L2",
        eps: float = 0.5,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        n_iterations: int = 50,
        history_size: int = 10,
        line_search_fn: str = "strong_wolfe",
        max_line_search: int = 20,
        initial_step: float = 1.0,
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the L-BFGS attack.

        Args:
            model: The model to attack.
            norm: Norm used for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum perturbation size.
            targeted: Whether to perform a targeted attack.
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            n_iterations: Maximum number of optimization iterations.
            history_size: Number of previous iterations to store for Hessian approximation.
            line_search_fn: Line search method ('strong_wolfe' or 'armijo').
            max_line_search: Maximum number of line search iterations.
            initial_step: Initial step size for line search.
            rand_init: Whether to initialize with random perturbation.
            init_std: Standard deviation for random initialization.
            early_stopping: Whether to stop early when attack succeeds.
            verbose: Whether to print progress information.
            device: The device to use (CPU or GPU).
        """
        # Initialize the base attack with the provided model and parameters.
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Instantiate the LBFGS optimizer with the desired configuration.
        self.optimizer = LBFGSOptimizer(
            norm=norm,
            eps=eps,
            n_iterations=n_iterations,
            history_size=history_size,
            line_search_fn=line_search_fn,
            max_line_search=max_line_search,
            initial_step=initial_step,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using the L-BFGS attack.

        The method moves the inputs and targets to the correct device, ensures batch sizes match,
        and defines helper functions to compute gradients, evaluate the loss, and determine if
        the adversarial goal is met. Finally, it calls the LBFGSOptimizer to perform the attack.

        Args:
            inputs: Input images to perturb.
            targets: Target labels for a targeted attack or true labels for an untargeted attack.

        Returns:
            A tuple (adversarial_examples, metrics), where metrics include iterations, gradient calls,
            total time, and success rate.
        """
        # Reset any previously stored metrics.
        self.reset_metrics()
        start_time = time.time()

        batch_size = inputs.shape[0]

        # Move inputs and targets to the attack device (e.g., GPU if available).
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions
        self.store_original_predictions(inputs)

        # Ensure that the number of targets matches the number of inputs.
        if inputs.size(0) != targets.size(0):
            # For targeted attacks with a single target class, expand the target tensor.
            if self.targeted and targets.size(0) == 1:
                targets = targets.expand(inputs.size(0))
            else:
                raise ValueError(
                    f"Input batch size {inputs.size(0)} doesn't match target batch size {targets.size(0)}"
                )

        # Store the targets to avoid passing them repeatedly in helper functions.
        self.original_targets = targets

        # Define a helper function to compute gradients of the loss with respect to inputs.
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            # Use corresponding targets if x is a subset of the original inputs.
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            # Detach x to avoid modifying the original input and enable gradient computation.
            x = x.detach().clone()
            x.requires_grad_(True)

            # Forward pass through the model.
            outputs = self.model(x)
            # Compute the loss (using mean reduction for a single scalar per batch).
            loss = self._compute_loss(outputs, curr_targets, reduction="mean")

            # Backpropagate to compute the gradients.
            self.model.zero_grad()
            loss.backward()

            # Clone the computed gradient.
            grad = x.grad.clone()

            # Turn off gradient tracking for x.
            x.requires_grad_(False)
            return grad

        # Define a helper function to compute per-example losses for the optimizer.
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            # Use the appropriate targets for x.
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            # Evaluate the loss without tracking gradients.
            with torch.no_grad():
                outputs = self.model(x)
                # Use 'none' reduction to obtain individual losses for each example.
                loss = self._compute_loss(outputs, curr_targets, reduction="none")
            return loss

        # Define a helper function to check if the adversarial example is successful.
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            # Retrieve the correct targets if x is a subset.
            if x.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x)
                # _check_success should return a Boolean tensor for each example.
                return self._check_success(outputs, curr_targets)

        # Call the LBFGS optimizer using the helper functions.
        x_adv, opt_metrics = self.optimizer.optimize(
            x_init=inputs,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=inputs,  # Use the original inputs as a reference for the perturbation constraint.
        )

        # Update internal metrics with those returned by the optimizer.
        self.total_iterations = opt_metrics["iterations"]
        self.total_gradient_calls = opt_metrics["gradient_calls"]
        self.total_time = time.time() - start_time

        # Compile final metrics, converting the success rate to a percentage.
        metrics = {
            **self.get_metrics(),
            "success_rate": opt_metrics["success_rate"],
        }

        return x_adv, metrics
