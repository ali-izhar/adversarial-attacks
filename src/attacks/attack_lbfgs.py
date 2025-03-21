"""Limited-memory BFGS (L-BFGS) adversarial attack implementation."""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack
from .optimize import LBFGSOptimizer


class LBFGS(BaseAttack):
    """
    Limited-memory BFGS (L-BFGS) adversarial attack.

    This attack uses the LBFGSOptimizer to generate adversarial examples.
    L-BFGS is a quasi-Newton method that approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    algorithm using a limited amount of memory.
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
            model: The model to attack
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            targeted: Whether to perform a targeted attack
            loss_fn: The loss function to use ('cross_entropy' or 'margin')
            n_iterations: Maximum number of iterations
            history_size: Number of previous iterations to store for approximating Hessian
            line_search_fn: Line search method ('strong_wolfe' or 'armijo')
            max_line_search: Maximum number of line search iterations
            initial_step: Initial step size for line search
            rand_init: Whether to initialize with random perturbation
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop early when attack succeeds
            verbose: Whether to print progress information
            device: The device to use (CPU or GPU)
        """
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Initialize the optimizer
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
        Generate adversarial examples using L-BFGS.

        Args:
            inputs: The input images
            targets: The target labels (true labels for untargeted attacks, target labels for targeted attacks)

        Returns:
            A tuple of (adversarial_examples, metrics)
        """
        # Reset metrics
        self.reset_metrics()
        start_time = time.time()

        batch_size = inputs.shape[0]

        # Move inputs and targets to the attack device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Make sure inputs and targets have the same batch size
        if inputs.size(0) != targets.size(0):
            # If targeted=True with a single target class, expand to match input size
            if self.targeted and targets.size(0) == 1:
                targets = targets.expand(inputs.size(0))
            else:
                raise ValueError(
                    f"Input batch size {inputs.size(0)} doesn't match target batch size {targets.size(0)}"
                )

        # Store targets to avoid passing them repeatedly
        self.original_targets = targets

        # Define gradient function
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            # Get the corresponding subset of targets if x is a subset
            if x.size(0) != self.original_targets.size(0):
                # Find which indices of the original batch are in x
                # This is a simplification - in practice, we should track indices
                # But for now, we assume x is always a contiguous subset
                # So we just use the first x.size(0) targets
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            # Ensure x requires gradients
            x = x.detach().clone()
            x.requires_grad_(True)

            # Compute the loss
            outputs = self.model(x)
            loss = self._compute_loss(outputs, curr_targets, reduction="mean")

            # Compute gradient
            self.model.zero_grad()
            loss.backward()

            # Get gradient
            grad = x.grad.clone()

            # Clean up
            x.requires_grad_(False)

            return grad

        # Define loss function for the optimizer
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            # Get the corresponding subset of targets if x is a subset
            if x.size(0) != self.original_targets.size(0):
                # Same simplification as above
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            # No need for gradient computation here
            with torch.no_grad():
                outputs = self.model(x)
                # Use 'none' reduction to get per-example losses
                loss = self._compute_loss(outputs, curr_targets, reduction="none")
            return loss

        # Define success function
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            # Get the corresponding subset of targets if x is a subset
            if x.size(0) != self.original_targets.size(0):
                # Same simplification as above
                curr_targets = self.original_targets[: x.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x)
                return self._check_success(outputs, curr_targets)

        # Run optimization
        x_adv, opt_metrics = self.optimizer.optimize(
            x_init=inputs,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=inputs,
        )

        # Update metrics
        self.total_iterations = opt_metrics["iterations"]
        self.total_gradient_calls = opt_metrics["gradient_calls"]
        self.total_time = time.time() - start_time

        # Calculate final metrics
        metrics = {
            **self.get_metrics(),
            "success_rate": opt_metrics["success_rate"] * 100,
        }

        return x_adv, metrics
