"""Projected Gradient Descent (PGD) adversarial attack implementation."""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack
from .optimization import PGDOptimizer


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    This attack uses the PGDOptimizer to generate adversarial examples.
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
            model: The model to attack
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            targeted: Whether to perform a targeted attack
            loss_fn: The loss function to use ('cross_entropy' or 'margin')
            n_iterations: Maximum number of iterations
            alpha_init: Initial step size
            alpha_type: Step size schedule ('constant' or 'diminishing')
            rand_init: Whether to initialize with random perturbation
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop early when attack succeeds
            verbose: Whether to print progress information
            device: The device to use (CPU or GPU)
        """
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Initialize the optimizer
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

        # Define gradient function
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            return self._compute_gradient(x, targets)

        # Define loss function for the optimizer
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            x.requires_grad_(True)
            outputs = self.model(x)
            loss = self._compute_loss(outputs, targets)
            return loss

        # Define success function
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                outputs = self.model(x)
                return self._check_success(outputs, targets)

        # Run optimization
        x_adv, opt_metrics = self.optimizer.optimize(
            x_init=inputs,
            gradient_fn=gradient_fn,
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
