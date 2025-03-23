"""Fast Gradient Sign Method (FGSM) adversarial attack implementation.

This file implements the FGSM attack, which is a simple and fast method for generating
adversarial examples. The method works by taking a single step in the direction of the
gradient of the loss function with respect to the input, making it much faster than
iterative methods like PGD.

Key features:
- Single-step gradient-based method (very fast)
- Supports both targeted and untargeted attacks
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
- Minimal hyperparameters (mainly epsilon)

Expected inputs:
- A neural network model to attack
- Input samples (images) to perturb
- Target labels (true labels for untargeted attacks, target labels for targeted attacks)
- Epsilon parameter (perturbation magnitude)

Expected outputs:
- Adversarial examples that aim to fool the model
- Performance metrics including success rate
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    This attack performs a single-step update in the direction of the gradient
    to create adversarial examples quickly. It's simpler and faster than iterative
    methods like PGD, but often creates more visible perturbations for the same
    perturbation budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "Linf",
        eps: float = 0.01,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the FGSM attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude.
            targeted: Whether to perform a targeted attack.
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            clip_min: Minimum value for pixel clipping.
            clip_max: Maximum value for pixel clipping.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Initialize the base attack
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using FGSM.

        The attack performs the following steps:
          1. Moves inputs and targets to the appropriate device.
          2. Computes the gradient of the loss with respect to the inputs.
          3. Takes a single step in the direction of the gradient.
          4. Clips the perturbation to ensure it's within the epsilon constraint.
          5. Returns the adversarial examples along with metrics.

        Args:
            inputs: Input images (clean samples) to perturb.
            targets: Target labels (for targeted attacks) or true labels (for untargeted attacks).

        Returns:
            A tuple (adversarial_examples, metrics), where metrics include the number of iterations,
            gradient calls, total time, and the attack success rate.
        """
        # Reset metrics from previous runs
        self.reset_metrics()
        start_time = time.time()

        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions for evaluation
        self.store_original_predictions(inputs)

        # Prepare for gradient computation
        inputs_with_grad = inputs.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs_with_grad)

        # Compute loss (depends on whether this is a targeted or untargeted attack)
        # For targeted attacks, we minimize the loss
        # For untargeted attacks, we maximize the loss
        loss = self._compute_loss(outputs, targets)

        # Compute gradients with respect to inputs
        self.model.zero_grad()
        loss.backward()

        # Extract gradients
        gradients = inputs_with_grad.grad.detach()

        # For untargeted attacks, we want to maximize the loss,
        # For targeted attacks, we want to minimize the loss
        if not self.targeted:
            # Maximize the loss (move in direction of gradient)
            gradient_direction = gradients
        else:
            # Minimize the loss (move opposite of gradient)
            gradient_direction = -gradients

        # Apply perturbation based on the norm
        if self.norm.lower() == "linf":
            # For Linf norm, take the sign of the gradient
            perturbation = self.eps * torch.sign(gradient_direction)
        else:  # L2 norm
            # For L2 norm, normalize the gradient and scale by epsilon
            # Add small constant to avoid division by zero
            l2_norm = torch.norm(
                gradient_direction.view(gradients.shape[0], -1), p=2, dim=1
            )
            l2_norm = torch.clamp(l2_norm, min=1e-12)

            # Normalize gradient to have unit L2 norm
            gradient_direction_norm = gradient_direction / l2_norm.view(-1, 1, 1, 1)

            # Scale by epsilon
            perturbation = self.eps * gradient_direction_norm

        # Create adversarial examples by applying the perturbation
        adv_inputs = inputs + perturbation

        # Clip the adversarial examples to ensure they're valid images
        adv_inputs = torch.clamp(adv_inputs, min=self.clip_min, max=self.clip_max)

        # Get predictions on adversarial examples
        with torch.no_grad():
            adv_outputs = self.model(adv_inputs)
            adv_predictions = adv_outputs.argmax(dim=1)

        # Calculate success rate
        if self.targeted:
            # For targeted attacks, success is when the model predicts the target class
            success = (adv_predictions == targets).float().mean().item() * 100
        else:
            # For untargeted attacks, success is when the model's prediction changes
            success = (adv_predictions != targets).float().mean().item() * 100

        # Calculate time taken
        self.total_time = time.time() - start_time
        self.total_gradient_calls = 1  # FGSM uses a single gradient call

        # Compile metrics
        metrics = {**self.get_metrics(), "success_rate": success}

        return adv_inputs, metrics
