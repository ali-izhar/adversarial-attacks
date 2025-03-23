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

    As described in the paper "Explaining and Harnessing Adversarial Examples"
    (Goodfellow et al., 2014), this attack performs a single-step update in the
    direction of the gradient to create adversarial examples quickly.

    The attack formula is:
        x_adv = x + epsilon * sign(∇J(x, y))

    Where:
        x is the original input
        y is the target (true label for untargeted, target label for targeted)
        J is the loss function
        epsilon is the perturbation magnitude
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
        original_predictions = self.store_original_predictions(inputs)

        # Create a copy of inputs that requires gradient
        inputs_with_grad = inputs.clone().detach().requires_grad_(True)

        # Forward pass through the model
        outputs = self.model(inputs_with_grad)

        # Compute loss
        loss = self._compute_loss(outputs, targets)

        # Backpropagate to get gradients
        self.model.zero_grad()
        loss.backward()

        # Get the gradients
        data_grad = inputs_with_grad.grad.data

        # Determine direction based on attack type
        # For untargeted attacks, we maximize the loss (follow gradient)
        # For targeted attacks, we minimize the loss (go opposite of gradient)
        if self.targeted:
            data_grad = -data_grad

        # First denormalize the input to apply perturbation in original pixel space
        # (this matches PyTorch tutorial's approach)
        denormalized_inputs = self._denormalize(inputs)

        # Apply perturbation based on norm constraint (in original pixel space)
        if self.norm.lower() == "linf":
            # FGSM original formulation: x_adv = x + epsilon * sign(grad)
            # We need to scale epsilon by std if normalization is used
            if self.std is not None:
                # Scale epsilon by the standard deviation for each channel
                scaled_eps = self.eps * self.std
                perturbation = scaled_eps * torch.sign(data_grad)
            else:
                perturbation = self.eps * torch.sign(data_grad)
        else:  # L2 norm
            # Normalize the gradient to have unit L2 norm
            grad_norms = torch.norm(data_grad.view(data_grad.shape[0], -1), p=2, dim=1)
            # Avoid division by zero
            grad_norms = torch.clamp(grad_norms, min=1e-12)
            # Normalize and scale by epsilon
            normalized_grad = data_grad / grad_norms.view(-1, 1, 1, 1)

            # Scale epsilon by std if normalization is used
            if self.std is not None:
                scaled_eps = self.eps * self.std
                perturbation = scaled_eps * normalized_grad
            else:
                perturbation = self.eps * normalized_grad

        # Apply perturbation to create adversarial examples in original pixel space
        denorm_adv_inputs = denormalized_inputs + perturbation

        # Clip to ensure valid image range in original pixel space
        denorm_adv_inputs = torch.clamp(
            denorm_adv_inputs, min=self.clip_min, max=self.clip_max
        )

        # Renormalize to feed back into the model
        adv_inputs = self._renormalize(denorm_adv_inputs)

        # Evaluate attack success
        with torch.no_grad():
            adv_outputs = self.model(adv_inputs)
            success_mask = self._check_success(adv_outputs, targets)
            success_rate = success_mask.float().mean().item() * 100

        # Update metrics
        self.total_time = time.time() - start_time
        self.total_gradient_calls = 1  # FGSM uses a single gradient call
        self.total_iterations = 1  # FGSM is a single-step attack

        # Compile metrics
        metrics = {**self.get_metrics(), "success_rate": success_rate}

        if self.verbose:
            print(
                f"FGSM Attack: Epsilon = {self.eps}, Success Rate = {success_rate:.2f}%"
            )

        return adv_inputs, metrics
