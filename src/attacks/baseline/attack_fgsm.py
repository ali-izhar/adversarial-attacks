"""Fast Gradient Sign Method (FGSM) adversarial attack implementation.

Implementation of the FGSM attack from the paper:
'Explaining and Harnessing Adversarial Examples'
[https://arxiv.org/abs/1412.6572]

This is a simple one-step method that generates adversarial examples by taking a single
step in the direction of the gradient sign of the loss function with respect to the input.
It's extremely efficient but produces relatively large perturbations compared to iterative methods.

Key features:
- Single-step gradient-based method (very fast)
- Supports both targeted and untargeted attacks
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from ..base import BaseAttack


class FGSM(BaseAttack):
    r"""
    Fast Gradient Sign Method (FGSM) adversarial attack.

    This attack implements the algorithm from the paper:
    "Explaining and Harnessing Adversarial Examples"
    [https://arxiv.org/abs/1412.6572]

    FGSM works by taking a single step in the direction of the gradient sign:

    .. math::
        x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(x, y))

    Where:
        - :math:`x` is the original input
        - :math:`y` is the target (true label for untargeted, target label for targeted)
        - :math:`J` is the loss function
        - :math:`\epsilon` is the perturbation magnitude

    For the L2 norm variant, the perturbation is normalized by its L2 norm:

    .. math::
        x_{adv} = x + \epsilon \cdot \frac{\nabla_x J(x, y)}{||\nabla_x J(x, y)||_2}

    Args:
        model: The neural network model to attack.
        norm: Norm for the perturbation constraint ('L2' or 'Linf').
            The Linf norm is used in the original paper.
        eps: Maximum allowed perturbation magnitude (default: 0.01).
            Common values in literature include 8/255 ≈ 0.031 for images in [0,1].
        targeted: Whether to perform a targeted attack.
            If True, tries to make the model predict a specific target class.
            If False, tries to make the model predict any class other than the true class.
        loss_fn: Loss function to use ('cross_entropy' or 'margin').
        clip_min: Minimum value for pixel clipping (default: 0.0).
        clip_max: Maximum value for pixel clipping (default: 1.0).
        verbose: Print progress updates.
        device: Device to run the attack on (e.g., CPU or GPU).

    Shape:
        - inputs: :math:`(N, C, H, W)` where `N = batch size`, `C = channels`, `H = height`, `W = width`.
          Input values should be in range [0, 1] or normalized according to model requirements.
        - targets: :math:`(N)` where each value is the class index.
        - output: :math:`(N, C, H, W)` containing adversarial examples.

    Example:
        >>> attack = FGSM(model, norm='Linf', eps=8/255)
        >>> adv_images, metrics = attack.generate(images, labels)
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
                The Linf norm is used in the original paper.
            eps: Maximum allowed perturbation magnitude (default: 0.01).
                Common values in literature include 8/255 ≈ 0.031 for images in [0,1].
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

        The algorithm computes the gradient of the loss with respect to the input,
        then takes a single step in the direction of the sign of this gradient
        (for Linf norm) or the normalized gradient (for L2 norm).

        For targeted attacks, we move in the opposite direction of the gradient to
        minimize the loss toward the target class. For untargeted attacks, we move
        in the direction of the gradient to maximize the loss away from the true class.

        Args:
            inputs: Input images (clean samples) to perturb, of shape (N, C, H, W)
            targets: Target labels (for targeted attacks) or true labels (for untargeted attacks),
                of shape (N)

        Returns:
            A tuple (adversarial_examples, metrics):
            - adversarial_examples: Tensor of same shape as inputs containing the adversarial examples
            - metrics: Dictionary with performance metrics including success rate
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
