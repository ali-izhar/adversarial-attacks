"""Fast-FGSM (Fast Fast Gradient Sign Method) adversarial attack implementation.

Implementation of the Fast-FGSM attack, a variant of FGSM with random initialization,
from the paper: 'Fast is better than free: Revisiting adversarial training'
[https://arxiv.org/abs/2001.03994]

FFGSM adds a random initialization step before applying the gradient step of FGSM,
which helps to improve the transferability of the adversarial examples while maintaining
the computational efficiency of single-step methods.

Key features:
- Single-step gradient-based method with random initialization
- More effective than FGSM with similar computational cost
- Supports both targeted and untargeted attacks
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from ..base import BaseAttack


class FFGSM(BaseAttack):
    r"""
    Fast-FGSM (Fast Fast Gradient Sign Method) adversarial attack.

    This attack is a variant of FGSM with random initialization, from the paper:
    "Fast is better than free: Revisiting adversarial training"
    [https://arxiv.org/abs/2001.03994]

    FFGSM first applies a random perturbation, then a single gradient step:

    .. math::
        x' = x + \alpha \cdot \epsilon \cdot \text{sign}(\mathcal{U}(-1, 1))

    .. math::
        x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(x', y))

    Where:
        - :math:`x` is the original input
        - :math:`\alpha` is the random step size ratio
        - :math:`\mathcal{U}(-1, 1)` is a uniform random tensor with values in [-1, 1]
        - :math:`y` is the target (true label for untargeted, target label for targeted)
        - :math:`J` is the loss function
        - :math:`\epsilon` is the perturbation magnitude

    For the L2 norm variant, the perturbation is normalized by its L2 norm.

    Args:
        model: The neural network model to attack.
        norm: Norm for the perturbation constraint ('L2' or 'Linf').
        eps: Maximum allowed perturbation magnitude (default: 0.01).
            Common values in literature include 8/255 ≈ 0.031 for images in [0,1].
        alpha: Random step size ratio (default: 0.2).
            Controls how much of the epsilon budget is used for random initialization.
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
        >>> attack = FFGSM(model, norm='Linf', eps=8/255, alpha=0.2)
        >>> adv_images, metrics = attack.generate(images, labels)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "Linf",
        eps: float = 0.01,
        alpha: float = 0.2,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the FFGSM attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude (default: 0.01).
                Common values in literature include 8/255 ≈ 0.031 for images in [0,1].
            alpha: Random step size ratio (default: 0.2).
                Controls how much of the epsilon budget is used for random initialization.
            targeted: Whether to perform a targeted attack.
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            clip_min: Minimum value for pixel clipping.
            clip_max: Maximum value for pixel clipping.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Initialize the base attack
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        self.alpha = alpha
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using FFGSM.

        The algorithm has two steps:
        1. Add a random perturbation to the input (scaled by alpha*epsilon)
        2. Compute the gradient at this perturbed point and take a step in the direction
           of the sign of this gradient (for Linf norm) or the normalized gradient (for L2 norm).

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

        # Denormalize the input to apply perturbation in original pixel space
        denormalized_inputs = self._denormalize(inputs)

        # Step 1: Add random noise to the input
        if self.norm.lower() == "linf":
            if self.std is not None:
                # Scale epsilon by the standard deviation for each channel
                scaled_eps = self.eps * self.std
                # Generate random perturbation scaled by alpha*epsilon
                random_noise = (
                    torch.FloatTensor(denormalized_inputs.shape)
                    .uniform_(-1, 1)
                    .to(self.device)
                )
                random_noise = self.alpha * scaled_eps * torch.sign(random_noise)
            else:
                # Generate random perturbation scaled by alpha*epsilon
                random_noise = (
                    torch.FloatTensor(denormalized_inputs.shape)
                    .uniform_(-1, 1)
                    .to(self.device)
                )
                random_noise = self.alpha * self.eps * torch.sign(random_noise)
        else:  # L2 norm
            # Generate random perturbation with unit L2 norm
            random_noise = (
                torch.FloatTensor(denormalized_inputs.shape).normal_().to(self.device)
            )
            random_norms = torch.norm(
                random_noise.view(random_noise.shape[0], -1), p=2, dim=1
            )
            # Avoid division by zero
            random_norms = torch.clamp(random_norms, min=1e-12)
            # Normalize and scale by alpha*epsilon
            random_noise = random_noise / random_norms.view(-1, 1, 1, 1)

            if self.std is not None:
                scaled_eps = self.eps * self.std
                random_noise = self.alpha * scaled_eps * random_noise
            else:
                random_noise = self.alpha * self.eps * random_noise

        # Apply random perturbation to create randomly perturbed examples
        denorm_perturbed_inputs = torch.clamp(
            denormalized_inputs + random_noise, min=self.clip_min, max=self.clip_max
        )

        # Renormalize to feed back into the model
        perturbed_inputs = self._renormalize(denorm_perturbed_inputs)

        # Step 2: Compute gradient at the perturbed inputs
        perturbed_inputs_grad = perturbed_inputs.clone().detach().requires_grad_(True)
        outputs = self.model(perturbed_inputs_grad)
        loss = self._compute_loss(outputs, targets)

        # Backpropagate to get gradients
        self.model.zero_grad()
        loss.backward()
        data_grad = perturbed_inputs_grad.grad.data

        # Determine direction based on attack type
        if self.targeted:
            data_grad = -data_grad

        # Apply gradient-based perturbation
        if self.norm.lower() == "linf":
            if self.std is not None:
                scaled_eps = self.eps * self.std
                perturbation = scaled_eps * torch.sign(data_grad)
            else:
                perturbation = self.eps * torch.sign(data_grad)
        else:  # L2 norm
            grad_norms = torch.norm(data_grad.view(data_grad.shape[0], -1), p=2, dim=1)
            grad_norms = torch.clamp(grad_norms, min=1e-12)
            normalized_grad = data_grad / grad_norms.view(-1, 1, 1, 1)

            if self.std is not None:
                scaled_eps = self.eps * self.std
                perturbation = scaled_eps * normalized_grad
            else:
                perturbation = self.eps * normalized_grad

        # Apply final perturbation to the original inputs (not the randomly perturbed ones)
        denorm_adv_inputs = torch.clamp(
            denormalized_inputs + perturbation, min=self.clip_min, max=self.clip_max
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
        self.total_gradient_calls = 1  # FFGSM uses a single gradient call
        self.total_iterations = 1  # FFGSM is a single-step attack

        # Compile metrics
        metrics = {**self.get_metrics(), "success_rate": success_rate}

        if self.verbose:
            print(
                f"FFGSM Attack: Epsilon = {self.eps}, Alpha = {self.alpha}, Success Rate = {success_rate:.2f}%"
            )

        return adv_inputs, metrics
