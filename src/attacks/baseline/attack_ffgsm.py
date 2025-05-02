#!/usr/bin/env python

"""Fast Fast Gradient Sign Method (FFGSM) adversarial attack implementation.

Some code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import time
import torch
import torch.nn as nn

from .attack import Attack


class FFGSM(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 6/255)

    Shape:
        - images: :math:`(N, C, H, W)` normalized images with ImageNet mean/std
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)` normalized adversarial images.

    Examples::
        >>> attack = FFGSM(model, eps=8/255, alpha=6/255)
        >>> adv_images = attack(images, labels)

    Note:
        The epsilon and alpha values are applied in normalized space. For standard values of
        8/255 and 6/255 in [0,1] space, they are scaled by the std values of ImageNet normalization.
    """

    def __init__(self, model, eps=8 / 255, alpha=6 / 255):
        """Initialize FFGSM attack.

        Args:
            model: Target model to attack
            eps: Maximum perturbation size in [0,1] space (default: 8/255)
            alpha: Step size for gradient update in [0,1] space (default: 6/255)

        Raises:
            ValueError: If alpha is greater than epsilon
        """
        super().__init__("FFGSM", model)

        # Store the original epsilon and alpha as specified in [0,1] space
        self.orig_eps = eps
        self.orig_alpha = alpha

        # Scale epsilon and alpha to normalized space by dividing by ImageNet std
        # This makes the perturbation magnitude consistent across channels
        mean_std = self.std.clone().detach().mean().item()
        self.eps = eps / mean_std  # Scaling to normalized space
        self.alpha = alpha / mean_std  # Alpha is the step size for the gradient update

        # Note: In Wong et al. "Fast is better than free: Revisiting adversarial training",
        # they recommend alpha = 1.25*epsilon, which is greater than epsilon.
        # Therefore, we don't enforce alpha <= epsilon constraint here.

        # FFGSM supports both untargeted and targeted attacks
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""Overridden."""
        # Start timer for performance tracking
        start_time = time.time()

        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.size(0)

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # Calculate normalized min/max bounds for valid pixel values
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )

        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Keep track of best adversarial examples (for metrics)
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        dim = len(images.shape)

        # Use cross-entropy loss for classification tasks
        # This implements L(f(x+δ), y) from the paper's formulation
        loss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss(reduction="none")  # Used for L2 distance calculation
        Flatten = (
            nn.Flatten()
        )  # Needed to flatten spatial dimensions for L2 calculation

        # FFGSM Step 1: Initialize with random noise within epsilon bound
        # This corresponds to x' = x + α·sign(N(0,1)) from the paper
        # But using uniform noise instead of normally distributed noise with sign
        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)

        # Ensure the initial perturbation is within valid normalized image range
        # This implements the projection step to keep x' within the valid image domain
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Enable gradient computation for the perturbed images
        adv_images.requires_grad = True

        # Get model predictions for the randomly perturbed images
        # This calls f(x') in the paper's notation and increments gradient call counter
        outputs = self.get_logits(adv_images)

        # FFGSM is a single-step method (after random initialization)
        self.total_iterations += batch_size

        # Calculate loss based on attack mode
        if self.targeted:
            # For targeted attacks, minimize loss with respect to target labels
            cost = -loss(outputs, target_labels)  # Negative sign to minimize
        else:
            # For untargeted attacks, maximize loss with respect to true labels
            # This implements max_δ L(f(x+δ), y) from the paper
            cost = loss(outputs, labels)

        # Compute gradients of loss with respect to perturbed images
        # This calculates ∇_x L(f(x'), y) from the paper
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        # FFGSM Step 2: Gradient step from the perturbed starting point
        # This corresponds to x_adv = x' + (ε-α)·sign(∇_x L(f(x'), y))
        # But using alpha directly instead of (ε-α) as in some implementations
        adv_images = adv_images + self.alpha * grad.sign()

        # Project the perturbation back to epsilon-ball around original images
        # This ensures ||δ||_∞ ≤ ε as required by the constraint
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)

        # Final adversarial images are original images plus bounded perturbation
        # This implements the final projection step to ensure valid images
        adv_images = torch.clamp(images + delta, min=min_bound, max=max_bound).detach()

        # Calculate L2 distance between original and adversarial images
        current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)

        # Evaluate success of the attack and track metrics
        with torch.no_grad():
            adv_outputs = self.get_output_with_eval_nograd(adv_images)
            pre = torch.argmax(adv_outputs, 1)

            # Different success conditions based on attack mode
            if self.targeted:
                # For targeted attacks, we want predictions to match target labels
                condition = (pre == target_labels).float()
            else:
                # For untargeted attacks, we want predictions to differ from true labels
                condition = (pre != labels).float()

            # Update best adversarial examples and track L2 norms
            mask = condition * (best_L2 > current_L2)
            best_L2 = mask * current_L2 + (1 - mask) * best_L2

            # Update best adversarial images (only keep successful attacks with better L2)
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images + (1 - mask) * best_adv_images

            # Update attack success count
            self.attack_success_count += condition.sum().item()
            self.total_samples += batch_size

        # Calculate perturbation metrics for final report
        self.compute_perturbation_metrics(images, best_adv_images, condition.bool())

        # Measure and record time taken
        end_time = time.time()
        self.total_time += end_time - start_time

        return best_adv_images
