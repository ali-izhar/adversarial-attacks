#!/usr/bin/env python

"""Fast Gradient Sign Method (FGSM) adversarial attack implementation.

Some code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import time
import torch
import torch.nn as nn

from .attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` normalized images with ImageNet mean/std
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)` normalized adversarial images.

    Examples::
        >>> attack = FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    Note:
        The epsilon value is applied in normalized space. For a standard epsilon of 8/255
        in [0,1] space, we scale by the std values of ImageNet normalization.
    """

    def __init__(self, model, eps=8 / 255):
        """Initialize FGSM attack.

        Args:
            model: Target model to attack
            eps: Maximum perturbation size in [0,1] space (default: 8/255)
                This will be appropriately scaled to the normalized space
        """
        super().__init__("FGSM", model)

        # Store the original epsilon as specified in [0,1] space
        self.orig_eps = eps

        # Scale epsilon to normalized space by dividing by ImageNet std
        # This makes the perturbation magnitude consistent across channels
        mean_std = self.std.clone().detach().mean().item()
        # Scale epsilon to normalized space for consistent perturbation
        self.eps = eps / mean_std

        # FGSM supports both untargeted and targeted attacks
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""Overridden."""
        # Track time for performance metrics
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

        # Enable gradient computation for input images
        images.requires_grad = True

        # Get model predictions - this increments gradient call counter in the base class
        outputs = self.get_logits(images)  # f(x) in the paper

        # FGSM is a single-step method (one gradient computation per sample)
        # We count this as one iteration per sample in our metrics
        self.total_iterations += batch_size

        # Calculate loss based on attack mode
        if self.targeted:
            # For targeted attacks, minimize loss with respect to target labels
            # This is an extension of the original FGSM but follows the same principle
            cost = -loss(
                outputs, target_labels
            )  # Negative sign to minimize instead of maximize
        else:
            # For untargeted attacks, maximize loss with respect to true labels
            # This implements max_δ L(f(x+δ), y) from the paper
            cost = loss(outputs, labels)

        # Compute gradients of loss with respect to input images
        # This calculates ∇_x L(f(x), y) from the paper
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        # Explicitly track gradient calls (one per sample in batch)
        self.track_gradient_calls(batch_size)

        # FGSM update: add signed gradients scaled by epsilon
        # This implements x_adv = x + ε * sign(∇_x L(f(x), y)) from the paper
        adv_images = images + self.eps * grad.sign()

        # Clamp to valid normalized range
        # This enforces the constraint ||δ||_∞ ≤ ε indirectly by ensuring
        # the adversarial examples remain valid images
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Calculate L2 distance between original and adversarial images
        # This is for tracking metrics - exactly like in CW
        current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)

        # Evaluate success of the attack and track metrics (exactly like CW)
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

            # Update best adversarial examples and track L2 norms - exactly like in CW
            mask = condition * (best_L2 > current_L2)
            best_L2 = mask * current_L2 + (1 - mask) * best_L2

            # Update best adversarial images (only keep successful attacks with better L2)
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images + (1 - mask) * best_adv_images

            # Update attack success count (following CW's approach)
            self.attack_success_count += condition.sum().item()
            self.total_samples += batch_size

        # Calculate perturbation metrics for final report
        self.compute_perturbation_metrics(images, best_adv_images, condition.bool())

        # Track how long the attack took
        end_time = time.time()
        self.total_time += end_time - start_time

        return best_adv_images
