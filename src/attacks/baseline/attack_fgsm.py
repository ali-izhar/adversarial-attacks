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

        # Calculate normalized min/max bounds for valid pixel values
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )

        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Use cross-entropy loss for classification tasks
        loss = nn.CrossEntropyLoss()

        # FGSM is a single-step method (one gradient computation per sample)
        # We count this as one iteration per sample in our metrics
        self.total_iterations += batch_size

        # Create adversarial examples with different logic for targeted vs untargeted attacks
        if self.targeted:
            # For targeted attacks, get the target labels and apply targeted perturbation
            target_labels = self.get_target_label(images, labels)

            # Enable gradient computation for input images
            images.requires_grad = True

            # Get model predictions
            outputs = self.get_logits(images)

            # For targeted attacks, we want to minimize the loss with respect to target class
            cost = loss(outputs, target_labels)

            # Compute gradients
            grad = torch.autograd.grad(
                cost, images, retain_graph=False, create_graph=False
            )[0]

            # For targeted attacks, move AWAY from gradient to move TOWARD target class
            adv_images = images - self.eps * grad.sign()

            # Clamp to valid normalized range
            adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

            # Track how long the attack took (before metric calculation)
            end_time = time.time()
            self.total_time += end_time - start_time

            # Evaluate attack success using the base class method
            # This updates self.attack_success_count and self.total_samples correctly
            _success_rate, success_mask, _predictions = self.evaluate_attack_success(
                images, adv_images, labels  # Use original labels for evaluation context
            )

            # Calculate and store perturbation metrics for successful attacks using base class method
            # This appends L2, Linf, and SSIM for successful attacks only
            self.compute_perturbation_metrics(images, adv_images, success_mask)

            # FGSM iteration count is already handled above for the batch

            # Return the generated adversarial images
            return adv_images
        else:
            # Original untargeted FGSM implementation
            images.requires_grad = True
            outputs = self.get_logits(images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(
                cost, images, retain_graph=False, create_graph=False
            )[0]
            adv_images = images + self.eps * grad.sign()

            # Clamp to valid normalized range
            adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

            # Track how long the attack took (before metric calculation)
            end_time = time.time()
            self.total_time += end_time - start_time

            # Evaluate success of the attack and track metrics using base class method
            _success_rate, success_mask, _predictions = self.evaluate_attack_success(
                images, adv_images, labels
            )

            # Calculate perturbation metrics using the base class method for final report
            # Using the direct adv_images and the success_mask from evaluate_attack_success
            self.compute_perturbation_metrics(images, adv_images, success_mask)

            # Return the directly calculated adversarial images
            return adv_images
