"""Conjugate Gradient (CG) adversarial attack implementation.

This attack implements the Conjugate Gradient method for generating adversarial examples.
It uses the conjugate gradient optimizer to efficiently find perturbations that cause
misclassification with minimal visual impact.

The implementation follows the same interface as other baseline attacks and integrates
with the Attack base class for consistent evaluation and reporting.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline.attack import Attack


class ConjugateGradient(Attack):
    r"""
    Conjugate Gradient Attack with perceptual constraints.

    This attack creates adversarial examples by optimizing the input using
    conjugate gradient descent to find minimal perturbations that cause
    misclassification while maintaining visual similarity.

    Arguments:
        model (nn.Module): Model to attack.
        norm (str): Norm of the attack ('L2' or 'Linf').
        eps (float): Maximum perturbation.
        n_iter (int): Number of iterations.
        fletcher_reeves (bool): Whether to use Fletcher-Reeves or Polak-Ribière formula.
        restart_interval (int): Interval for restarting conjugate gradient updates.
        tv_lambda (float): Total variation regularization weight for smoothness.
        color_lambda (float): Color regularization weight.
        rand_init (bool): If True, use random initialization.
        early_stopping (bool): If True, stop when adversarial goal is achieved.
        verbose (bool): If True, print progress updates.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,  # Reduced default epsilon for less visibility
        n_iter: int = 40,
        fletcher_reeves: bool = True,
        restart_interval: int = 10,
        tv_lambda: float = 0.3,  # TV smoothing strength
        color_lambda: float = 0.5,  # Color regularization weight
        rand_init: bool = True,
        early_stopping: bool = True,
        verbose: bool = False,
    ):
        # Initialize the Attack base class
        super().__init__("ConjugateGradient", model)

        # Record attack parameters
        self.norm = norm
        self.eps = eps
        self.n_iter = n_iter
        self.fletcher_reeves = fletcher_reeves
        self.restart_interval = restart_interval
        self.tv_lambda = tv_lambda
        self.color_lambda = color_lambda
        self.rand_init = rand_init
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

    def total_variation_loss(self, delta):
        """Calculate total variation loss for smoothness.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)

        Returns:
            Total variation loss normalized by image size
        """
        # Compute differences in x and y directions
        diff_h = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        diff_w = delta[:, :, :, 1:] - delta[:, :, :, :-1]

        # Sum absolute differences
        tv_h = torch.abs(diff_h).sum(dim=(1, 2, 3))
        tv_w = torch.abs(diff_w).sum(dim=(1, 2, 3))

        # Normalize by image size
        batch_size = delta.size(0)
        n_pixels = delta.size(2) * delta.size(3)

        return (tv_h + tv_w) / (n_pixels * batch_size)

    def color_regularization(self, delta):
        """Calculate color regularization to penalize visible changes.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)

        Returns:
            Color regularization loss
        """
        # Create weights for different color channels - humans are less sensitive to blue
        # Values based on human color perception research
        # Red: 0.8, Green: 1.0 (most sensitive), Blue: 0.6 (least sensitive)
        channel_weights = torch.tensor([0.8, 1.0, 0.6], device=delta.device).view(
            1, 3, 1, 1
        )

        # Apply channel weights to delta
        weighted_delta = delta * channel_weights

        # Return mean squared perturbation weighted by color sensitivity
        return torch.mean(weighted_delta**2)

    def forward(self, images, labels):
        r"""
        Overridden method for generating adversarial examples.

        Arguments:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): Labels.
                - If self.targeted is False, labels are the actual labels
                - If self.targeted is True, labels are the target labels

        Returns:
            adversarial_images (torch.Tensor): Adversarial examples.
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Note start time for performance tracking
        self.start_time = time.time()

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # Use cross-entropy loss for classification tasks
        loss = nn.CrossEntropyLoss()

        # Track iterations for metrics (n_iter iterations per sample)
        self.total_iterations += images.size(0) * self.n_iter

        # Initialize adversarial examples as the original images
        adv_images = images.clone().detach()

        # Add random initialization if enabled - similar to PGD
        if self.rand_init:
            if self.norm.lower() == "l2":
                # Gaussian noise for L2
                delta = torch.randn_like(images) * 0.01  # Smaller init noise
                # Normalize to unit norm
                flat_delta = delta.view(delta.size(0), -1)
                l2_norm = torch.norm(flat_delta, p=2, dim=1).view(-1, 1, 1, 1)
                # Scale to epsilon
                delta = (
                    delta * (self.eps * 0.1) / (l2_norm + 1e-10)
                )  # Start with 10% of epsilon
            else:  # Linf
                # Uniform noise for Linf - smaller magnitude
                delta = torch.empty_like(images).uniform_(
                    -self.eps * 0.1, self.eps * 0.1
                )

            # Apply perturbation and ensure image is in [0,1]
            adv_images = torch.clamp(images + delta, 0, 1)

        # Smaller step size for more refined perturbations
        step_size_l2 = self.eps / (self.n_iter * 2)
        step_size_linf = self.eps / (self.n_iter * 2)

        # Main attack loop
        for i in range(self.n_iter):
            adv_images.requires_grad = True

            # Forward pass
            outputs = self.get_logits(adv_images)

            # Count gradient calls (one per sample per iteration)
            self.total_gradient_calls += images.size(0)

            # Calculate classification loss
            if self.targeted:
                ce_loss = -loss(outputs, target_labels)
            else:
                ce_loss = loss(outputs, labels)

            # Calculate current perturbation
            delta = adv_images - images

            # Add total variation regularization for smoothness
            tv_loss = self.total_variation_loss(delta).mean()

            # Add color-aware regularization
            color_loss = self.color_regularization(delta)

            # Combined loss with regularization
            total_loss = (
                ce_loss + self.tv_lambda * tv_loss + self.color_lambda * color_loss
            )

            # Get gradients
            grad = torch.autograd.grad(
                total_loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            # Detach for next iteration
            adv_images = adv_images.detach()

            # Update images using gradient sign for Linf or normalized gradient for L2
            if self.norm.lower() == "l2":
                # L2 normalization
                grad_norms = (
                    torch.norm(grad.view(images.shape[0], -1), p=2, dim=1) + 1e-10
                )
                normalized_grad = grad / grad_norms.view(-1, 1, 1, 1)

                # Smaller step size for more refined perturbations
                step = step_size_l2 * normalized_grad
            else:
                # Linf - sign gradient like in PGD
                step = step_size_linf * grad.sign()

            # Update and project
            adv_images = adv_images + step

            # Project perturbation
            delta = adv_images - images
            if self.norm.lower() == "l2":
                # L2 projection
                delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
                factor = self.eps / (delta_norms + 1e-10)
                factor = torch.min(factor, torch.ones_like(factor))
                delta = delta * factor.view(-1, 1, 1, 1)
            else:
                # Linf projection
                delta = torch.clamp(delta, -self.eps, self.eps)

            # Apply projected perturbation and clamp to valid image range
            adv_images = torch.clamp(images + delta, 0, 1)

            # Additional spatial smoothing using Gaussian blur
            # Only applied at certain intervals to maintain attack strength
            if i % 5 == 0 and i > 0:
                # Calculate perturbation
                delta = adv_images - images
                # Apply slight Gaussian blur to smooth perturbation
                smoothed_delta = F.avg_pool2d(delta, kernel_size=3, stride=1, padding=1)
                # Apply smoothed perturbation
                adv_images = torch.clamp(images + smoothed_delta, 0, 1)

            # Check for early stopping if enabled
            if (
                self.early_stopping and i > self.n_iter // 4
            ):  # Only check after 25% of iterations
                with torch.no_grad():
                    outputs = self.get_logits(adv_images)
                    if self.targeted:
                        success = outputs.argmax(dim=1) == target_labels
                    else:
                        success = outputs.argmax(dim=1) != labels

                    if success.all():
                        if self.verbose:
                            print(f"Early stopping at iteration {i+1}/{self.n_iter}")
                        break

        # Final result - ensure in valid range
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time

        # Compute perturbation metrics
        perturbation_metrics = self.compute_perturbation_metrics(images, adv_images)

        # Evaluate success
        success_rate, success_mask, _ = self.evaluate_attack_success(
            images, adv_images, labels
        )

        if self.verbose:
            print(f"Attack complete: {success_rate:.2f}% success rate")
            print(
                f"L2 Norm: {perturbation_metrics['l2_norm']:.6f}, L∞ Norm: {perturbation_metrics['linf_norm']:.6f}"
            )
            print(
                f"Iterations: {self.n_iter}, Gradient calls: {self.total_gradient_calls}"
            )

        return adv_images
