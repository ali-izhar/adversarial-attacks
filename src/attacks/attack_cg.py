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

from .baseline.attack import Attack
from .optimize.cg import ConjugateGradientOptimizer


class ConjugateGradient(Attack):
    r"""
    Enhanced Conjugate Gradient Attack with perceptual constraints.

    This attack creates adversarial examples by optimizing the input using
    conjugate gradient descent to find minimal perturbations that cause
    misclassification while maintaining visual similarity. The improved implementation
    uses proper conjugate gradient formulas for better convergence.

    Arguments:
        model (nn.Module): Model to attack.
        norm (str): Norm of the attack ('L2' or 'Linf').
        eps (float): Maximum perturbation.
        n_iter (int): Number of iterations.
        beta_method (str): Formula to use for conjugate updates ('FR', 'PR', or 'HS').
        restart_interval (int): Interval for restarting conjugate gradient updates.
        tv_lambda (float): Total variation regularization weight for smoothness.
        color_lambda (float): Color regularization weight for perceptual quality.
        perceptual_lambda (float): Weight for frequency domain perceptual loss.
        rand_init (bool): If True, use random initialization.
        fgsm_init (bool): If True, use FGSM initialization for better starting point.
        adaptive_restart (bool): If True, dynamically restart based on conjugacy loss.
        early_stopping (bool): If True, stop when adversarial goal is achieved.
        verbose (bool): If True, print progress updates.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iter: int = 50,
        beta_method: str = "HS",  # Hestenes-Stiefel method (best for nonlinear CG)
        restart_interval: int = 10,
        tv_lambda: float = 0.2,  # Reduced TV strength for better convergence
        color_lambda: float = 0.3,  # Reduced color regularization
        perceptual_lambda: float = 0.4,  # Weight for frequency domain perceptual loss
        rand_init: bool = False,  # Don't use random init by default
        fgsm_init: bool = True,  # Use FGSM initialization by default
        adaptive_restart: bool = True,  # Enable adaptive restart for better convergence
        early_stopping: bool = True,
        verbose: bool = False,
    ):
        # Initialize the Attack base class
        super().__init__("ConjugateGradient", model)

        # Record attack parameters
        self.norm = norm
        self.eps = eps
        self.n_iter = n_iter
        self.beta_method = beta_method
        self.restart_interval = restart_interval
        self.tv_lambda = tv_lambda
        self.color_lambda = color_lambda
        self.perceptual_lambda = perceptual_lambda
        self.rand_init = rand_init
        self.fgsm_init = fgsm_init
        self.adaptive_restart = adaptive_restart
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer with correct parameters
        self.optimizer = ConjugateGradientOptimizer(
            norm=norm,
            eps=eps,
            n_iterations=n_iter,
            beta_method=beta_method,
            restart_interval=restart_interval,
            rand_init=rand_init,
            early_stopping=early_stopping,
            verbose=verbose,
            fgsm_init=fgsm_init,
            adaptive_restart=adaptive_restart,
            # Don't use momentum for proper CG
            momentum=0.0,
            # Use true CG line search parameters
            line_search_factor=0.5,
            sufficient_decrease=1e-4,
        )

    def total_variation_loss(self, delta):
        """Calculate total variation loss for smoothness.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)

        Returns:
            Total variation loss normalized by image size (per-sample)
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
            Color regularization loss (per-sample)
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
        return torch.mean(weighted_delta**2, dim=(1, 2, 3))

    def perceptual_loss(self, delta, input_images):
        """Calculate perceptual loss to preserve image structure.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)
            input_images: Original images of shape (B, C, H, W)

        Returns:
            Perceptual loss value (per-sample)
        """
        # Compute frequency domain representation
        fft_orig = torch.fft.fft2(input_images, dim=(2, 3))
        fft_pert = torch.fft.fft2(input_images + delta, dim=(2, 3))

        # Compute magnitude difference in frequency domain
        mag_orig = torch.abs(fft_orig)
        mag_pert = torch.abs(fft_pert)

        # Focus more on low frequency differences (structural)
        batch_size, channels, height, width = input_images.shape

        # Create a frequency weighting mask (emphasize low frequencies)
        y_coords = torch.arange(height, device=input_images.device).view(1, 1, -1, 1)
        x_coords = torch.arange(width, device=input_images.device).view(1, 1, 1, -1)

        # Compute distance from center (DC component)
        center_y, center_x = height // 2, width // 2
        dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # Create a weight mask that emphasizes low frequencies (close to DC)
        weight_mask = torch.exp(-dist / (min(height, width) / 4))
        weight_mask = weight_mask.expand(batch_size, channels, -1, -1)

        # Apply weighting and compute difference
        weighted_diff = ((mag_orig - mag_pert).abs() * weight_mask).sum(dim=(1, 2, 3))

        # Normalize by image size
        return weighted_diff / (channels * height * width)

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

        # Use cross-entropy loss for classification
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Define the gradient function for the optimizer
        def gradient_fn(x):
            # Ensure gradients are enabled
            x.requires_grad_(True)

            # Forward pass
            outputs = self.get_logits(x)

            # Get appropriate labels for the current batch
            # This handles cases when we're only processing a subset of samples
            if self.targeted:
                # For targeted, use target labels matching the current batch size
                curr_labels = target_labels[: x.size(0)]
            else:
                # For untargeted, use true labels matching the current batch size
                curr_labels = labels[: x.size(0)]

            # Classification loss
            if self.targeted:
                # For targeted attacks, minimize negative CE loss to target class
                class_loss = -ce_loss(outputs, curr_labels)
            else:
                # For untargeted attacks, maximize CE loss to true class
                class_loss = ce_loss(outputs, curr_labels)

            # Compute current perturbation
            delta = x - images[: x.size(0)]

            # Add regularization losses - all should return per-sample values
            tv_loss = self.total_variation_loss(delta)
            color_loss = self.color_regularization(delta)
            percept_loss = self.perceptual_loss(delta, images[: x.size(0)])

            # Combine losses with weights - keeping per-sample values
            total_loss = (
                class_loss
                + self.tv_lambda * tv_loss
                + self.color_lambda * color_loss
                + self.perceptual_lambda * percept_loss
            )

            # Compute gradients for each sample
            grads = []
            for i in range(x.size(0)):
                grad = torch.autograd.grad(
                    total_loss[i], x, retain_graph=(i < x.size(0) - 1)
                )[0][i : i + 1]
                grads.append(grad)

            # Combine gradients
            full_grad = torch.cat(grads, dim=0)
            return full_grad

        # Define the loss function that returns per-sample loss values
        def loss_fn(x):
            with torch.no_grad():
                # Forward pass
                outputs = self.get_logits(x)

                # Get matching labels for current batch size
                if self.targeted:
                    curr_labels = target_labels[: x.size(0)]
                else:
                    curr_labels = labels[: x.size(0)]

                # Classification loss
                if self.targeted:
                    class_loss = -ce_loss(outputs, curr_labels)
                else:
                    class_loss = ce_loss(outputs, curr_labels)

                # Compute current perturbation
                delta = x - images[: x.size(0)]

                # Add regularization losses
                tv_loss = self.total_variation_loss(delta)
                color_loss = self.color_regularization(delta)
                percept_loss = self.perceptual_loss(delta, images[: x.size(0)])

                # Combine losses with weights
                total_loss = (
                    class_loss
                    + self.tv_lambda * tv_loss
                    + self.color_lambda * color_loss
                    + self.perceptual_lambda * percept_loss
                )

                return total_loss

        # Define success function for early stopping
        def success_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)

                # Get matching labels for the current batch
                if self.targeted:
                    curr_labels = target_labels[: x.size(0)]
                else:
                    curr_labels = labels[: x.size(0)]

                if self.targeted:
                    # Attack succeeds if model predicts target class
                    return outputs.argmax(dim=1) == curr_labels
                else:
                    # Attack succeeds if model predicts any class other than the true label
                    return outputs.argmax(dim=1) != curr_labels

        # Run the optimizer to generate adversarial examples
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
            targeted=self.targeted,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += int(metrics["iterations"] * images.size(0))
        self.total_gradient_calls += int(metrics["gradient_calls"] * images.size(0))

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time

        # Ensure outputs are properly clamped
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Compute perturbation metrics - now passing the success mask
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, success_mask
        )

        # Update attack_success_count for metrics
        self.attack_success_count += success_mask.sum().item()
        self.total_samples += images.size(0)

        # Log metrics if verbose
        if self.verbose:
            orig_preds, adv_preds = pred_info
            print(f"Attack complete: {success_rate:.2f}% success rate")
            print(
                f"Metrics: L2={perturbation_metrics['l2_norm']:.6f}, "
                f"L∞={perturbation_metrics['linf_norm']:.6f}, "
                f"SSIM={perturbation_metrics['ssim']:.4f}"
            )
            print(
                f"Iterations: {metrics['iterations']:.1f}, Gradient calls: {metrics['gradient_calls']:.1f}"
            )
            print(f"Time per sample: {metrics['time_per_sample']*1000:.2f}ms")

        return adv_images
