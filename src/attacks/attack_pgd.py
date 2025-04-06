"""Projected Gradient Descent (PGD) adversarial attack implementation.

This file implements the PGD attack, which is a powerful first-order iterative method for
generating adversarial examples. The PGD method works by taking steps in the direction of the
gradient and then projecting back onto the constraint set defined by the perturbation norm ball,
making it an effective and widely used attack in adversarial machine learning.

Key features:
- Simple yet powerful first-order optimization method
- Supports both targeted and untargeted attacks
- Configurable step size schedules (constant or diminishing)
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
- Optional random initialization within the perturbation space
- Early stopping capability when adversarial criteria are met

Expected inputs:
- A neural network model to attack
- Input samples (images) to perturb
- Target labels (true labels for untargeted attacks, target labels for targeted attacks)
- Configuration parameters for the attack and optimizer

Expected outputs:
- Adversarial examples that aim to fool the model
- Performance metrics (iterations, gradient calls, time, success rate)
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from .baseline.attack import Attack
from .optimize.pgd import PGDOptimizer


class PGD(Attack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    This attack uses the PGDOptimizer to generate adversarial examples by iteratively taking
    steps in the gradient direction (to maximize or minimize the loss depending on the attack)
    and then projecting the perturbed input back onto the constraint set (the eps-ball).
    It supports both targeted and untargeted attacks and allows for different step size schedules.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 100,
        alpha_init: float = 0.1,
        alpha_type: str = "diminishing",
        loss_fn: str = "cross_entropy",
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        normalize_grad: bool = True,  # Whether to normalize gradient for better convergence
        debug: bool = False,  # Enable debugging output
    ):
        """
        Initialize the PGD attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude.
            n_iterations: Maximum number of iterations to run.
            alpha_init: Initial step size for the updates.
            alpha_type: Step size schedule ('constant' or 'diminishing').
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            rand_init: Whether to initialize the attack with random noise.
            init_std: Standard deviation for random initialization.
            early_stopping: Stop early if adversarial criteria are met.
            verbose: Print progress updates.
            normalize_grad: Whether to normalize gradients for L2 attacks.
            debug: Print debug information during the attack.
        """
        # Initialize the base Attack class
        super().__init__("PGD", model)

        # Record attack parameters
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations
        self.alpha_init = alpha_init
        self.alpha_type = alpha_type
        self.loss_fn_type = loss_fn
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.normalize_grad = normalize_grad
        self.debug = debug

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Adjust step size for Linf attacks
        if norm.lower() == "linf" and alpha_init > eps:
            if verbose:
                print(
                    f"Warning: Reducing step size to eps/10 for Linf attack (was: {alpha_init}, eps: {eps})"
                )
            alpha_init = eps / 10.0

        # Create the optimizer for PGD
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
            maximize=True,  # Default for untargeted attacks, will be set in forward()
            normalize_grad=normalize_grad,
            debug=debug,
        )

    def forward(self, images, labels):
        r"""
        Overridden method for generating adversarial examples using PGD.

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

        # Set maximize parameter based on attack mode (maximize loss for untargeted, minimize for targeted)
        self.optimizer.maximize = not self.targeted

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
                loss = -ce_loss(outputs, curr_labels)
            else:
                # For untargeted attacks, maximize CE loss to true class
                loss = ce_loss(outputs, curr_labels)

            # Compute mean loss for this batch
            mean_loss = loss.mean()

            # Compute gradient
            grad = torch.autograd.grad(mean_loss, x)[0]

            return grad

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
                    # For targeted attacks, minimize CE to target (track as negative for consistency)
                    loss = -ce_loss(outputs, curr_labels)
                else:
                    # For untargeted attacks, maximize CE from true label
                    loss = ce_loss(outputs, curr_labels)

            return loss

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

        # Print debug information about the attack configuration
        if self.debug:
            print(f"Attack configuration:")
            print(f"- Norm: {self.norm}")
            print(f"- Epsilon: {self.eps}")
            print(f"- Step size: {self.optimizer.alpha_init}")
            print(f"- Targeted: {self.targeted}")
            print(f"- Gradient normalization: {self.normalize_grad}")
            print(f"- Random initialization: {self.rand_init}")
            print(f"- Iterations: {self.n_iterations}")

        # Run the optimizer to generate adversarial examples
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += int(metrics["iterations"] * images.size(0))
        self.total_gradient_calls += int(metrics["gradient_calls"] * images.size(0))

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time

        # Calculate time per sample for metrics
        time_per_sample = (self.end_time - self.start_time) / images.size(0)
        metrics["time_per_sample"] = time_per_sample

        # Ensure outputs are properly clamped
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Compute perturbation metrics
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, success_mask
        )

        # Debug information about final perturbation
        if self.debug:
            delta = adv_images - images
            if self.norm.lower() == "l2":
                norm_values = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                print(
                    f"Final L2 norms: min={norm_values.min().item():.4f}, mean={norm_values.mean().item():.4f}, max={norm_values.max().item():.4f}, eps={self.eps:.4f}"
                )
            elif self.norm.lower() == "linf":
                norm_values = torch.norm(
                    delta.view(delta.size(0), -1), p=float("inf"), dim=1
                )
                print(
                    f"Final Linf norms: min={norm_values.min().item():.4f}, mean={norm_values.mean().item():.4f}, max={norm_values.max().item():.4f}, eps={self.eps:.4f}"
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
