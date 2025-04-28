#!/usr/bin/env python

"""Conjugate Gradient (CG) adversarial attack implementation."""

import time
import torch
import torch.nn as nn
import numpy as np

from .baseline.attack import Attack
from .optimize.cg import ConjugateGradientOptimizer
from .utils import (
    project_adversarial_example,
    total_variation_loss,
    color_regularization,
    perceptual_loss,
    refine_perturbation,
)


class CG(Attack):
    r"""
    Enhanced Conjugate Gradient Attack with perceptual constraints.

    # Paper: "For adversarial attacks, we formulate the optimization problem as:
    # min_δ L(f(x + δ), y) subject to ||δ||_p ≤ ε"

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
        strict_epsilon_constraint (bool): If True, strictly enforce the epsilon constraint.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iter: int = 50,
        beta_method: str = "HS",
        restart_interval: int = 10,
        tv_lambda: float = 0.05,
        color_lambda: float = 0.05,
        perceptual_lambda: float = 0.05,
        rand_init: bool = True,
        fgsm_init: bool = True,
        adaptive_restart: bool = True,
        early_stopping: bool = True,
        verbose: bool = False,
        strict_epsilon_constraint: bool = True,
        refine_perturbation: bool = False,
    ):
        # Initialize the Attack base class
        super().__init__("CG", model)

        # Record attack parameters
        self.norm = norm
        self.orig_eps = eps  # Store original epsilon specified in [0,1] space

        # Scale epsilon to normalized space by dividing by ImageNet std
        mean_std = self.std.clone().detach().mean().item()
        self.eps = (
            eps / mean_std
        )  # Scaling to normalized space for consistent perturbation

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
        self.strict_epsilon_constraint = strict_epsilon_constraint
        self.refine_perturbation = refine_perturbation
        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer with correct parameters
        self.optimizer = ConjugateGradientOptimizer(
            norm=norm,
            eps=self.eps,  # Pass the scaled epsilon for normalized space
            n_iterations=n_iter,
            beta_method=beta_method,
            restart_interval=restart_interval,
            rand_init=rand_init,
            early_stopping=early_stopping,
            verbose=verbose,
            fgsm_init=fgsm_init,
            adaptive_restart=adaptive_restart,
            momentum=0.0,  # Don't use momentum for proper CG
            line_search_factor=0.5,
            sufficient_decrease=1e-4,
        )

    def forward(self, images, labels):
        r"""
        Overridden method for generating adversarial examples.

        # Paper: "min_δ L(f(x + δ), y) subject to ||δ||_p ≤ ε"
        # This is the main attack method implementing the paper's approach

        Arguments:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): Labels.
                - If self.targeted is False, labels are the actual labels
                - If self.targeted is True, labels are the target labels

        Returns:
            adversarial_images (torch.Tensor): Adversarial examples.
        """
        # Track time for performance metrics
        start_time = time.time()

        # Reset per-batch success tracking
        batch_attack_success_count = 0

        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # Calculate normalized min/max bounds for valid pixel ranges after denormalization
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

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
            # Note: These regularization terms are enhancements beyond the paper
            tv_loss = total_variation_loss(delta)
            color_loss = color_regularization(delta)
            percept_loss = perceptual_loss(delta, images[: x.size(0)])

            # Combine losses with weights - keeping per-sample values
            total_loss = (
                class_loss
                + self.tv_lambda * tv_loss
                + self.color_lambda * color_loss
                + self.perceptual_lambda * percept_loss
            )

            # Compute gradients for each sample
            # Paper: "g_t = ∇_δ L(f(x + δ_t), y)" - computing the gradient
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
                tv_loss = total_variation_loss(delta)
                color_loss = color_regularization(delta)
                percept_loss = perceptual_loss(delta, images[: x.size(0)])

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
                    # Paper: "break if ||r_{k+1}|| < tol or arg max_j f_j(x + δ_{k+1}) ≠ y_true"
                    # This is the second condition for early stopping
                    return outputs.argmax(dim=1) != curr_labels

        # Run the optimizer to generate adversarial examples
        # This executes the Algorithm 1 from the paper (Efficient Conjugate Gradient Attack)
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
            targeted=self.targeted,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += int(metrics["iterations"] * images.size(0))
        self.total_gradient_calls += int(metrics["gradient_calls"] * images.size(0))

        # Update success count from optimizer metrics - this is critical for targeted attacks
        if "success_rate" in metrics and metrics["success_rate"] > 0:
            # Calculate how many samples were successful based on optimizer's success rate
            batch_size = images.size(0)
            success_count = int((metrics["success_rate"] / 100) * batch_size)
            self.attack_success_count += success_count
            self.total_samples += batch_size

        # Strictly enforce epsilon constraint if enabled, but ensure minimal perturbation size
        if self.strict_epsilon_constraint:
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Calculate current perturbation magnitude
                delta = adv_images[i : i + 1] - images[i : i + 1]
                if self.norm.lower() == "l2":
                    delta_norm = torch.norm(delta.flatten(), p=2).item()
                else:  # Linf
                    delta_norm = torch.norm(delta.flatten(), p=float("inf")).item()

                # Only project if perturbation exceeds epsilon
                if delta_norm > self.eps:
                    adv_images[i : i + 1] = project_adversarial_example(
                        adv_images[i : i + 1],
                        images[i : i + 1],
                        self.eps,
                        self.norm,
                        min_bound=min_bound,
                        max_bound=max_bound,
                    )

        # Ensure outputs are properly clamped to valid input range
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Apply post-processing refinement to improve SSIM and reduce perturbation
        # Only refine if we have successful examples
        if metrics["success_rate"] > 0 and self.refine_perturbation:
            adv_images = refine_perturbation(
                model=self.model,
                original_images=images,
                adv_images=adv_images,
                labels=labels,
                targeted=self.targeted,
                get_target_label_fn=self.get_target_label if self.targeted else None,
                refinement_steps=10,
                min_bound=min_bound,
                max_bound=max_bound,
                device=self.device,
                verbose=self.verbose,
            )

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Manually update attack success statistics for accurate metrics
        self.attack_success_count += success_mask.sum().item()
        self.total_samples += images.size(0)

        # Compute perturbation metrics on all samples, not just successful ones
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, None  # Pass None to compute metrics on all samples
        )

        # Update final metrics counters
        self.attack_success_count += batch_attack_success_count
        self.total_samples += images.size(0)

        # Log metrics if verbose
        if self.verbose:
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

        # End timing
        end_time = time.time()
        self.total_time += end_time - start_time

        return adv_images
