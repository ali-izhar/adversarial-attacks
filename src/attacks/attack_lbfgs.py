"""Simplified Limited-memory BFGS (L-BFGS) adversarial attack implementation.

This file implements the L-BFGS attack, which leverages scipy's L-BFGS-B optimizer
to create adversarial examples with minimal perturbation. The L-BFGS method
approximates second-order curvature information without explicitly forming the Hessian matrix.

Key features:
- Efficient quasi-Newton optimization for adversarial examples
- Binary search for optimal trade-off between perturbation size and attack success
- Supports both targeted and untargeted attacks
- Compatible with different loss functions
- Constraint handling for both L2 and Linf perturbation norms
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
from typing import Optional

from .baseline.attack import Attack
from .optimize.lbfgs import LBFGSOptimizer


class LBFGS(Attack):
    """
    Simplified L-BFGS adversarial attack.

    This attack uses scipy's L-BFGS-B optimizer to generate adversarial examples
    by efficiently searching for minimal perturbations that cause misclassification.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 50,
        history_size: int = 10,
        initial_const: float = 1e-2,
        binary_search_steps: int = 5,
        const_factor: float = 10.0,
        repeat_search: bool = True,
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the L-BFGS attack.

        Args:
            model: The model to attack
            norm: Norm for the perturbation constraint ('L2' or 'Linf')
            eps: Maximum allowed perturbation magnitude
            n_iterations: Maximum number of iterations for L-BFGS
            history_size: Size of history used for Hessian approximation
            initial_const: Initial trade-off constant between perturbation and loss
            binary_search_steps: Number of binary search steps to find optimal constant
            const_factor: Factor to multiply constant by when no solution is found
            repeat_search: Whether to repeat search with upper bound on last step
            rand_init: Whether to initialize with random noise
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop when adversarial criteria are met
            verbose: Whether to print progress information
            device: Device to run the attack on
        """
        # Initialize the base Attack class
        super().__init__("LBFGS", model)

        # Record attack parameters
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations
        self.history_size = history_size
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.const_factor = const_factor
        self.repeat_search = repeat_search
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer
        self.optimizer = LBFGSOptimizer(
            norm=norm,
            eps=eps,
            n_iterations=n_iterations,
            history_size=history_size,
            initial_const=initial_const,
            binary_search_steps=binary_search_steps,
            const_factor=const_factor,
            repeat_search=repeat_search,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def forward(self, images, labels):
        """
        Generate adversarial examples using L-BFGS optimization.

        Args:
            images: Input images to perturb
            labels: Target labels

        Returns:
            Adversarial examples
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Note start time for performance tracking
        start_time = time.time()

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # Define the loss function
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Define the gradient function
        def gradient_fn(x):
            x.requires_grad_(True)
            outputs = self.get_logits(x)

            # Get appropriate labels for the current batch
            curr_labels = target_labels[: x.size(0)]

            # Classification loss
            if self.targeted:
                # For targeted attacks, minimize negative CE loss to target class
                loss = -ce_loss(outputs, curr_labels)
            else:
                # For untargeted attacks, maximize CE loss to true class
                loss = ce_loss(outputs, curr_labels)

            # Compute gradient
            loss.mean().backward()
            grad = x.grad.clone()
            x.grad = None

            return grad

        # Define the loss function that returns per-sample loss values
        def loss_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)
                curr_labels = target_labels[: x.size(0)]

                if self.targeted:
                    loss = -ce_loss(outputs, curr_labels)
                else:
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

        # Run the optimizer to generate adversarial examples
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += metrics["iterations"]
        self.total_gradient_calls += metrics["gradient_calls"]

        # End timing
        end_time = time.time()
        self.total_time += end_time - start_time

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

        # Add mean L2 distortion of successful examples from optimizer metrics
        if "mean_successful_l2" in metrics and metrics["mean_successful_l2"] > 0:
            perturbation_metrics["mean_successful_l2"] = metrics["mean_successful_l2"]

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

            if "mean_successful_l2" in perturbation_metrics:
                print(
                    f"Mean L2 of successful examples: {perturbation_metrics['mean_successful_l2']:.6f}"
                )

            print(
                f"Iterations: {metrics['iterations']}, Gradient calls: {metrics['gradient_calls']}"
            )
            print(f"Time per sample: {metrics['time_per_sample']*1000:.2f}ms")

        return adv_images
