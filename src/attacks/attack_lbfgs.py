"""Limited-memory BFGS (L-BFGS) adversarial attack implementation.

This file implements the L-BFGS attack, which leverages the quasi-Newton L-BFGS optimization
algorithm to create adversarial examples. The L-BFGS method approximates second-order curvature
information without explicitly forming the Hessian matrix, making it efficient for high-dimensional
optimization problems like adversarial attacks against deep neural networks.

Key features:
- Efficient second-order optimization using limited memory approximation
- Supports both targeted and untargeted attacks
- Configurable line search methods (strong Wolfe conditions or Armijo)
- Compatible with different loss functions (cross-entropy and margin loss)
- Constraint handling for both L2 and Linf perturbation norms
- Adjustable memory usage for Hessian approximation via history size
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
from typing import Optional

from .baseline.attack import Attack
from .optimize.lbfgs import LBFGSOptimizer


class LBFGS(Attack):
    """
    Limited-memory BFGS (L-BFGS) adversarial attack.

    This attack uses the LBFGSOptimizer to generate adversarial examples by approximating
    second-order information using a limited history of gradients and position differences.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        n_iterations: int = 50,
        history_size: int = 10,
        line_search_fn: str = "armijo",
        max_line_search: int = 20,
        initial_step: float = 3.0,
        rand_init: bool = True,
        init_std: float = 0.05,
        grad_scale: float = 1.0,
        restart_freq: int = 10,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the L-BFGS attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude.
            targeted: Whether to perform a targeted attack.
            loss_fn: Loss function to use ('cross_entropy' or 'margin').
            n_iterations: Maximum number of iterations to run.
            history_size: Number of past iterations to store for Hessian approximation.
            line_search_fn: Line search method ('strong_wolfe' or 'armijo').
            max_line_search: Maximum number of iterations for the line search.
            initial_step: Initial step size for the line search.
            rand_init: Whether to initialize the attack with random noise.
            init_std: Standard deviation for random initialization.
            grad_scale: Factor to scale gradients for more aggressive updates.
            restart_freq: How often to restart from steepest descent direction.
            early_stopping: Stop early if adversarial criteria are met.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Initialize the base Attack class
        super().__init__("LBFGS", model)

        # Record attack parameters
        self.norm = norm
        self.eps = eps
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.n_iterations = n_iterations
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.max_line_search = max_line_search
        self.initial_step = initial_step
        self.rand_init = rand_init
        self.init_std = init_std
        self.grad_scale = grad_scale
        self.restart_freq = restart_freq
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer with correct parameters
        self.optimizer = LBFGSOptimizer(
            norm=norm,
            eps=eps,
            n_iterations=n_iterations,
            history_size=history_size,
            line_search_fn=line_search_fn,
            max_line_search=max_line_search,
            initial_step=initial_step,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def forward(self, images, labels):
        """
        Overridden method for generating adversarial examples using L-BFGS.

        Args:
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
            if self.targeted:
                curr_labels = target_labels[: x.size(0)]
            else:
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

        # Compute perturbation metrics
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
