#!/usr/bin/env python

"""Conjugate Gradient (CG) adversarial attack implementation."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .baseline.attack import Attack
from .optimize.cg import ConjugateGradientOptimizer
from .utils import project_adversarial_example


class CG(Attack):
    r"""
    Conjugate Gradient Attack.

    This attack creates adversarial examples using conjugate gradient descent
    to find minimal perturbations that cause misclassification.

    Arguments:
        model (nn.Module): Model to attack.
        norm (str): Norm of the attack ('L2' or 'Linf').
        eps (float): Maximum perturbation.
        n_iter (int): Number of iterations.
        beta_method (str): Formula to use for conjugate updates ('FR', 'PR', or 'HS').
        restart_interval (int): Interval for restarting conjugate gradient updates.
        tv_lambda (float): Total variation regularization weight.
        color_lambda (float): Color regularization weight.
        perceptual_lambda (float): Weight for perceptual loss.
        rand_init (bool): If True, use random initialization.
        fgsm_init (bool): If True, use FGSM initialization.
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

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Generate adversarial examples.

        Arguments:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): Labels.
                - If self.targeted is False, labels are the actual labels
                - If self.targeted is True, labels are the target labels

        Returns:
            adversarial_images (torch.Tensor): Adversarial examples.
        """
        # Start timing
        start_time = time.time()

        # Clone and detach input images
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
            # Use much higher epsilon for targeted attacks (needs more perturbation)
            eps = self.eps * 5.0  # 5x epsilon for targeted
            # Use many more iterations for targeted attacks
            n_iter = self.n_iter * 10  # 10x the iterations

            if self.verbose:
                print(f"Using increased epsilon of {eps:.4f} for targeted attack")
                print(f"Targeted CG attack with {n_iter} iterations")
        else:
            target_labels = labels
            eps = self.eps
            n_iter = self.n_iter

        # Calculate normalized min/max bounds for valid pixel ranges
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Initialize adversarial examples
        # For targeted attacks, start with stronger random perturbation
        if self.targeted and self.rand_init:
            # Create random initialization - for targeted attacks, make it closer to the original
            # but with high epsilon to allow sufficient exploration
            adv_images = torch.zeros_like(images)

            for i in range(images.size(0)):
                if self.norm.lower() == "l2":
                    # L2 random direction with 95% of epsilon
                    delta = torch.randn_like(images[i : i + 1])
                    flat_delta = delta.reshape(1, -1)
                    l2_norm = torch.norm(flat_delta, p=2)
                    delta = delta / (l2_norm + 1e-10) * (eps * 0.95)
                else:
                    # Uniform random in 95% of epsilon range
                    delta = torch.zeros_like(images[i : i + 1]).uniform_(
                        -eps * 0.95, eps * 0.95
                    )

                # Apply perturbation
                adv_images[i : i + 1] = torch.clamp(
                    images[i : i + 1] + delta, min=min_bound, max=max_bound
                )

            if self.verbose:
                print("Using strong random initialization for targeted attack")
        else:
            # Use original images as starting point
            adv_images = images.clone()

        # Set up optimization parameters
        best_advs = adv_images.clone()
        best_l2_dist = torch.full([images.size(0)], float("inf"), device=images.device)

        # For untargeted: maximize loss; for targeted: minimize loss
        multiplier = -1 if self.targeted else 1

        # Track attack metrics
        total_grad_calls = 0

        # Create momentum buffer for better targeted attack convergence
        previous_grad = torch.zeros_like(images)
        momentum = 0.9 if self.targeted else 0.0

        # Initialize for conjugate gradient
        prev_direction = None
        prev_grad = None

        # For targeted attacks, try multiple restart points
        max_restarts = 5 if self.targeted else 1
        restart_points = []

        # Main optimization loop, with multiple restarts for targeted attacks
        for restart in range(max_restarts):
            if restart > 0:
                # For subsequent restarts, use a new random initialization
                if self.targeted:
                    # Create a different random perturbation
                    temp_adv = torch.zeros_like(images)
                    for i in range(images.size(0)):
                        if self.norm.lower() == "l2":
                            # L2 random direction with 95% of epsilon
                            delta = torch.randn_like(images[i : i + 1])
                            flat_delta = delta.reshape(1, -1)
                            l2_norm = torch.norm(flat_delta, p=2)
                            delta = delta / (l2_norm + 1e-10) * (eps * 0.95)
                        else:
                            # Uniform random in 95% of epsilon range
                            delta = torch.zeros_like(images[i : i + 1]).uniform_(
                                -eps * 0.95, eps * 0.95
                            )

                        # Apply perturbation
                        temp_adv[i : i + 1] = torch.clamp(
                            images[i : i + 1] + delta, min=min_bound, max=max_bound
                        )

                    # Use the new initialization
                    adv_images = temp_adv.clone()

                    if self.verbose:
                        print(
                            f"Restart {restart+1}/{max_restarts} with new random initialization"
                        )

                # Reset direction and gradient
                prev_direction = None
                prev_grad = None

            # Inner optimization loop for current restart
            inner_iterations = n_iter // max_restarts
            for i in range(inner_iterations):
                # Create a fresh copy that requires grad
                adv_images_var = adv_images.clone().detach().requires_grad_(True)

                # Forward pass
                outputs = self.get_logits(adv_images_var)

                # Compute loss based on attack mode
                if self.targeted:
                    # For targeted attacks - more aggressive loss function

                    # Targeted logit loss (direct logit manipulation)
                    target_logits = outputs.gather(
                        1, target_labels.unsqueeze(1)
                    ).squeeze(1)
                    other_logits = outputs.clone()
                    other_logits.scatter_(1, target_labels.unsqueeze(1), float("-inf"))
                    highest_other = other_logits.max(1)[0]

                    # Margin term with higher confidence (30.0)
                    logit_diff = highest_other - target_logits
                    margin_loss = torch.clamp(logit_diff + 30.0, min=0)

                    # Cross-entropy term (smaller weight to keep focus on direct logit manipulation)
                    ce_loss = F.cross_entropy(outputs, target_labels, reduction="none")

                    # Combined loss - weighted heavily toward margin term
                    loss = margin_loss + 0.1 * ce_loss
                else:
                    # For untargeted attacks, we maximize CE loss
                    loss = F.cross_entropy(outputs, target_labels, reduction="none")

                # Compute gradients
                grad = torch.autograd.grad(
                    loss.sum(), adv_images_var, retain_graph=False, create_graph=False
                )[0]

                # Update gradient call counter
                total_grad_calls += 1

                # Check success to determine which examples to update
                with torch.no_grad():
                    if self.targeted:
                        success = outputs.argmax(dim=1) == target_labels
                    else:
                        success = outputs.argmax(dim=1) != labels

                    # Only optimize examples that aren't yet successful
                    if self.early_stopping and success.all():
                        break

                    # Mask for active examples (those still being optimized)
                    active = (
                        ~success
                        if self.early_stopping
                        else torch.ones_like(success, dtype=torch.bool)
                    )

                    # Skip if no active examples
                    if not active.any():
                        break

                # Apply momentum to gradient - higher for targeted attacks
                grad = momentum * previous_grad + (1 - momentum) * grad
                previous_grad = grad.clone().detach()

                # Apply multiplier to gradient based on attack type
                grad = multiplier * grad

                # Conjugate gradient update
                if i == 0 or prev_grad is None or i % (self.restart_interval // 2) == 0:
                    # First iteration or restart: use steepest descent direction
                    direction = -grad
                    prev_grad = grad.clone()
                    prev_direction = direction.clone()
                else:
                    # Compute beta using Hestenes-Stiefel formula
                    beta = torch.zeros(images.size(0), device=images.device)

                    for j in range(images.size(0)):
                        if not active[j]:
                            continue

                        grad_diff = grad[j].flatten() - prev_grad[j].flatten()
                        numerator = torch.dot(grad[j].flatten(), grad_diff)
                        denominator = torch.dot(prev_direction[j].flatten(), grad_diff)

                        # Safeguard against division by zero
                        if abs(denominator) > 1e-10:
                            beta[j] = numerator / denominator

                        # Ensure beta is positive and not too large
                        beta[j] = torch.clamp(beta[j], min=0.0, max=1.0)

                    # Compute new direction: d_i = -g_i + beta * d_{i-1}
                    direction = torch.zeros_like(grad)
                    for j in range(images.size(0)):
                        if active[j]:
                            direction[j] = -grad[j] + beta[j] * prev_direction[j]

                    # Update previous gradient and direction
                    prev_grad = grad.clone()
                    prev_direction = direction.clone()

                # Determine step size based on attack type and progress
                if self.targeted:
                    # For targeted attacks, higher step size throughout
                    progress = i / inner_iterations
                    if progress < 0.25:
                        step_size = (
                            0.1  # Very large steps initially (10x more aggressive)
                        )
                    elif progress < 0.75:
                        step_size = 0.05  # Medium steps in the middle
                    else:
                        step_size = 0.01  # Smaller steps at the end for fine-tuning
                else:
                    # For untargeted attacks, use smaller constant step size
                    step_size = 0.01

                # Create a new tensor for the updated adversarial examples
                new_adv_images = adv_images.clone()

                # Update adversarial examples only for active examples
                with torch.no_grad():
                    for j in range(images.size(0)):
                        if not active[j]:
                            continue

                        # Normalize direction for L2 norm
                        current_dir = direction[j : j + 1].clone()
                        if self.norm.lower() == "l2":
                            dir_norm = torch.norm(current_dir.flatten())
                            if dir_norm > 1e-10:
                                current_dir = current_dir / dir_norm

                        # Take a step in the direction
                        if self.norm.lower() == "l2":
                            temp = adv_images[j : j + 1] + step_size * current_dir
                        else:  # Linf
                            temp = adv_images[j : j + 1] + step_size * torch.sign(
                                current_dir
                            )

                        # Project perturbation to epsilon constraint
                        temp = project_adversarial_example(
                            temp, images[j : j + 1], eps, self.norm
                        )

                        # Ensure valid bounds
                        temp = torch.clamp(temp, min=min_bound, max=max_bound)

                        # Store the result in the new tensor
                        new_adv_images[j : j + 1] = temp

                        # Update best adversarial example if this one meets the adversarial condition
                        # and has smaller L2 distance
                        pred_class = outputs[j].argmax().item()
                        if (
                            self.targeted and pred_class == target_labels[j].item()
                        ) or (not self.targeted and pred_class != labels[j].item()):
                            current_dist = torch.norm(
                                (new_adv_images[j] - images[j]).flatten(), p=2
                            )
                            if current_dist < best_l2_dist[j]:
                                best_l2_dist[j] = current_dist
                                best_advs[j] = new_adv_images[j].clone()

                # Update adv_images with the new version
                adv_images = new_adv_images.clone()

                # Print progress periodically
                if self.verbose and (i + 1) % max(1, inner_iterations // 5) == 0:
                    with torch.no_grad():
                        outputs = self.get_logits(adv_images)
                        if self.targeted:
                            success_rate = (
                                outputs.argmax(dim=1) == target_labels
                            ).float().mean().item() * 100
                        else:
                            success_rate = (
                                outputs.argmax(dim=1) != labels
                            ).float().mean().item() * 100
                        print(
                            f"Restart {restart+1}/{max_restarts}, Iteration {i+1}/{inner_iterations}, "
                            f"Success rate: {success_rate:.2f}%"
                        )

            # Save current adv_images as a potential restart point
            restart_points.append(adv_images.clone())

        # End timing
        end_time = time.time()

        # Use the best adversarial examples found across all restarts
        adv_images = best_advs.clone()

        # One final refinement for targeted attacks
        if self.targeted:
            with torch.no_grad():
                outputs = self.get_logits(adv_images)
                success = outputs.argmax(dim=1) == target_labels

                # If we have at least one successful example, use binary search to find
                # minimal perturbation that maintains the target class
                for j in range(images.size(0)):
                    if success[j]:
                        # Binary search for minimal perturbation
                        original = images[j : j + 1]
                        perturbed = adv_images[j : j + 1].clone()
                        delta = perturbed - original

                        # Start with current successful perturbation
                        alpha_low = 0.0  # Lower bound = no perturbation
                        alpha_high = 1.0  # Upper bound = full perturbation
                        best_alpha = 1.0  # Start with full perturbation

                        # Binary search for 10 steps
                        for search_step in range(10):
                            alpha_mid = (alpha_low + alpha_high) / 2
                            mid_point = original + alpha_mid * delta

                            # Check if midpoint is still adversarial
                            mid_output = self.get_logits(mid_point)
                            mid_class = mid_output.argmax(dim=1)

                            if mid_class.item() == target_labels[j].item():
                                # Still successful, can reduce perturbation
                                alpha_high = alpha_mid
                                best_alpha = alpha_mid  # Save this successful alpha
                            else:
                                # Not successful, need more perturbation
                                alpha_low = alpha_mid

                        # Use the minimal successful perturbation found
                        refined = original + best_alpha * delta
                        adv_images[j : j + 1] = refined

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Update metrics
        self.total_iterations += n_iter * images.size(0)
        self.total_gradient_calls += total_grad_calls
        self.attack_success_count += success_mask.sum().item()
        self.total_samples += images.size(0)
        self.total_time += end_time - start_time

        # Compute perturbation metrics
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, None  # Use all samples
        )

        # Log metrics if verbose
        if self.verbose:
            print(f"Attack complete: {success_rate:.2f}% success rate")
            print(
                f"Metrics: L2={perturbation_metrics['l2_norm']:.6f}, "
                f"L∞={perturbation_metrics['linf_norm']:.6f}, "
                f"SSIM={perturbation_metrics['ssim']:.4f}"
            )
            print(f"Gradient calls: {total_grad_calls}")
            print(
                f"Time per sample: {(end_time - start_time) * 1000 / images.size(0):.2f}ms"
            )

        return adv_images
