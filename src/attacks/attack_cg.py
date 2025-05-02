#!/usr/bin/env python

"""Conjugate Gradient (CG) adversarial attack implementation."""

import time
import torch
import torch.nn as nn

from .baseline.attack import Attack


class CG(Attack):
    r"""
    Conjugate Gradient (CG) inspired adversarial attack.
    This is not a direct implementation of the CG algorithm to solve Ax=b,
    but rather uses the CG update rule for search directions within a
    projected gradient ascent/descent framework for loss maximization/minimization.
    This implementation keeps track of the best attack found across iterations.

    Distance Measure : L2 or Linf

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Norm to use for projection ("L2" or "Linf"). (Default: "L2")
        eps (float): maximum perturbation in the specified norm. (Default: 0.5)
        steps (int): number of steps. (Default: 40)
        alpha (float): step size in normalized space. (Default: 0.05)
        beta_method (str): Method for calculating beta ('FR' for Fletcher-Reeves or 'PR' for Polak-Ribière). (Default: 'PR')
        rand_init (bool): Whether to use random initialization for perturbation. (Default: False)
        alpha_multiplier (float): Multiplier for alpha when using Linf norm. (Default: 1.0)

    Shape:
        - images: :math:`(N, C, H, W)` normalized images with ImageNet mean/std
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)` normalized adversarial images.

    Examples::
        >>> attack = CG(model, norm="L2", eps=0.5, steps=40, alpha=0.05) # Tuned defaults L2
        >>> attack = CG(model, norm="Linf", eps=8/255, steps=40, alpha=2/255) # Tuned defaults Linf
        >>> adv_images = attack(images, labels) # Untargeted
        >>> attack.set_mode_targeted_random()
        >>> adv_images = attack(images, labels) # Targeted

    Note:
        Epsilon and alpha should be specified in normalized space to match
        the behavior of other attacks.
    """

    def __init__(
        self,
        model,
        norm="L2",
        eps=0.5,
        steps=40,
        alpha=0.05,
        beta_method="PR",
        rand_init=False,
        alpha_multiplier=1.0,
    ):
        """Initialize CG attack.

        Args:
            model: Target model to attack
            norm: Norm to use for projection ("L2" or "Linf"). (Default: "L2")
            eps: Maximum perturbation norm in normalized space (default: 0.5)
            steps: Number of iterations (default: 40 - tuned)
            alpha: Step size in normalized space (default: 0.05 - tuned for L2)
            beta_method: Method for beta calculation ('FR' or 'PR') (default: 'PR')
            rand_init: Whether to use random initialization (default: False)
            alpha_multiplier: Multiplier for alpha when using Linf norm (default: 1.0)
        """
        super().__init__("CG", model)
        self.norm = norm.upper()
        if self.norm not in ["L2", "LINF"]:
            raise ValueError(f"Norm {norm} not supported. Use 'L2' or 'Linf'.")
        self.eps = eps  # Perturbation budget in normalized space
        self.steps = steps
        self.alpha = alpha  # Step size in normalized space

        # Adjust alpha for Linf attacks if needed
        if self.norm == "LINF":
            self.alpha = alpha * alpha_multiplier

        if beta_method not in ["FR", "PR"]:
            raise ValueError("beta_method must be 'FR' or 'PR'")
        self.beta_method = beta_method
        self.rand_init = rand_init
        # Supports both untargeted and targeted modes
        self.supported_mode = ["default", "targeted"]
        # _attack_mode is managed by the base class methods like set_mode_default() etc.

    def forward(self, images, labels):
        r"""Overridden."""
        # Track time for performance metrics
        start_time = time.time()

        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.size(0)

        # Get target labels if in targeted mode
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # Calculate normalized min/max bounds for valid pixel values
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1).expand_as(images)
        max_bound = max_bound.view(1, 3, 1, 1).expand_as(images)

        # Initialize perturbation delta with random noise if enabled
        if self.rand_init:
            if self.norm == "L2":
                # Random initialization on unit sphere
                delta = torch.randn_like(images)
                delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                delta = delta * self.eps * 0.5 / delta_norms.view(batch_size, 1, 1, 1)
            else:  # LINF
                # Random initialization in [-eps, eps]
                delta = torch.rand_like(images) * 2 * self.eps - self.eps

            # Ensure initial perturbed images are valid
            adv_images = images + delta
            delta = torch.clamp(adv_images, min_bound, max_bound) - images
        else:
            # Otherwise initialize with zeros
            delta = torch.zeros_like(images, requires_grad=False)

        # Initialize tracking for best adversarial examples found
        best_adv_images = images.clone().detach()
        best_dist = torch.full(
            (batch_size,), float("inf"), device=self.device, dtype=images.dtype
        )

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Previous gradient and direction for CG updates
        g_prev = None
        d_prev = None

        # Main optimization loop
        for step in range(self.steps):
            # Add perturbation and clamp to valid range
            adv_images = images + delta
            adv_images.clamp_(min_bound, max_bound)  # In-place clamp
            adv_images.requires_grad = True

            # Get model predictions and loss
            # Calls get_logits which increments gradient counter
            outputs = self.get_logits(adv_images)

            # Calculate loss based on attack mode
            if self.targeted:
                # Minimize loss w.r.t target labels for targeted attacks
                cost = -loss_fn(outputs, target_labels)
            else:
                # Maximize loss w.r.t true labels for untargeted attacks
                cost = loss_fn(outputs, labels)

            # Compute gradient w.r.t. adversarial image (gradient of cost)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # --- Calculate CG Direction ---
            if (
                step == 0 or g_prev is None or d_prev is None
            ):  # Ensure g_prev/d_prev exist
                beta = 0.0
                d_k = grad  # First step is just gradient ascent/descent
            else:
                g_k_flat = grad.view(batch_size, -1)
                g_prev_flat = g_prev.view(batch_size, -1)
                dot_gk_gk = torch.sum(g_k_flat * g_k_flat, dim=1)
                # Prevent division by zero/very small numbers for dot_gprev_gprev
                dot_gprev_gprev = torch.sum(g_prev_flat * g_prev_flat, dim=1)
                dot_gprev_gprev = torch.clamp(dot_gprev_gprev, min=1e-12)

                if self.beta_method == "FR":
                    # Fletcher-Reeves
                    beta = dot_gk_gk / dot_gprev_gprev
                elif self.beta_method == "PR":
                    # Polak-Ribière
                    dot_gk_diff = torch.sum(g_k_flat * (g_k_flat - g_prev_flat), dim=1)
                    beta = dot_gk_diff / dot_gprev_gprev
                    # Use non-negative PR variant for stability in ascent/descent
                    beta = torch.relu(beta)

                beta = beta.view(-1, 1, 1, 1)
                # Update search direction using conjugate direction
                d_k = grad + beta * d_prev

            # Store current gradient and direction for next iteration
            g_prev = grad.detach()
            d_prev = d_k.detach()

            # --- Update delta and project ---
            # Normalize direction for consistent step size
            if self.norm == "LINF":
                # For Linf, use sign of gradient (like FGSM)
                direction = torch.sign(d_k)
            else:
                # For L2, normalize by L2 norm
                direction = d_k

            # Apply step
            delta = delta + self.alpha * direction.detach()
            delta = delta.detach()

            # Project delta onto appropriate norm ball
            if self.norm == "L2":
                # L2 projection
                delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                factor = self.eps / delta_norms.clamp(min=1e-12)
                factor = torch.min(
                    factor, torch.ones_like(delta_norms)
                )  # Clamp factor to 1
                delta = delta * factor.view(batch_size, 1, 1, 1)
            elif self.norm == "LINF":
                # Linf projection
                delta = torch.clamp(delta, -self.eps, self.eps)

            # Ensure perturbed images are valid
            adv_images = torch.clamp(images + delta, min_bound, max_bound)
            # Recompute delta to account for image bounds
            delta = adv_images - images

            # --- Track Best Adversarial Example ---
            # Current adversarial example after step and projection
            current_adv_images = adv_images.detach()

            # Calculate distance for current adversarial examples
            current_pert = current_adv_images - images
            if self.norm == "L2":
                current_dist = torch.norm(current_pert.view(batch_size, -1), p=2, dim=1)
            else:  # "LINF"
                current_dist = torch.norm(
                    current_pert.view(batch_size, -1), p=float("inf"), dim=1
                )

            # Check success of current adversarial examples
            with torch.no_grad():
                # Need current predictions to check success
                current_outputs = self.model(current_adv_images)
                if self.targeted:
                    current_success_mask = (
                        current_outputs.argmax(dim=1) == target_labels
                    )
                else:
                    current_success_mask = current_outputs.argmax(dim=1) != labels

            # Identify which examples to update (successful AND lower distance)
            update_mask = (current_success_mask & (current_dist < best_dist)).float()
            update_mask_view = update_mask.view(batch_size, 1, 1, 1)

            # Update best distances and best adversarial images
            best_dist = update_mask * current_dist + (1 - update_mask) * best_dist
            best_adv_images = (
                update_mask_view * current_adv_images.detach()
                + (1 - update_mask_view) * best_adv_images
            )

            # Track iterations (one outer step per sample)
            self.total_iterations += batch_size

            # Gradient call counter is handled by self.get_logits()

        # --- Final Steps ---
        # Use the best adversarial images found across all iterations
        final_adv_images = best_adv_images.detach()

        # Update total time
        end_time = time.time()
        self.total_time += end_time - start_time

        # --- Use Base Class for Final Evaluation & Metric Calculation ---
        # Evaluate success on the *best* images found
        _success_rate, success_mask, _predictions = self.evaluate_attack_success(
            images,
            final_adv_images,
            labels,  # Pass original labels for evaluation context
        )
        # Calculate metrics on the *best* images found
        self.compute_perturbation_metrics(images, final_adv_images, success_mask)

        return final_adv_images
