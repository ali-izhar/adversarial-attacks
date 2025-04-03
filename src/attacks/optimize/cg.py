"""Conjugate Gradient optimization method.

This file implements the Conjugate Gradient optimizer for generating adversarial examples
against neural networks. The optimizer finds minimal perturbations that cause model
misclassification by efficiently navigating the loss landscape using conjugate search
directions.
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable

from .projections import project_adversarial_example


class ConjugateGradientOptimizer:
    """
    Conjugate Gradient optimization method for generating adversarial examples.

    The algorithm uses conjugate directions to efficiently search the loss landscape.
    Depending on the setting, it uses either the Fletcher-Reeves or the Polak-Ribière formula
    for updating the search direction.
    """

    def __init__(
        self,
        norm: str = "L2",
        eps: float = 10.0,  # Increased default
        n_iterations: int = 150,
        fletcher_reeves: bool = True,
        restart_interval: int = 10,
        backtracking_factor: float = 0.5,
        sufficient_decrease: float = 1e-7,
        line_search_max_iter: int = 15,
        rand_init: bool = True,
        init_std: float = 0.1,
        early_stopping: bool = True,
        verbose: bool = False,
        momentum: float = 0.2,
    ):
        """
        Initialize the optimizer with parameters that control:
        - The type of norm constraint (L2 or Linf)
        - Maximum allowed perturbation size (eps)
        - Maximum number of iterations and other algorithmic details
        """
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations
        self.fletcher_reeves = fletcher_reeves
        self.restart_interval = restart_interval
        self.backtracking_factor = backtracking_factor
        self.sufficient_decrease = sufficient_decrease
        self.line_search_max_iter = line_search_max_iter
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.momentum = momentum

    def _compute_beta(
        self, grad_new: torch.Tensor, grad_old: torch.Tensor, d_old: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conjugate gradient update parameter beta.

        For Fletcher-Reeves:
            beta = ||grad_new||^2 / ||grad_old||^2
        For Polak-Ribière:
            beta = (grad_new·(grad_new - grad_old)) / ||grad_old||^2, then clamped to be non-negative.

        Args:
            grad_new: New gradient at the current point.
            grad_old: Gradient at the previous point.
            d_old: Previous search direction (unused in these formulas, but kept for API consistency).

        Returns:
            beta: A per-example tensor for the conjugate update.
        """
        # Flatten gradients per example and compute squared norms.
        grad_new_sq = (grad_new.view(grad_new.shape[0], -1) ** 2).sum(dim=1)
        grad_old_sq = (grad_old.view(grad_old.shape[0], -1) ** 2).sum(dim=1)

        if self.fletcher_reeves:
            # Fletcher-Reeves update: ratio of squared norms.
            beta = grad_new_sq / (grad_old_sq + 1e-10)
        else:
            # Polak-Ribière update: measures change in gradients.
            grad_diff = grad_new - grad_old
            # Compute inner product: grad_new dot (grad_new - grad_old)
            numerator = (
                grad_new.view(grad_new.shape[0], -1)
                * grad_diff.view(grad_diff.shape[0], -1)
            ).sum(dim=1)
            beta = numerator / (grad_old_sq + 1e-10)
            # Use the Polak-Ribière+ variant: non-negative beta.
            beta = torch.clamp(beta, min=0)

        return beta

    def _line_search(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        current_grad: torch.Tensor,
        current_loss: torch.Tensor,
        x_original: Optional[torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Backtracking line search to determine a suitable step size alpha.
        If line search doesn't make progress, falls back to PGD-like fixed step size.

        Args:
            x: Current point.
            direction: Descent direction.
            current_grad: Gradient at the current point.
            current_loss: Loss at the current point (per example).
            x_original: Original images (for projection constraints).
            loss_fn: Function to compute loss; should return a tensor of per-example losses.

        Returns:
            alpha: A per-example tensor containing the step sizes.
        """
        batch_size = x.shape[0]
        # Start with a much larger step size to be more aggressive
        alpha = torch.ones(batch_size, device=x.device) * 2.0

        # Compute the directional derivative: d^T * grad.
        dir_deriv = (
            direction.view(batch_size, -1) * current_grad.view(batch_size, -1)
        ).sum(dim=1)

        # Compute the Armijo threshold.
        armijo_threshold = current_loss + self.sufficient_decrease * alpha * dir_deriv

        # Track if we've made progress for each sample
        made_progress = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Normalize direction for better step size control
        direction_sign = direction.sign()

        # Iterate to find a step size that satisfies the condition.
        for _ in range(self.line_search_max_iter):
            # Update candidate point: take a step of size alpha in the descent direction.
            x_new = x + alpha.view(-1, 1, 1, 1) * direction

            # Project back to the allowed perturbation region if original images provided.
            if x_original is not None:
                x_new = project_adversarial_example(
                    x_new, x_original, self.eps, self.norm
                )

            # Evaluate the loss at the new point.
            new_loss = loss_fn(x_new)

            # Check the Armijo condition (per example).
            armijo_condition = new_loss <= armijo_threshold

            # Update progress tracker
            made_progress = made_progress | armijo_condition

            # If all examples meet the condition, exit the loop.
            if armijo_condition.all():
                break

            # For examples that failed, reduce the step size more aggressively
            alpha[~armijo_condition] *= self.backtracking_factor

            # Update the Armijo threshold for those examples with the new alpha.
            armijo_threshold[~armijo_condition] = current_loss[~armijo_condition] + (
                self.sufficient_decrease
                * alpha[~armijo_condition]
                * dir_deriv[~armijo_condition]
            )

        # For examples where line search made no progress, fall back to PGD-like step
        if not made_progress.all() and x_original is not None:
            no_progress_mask = ~made_progress

            # Use sign of gradient with fixed step size (PGD approach)
            # For samples where line search failed
            alpha_pgd = torch.ones_like(alpha) * 0.1  # Fixed step relative to epsilon

            # Create a new alpha array
            alpha_new = alpha.clone()

            # For samples with no progress, use PGD-style update
            alpha_new[no_progress_mask] = alpha_pgd[no_progress_mask]

            # Also use the sign direction rather than the full gradient
            return alpha_new, direction_sign

        # Return the found alpha values and original direction
        return alpha, direction

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run the conjugate gradient optimization procedure to generate adversarial examples.

        The algorithm:
          1. (Optionally) initializes with random noise at the epsilon boundary.
          2. Computes the gradient and sets the initial search direction to -grad.
          3. Uses line search to find a good step size satisfying the Armijo condition.
          4. Updates the point and then computes the new gradient.
          5. Uses either a restart or the conjugate update for the search direction.
          6. Optionally stops early if the adversarial goal is met.

        Args:
            x_init: Initial images (clean images to be perturbed).
            gradient_fn: Function that returns the gradient of the adversarial loss.
            loss_fn: Function that computes the loss (must return per-example losses).
            success_fn: Function to check if adversarial criteria are met (returns boolean tensor).
            x_original: Original images for projection constraints.

        Returns:
            A tuple (x_adv, metrics) where:
              - x_adv: The optimized adversarial examples.
              - metrics: Dictionary containing iterations, gradient calls, total time, and success rate.
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Initialize adversarial examples with random noise if enabled.
        if self.rand_init and x_original is not None:
            # PGD-style initialization at the epsilon boundary
            if self.norm.lower() == "l2":
                # Generate random direction from uniform distribution
                delta = torch.rand_like(x_init) * 2 - 1  # Values between -1 and 1
                # Normalize to unit norm
                flat_delta = delta.reshape(delta.shape[0], -1)
                l2_norm = torch.norm(flat_delta, p=2, dim=1, keepdim=True)
                normalized_delta = flat_delta / (l2_norm + 1e-10)
                # Scale to epsilon magnitude
                delta = normalized_delta.reshape(delta.shape) * self.eps
                # Apply the perturbation
                x_adv = x_original + delta
            else:  # Linf norm
                # Uniform random perturbation in [-eps, eps]
                delta = torch.zeros_like(x_init).uniform_(-self.eps, self.eps)
                x_adv = x_original + delta

            # Ensure the result is within valid bounds (will be handled by projection)
            x_adv = project_adversarial_example(x_adv, x_original, self.eps, self.norm)
        else:
            x_adv = x_init.clone()

        # Initialize metrics for reporting.
        start_time = time.time()
        iterations = 0
        gradient_calls = 0
        successful = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Check initial success (if a success criterion is provided).
        if success_fn is not None:
            success = success_fn(x_adv)
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Store the initial success state for metrics
        initial_success = success.clone()

        # Compute the initial gradient and set the initial search direction.
        grad = gradient_fn(x_adv)
        gradient_calls += batch_size
        d = -grad  # Initial search direction is the steepest descent.

        # Initialize momentum buffer
        prev_direction = torch.zeros_like(d)

        loss_current = loss_fn(x_adv)  # Compute initial loss (per-example).

        # Main optimization loop.
        for t in range(self.n_iterations):
            iterations += 1

            # Early stopping: break if all examples have already met the adversarial criteria.
            if self.early_stopping and success.all():
                break

            # Save the current gradient for the conjugate update.
            grad_old = grad.clone()

            # Apply momentum to the search direction
            if t > 0:  # Skip first iteration since no previous direction yet
                d = d + self.momentum * prev_direction

            # Line search: determine a suitable step size alpha.
            if self.early_stopping:
                working_examples = ~success  # Only update examples not yet successful.
                if working_examples.sum() == 0:
                    break

                # Perform line search only for working examples
                x_adv_working = x_adv[working_examples]
                d_working = d[working_examples]
                grad_old_working = grad_old[working_examples]
                loss_current_working = loss_current[working_examples]
                x_original_working = (
                    x_original[working_examples] if x_original is not None else None
                )

                # Create a wrapper for working examples
                def subset_loss_fn(x_subset):
                    # Map back to full batch indices
                    full_x = x_adv.clone()
                    full_x[working_examples] = x_subset
                    # Get loss for all samples
                    full_loss = loss_fn(full_x)
                    # Return only the relevant losses
                    return full_loss[working_examples]

                # Line search returns alpha and potentially modified direction
                alpha_working, dir_working = self._line_search(
                    x_adv_working,
                    d_working,
                    grad_old_working,
                    loss_current_working,
                    x_original_working,
                    subset_loss_fn,
                )

                # Update only the working examples
                step = torch.zeros_like(x_adv)
                # Use the new direction (sign or original) based on what line search returned
                step[working_examples] = alpha_working.view(-1, 1, 1, 1) * dir_working
                x_adv = x_adv + step
            else:
                # Full batch line search
                alpha, direction_to_use = self._line_search(
                    x_adv, d, grad_old, loss_current, x_original, loss_fn
                )

                # Save current direction for momentum
                prev_direction = d.clone()

                # Update adversarial examples using the alpha and direction
                x_adv = x_adv + alpha.view(-1, 1, 1, 1) * direction_to_use

            # Project back to the allowed perturbation region.
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )

            # Update the gradient
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Create a wrapper for the gradient function that can work with subsets
                def subset_gradient_fn(x_subset):
                    # Create a full batch tensor but only update the working examples
                    x_full = x_adv.clone()
                    x_full[working_examples] = x_subset
                    # Compute gradient on full batch then extract only the working examples
                    full_grad = gradient_fn(x_full)
                    return full_grad[working_examples]

                grad_full = torch.zeros_like(x_adv)
                grad_working = subset_gradient_fn(x_adv[working_examples])
                grad_full[working_examples] = grad_working
                grad = grad_full

                # Count gradient calls for working examples only
                gradient_calls += working_examples.sum().item()

                # Track which examples were successful during optimization
                success_update = success_fn(x_adv)
                newly_successful = success_update & ~success
                successful = successful | newly_successful
                success = success_update
            else:
                grad = gradient_fn(x_adv)
                gradient_calls += batch_size

                # Track which examples became successful in this iteration
                if success_fn is not None:
                    success_update = success_fn(x_adv)
                    newly_successful = success_update & ~success
                    successful = successful | newly_successful
                    success = success_update

            # Compute the new loss after the update.
            loss_current = loss_fn(x_adv)

            # Log progress if requested
            if self.verbose and (t + 1) % 10 == 0:
                if success_fn is not None:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

            # Update the search direction.
            if (t + 1) % self.restart_interval == 0:
                # Restart: use the negative gradient as the new search direction.
                d = -grad
            else:
                # Compute beta for the conjugate update.
                beta = self._compute_beta(grad, grad_old, d)
                # New direction: combine the steepest descent and the previous direction.
                d = -grad + beta.view(-1, 1, 1, 1) * d

        # Compile metrics.
        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "gradient_calls": gradient_calls,
            "time": total_time,
            "success_rate": (
                success.float().mean().item()
                * 100  # Track overall success rate, not just new successes
                if success_fn is not None
                else 0.0
            ),
            "initial_success_rate": (
                initial_success.float().mean().item() * 100
                if success_fn is not None
                else 0.0
            ),
        }

        return x_adv, metrics
