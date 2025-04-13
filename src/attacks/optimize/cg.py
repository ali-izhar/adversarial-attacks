"""Conjugate Gradient optimization method."""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable

from .projections import project_adversarial_example


class ConjugateGradientOptimizer:
    """Implementation of nonlinear conjugate gradient descent."""

    def __init__(
        self,
        norm: str = "L2",
        eps: float = 5.0,  # perturbation budget ε constraint
        n_iterations: int = 50,  # iterations T parameter
        beta_method: str = "HS",  # Options: "FR", "PR", "HS" (Hestenes-Stiefel)
        restart_interval: int = 10,
        line_search_factor: float = 0.5,
        sufficient_decrease: float = 1e-4,
        line_search_max_iter: int = 10,
        rand_init: bool = True,  # δ₀ ← U(-0.01, 0.01)ⁿ random initialization
        init_std: float = 0.1,
        early_stopping: bool = True,  # early stopping if successful attack
        verbose: bool = False,
        adaptive_restart: bool = True,
        momentum: float = 0.0,  # Should be 0 for pure CG
        fgsm_init: bool = True,
        multi_stage: bool = False,
        auto_tune_eps: bool = False,
    ):
        """
        Initialize the optimizer with parameters that control:
        - The type of norm constraint (L2 or Linf)
        - Maximum allowed perturbation size (eps)
        - Maximum number of iterations and other algorithmic details
        """
        # ||δ||_p ≤ ε constraint
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations  # iterations T parameter
        # β formulas - FR, PR, or HS
        self.beta_method = beta_method
        self.restart_interval = restart_interval
        self.line_search_factor = line_search_factor
        self.sufficient_decrease = sufficient_decrease
        self.line_search_max_iter = line_search_max_iter
        # δ₀ ← U(-0.01, 0.01)ⁿ random initialization
        self.rand_init = rand_init
        self.init_std = init_std
        # break if ||r_{k+1}|| < tol or arg max_j f_j(x + δ_{k+1}) ≠ y_true
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.momentum = momentum  # Pure CG uses no momentum

        # Additional parameters
        self.adaptive_restart = adaptive_restart
        self.fgsm_init = fgsm_init
        self.multi_stage = multi_stage
        self.auto_tune_eps = auto_tune_eps

        # Tracking variables
        self._last_iter_decrease = 0
        self._avg_progress_rate = 0

        # Conjugacy threshold - used to detect when to restart
        self.conjugacy_threshold = 0.2

    def _compute_beta(
        self,
        grad_new: torch.Tensor,
        grad_old: torch.Tensor,
        d_old: torch.Tensor,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute conjugate gradient beta parameter using various formulas.

        # Paper directly implements these three β formulas:
        # β_t^FR = ||g_{t+1}||²/||g_t||² (Fletcher-Reeves)
        # β_t^PR = g_{t+1}ᵀ(g_{t+1} - g_t)/(g_tᵀg_t) (Polak-Ribière)
        # β_t^HS = g_{t+1}ᵀ(g_{t+1} - g_t)/(d_tᵀ(g_{t+1} - g_t)) (Hestenes-Stiefel)

        Args:
            grad_new: New gradient at current point
            grad_old: Previous gradient
            d_old: Previous search direction
            y: Gradient difference (grad_new - grad_old), precomputed for efficiency

        Returns:
            beta: Per-example tensor for conjugate update
        """
        batch_size = grad_new.shape[0]
        grad_new_flat = grad_new.view(batch_size, -1)
        grad_old_flat = grad_old.view(batch_size, -1)
        d_old_flat = d_old.view(batch_size, -1)

        # Compute y = grad_new - grad_old if not provided
        if y is None:
            y = grad_new_flat - grad_old_flat

        # Initialize beta tensor
        beta = torch.zeros(batch_size, device=grad_new.device)

        if self.beta_method == "FR":  # Fletcher-Reeves
            # β_t^FR = ||g_{t+1}||²/||g_t||²
            grad_new_sq = (grad_new_flat**2).sum(dim=1)
            grad_old_sq = (grad_old_flat**2).sum(dim=1)

            # Avoid division by zero
            safe_mask = grad_old_sq > 1e-10
            beta[safe_mask] = grad_new_sq[safe_mask] / grad_old_sq[safe_mask]

        elif self.beta_method == "PR":  # Polak-Ribière
            # β_t^PR = g_{t+1}ᵀ(g_{t+1} - g_t)/(g_tᵀg_t)
            grad_old_sq = (grad_old_flat**2).sum(dim=1)
            numerator = (grad_new_flat * y).sum(dim=1)

            # Avoid division by zero
            safe_mask = grad_old_sq > 1e-10
            beta[safe_mask] = numerator[safe_mask] / grad_old_sq[safe_mask]

            # Polak-Ribière+ variant: ensure non-negative beta
            beta = torch.clamp(beta, min=0)

        elif self.beta_method == "HS":  # Hestenes-Stiefel
            # β_t^HS = g_{t+1}ᵀ(g_{t+1} - g_t)/(d_tᵀ(g_{t+1} - g_t))
            # First compute denominator: d_k · y
            denominator = (d_old_flat * y).sum(dim=1)
            numerator = (grad_new_flat * y).sum(dim=1)

            # Avoid division by zero with a threshold
            safe_mask = torch.abs(denominator) > 1e-10
            beta[safe_mask] = numerator[safe_mask] / denominator[safe_mask]

        else:
            raise ValueError(f"Unknown beta method: {self.beta_method}")

        # Handle cases where beta calculation failed with steepest descent
        beta[torch.isnan(beta)] = 0.0

        # Safeguard against very large beta values for stability
        # The Powell restart condition: if beta becomes too large or direction not descent
        beta = torch.clamp(beta, min=0.0, max=1.0)

        return beta

    def _check_conjugacy_loss(
        self, grad_new: torch.Tensor, d_old: torch.Tensor, y: torch.Tensor = None
    ) -> torch.BoolTensor:
        """
        Check if conjugacy is lost and restart is needed.

        # Paper mentions: "periodic restart every restart_interval iterations"
        # This is an enhancement of the paper's approach with more sophisticated restart criteria

        Conditions for restart (any of):
        1. Direction and gradient nearly parallel (> 0.2 cosine similarity)
        2. Powell's restart criterion: βₖ ≥ 0 and gₖᵀgₖ₋₁ / ||gₖ₋₁||² ≥ 0.2

        Args:
            grad_new: Current gradient
            d_old: Previous search direction
            y: Gradient difference (optional)

        Returns:
            loss_conjugacy: Boolean tensor, True if conjugacy is lost
        """
        batch_size = grad_new.shape[0]
        grad_flat = grad_new.view(batch_size, -1)
        d_flat = d_old.view(batch_size, -1)

        # Compute normalized vectors for cosine similarity
        grad_norm = torch.norm(grad_flat, dim=1, keepdim=True)
        d_norm = torch.norm(d_flat, dim=1, keepdim=True)

        # Avoid division by zero
        safe_grad_norm = torch.where(
            grad_norm > 1e-10, grad_norm, torch.ones_like(grad_norm)
        )
        safe_d_norm = torch.where(d_norm > 1e-10, d_norm, torch.ones_like(d_norm))

        # Compute absolute cosine similarity - near 1 means vectors are nearly parallel
        cos_sim = torch.abs(
            (grad_flat * d_flat).sum(dim=1) / (safe_grad_norm * safe_d_norm).squeeze()
        )

        # First restart condition: if direction and gradient are nearly parallel
        condition1 = cos_sim > self.conjugacy_threshold

        # Second restart condition (Powell): if beta would be large, restart
        # We don't actually compute beta here for efficiency
        if y is not None:
            grad_prod = (grad_flat * y).sum(dim=1)
            y_norm_sq = (y**2).sum(dim=1)
            powell_metric = torch.abs(grad_prod) / (
                torch.sqrt(y_norm_sq) * safe_grad_norm.squeeze()
            )
            condition2 = powell_metric > 0.8
        else:
            condition2 = torch.zeros_like(condition1, dtype=torch.bool)

        # Restart if either condition is met
        return condition1 | condition2

    def _line_search(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        current_grad: torch.Tensor,
        current_loss: torch.Tensor,
        x_original: Optional[torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        eps: torch.Tensor,
        min_bound: Optional[torch.Tensor] = None,
        max_bound: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Improved backtracking line search satisfying Armijo condition.

        # Paper uses a simple optimal step size: α_k = r_k^T r_k / (p_k^T A p_k)
        # This is an enhancement that uses a more robust line search approach
        # for the nonlinear case where the simple formula may not be optimal

        Args:
            x: Current point
            direction: Search direction
            current_grad: Gradient at current point
            current_loss: Loss at current point
            x_original: Original images for projection
            loss_fn: Function to evaluate loss
            eps: Per-sample epsilon constraint
            min_bound: Minimum valid pixel values (for normalized images)
            max_bound: Maximum valid pixel values (for normalized images)

        Returns:
            Tuple of (step sizes, direction to use)
        """
        batch_size = x.shape[0]

        # Use default bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

        # Reshape for more efficient computation
        direction_flat = direction.view(batch_size, -1)
        current_grad_flat = current_grad.view(batch_size, -1)

        # Compute directional derivative ∇f(x)ᵀd
        dir_deriv = (current_grad_flat * direction_flat).sum(dim=1)

        # If direction is not a descent direction (gradient and direction form obtuse angle),
        # we should use steepest descent instead
        non_descent = dir_deriv >= 0

        direction_modified = direction.clone()
        if non_descent.any():
            direction_modified[non_descent] = -current_grad[non_descent]
            # Recompute directional derivative for corrected directions
            dir_deriv_fixed = -(current_grad_flat[non_descent] ** 2).sum(dim=1)
            dir_deriv[non_descent] = dir_deriv_fixed

        # Choose appropriate initial step size based on norm type and epsilon
        if self.norm.lower() == "l2":
            # For L2, use a smaller fraction of epsilon
            initial_step = eps.squeeze() * 0.2  # 20% of epsilon
        else:
            # For Linf, we can be more aggressive
            initial_step = eps.squeeze() * 0.5  # 50% of epsilon

        # Ensure positive step size and make sure it's a tensor with batch dimension
        initial_step = torch.maximum(initial_step, torch.ones_like(initial_step) * 0.01)

        # Ensure initial_step has a batch dimension
        if initial_step.dim() == 0:
            initial_step = initial_step.unsqueeze(0).expand(batch_size)

        # Create batch-specific step sizes
        alpha = initial_step.clone()

        # Armijo condition threshold: f(x) + c*α*∇f(x)ᵀd where c is the sufficient decrease param
        armijo_threshold = current_loss + self.sufficient_decrease * alpha * dir_deriv

        # Track which samples have made progress
        made_progress = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Line search loop
        for i in range(self.line_search_max_iter):
            # Apply step and project
            x_new = x + alpha.view(-1, 1, 1, 1) * direction_modified

            # Project to constraint set if needed
            if x_original is not None:
                # Project each sample using its specific epsilon
                for j in range(batch_size):
                    # δ_{k+1} ← Π_{||·||_p ≤ ε}(δ_k + α_k p_k) projection step
                    x_new[j : j + 1] = project_adversarial_example(
                        x_new[j : j + 1],
                        x_original[j : j + 1],
                        eps[j].item(),
                        self.norm,
                        min_bound=min_bound,
                        max_bound=max_bound,
                    )

            # Evaluate loss at new point
            new_loss = loss_fn(x_new)

            # Check Armijo condition
            armijo_success = new_loss <= armijo_threshold

            # Update progress tracking
            made_progress = made_progress | armijo_success

            # If all examples satisfy the condition, exit
            if armijo_success.all():
                break

            # For failed examples, reduce step size
            reduction_factor = self.line_search_factor * (
                0.95**i
            )  # Gradually more aggressive

            # Handle the case where batch_size is 1 and armijo_success is a 0-dim tensor
            if batch_size == 1:
                if not armijo_success.item():
                    alpha = alpha * reduction_factor
                    armijo_threshold = (
                        current_loss + self.sufficient_decrease * alpha * dir_deriv
                    )
            else:
                alpha[~armijo_success] *= reduction_factor
                # Update Armijo thresholds for failed samples
                armijo_threshold[~armijo_success] = current_loss[~armijo_success] + (
                    self.sufficient_decrease
                    * alpha[~armijo_success]
                    * dir_deriv[~armijo_success]
                )

            # Exit early if step sizes become too small
            if (alpha < 1e-4 * initial_step).all():
                break

        # For examples where line search still failed, use a small fixed step with gradient
        if not made_progress.all() and x_original is not None:
            failed = ~made_progress

            # Handle the case where batch_size is 1
            if batch_size == 1:
                if failed.item():
                    alpha = torch.ones_like(alpha) * 0.01
                    direction_final = -current_grad / (
                        torch.norm(
                            current_grad.reshape(1, -1), dim=1, keepdim=True
                        ).view(-1, 1, 1, 1)
                        + 1e-10
                    )
                else:
                    direction_final = direction_modified
            else:
                # Small fixed step for failed samples
                alpha[failed] = torch.ones_like(alpha[failed]) * 0.01
                # Use negative gradient (steepest descent) for failed samples
                direction_final = torch.where(
                    failed.view(-1, 1, 1, 1).expand_as(direction),
                    -current_grad
                    / (
                        torch.norm(
                            current_grad.reshape(batch_size, -1), dim=1, keepdim=True
                        ).view(-1, 1, 1, 1)
                        + 1e-10
                    ),
                    direction_modified,
                )
            return alpha, (
                direction_final if "direction_final" in locals() else direction_modified
            )

        # Return successful step sizes and (potentially) modified directions
        return alpha, direction_modified

    def fgsm_initialization(
        self,
        x_original: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        targeted: bool = False,
        alpha: float = 0.7,  # More aggressive initialization (70% of epsilon)
        eps: torch.Tensor = None,
        min_bound: Optional[torch.Tensor] = None,
        max_bound: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize adversarial examples using Fast Gradient Sign Method.

        # This is an enhancement beyond the paper's random initialization
        # to provide a better starting point closer to the decision boundary

        Args:
            x_original: Original clean images
            gradient_fn: Function to compute gradient
            targeted: Whether this is a targeted attack
            alpha: Fraction of epsilon to use for initialization
            eps: Per-sample epsilon values
            min_bound: Minimum valid pixel values (for normalized images)
            max_bound: Maximum valid pixel values (for normalized images)

        Returns:
            Initialized adversarial examples
        """
        batch_size = x_original.shape[0]

        # Compute initial gradient with gradient tracking
        x_copy = x_original.clone().detach().requires_grad_(True)
        grad = gradient_fn(x_copy)

        # For targeted attacks, gradient direction is reversed
        if targeted:
            grad = -grad

        # Initialize perturbation tensor
        delta = torch.zeros_like(x_original)

        # Apply FGSM perturbation with per-sample epsilon
        for i in range(batch_size):
            # Get epsilon for this sample
            sample_eps = eps[i].item() if eps is not None else self.eps

            if self.norm.lower() == "l2":
                # For L2, normalize gradient to unit norm and scale
                grad_flat = grad[i : i + 1].reshape(1, -1)
                grad_norm = torch.norm(grad_flat, p=2) + 1e-10
                normalized_grad = grad[i : i + 1] / grad_norm

                # Scale by alpha * epsilon
                delta[i : i + 1] = normalized_grad * (sample_eps * alpha)
            else:  # Linf norm
                # For Linf, use sign and scale
                delta[i : i + 1] = torch.sign(grad[i : i + 1]) * (sample_eps * alpha)

        # Apply perturbation
        x_adv = x_original + delta

        # Project each sample individually to ensure constraints
        for i in range(batch_size):
            sample_eps = eps[i].item() if eps is not None else self.eps
            x_adv[i : i + 1] = project_adversarial_example(
                x_adv[i : i + 1],
                x_original[i : i + 1],
                sample_eps,
                self.norm,
                min_bound=min_bound,
                max_bound=max_bound,
            )

        # Ensure result is valid image
        x_adv = torch.clamp(x_adv, min=min_bound, max=max_bound)

        return x_adv

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
        targeted: bool = False,
        min_bound: Optional[torch.Tensor] = None,
        max_bound: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run conjugate gradient optimization to generate adversarial examples.

        # This is the main function that implements Algorithm 1 from the paper:
        # "Efficient Conjugate Gradient Attack"

        Args:
            x_init: Initial images
            gradient_fn: Function to compute gradient of loss
            loss_fn: Function to compute loss
            success_fn: Function to check if adversarial criteria are met
            x_original: Original images for projection
            targeted: Whether this is a targeted attack
            min_bound: Minimum valid pixel values (for normalized images)
            max_bound: Maximum valid pixel values (for normalized images)

        Returns:
            Tuple of (adversarial examples, metrics)
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Use default bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

        # Initialize timing and metrics
        start_time = time.time()
        iterations_done = torch.zeros(batch_size, dtype=torch.long, device=device)
        gradient_calls = torch.zeros(batch_size, dtype=torch.long, device=device)
        successful = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Create per-sample epsilon tensor
        per_sample_eps = torch.ones(batch_size, device=device) * self.eps

        # Initialize adversarial examples
        if self.fgsm_init and x_original is not None:
            # Use FGSM for initialization with per-sample epsilon
            x_adv = self.fgsm_initialization(
                x_original,
                gradient_fn,
                targeted=targeted,
                alpha=0.7,  # Use 70% of epsilon for initialization
                eps=per_sample_eps,
                min_bound=min_bound,
                max_bound=max_bound,
            )
            # Count gradient calls for initialization
            gradient_calls += 1
        elif self.rand_init and x_original is not None:
            # δ₀ ← U(-0.01, 0.01)ⁿ - Random initialization
            x_adv = torch.zeros_like(x_original)
            for i in range(batch_size):
                if self.norm.lower() == "l2":
                    # Random direction with L2 normalization
                    delta = torch.randn_like(x_original[i : i + 1])
                    flat_delta = delta.reshape(1, -1)
                    l2_norm = torch.norm(flat_delta, p=2)
                    delta = delta / (l2_norm + 1e-10) * per_sample_eps[i]
                else:  # Linf
                    # Uniform random in [-eps, eps]
                    delta = torch.zeros_like(x_original[i : i + 1]).uniform_(
                        -per_sample_eps[i], per_sample_eps[i]
                    )

                # Apply perturbation and project
                x_adv[i : i + 1] = project_adversarial_example(
                    x_original[i : i + 1] + delta,
                    x_original[i : i + 1],
                    per_sample_eps[i],
                    self.norm,
                    min_bound=min_bound,
                    max_bound=max_bound,
                )
        else:
            # Start from original image
            x_adv = x_original.clone() if x_original is not None else x_init.clone()

        # Ensure inputs are on the correct device
        x_adv = x_adv.to(device)

        # Check initial success
        if success_fn is not None:
            success = success_fn(x_adv)
            initial_success = success.clone()
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            initial_success = success.clone()

        # Storage for previous iteration data
        prev_grad = None
        prev_d = None

        # Create active sample mask - samples we're still optimizing
        active = (
            ~success
            if self.early_stopping
            else torch.ones(batch_size, dtype=torch.bool, device=device)
        )

        # Main optimization loop
        for i in range(self.n_iterations):
            # Skip if no active samples
            if not active.any():
                break

            # Increment iteration counter for active samples
            iterations_done[active] += 1

            # Current x for active samples
            x_current = x_adv[active]

            # Compute gradient for active samples
            x_current.requires_grad_(True)
            grad_all = torch.zeros_like(x_adv)

            # Get gradient only for active samples to save computation
            # g_t = ∇_δ L(f(x + δ_t), y) - computing the gradient
            grad_active = gradient_fn(x_current)
            gradient_calls[active] += 1

            # Store in full gradient tensor
            grad_all[active] = grad_active

            # Initialize search direction or compute conjugate direction
            if i == 0 or prev_grad is None:
                # First iteration: use negative gradient (steepest descent)
                # p_0 = r_0 where r_0 is -gradient
                d_all = -grad_all.clone()
                prev_grad = grad_all.clone()
                prev_d = d_all.clone()
            else:
                # For active samples, check conjugacy loss for restart
                d_all = torch.zeros_like(x_adv)

                if self.adaptive_restart:
                    # Check only active samples for restart
                    grad_active = grad_all[active]
                    prev_grad_active = prev_grad[active]
                    prev_d_active = prev_d[active]

                    # Compute gradient difference for active samples
                    y_active = grad_active.reshape(
                        grad_active.shape[0], -1
                    ) - prev_grad_active.reshape(prev_grad_active.shape[0], -1)

                    # Check for restart condition
                    to_restart = self._check_conjugacy_loss(
                        grad_active, prev_d_active, y_active
                    )

                    # Create masks for restarting and continuing samples
                    restart_mask = to_restart
                    continue_mask = ~restart_mask

                    # For samples needing restart, use steepest descent
                    if restart_mask.any():
                        d_active = torch.zeros_like(grad_active)
                        d_active[restart_mask] = -grad_active[restart_mask]
                    else:
                        d_active = torch.zeros_like(grad_active)

                    # For samples continuing CG, compute beta and update direction
                    if continue_mask.any():
                        # Compute beta only for continuing samples
                        # β_{k+1} = ||g_{t+1}||²/||g_t||² (for FR method)
                        beta = self._compute_beta(
                            grad_active[continue_mask],
                            prev_grad_active[continue_mask],
                            prev_d_active[continue_mask],
                            y_active[continue_mask],
                        )

                        # Update direction
                        # p_{k+1} = -r_{k+1} + β_{k+1} p_k
                        d_active[continue_mask] = (
                            -grad_active[continue_mask]
                            + beta.view(-1, 1, 1, 1) * prev_d_active[continue_mask]
                        )

                    # Store in full direction tensor
                    d_all[active] = d_active
                else:
                    # Simple approach without per-sample restart checks
                    # Periodic restart every restart_interval iterations or when normal iteration
                    if i % self.restart_interval == 0:
                        d_all = -grad_all
                    else:
                        # Compute beta for all active samples
                        grad_active = grad_all[active]
                        prev_grad_active = prev_grad[active]
                        prev_d_active = prev_d[active]

                        beta = self._compute_beta(
                            grad_active, prev_grad_active, prev_d_active
                        )
                        d_active = -grad_active + beta.view(-1, 1, 1, 1) * prev_d_active
                        d_all[active] = d_active

            # Update previous values for next iteration
            prev_grad = grad_all.clone()
            prev_d = d_all.clone()

            # Compute current loss (for line search)
            loss_current = loss_fn(x_adv)

            # Perform line search for active samples
            if active.sum() > 0:
                # Number of active samples
                num_active = active.sum()

                # Create a loss function wrapper that handles the active samples correctly
                def wrapped_loss_fn(x_active):
                    # Create a full batch with both inactive and active samples
                    x_full = x_adv.clone()

                    # Place active samples in their correct positions
                    x_full[active] = x_active

                    # Compute full loss
                    full_loss = loss_fn(x_full)

                    # Extract just the losses for active samples
                    # For a single active sample, we need special handling
                    if num_active == 1:
                        active_idx = active.nonzero().item()
                        return full_loss[active_idx : active_idx + 1]
                    else:
                        return full_loss[active]

                # Line search with per-sample epsilon
                # α_k = r_k^T r_k / (p_k^T A p_k) optimal step size
                # Our implementation uses a more robust line search for nonlinear case
                alpha, d_search = self._line_search(
                    x_adv[active],
                    d_all[active],
                    grad_all[active],
                    loss_current[active],
                    x_original[active] if x_original is not None else None,
                    wrapped_loss_fn,
                    (
                        per_sample_eps[active].view(-1, 1)
                        if num_active == 1
                        else per_sample_eps[active]
                    ),
                    min_bound=min_bound,
                    max_bound=max_bound,
                )

                # Apply update only to active samples
                x_adv_new = x_adv.clone()

                # Shape alpha properly based on number of active samples
                if num_active == 1:
                    # For a single active sample
                    active_idx = active.nonzero().item()
                    x_adv_new[active_idx : active_idx + 1] = (
                        x_adv[active_idx : active_idx + 1]
                        + alpha.view(-1, 1, 1, 1) * d_search
                    )
                else:
                    # For multiple active samples
                    x_adv_new[active] = (
                        x_adv[active] + alpha.view(-1, 1, 1, 1) * d_search
                    )

                # Project to valid range
                if x_original is not None:
                    for i in range(batch_size):
                        if active[i]:
                            # δ_{k+1} ← Π_{||·||_p ≤ ε}(δ_k + α_k p_k)
                            x_adv_new[i : i + 1] = project_adversarial_example(
                                x_adv_new[i : i + 1],
                                x_original[i : i + 1],
                                per_sample_eps[i],
                                self.norm,
                                min_bound=min_bound,
                                max_bound=max_bound,
                            )

                # Ensure valid pixel range
                x_adv = torch.clamp(x_adv_new, min=min_bound, max=max_bound)

            # Check success and update active samples
            if success_fn is not None:
                # See which samples are now successful
                new_success = success_fn(x_adv)
                newly_successful = new_success & ~success

                # Update overall success tracking
                successful = successful | newly_successful
                success = new_success

                # Update active samples mask for early stopping
                if self.early_stopping:
                    # break if ||r_{k+1}|| < tol or arg max_j f_j(x + δ_{k+1}) ≠ y_true
                    # This implements the early stopping if attack succeeds
                    active = ~success

            # Verbose logging
            if self.verbose and (i + 1) % 10 == 0:
                if success_fn is not None:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {i+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

        # Final projection to ensure constraints
        if x_original is not None:
            for i in range(batch_size):
                x_adv[i : i + 1] = project_adversarial_example(
                    x_adv[i : i + 1],
                    x_original[i : i + 1],
                    per_sample_eps[i],
                    self.norm,
                    min_bound=min_bound,
                    max_bound=max_bound,
                )

        # Ensure final result is within valid pixel range
        x_adv = torch.clamp(x_adv, min=min_bound, max=max_bound)

        # Compute final metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Compute average metrics, handling the case where no samples were processed
        avg_iterations = (
            iterations_done.float().mean().item() if iterations_done.size(0) > 0 else 0
        )
        avg_gradient_calls = (
            gradient_calls.float().mean().item() if gradient_calls.size(0) > 0 else 0
        )
        success_rate = success.float().mean().item() * 100 if success.size(0) > 0 else 0
        initial_success_rate = (
            initial_success.float().mean().item() * 100
            if initial_success.size(0) > 0
            else 0
        )

        metrics = {
            "iterations": avg_iterations,
            "gradient_calls": avg_gradient_calls,
            "time": total_time,
            "time_per_sample": total_time / batch_size,
            "success_rate": success_rate,
            "initial_success_rate": initial_success_rate,
        }

        return x_adv, metrics
