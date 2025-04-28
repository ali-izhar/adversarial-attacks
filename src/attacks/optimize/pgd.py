#!/usr/bin/env python

"""Projected Gradient Descent (PGD) optimization method."""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable


class PGDOptimizer:
    """PGD is a first-order method that iteratively updates the adversarial examples
    by taking a step in the gradient direction and then projecting the perturbed
    inputs back onto a feasible set defined by a norm constraint."""

    def __init__(
        self,
        norm: str = "L2",  # Norm constraint type as in paper (L2 or Linf)
        eps: float = 0.5,  # Perturbation budget ε as defined in the paper
        n_iterations: int = 100,  # Maximum iterations T in the PGD algorithm
        step_size: float = 0.1,  # Step size α for gradient updates
        rand_init: bool = True,  # Whether to use random initialization (δ₀ ~ U(-0.01,0.01)ⁿ)
        early_stopping: bool = True,  # Implements early stopping from paper's algorithm
        verbose: bool = False,
        maximize: bool = True,  # Whether to maximize loss (untargeted) or minimize (targeted)
    ):
        """
        Initialize the PGD optimizer with minimal parameters.

        Args:
            norm: Norm for the perturbation constraint ('L2' or 'Linf')
            eps: Maximum allowed perturbation magnitude
            n_iterations: Maximum number of iterations to run
            step_size: Step size for gradient updates
            rand_init: Whether to initialize with random noise
            early_stopping: Whether to stop when adversarial examples are found
            verbose: Whether to print progress information
            maximize: Whether to maximize or minimize the loss function
        """
        self.norm = norm.lower()
        self.eps = eps
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.rand_init = rand_init
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.maximize = maximize

        # Ensure step size is not larger than epsilon for Linf attacks
        if self.norm == "linf" and self.step_size > self.eps:
            print(
                f"WARNING: step size ({self.step_size}) is larger than epsilon ({self.eps}); reducing to epsilon/10"
            )
            self.step_size = self.eps / 10.0

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
        min_bound: Optional[torch.Tensor] = None,
        max_bound: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run PGD optimization to generate adversarial examples.

        Args:
            x_init: Initial input tensor to perturb
            gradient_fn: Function to compute gradients of the loss w.r.t input
            success_fn: Optional function to determine if adversarial criteria are met
            x_original: Original input tensor (for projection constraints)
            min_bound: Optional minimum bound for clamping (for normalized inputs)
            max_bound: Optional maximum bound for clamping (for normalized inputs)

        Returns:
            Tuple of (adversarial examples, metrics dictionary)
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Perform basic sanity checks
        if self.eps < 0:
            raise ValueError(f"eps must be non-negative: {self.eps}")
        if self.step_size < 0:
            raise ValueError(f"step_size must be non-negative: {self.step_size}")
        if self.norm not in ["l2", "linf"]:
            raise ValueError(f"Unsupported norm: {self.norm}. Use 'L2' or 'Linf'")

        # Use original images as reference if provided, otherwise use initial images
        if x_original is None:
            x_original = x_init

        # Use default clamping bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

        # Initialize perturbation - this corresponds to the initialization step in the algorithm
        # "Initialize δ₀ ~ U(-0.01,0.01)ⁿ" (random initialization within small range)
        if self.rand_init:
            if self.norm == "linf":
                # Uniform initialization in epsilon ball for Linf
                eta = torch.zeros_like(x_init).uniform_(-self.eps, self.eps)
            else:  # L2 norm
                # Random direction with magnitude eps for L2
                eta = torch.randn_like(x_init)
                eta_flat = eta.view(eta.shape[0], -1)
                eta_norm = torch.norm(eta_flat, p=2, dim=1).view(-1, 1, 1, 1)
                eta = eta * self.eps / (eta_norm + 1e-12)
        else:
            eta = torch.zeros_like(x_init)

        # Apply initial perturbation and clip to valid image range
        # This ensures x + δ ∈ [min_bound, max_bound] for normalized inputs
        x_adv = torch.clamp(x_original + eta, min=min_bound, max=max_bound)

        # Initialize metrics
        start_time = time.time()
        iterations = 0

        # Initialize success tracking
        if success_fn is not None:
            success = success_fn(x_adv)
            initial_success = success.clone()
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            initial_success = success.clone()

        # Main optimization loop - the "for t = 0 to T-1" loop in the algorithm
        for i in range(self.n_iterations):
            iterations += 1

            # Early stopping if all examples are successful
            # This implements the paper's early stopping condition:
            # "If arg max_j f_j(x_{t+1}) ≠ y_{true} and ||δ_{t+1} - δ_t||_p < 0.01 ||δ_t||_p"
            if self.early_stopping and success_fn is not None and success.all():
                if self.verbose:
                    print(f"Early stopping at iteration {i}: all examples successful")
                break

            # Track non-successful examples for selective updating
            if self.early_stopping and success_fn is not None:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break  # All examples are successful
            else:
                working_examples = torch.ones(
                    batch_size, dtype=torch.bool, device=device
                )

            # Compute gradient - corresponds to "g_t ← ∇_δ L(f(x + δ_t), y)" in the algorithm
            grad = gradient_fn(x_adv)

            # Update only working examples if early stopping is enabled
            if self.early_stopping and not working_examples.all():
                grad = grad * working_examples.view(-1, 1, 1, 1).float()

            # Normalize gradient for Linf or L2 attacks
            # This implements the normalization step in the algorithm:
            # "d_t ← sign(g_t) for ℓ_∞ or d_t ← g_t/||g_t||_2 for ℓ_2"
            if self.norm == "linf":
                # Sign gradient for Linf norm - implements Eq. (3) for Linf update
                grad_step = torch.sign(grad)
            else:  # L2 norm
                # Normalize gradient by L2 norm - implements Eq. (5) for L2 update
                grad_flat = grad.view(grad.shape[0], -1)
                grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1, 1, 1, 1)
                grad_step = grad / (grad_norm + 1e-12)

            # Update in the appropriate direction
            # This implements the gradient step in the algorithm:
            # "δ_{t+1} ← δ_t + α · d_t" for untargeted attacks (maximize=True)
            # "δ_{t+1} ← δ_t - α · d_t" for targeted attacks (maximize=False)
            if self.maximize:
                x_adv = x_adv + self.step_size * grad_step
            else:
                x_adv = x_adv - self.step_size * grad_step

            # Project perturbation onto epsilon ball
            # This implements the projection step in the algorithm:
            # "δ_{t+1} ← Π_{||·||_p ≤ ε}(δ_t + α · d_t)"
            delta = x_adv - x_original
            if self.norm == "linf":
                # Element-wise clipping for Linf - implements Eq. (4) Π_{||·||_∞ ≤ ε}
                delta = torch.clamp(delta, -self.eps, self.eps)
            else:  # L2 norm
                # Project onto L2 ball - implements Eq. (6) Π_{||·||_2 ≤ ε}
                delta_flat = delta.view(delta.shape[0], -1)
                delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1, 1, 1, 1)
                factor = torch.min(
                    torch.ones_like(delta_norm), self.eps / (delta_norm + 1e-12)
                )
                delta = delta * factor

            # Apply projected perturbation and clip to valid image range
            # This implements the final step in the algorithm:
            # "x_{t+1} ← Π_{[min_bound,max_bound]}(x + δ_{t+1})" for clamping to valid range
            x_adv = torch.clamp(x_original + delta, min=min_bound, max=max_bound)

            # Check for success - part of the early stopping condition
            if success_fn is not None:
                new_success = success_fn(x_adv)
                if self.early_stopping:
                    success = success | new_success
                else:
                    success = new_success

                if self.verbose and (i + 1) % 10 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {i+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

        # Calculate metrics
        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "time": total_time,
            "time_per_sample": total_time / batch_size,
        }

        if success_fn is not None:
            metrics["success_rate"] = success.float().mean().item() * 100
            metrics["initial_success_rate"] = (
                initial_success.float().mean().item() * 100
            )

        return x_adv, metrics
