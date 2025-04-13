"""Simplified Limited-memory BFGS (L-BFGS) optimization method.

This file implements an efficient L-BFGS optimizer for generating adversarial examples
against neural networks. L-BFGS is a quasi-Newton method that approximates the
inverse Hessian matrix using limited memory, making it suitable for adversarial attacks.

Key features:
- Efficient implementation using scipy's L-BFGS-B optimizer
- Binary search to find optimal trade-off between perturbation size and attack success
- Support for L2 and Linf perturbation norms
- Early stopping capability
- Tracking of best adversarial examples found

Expected inputs:
- Initial images to perturb
- Gradient function that computes gradients of the adversarial loss
- Loss function that returns per-example losses
- Success function that determines if adversarial criteria are met
- Original images (for projection constraints)

Expected outputs:
- Optimized adversarial examples
- Basic metrics (iterations, gradient calls, time, success rate)
"""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable


class LBFGSOptimizer:
    """
    Simplified Limited-memory BFGS (L-BFGS) optimization method.

    This optimizer uses scipy's L-BFGS-B implementation to efficiently find
    adversarial examples with minimal perturbation.
    """

    def __init__(
        self,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 50,
        history_size: int = 10,
        line_search_fn: str = "strong_wolfe",
        max_line_search: int = 20,
        initial_step: float = 1.0,
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        initial_const: float = 1e-2,
        binary_search_steps: int = 5,
        const_factor: float = 10.0,  # Factor to multiply constant by when no solution is found
        repeat_search: bool = False,  # Whether to repeat search with upper bound on last step
        verbose: bool = False,
    ):
        """
        Initialize the L-BFGS optimizer with parameters.

        Args:
            norm: Norm for the perturbation constraint ('L2' or 'Linf')
            eps: Maximum allowed perturbation magnitude
            n_iterations: Maximum number of L-BFGS iterations
            history_size: Size of the history used for Hessian approximation
            line_search_fn: Line search method ('strong_wolfe' or 'armijo')
            max_line_search: Maximum line search iterations
            initial_step: Initial step size for line search
            rand_init: Whether to initialize with random noise
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop when adversarial examples are found
            initial_const: Initial constant for balancing perturbation and loss
            binary_search_steps: Number of binary search steps to find optimal constant
            const_factor: Factor to multiply constant by when no solution is found
            repeat_search: Whether to repeat search with upper bound on last step
            verbose: Whether to print progress information
        """
        self.norm = norm.lower()
        self.eps = eps
        self.n_iterations = n_iterations
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.max_line_search = max_line_search
        self.initial_step = initial_step
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.const_factor = const_factor
        self.repeat_search = repeat_search
        self.verbose = verbose

    def _project_perturbation(
        self,
        x_adv: torch.Tensor,
        x_original: torch.Tensor,
        min_bound=None,
        max_bound=None,
    ) -> torch.Tensor:
        """
        Project perturbation to the epsilon ball using the appropriate norm.

        Args:
            x_adv: Current adversarial examples
            x_original: Original unperturbed inputs
            min_bound: Minimum valid pixel values for normalized images
            max_bound: Maximum valid pixel values for normalized images

        Returns:
            Projected adversarial examples
        """
        # Use default bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

        # Compute perturbation
        delta = x_adv - x_original

        if self.norm == "l2":
            # L2 projection
            delta_flat = delta.view(delta.size(0), -1)
            delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1, 1, 1, 1)

            # Project onto epsilon ball
            factor = torch.min(
                torch.ones_like(delta_norm, device=delta.device),
                self.eps / (delta_norm + 1e-12),
            )
            delta = delta * factor

        elif self.norm == "linf":
            # Linf projection is element-wise clipping
            delta = torch.clamp(delta, -self.eps, self.eps)

        # Apply projected perturbation to original input
        projected = x_original + delta

        # Ensure it's in valid range
        projected = torch.clamp(projected, min=min_bound, max=max_bound)

        return projected

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
        min_bound: Optional[torch.Tensor] = None,
        max_bound: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run L-BFGS optimization using scipy's implementation.

        Args:
            x_init: Initial input tensor to perturb
            gradient_fn: Function to compute gradients of the loss w.r.t input
            loss_fn: Function to compute per-example loss values
            success_fn: Function to determine if adversarial criteria are met
            x_original: Original input tensor (for projection constraints)
            min_bound: Minimum valid pixel values for normalized images
            max_bound: Maximum valid pixel values for normalized images

        Returns:
            Tuple of (adversarial examples, metrics dictionary)
        """
        try:
            from scipy.optimize import fmin_l_bfgs_b
        except ImportError:
            raise ImportError(
                "scipy is required for L-BFGS optimization. Install it using 'pip install scipy'."
            )

        device = x_init.device
        batch_size = x_init.shape[0]

        # Use default bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

        # Ensure bounds are properly shaped for broadcasting
        if isinstance(min_bound, torch.Tensor) and min_bound.dim() == 0:
            min_bound = min_bound.view(1, 1, 1, 1)
        if isinstance(max_bound, torch.Tensor) and max_bound.dim() == 0:
            max_bound = max_bound.view(1, 1, 1, 1)

        # Use original images as reference if provided, otherwise use initial images
        if x_original is None:
            x_original = x_init

        # Initialize with random noise if specified
        if self.rand_init:
            if self.norm == "linf":
                # Uniform initialization in epsilon ball
                eta = torch.zeros_like(x_init).uniform_(-self.eps, self.eps)
            else:  # L2 norm
                # Random direction with magnitude eps
                eta = torch.randn_like(x_init) * self.init_std
                eta_flat = eta.view(eta.shape[0], -1)
                eta_norm = torch.norm(eta_flat, p=2, dim=1).view(-1, 1, 1, 1)
                eta = eta * self.eps / (eta_norm + 1e-12)

            # Apply initial perturbation and clip to valid range
            x_init = torch.clamp(x_original + eta, min=min_bound, max=max_bound)

        # Initialize metrics
        start_time = time.time()
        gradient_calls = 0

        # Track best adversarial examples found
        best_adv = x_init.clone()
        best_l2 = torch.ones(batch_size, device=device) * 1e10

        # Binary search for optimal constants
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device) * 1e10
        const = torch.ones(batch_size, device=device) * self.initial_const

        # Function to check success
        if success_fn is None:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)
            initial_success = success.clone()
        else:
            success = success_fn(x_init)
            initial_success = success.clone()

        # Enable repeat search flag based on binary search steps
        repeat = self.repeat_search or self.binary_search_steps >= 10

        # Convert min_bound and max_bound to numpy for scipy LBFGS-B
        if isinstance(min_bound, torch.Tensor):
            min_bound_np = min_bound.cpu().numpy()
        else:
            min_bound_np = min_bound

        if isinstance(max_bound, torch.Tensor):
            max_bound_np = max_bound.cpu().numpy()
        else:
            max_bound_np = max_bound

        # Binary search to find optimal constant
        for binary_step in range(self.binary_search_steps):
            if self.verbose:
                print(f"Binary search step {binary_step+1}/{self.binary_search_steps}")

            # Repeat search with upper bound on last step (like in cleverhans)
            if repeat and binary_step == self.binary_search_steps - 1:
                const = upper_bound.clone()
                if self.verbose:
                    print(f"  Final search step: using upper bound constants")

            # Define the L-BFGS objective function with combined loss
            def lbfgs_objective(x_flat):
                # Reshape flattened input to tensor shape
                x_tensor = torch.tensor(
                    x_flat.reshape(x_init.shape), device=device, dtype=torch.float32
                )

                # Apply constraints to ensure valid range
                x_tensor = torch.clamp(x_tensor, min=min_bound, max=max_bound)

                # Project perturbation if needed
                if self.norm in ["l2", "linf"]:
                    x_tensor = self._project_perturbation(
                        x_tensor, x_original, min_bound=min_bound, max_bound=max_bound
                    )

                # Calculate adversarial loss (per sample)
                adv_loss = loss_fn(x_tensor)

                # Calculate L2 distance penalty
                delta = x_tensor - x_original
                delta_flat = delta.view(batch_size, -1)
                l2_dist = torch.sum(delta_flat**2, dim=1)

                # Current constants as tensor
                const_tensor = const.clone()

                # Combined loss: adversarial loss + constant * L2 penalty
                combined_loss = adv_loss + const_tensor * l2_dist
                loss_value = combined_loss.mean().item()

                # Compute gradient for scipy
                x_tensor.requires_grad_(True)
                combined_loss.mean().backward()
                grad = x_tensor.grad.cpu().numpy().flatten().astype(np.float64)

                # Track gradient calls
                nonlocal gradient_calls
                gradient_calls += 1

                return loss_value, grad

            # Call scipy's L-BFGS implementation
            x_np = x_init.cpu().detach().numpy()

            # Create proper bounds for each element based on min_bound and max_bound
            # This ensures we respect the normalized image space
            if isinstance(min_bound_np, np.ndarray) and min_bound_np.size > 1:
                clip_min = np.broadcast_to(min_bound_np, x_np.shape).flatten()
            else:
                clip_min = np.ones_like(x_np.flatten()) * min_bound_np

            if isinstance(max_bound_np, np.ndarray) and max_bound_np.size > 1:
                clip_max = np.broadcast_to(max_bound_np, x_np.shape).flatten()
            else:
                clip_max = np.ones_like(x_np.flatten()) * max_bound_np

            bounds = list(zip(clip_min, clip_max))

            try:
                adv_x_flat, f, d = fmin_l_bfgs_b(
                    lbfgs_objective,
                    x_np.flatten().astype(np.float64),
                    bounds=bounds,
                    maxiter=self.n_iterations,
                    maxfun=self.n_iterations * 2,  # Maximum function evaluations
                    m=self.history_size,
                    factr=1e7,  # Tolerance: higher is less accurate but faster
                    iprint=1 if self.verbose else -1,
                )

                # Convert back to torch tensor
                adv_x = torch.tensor(adv_x_flat.reshape(x_init.shape), device=device)

                # Ensure it's in valid range and properly projected
                adv_x = torch.clamp(adv_x, min=min_bound, max=max_bound)
                if self.norm in ["l2", "linf"]:
                    adv_x = self._project_perturbation(
                        adv_x, x_original, min_bound=min_bound, max_bound=max_bound
                    )

                # Check for success
                if success_fn is not None:
                    current_success = success_fn(adv_x)
                else:
                    current_success = torch.zeros_like(success)

                # Calculate L2 distance
                delta = adv_x - x_original
                delta_flat = delta.view(batch_size, -1)
                current_l2 = torch.sum(delta_flat**2, dim=1)

                # Update best results where successful
                for i in range(batch_size):
                    if current_success[i] and current_l2[i] < best_l2[i]:
                        best_adv[i] = adv_x[i]
                        best_l2[i] = current_l2[i]

                # Update constants using binary search
                for i in range(batch_size):
                    if current_success[i]:
                        # Success - reduce constant to find smaller perturbation
                        upper_bound[i] = min(upper_bound[i], const[i])
                        if upper_bound[i] < 1e9:
                            const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        # Failure - increase constant to prioritize adversarial success
                        lower_bound[i] = max(lower_bound[i], const[i])
                        if upper_bound[i] < 1e9:
                            const[i] = (lower_bound[i] + upper_bound[i]) / 2
                        else:
                            # No solution found yet, increase more aggressively
                            const[i] *= self.const_factor

                # Track overall success
                success = success | current_success

                if self.verbose:
                    print(
                        f"  Current success rate: {current_success.float().mean().item()*100:.2f}%"
                    )
                    print(f"  Current mean L2 distance: {current_l2.mean().item():.6f}")

                    # Calculate mean of successful distortions (like in cleverhans)
                    successful_l2 = current_l2[current_success]
                    if len(successful_l2) > 0:
                        mean_successful_l2 = successful_l2.mean().item()
                        print(
                            f"  Mean successful L2 distance: {mean_successful_l2:.6f}"
                        )

                # Early stopping if all examples are successful
                if self.early_stopping and success.all():
                    if self.verbose:
                        print(
                            f"Early stopping at binary step {binary_step+1}: all examples successful"
                        )
                    break

            except Exception as e:
                if self.verbose:
                    print(f"L-BFGS optimization failed: {str(e)}")
                # Continue to next binary search step

        # If we didn't find any successful adversarial examples,
        # just use the last iteration as output but make some perturbation
        # This ensures metrics can be calculated even with 0% success rate
        if not success.any():
            # Apply a small perturbation to ensure some metrics can be calculated
            noise = torch.zeros_like(x_original).uniform_(-self.eps / 10, self.eps / 10)
            best_adv = torch.clamp(x_original + noise, min=min_bound, max=max_bound)
            if self.norm in ["l2", "linf"]:
                best_adv = self._project_perturbation(
                    best_adv, x_original, min_bound=min_bound, max_bound=max_bound
                )

        # Calculate mean L2 norm of successful examples
        mean_successful_l2 = 0.0
        if success.any():
            mean_successful_l2 = best_l2[success].mean().item()

        # Calculate metrics
        total_time = time.time() - start_time
        metrics = {
            "iterations": self.n_iterations * (binary_step + 1),
            "gradient_calls": gradient_calls,
            "time": total_time,
            "time_per_sample": total_time / batch_size,
            "mean_successful_l2": mean_successful_l2,
        }

        if success_fn is not None:
            metrics["success_rate"] = success.float().mean().item() * 100
            metrics["initial_success_rate"] = (
                initial_success.float().mean().item() * 100
            )

        return best_adv, metrics
