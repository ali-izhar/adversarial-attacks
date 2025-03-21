"""Conjugate Gradient optimization method implementation."""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable

from src.utils.projections import project_adversarial_example


class ConjugateGradientOptimizer:
    """
    Conjugate Gradient optimization method.

    CG achieves faster convergence than standard gradient descent by using
    conjugate search directions that produce more efficient traversal
    of the optimization landscape.
    """

    def __init__(
        self,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 100,
        fletcher_reeves: bool = True,
        restart_interval: int = 20,
        backtracking_factor: float = 0.7,
        sufficient_decrease: float = 1e-4,
        line_search_max_iter: int = 10,
        rand_init: bool = True,
        init_std: float = 0.01,
        early_stopping: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the Conjugate Gradient optimizer.

        Args:
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            n_iterations: Maximum number of iterations
            fletcher_reeves: Whether to use Fletcher-Reeves formula (True) or Polak-Ribière (False)
            restart_interval: Restart conjugacy every N iterations
            backtracking_factor: Factor to reduce step size in line search
            sufficient_decrease: Sufficient decrease parameter for Armijo condition
            line_search_max_iter: Maximum number of iterations for line search
            rand_init: Whether to initialize with random perturbation
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop early when objective is achieved
            verbose: Whether to print progress information
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

    def _compute_beta(
        self, grad_new: torch.Tensor, grad_old: torch.Tensor, d_old: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the beta parameter for conjugate gradient.

        Args:
            grad_new: New gradient
            grad_old: Previous gradient
            d_old: Previous search direction

        Returns:
            Beta parameter for each example in the batch
        """
        # Compute squared norm of gradients
        grad_new_sq = (grad_new.view(grad_new.shape[0], -1) ** 2).sum(dim=1)
        grad_old_sq = (grad_old.view(grad_old.shape[0], -1) ** 2).sum(dim=1)

        if self.fletcher_reeves:
            # Fletcher-Reeves formula: beta = ||grad_new||^2 / ||grad_old||^2
            beta = grad_new_sq / (grad_old_sq + 1e-10)
        else:
            # Polak-Ribière formula: beta = (grad_new·(grad_new-grad_old)) / ||grad_old||^2
            grad_diff = grad_new - grad_old
            numerator = (
                grad_new.view(grad_new.shape[0], -1)
                * grad_diff.view(grad_diff.shape[0], -1)
            ).sum(dim=1)
            beta = numerator / (grad_old_sq + 1e-10)

            # Polak-Ribière+ variant: max(0, beta)
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
        Perform backtracking line search to find a good step size.

        Args:
            x: Current point
            direction: Search direction
            current_grad: Gradient at current point
            current_loss: Loss at current point
            x_original: Original input for projection (if provided)
            loss_fn: Function to compute loss

        Returns:
            Step size
        """
        batch_size = x.shape[0]
        alpha = torch.ones(batch_size, device=x.device)

        # Calculate initial directional derivative
        dir_deriv = (
            direction.view(batch_size, -1) * current_grad.view(batch_size, -1)
        ).sum(dim=1)

        # Store initial values
        armijo_threshold = current_loss - self.sufficient_decrease * alpha * dir_deriv

        # Perform backtracking line search
        for _ in range(self.line_search_max_iter):
            # Take a step
            x_new = x + alpha.view(-1, 1, 1, 1) * direction

            # Project if necessary
            if x_original is not None:
                x_new = project_adversarial_example(
                    x_new, x_original, self.eps, self.norm
                )

            # Ensure valid image range
            x_new = torch.clamp(x_new, 0.0, 1.0)

            # Compute new loss
            new_loss = loss_fn(x_new)

            # Check Armijo condition
            armijo_condition = new_loss <= armijo_threshold

            # If all satisfy Armijo, we're done
            if armijo_condition.all():
                break

            # Otherwise, reduce step size for examples that failed
            alpha[~armijo_condition] *= self.backtracking_factor

            # Update threshold
            armijo_threshold = (
                current_loss - self.sufficient_decrease * alpha * dir_deriv
            )

        return alpha

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run Conjugate Gradient optimization.

        Args:
            x_init: Initial point
            gradient_fn: Function that computes gradient given current x
            loss_fn: Function that computes loss given current x
            success_fn: Function that returns True if optimization goal is achieved
            x_original: Original input (for projection)

        Returns:
            Tuple of (optimized_x, metrics_dict)
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Initialize point
        if self.rand_init and self.init_std > 0:
            # Random initialization
            noise = torch.randn_like(x_init) * self.init_std
            x_adv = x_init + noise
            if x_original is not None:
                # Project back to eps-ball
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
        else:
            x_adv = x_init.clone()

        # Initialize metrics
        start_time = time.time()
        iterations = 0
        gradient_calls = 0

        # For early stopping
        if success_fn is not None:
            success = success_fn(x_adv)
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize CG algorithm
        grad = gradient_fn(x_adv)
        gradient_calls += 1
        d = -grad  # Initial search direction
        loss_current = loss_fn(x_adv)

        # Main optimization loop
        for t in range(self.n_iterations):
            iterations += 1

            # Skip already successful examples if early stopping is enabled
            if self.early_stopping and success.all():
                break

            # Store old gradient for beta calculation
            grad_old = grad.clone()

            # Perform line search to find optimal step size
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Compute line search only for examples that are still being optimized
                alpha_full = torch.zeros(batch_size, device=device)

                # Perform line search for working examples
                alpha_working = self._line_search(
                    x_adv[working_examples],
                    d[working_examples],
                    grad_old[working_examples],
                    loss_current[working_examples],
                    x_original[working_examples] if x_original is not None else None,
                    lambda x: loss_fn(x),
                )

                alpha_full[working_examples] = alpha_working
                alpha = alpha_full
            else:
                alpha = self._line_search(
                    x_adv, d, grad_old, loss_current, x_original, loss_fn
                )

            # Take a step in the search direction
            x_adv = x_adv + alpha.view(-1, 1, 1, 1) * d

            # Project back to eps-ball
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )

            # Ensure valid image range [0, 1]
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # Compute new gradient
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Compute gradient only for examples that are still being optimized
                grad_full = torch.zeros_like(x_adv)
                grad_working = gradient_fn(x_adv[working_examples])
                grad_full[working_examples] = grad_working
                grad = grad_full
            else:
                grad = gradient_fn(x_adv)

            gradient_calls += 1

            # Compute new loss for line search
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Compute loss only for examples that are still being optimized
                loss_full = torch.zeros(batch_size, device=device)
                loss_working = loss_fn(x_adv[working_examples])
                loss_full[working_examples] = loss_working
                loss_current = loss_full
            else:
                loss_current = loss_fn(x_adv)

            # Check for success
            if success_fn is not None:
                success = success_fn(x_adv)
                if self.verbose and (t + 1) % 10 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

            # Restart conjugate directions periodically or use conjugate update
            if (t + 1) % self.restart_interval == 0:
                # Restart with steepest descent direction
                d = -grad
            else:
                # Compute beta (conjugate direction coefficient)
                beta = self._compute_beta(grad, grad_old, d)

                # Update search direction
                d = -grad + beta.view(-1, 1, 1, 1) * d

        # Return the optimized point and metrics
        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "gradient_calls": gradient_calls,
            "time": total_time,
            "success_rate": (
                success.float().mean().item() if success_fn is not None else 0.0
            ),
        }

        return x_adv, metrics
