"""Limited-memory BFGS (L-BFGS) optimization method implementation."""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable, List

from src.utils.projections import project_adversarial_example


class LBFGSOptimizer:
    """
    Limited-memory BFGS (L-BFGS) optimization method.

    L-BFGS is a quasi-Newton method that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS)
    algorithm using a limited amount of memory. It uses the history of gradients and
    updates to construct an approximation of the inverse Hessian matrix.
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
        verbose: bool = False,
    ):
        """
        Initialize the L-BFGS optimizer.

        Args:
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            n_iterations: Maximum number of iterations
            history_size: Number of previous iterations to store for approximating Hessian
            line_search_fn: Line search method ('strong_wolfe' or 'armijo')
            max_line_search: Maximum number of line search iterations
            initial_step: Initial step size for line search
            rand_init: Whether to initialize with random perturbation
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop early when objective is achieved
            verbose: Whether to print progress information
        """
        self.norm = norm
        self.eps = eps
        self.n_iterations = n_iterations
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.max_line_search = max_line_search
        self.initial_step = initial_step
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Constants for line search
        self.c1 = 1e-4  # Sufficient decrease parameter (Armijo condition)
        self.c2 = 0.9  # Curvature parameter (Wolfe condition)

    def _two_loop_recursion(
        self,
        gradient: torch.Tensor,
        s_history: List[torch.Tensor],
        y_history: List[torch.Tensor],
        rho_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the search direction using two-loop recursion.

        Args:
            gradient: Current gradient
            s_history: History of position differences
            y_history: History of gradient differences
            rho_history: History of reciprocal inner products (1 / (s_i^T * y_i))

        Returns:
            Search direction
        """
        device = gradient.device
        batch_size = gradient.shape[0]

        # Initialize search direction with negative gradient
        q = gradient.clone()

        # First loop (backward)
        alpha_list = []
        history_size = len(s_history)

        for i in range(history_size - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = rho_history[i]

            # Ensure s_i and y_i have the correct batch dimension
            if s_i.shape[0] != batch_size:
                s_i = s_i.expand(batch_size, *s_i.shape[1:])

            if y_i.shape[0] != batch_size:
                y_i = y_i.expand(batch_size, *y_i.shape[1:])

            # Compute alpha_i
            alpha_i = torch.bmm(
                s_i.view(batch_size, 1, -1), q.view(batch_size, -1, 1)
            ).squeeze(-1)

            # Handle case where rho_i has different batch size
            if isinstance(rho_i, torch.Tensor):
                if rho_i.numel() != batch_size:
                    # If rho_i has wrong size, use the mean value
                    rho_i = torch.mean(rho_i).expand(batch_size)
                elif rho_i.dim() == 1 and rho_i.shape[0] != batch_size:
                    rho_i = rho_i[0].expand(batch_size)

            # Apply rho_i to alpha_i
            alpha_i = rho_i * alpha_i
            alpha_list.append(alpha_i)

            # Update q - ensure alpha_i has the right shape
            if alpha_i.dim() == 1:
                alpha_i = alpha_i.view(batch_size, 1, 1, 1)
            q = q - alpha_i * y_i

        # Scale initial Hessian approximation
        # H_k^0 = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1}) * I
        if history_size > 0:
            s_last = s_history[-1]
            y_last = y_history[-1]

            # Ensure s_last and y_last have the correct batch dimension
            if s_last.shape[0] != batch_size:
                s_last = s_last.expand(batch_size, *s_last.shape[1:])

            if y_last.shape[0] != batch_size:
                y_last = y_last.expand(batch_size, *y_last.shape[1:])

            s_dot_y = torch.bmm(
                s_last.view(batch_size, 1, -1), y_last.view(batch_size, -1, 1)
            ).squeeze()
            y_dot_y = torch.bmm(
                y_last.view(batch_size, 1, -1), y_last.view(batch_size, -1, 1)
            ).squeeze()

            # Avoid division by zero
            scale = s_dot_y / (y_dot_y + 1e-10)
            if scale.dim() > 0:
                scale = scale.view(batch_size, 1, 1, 1)
            r = q * scale
        else:
            r = q

        # Second loop (forward)
        for i in range(history_size):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = rho_history[i]
            alpha_i = alpha_list[history_size - 1 - i]

            # Ensure s_i and y_i have the correct batch dimension
            if s_i.shape[0] != batch_size:
                s_i = s_i.expand(batch_size, *s_i.shape[1:])

            if y_i.shape[0] != batch_size:
                y_i = y_i.expand(batch_size, *y_i.shape[1:])

            # Handle case where rho_i has different batch size
            if isinstance(rho_i, torch.Tensor):
                if rho_i.numel() != batch_size:
                    # If rho_i has wrong size, use the mean value
                    rho_i = torch.mean(rho_i).expand(batch_size)
                elif rho_i.dim() == 1 and rho_i.shape[0] != batch_size:
                    rho_i = rho_i[0].expand(batch_size)

            # Compute beta
            beta_i = torch.bmm(
                y_i.view(batch_size, 1, -1), r.view(batch_size, -1, 1)
            ).squeeze(-1)

            # Apply rho_i to beta_i
            beta_i = rho_i * beta_i

            # Update r - ensure alpha_i and beta_i have right shape
            if alpha_i.dim() == 1:
                alpha_i = alpha_i.view(batch_size, 1, 1, 1)
            if beta_i.dim() == 1:
                beta_i = beta_i.view(batch_size, 1, 1, 1)

            r = r + (alpha_i - beta_i) * s_i

        # Return search direction
        return -r

    def _line_search(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        current_loss: torch.Tensor,
        current_grad: torch.Tensor,
        x_original: Optional[torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        grad_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Perform line search to find a step size satisfying Wolfe conditions.

        Args:
            x: Current point
            direction: Search direction
            current_loss: Loss at current point
            current_grad: Gradient at current point
            x_original: Original input for projection
            loss_fn: Function to compute loss
            grad_fn: Function to compute gradient

        Returns:
            Tuple of (step_size, new_x, new_grad, success)
        """
        batch_size = x.shape[0]
        device = x.device

        # Initial step size
        alpha = torch.ones(batch_size, device=device) * self.initial_step

        # Calculate initial directional derivative
        dir_deriv = torch.bmm(
            current_grad.view(batch_size, 1, -1), direction.view(batch_size, -1, 1)
        ).squeeze(-1)

        # Ensure dir_deriv is a 1D tensor of shape (batch_size,)
        if dir_deriv.dim() > 1:
            dir_deriv = dir_deriv.view(batch_size)

        # For Armijo condition
        armijo_threshold = current_loss - self.c1 * alpha * dir_deriv

        # For Wolfe condition
        if self.line_search_fn == "strong_wolfe":
            wolfe_threshold = self.c2 * dir_deriv

        # Flag to track successful search
        success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Best step size and values
        best_alpha = alpha.clone()
        best_x = x.clone()
        best_grad = current_grad.clone()

        # Line search iterations
        for i in range(self.max_line_search):
            # Take step
            x_new = x + alpha.view(-1, 1, 1, 1) * direction

            # Project if necessary
            if x_original is not None:
                x_new = project_adversarial_example(
                    x_new, x_original, self.eps, self.norm
                )

            # Ensure valid image range
            x_new = torch.clamp(x_new, 0.0, 1.0)

            # Compute new loss and gradient
            loss_new = loss_fn(x_new)

            # Check Armijo condition (sufficient decrease)
            armijo_satisfied = loss_new <= armijo_threshold

            if self.line_search_fn == "armijo":
                # For Armijo-only line search
                success = armijo_satisfied

                # Update best values for successful steps
                best_alpha[armijo_satisfied] = alpha[armijo_satisfied]
                best_x[armijo_satisfied] = x_new[armijo_satisfied]

                if success.all():
                    # Compute gradient only for successful points to avoid extra computation
                    best_grad = grad_fn(best_x)
                    break

                # Backtracking: reduce step size for unsuccessful points
                alpha[~armijo_satisfied] *= 0.5
                armijo_threshold[~armijo_satisfied] = (
                    current_loss[~armijo_satisfied]
                    - self.c1 * alpha[~armijo_satisfied] * dir_deriv[~armijo_satisfied]
                )

            else:  # strong_wolfe
                # Only compute gradient for points that satisfy Armijo
                if armijo_satisfied.any():
                    # Compute gradient for points that satisfy Armijo
                    grad_new_full = torch.zeros_like(current_grad)

                    # Make sure armijo_satisfied has the correct shape for indexing
                    armijo_flat = armijo_satisfied
                    if armijo_satisfied.dim() > 1:
                        armijo_flat = armijo_satisfied.view(batch_size)

                    grad_new_partial = grad_fn(x_new[armijo_flat])
                    grad_new_full[armijo_flat] = grad_new_partial

                    # Compute new directional derivative
                    dir_deriv_new = torch.bmm(
                        grad_new_full.view(batch_size, 1, -1),
                        direction.view(batch_size, -1, 1),
                    ).squeeze(-1)

                    # Ensure dir_deriv_new is a 1D tensor
                    if dir_deriv_new.dim() > 1:
                        dir_deriv_new = dir_deriv_new.view(batch_size)

                    # Check Wolfe condition (curvature condition)
                    wolfe_satisfied = torch.abs(dir_deriv_new) <= torch.abs(
                        wolfe_threshold
                    )

                    # Make sure all boolean masks have the same shape
                    if wolfe_satisfied.dim() > 1:
                        wolfe_satisfied = wolfe_satisfied.view(batch_size)

                    # Combined conditions
                    new_success = armijo_flat & wolfe_satisfied

                    # Update best values for successful steps
                    best_alpha[new_success] = alpha[new_success]
                    best_x[new_success] = x_new[new_success]
                    best_grad[new_success] = grad_new_full[new_success]

                    # Update success flags
                    success = success | new_success

                    if success.all():
                        break

                    # For Armijo but not Wolfe: need to increase step size
                    increase_idx = armijo_flat & ~wolfe_satisfied & ~success
                    if increase_idx.any():
                        alpha[increase_idx] *= 2.0

                # For points not satisfying Armijo: decrease step size
                if armijo_satisfied.dim() > 1:
                    armijo_flat = armijo_satisfied.view(batch_size)
                else:
                    armijo_flat = armijo_satisfied

                decrease_idx = ~armijo_flat & ~success
                if decrease_idx.any():
                    alpha[decrease_idx] *= 0.5
                    armijo_threshold[decrease_idx] = (
                        current_loss[decrease_idx]
                        - self.c1 * alpha[decrease_idx] * dir_deriv[decrease_idx]
                    )

        # For any unsuccessful searches, use the best values found
        if not success.all():
            # Compute gradient for remaining points
            remaining_grad = grad_fn(best_x[~success])
            best_grad[~success] = remaining_grad

        return best_alpha, best_x, best_grad, success.all()

    def optimize(
        self,
        x_init: torch.Tensor,
        gradient_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        x_original: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run L-BFGS optimization.

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

        # History for L-BFGS
        s_history = []  # Position differences
        y_history = []  # Gradient differences
        rho_history = []  # 1 / (s_i^T * y_i)

        # Initialize with current gradient and loss
        grad = gradient_fn(x_adv)
        loss_current = loss_fn(x_adv)
        gradient_calls += 1

        # Main optimization loop
        for t in range(self.n_iterations):
            iterations += 1

            # Skip already successful examples if early stopping is enabled
            if self.early_stopping and success.all():
                break

            # If no history, use steepest descent
            if len(s_history) == 0:
                direction = -grad
            else:
                # Compute search direction using two-loop recursion
                direction = self._two_loop_recursion(
                    grad, s_history, y_history, rho_history
                )

            # Store current position and gradient
            x_old = x_adv.clone()
            grad_old = grad.clone()

            # Perform line search
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Only perform line search for examples still being optimized
                alpha_full = torch.zeros(batch_size, device=device)
                x_new_full = x_adv.clone()
                grad_new_full = grad.clone()

                # Line search for working examples
                alpha_working, x_new_working, grad_new_working, _ = self._line_search(
                    x_adv[working_examples],
                    direction[working_examples],
                    loss_current[working_examples],
                    grad_old[working_examples],
                    x_original[working_examples] if x_original is not None else None,
                    lambda x: loss_fn(x),
                    lambda x: gradient_fn(x),
                )

                alpha_full[working_examples] = alpha_working
                x_new_full[working_examples] = x_new_working
                grad_new_full[working_examples] = grad_new_working

                alpha = alpha_full
                x_adv = x_new_full
                grad = grad_new_full
            else:
                alpha, x_adv, grad, _ = self._line_search(
                    x_adv,
                    direction,
                    loss_current,
                    grad_old,
                    x_original,
                    loss_fn,
                    gradient_fn,
                )

            gradient_calls += 1

            # Compute loss for next iteration
            loss_current = loss_fn(x_adv)

            # Check for success
            if success_fn is not None:
                success = success_fn(x_adv)
                if self.verbose and (t + 1) % 5 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

            # Update L-BFGS history
            s_k = x_adv - x_old
            y_k = grad - grad_old

            # Compute ρₖ = 1/(sₖᵀyₖ)
            s_dot_y = (
                torch.bmm(
                    s_k.view(batch_size, 1, -1), y_k.view(batch_size, -1, 1)
                ).squeeze()
                + 1e-10
            )
            rho_k = 1.0 / s_dot_y

            # Only update history with meaningful steps (where s_dot_y is sufficiently large)
            valid_update = torch.abs(s_dot_y) > 1e-10

            if valid_update.any():
                # Add to history
                s_history.append(s_k)
                y_history.append(y_k)
                rho_history.append(rho_k)

                # Limit history size
                if len(s_history) > self.history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)

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
