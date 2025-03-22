"""Limited-memory BFGS (L-BFGS) optimization method implementation.

This file implements the L-BFGS optimizer for generating adversarial examples
against neural networks. L-BFGS is a quasi-Newton method that approximates the
inverse Hessian matrix using a limited history of position and gradient differences,
enabling more efficient optimization in high-dimensional spaces.

Key features:
- Two-loop recursion algorithm that avoids explicitly forming the inverse Hessian
- Multiple line search methods (Armijo and Strong Wolfe conditions)
- Batch processing for simultaneous optimization of multiple examples
- Configurable memory usage with adjustable history size
- Dynamic scaling of the initial Hessian approximation
- Constraint handling for both L2 and Linf perturbation norms
- Early stopping capability when adversarial criteria are met

Expected inputs:
- Initial images (usually clean images to be perturbed)
- Gradient function that computes gradients of the adversarial loss
- Loss function that returns per-example losses
- Success function that determines if adversarial criteria are met
- Original images (for projection constraints)

Expected outputs:
- Optimized adversarial examples
- Optimization metrics (iterations, gradient calls, time, success rate)
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional, Callable, List

from src.utils.projections import project_adversarial_example


class LBFGSOptimizer:
    """
    Limited-memory BFGS (L-BFGS) optimization method.

    L-BFGS is a quasi-Newton method that approximates the BFGS algorithm using
    a limited amount of memory. It uses the history of recent position and gradient
    differences to construct an approximation of the inverse Hessian, which is then
    used to compute a more informed search direction.
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
        Initialize the L-BFGS optimizer with parameters controlling:
          - the perturbation norm and maximum allowed perturbation (eps)
          - the number of iterations and the history length for Hessian approximation
          - the line search method and its parameters
          - optional random initialization and early stopping behavior
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

        # Constants for line search conditions.
        self.c1 = 1e-4  # Sufficient decrease (Armijo) parameter.
        self.c2 = 0.9  # Curvature parameter for the Wolfe condition.

    def _two_loop_recursion(
        self,
        gradient: torch.Tensor,
        s_history: List[torch.Tensor],
        y_history: List[torch.Tensor],
        rho_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the search direction using the two-loop recursion.

        This method approximates the product of the inverse Hessian with the gradient,
        i.e. d = -H_k * gradient, without forming H_k explicitly.

        Args:
            gradient: Current gradient (shape: [batch, ...])
            s_history: List of past differences in positions (s_i = x_{i+1} - x_i)
            y_history: List of past differences in gradients (y_i = grad_{i+1} - grad_i)
            rho_history: List of reciprocal inner products (1 / (s_i^T y_i))

        Returns:
            Search direction d (negative of the approximate Hessian-gradient product)
        """
        device = gradient.device
        batch_size = gradient.shape[0]

        # Initialize q as the current gradient.
        q = gradient.clone()

        # First loop: loop backward over the history.
        alpha_list = []
        history_size = len(s_history)
        for i in range(history_size - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = rho_history[i]

            # Ensure that s_i and y_i have a batch dimension.
            if s_i.shape[0] != batch_size:
                s_i = s_i.expand(batch_size, *s_i.shape[1:])
            if y_i.shape[0] != batch_size:
                y_i = y_i.expand(batch_size, *y_i.shape[1:])

            # Compute alpha_i = rho_i * (s_i^T * q)
            alpha_i = torch.bmm(
                s_i.view(batch_size, 1, -1), q.view(batch_size, -1, 1)
            ).squeeze(-1)
            # Adjust rho_i if its shape does not match
            if isinstance(rho_i, torch.Tensor):
                if rho_i.numel() != batch_size:
                    rho_i = torch.mean(rho_i).expand(batch_size)
                elif rho_i.dim() == 1 and rho_i.shape[0] != batch_size:
                    rho_i = rho_i[0].expand(batch_size)
            alpha_i = rho_i * alpha_i
            alpha_list.append(alpha_i)

            # Update q: subtract the curvature information scaled by y_i.
            # If alpha_i is 1D, add extra dimensions to match y_i.
            if alpha_i.dim() == 1:
                alpha_i = alpha_i.view(batch_size, 1, 1, 1)
            q = q - alpha_i * y_i

        # Scaling of the initial Hessian approximation:
        # H_0 = (s_{last}^T y_{last}) / (y_{last}^T y_{last}) * I
        if history_size > 0:
            s_last = s_history[-1]
            y_last = y_history[-1]
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
            scale = s_dot_y / (y_dot_y + 1e-10)  # Avoid division by zero.
            if scale.dim() > 0:
                scale = scale.view(batch_size, 1, 1, 1)
            # Multiply q by the scaling factor to form the initial r.
            r = q * scale
        else:
            # If no history, use the negative gradient direction.
            r = q

        # Second loop: loop forward over the history.
        for i in range(history_size):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = rho_history[i]
            # Retrieve alpha in the reverse order.
            alpha_i = alpha_list[history_size - 1 - i]

            if s_i.shape[0] != batch_size:
                s_i = s_i.expand(batch_size, *s_i.shape[1:])
            if y_i.shape[0] != batch_size:
                y_i = y_i.expand(batch_size, *y_i.shape[1:])
            if isinstance(rho_i, torch.Tensor):
                if rho_i.numel() != batch_size:
                    rho_i = torch.mean(rho_i).expand(batch_size)
                elif rho_i.dim() == 1 and rho_i.shape[0] != batch_size:
                    rho_i = rho_i[0].expand(batch_size)

            # Compute beta_i = rho_i * (y_i^T * r)
            beta_i = torch.bmm(
                y_i.view(batch_size, 1, -1), r.view(batch_size, -1, 1)
            ).squeeze(-1)
            beta_i = rho_i * beta_i

            # Ensure correct shapes before updating r.
            if alpha_i.dim() == 1:
                alpha_i = alpha_i.view(batch_size, 1, 1, 1)
            if beta_i.dim() == 1:
                beta_i = beta_i.view(batch_size, 1, 1, 1)

            # Update r with the correction term.
            r = r + (alpha_i - beta_i) * s_i

        # The search direction is the negative of the approximated Hessian–gradient product.
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
        Perform a line search to find a step size that satisfies either Armijo or strong Wolfe conditions.

        The Armijo condition ensures sufficient decrease:
          f(x + αd) <= f(x) + c1 * α * (d^T grad)
        and the strong Wolfe condition additionally requires:
          |d^T grad(x + αd)| <= c2 * |d^T grad|

        Args:
            x: Current point.
            direction: Search direction.
            current_loss: Loss at x (per-example).
            current_grad: Gradient at x.
            x_original: Original input for projection (if applicable).
            loss_fn: Function that computes the loss.
            grad_fn: Function that computes the gradient.

        Returns:
            A tuple containing:
              - step_size: Final step size (per-example tensor)
              - new_x: New point after taking the step.
              - new_grad: Gradient at the new point.
              - success: Boolean flag indicating if the conditions were met for all examples.
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize step sizes.
        alpha = torch.ones(batch_size, device=device) * self.initial_step

        # Compute directional derivative d^T * grad at x.
        dir_deriv = torch.bmm(
            current_grad.view(batch_size, 1, -1), direction.view(batch_size, -1, 1)
        ).squeeze(-1)
        if dir_deriv.dim() > 1:
            dir_deriv = dir_deriv.view(batch_size)

        # Set the Armijo threshold: note that if direction is descent then d^T grad < 0.
        armijo_threshold = current_loss + self.c1 * alpha * dir_deriv

        # For strong Wolfe, define a curvature threshold.
        if self.line_search_fn == "strong_wolfe":
            wolfe_threshold = self.c2 * dir_deriv  # Later used with absolute values.

        # Initialize flags and store best values seen so far.
        success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        best_alpha = alpha.clone()
        best_x = x.clone()
        best_grad = current_grad.clone()

        # Perform line search iterations.
        for i in range(self.max_line_search):
            # Compute candidate point: take a step along the search direction.
            x_new = x + alpha.view(-1, 1, 1, 1) * direction

            # Project back into the valid perturbation ball if needed.
            if x_original is not None:
                x_new = project_adversarial_example(
                    x_new, x_original, self.eps, self.norm
                )

            # Clamp x_new to ensure pixel values remain in [0, 1].
            x_new = torch.clamp(x_new, 0.0, 1.0)

            # Evaluate the loss at the new point.
            loss_new = loss_fn(x_new)

            # Check the Armijo (sufficient decrease) condition.
            armijo_satisfied = loss_new <= armijo_threshold

            if self.line_search_fn == "armijo":
                # For Armijo-only, update success and backtrack if necessary.
                success = armijo_satisfied

                # Make sure armijo_satisfied has the right batch dimension
                if armijo_satisfied.shape[0] != batch_size:
                    # If the shapes don't match, expand to match
                    if armijo_satisfied.numel() == 1:
                        armijo_satisfied = armijo_satisfied.expand(batch_size)

                # Update best values for examples that satisfy the Armijo condition
                best_alpha[armijo_satisfied] = alpha[armijo_satisfied]
                best_x[armijo_satisfied] = x_new[armijo_satisfied]

                if success.all():
                    best_grad = grad_fn(best_x)
                    break

                # Reduce step size for examples that do not satisfy Armijo.
                alpha[~armijo_satisfied] *= 0.5
                armijo_threshold[~armijo_satisfied] = (
                    current_loss[~armijo_satisfied]
                    + self.c1 * alpha[~armijo_satisfied] * dir_deriv[~armijo_satisfied]
                )
            else:  # strong_wolfe branch.
                # For points satisfying Armijo, compute the new gradient.
                if armijo_satisfied.any():
                    # Flatten boolean mask if needed.
                    armijo_mask = armijo_satisfied.view(batch_size)
                    grad_new_full = torch.zeros_like(current_grad)
                    grad_new_partial = grad_fn(x_new[armijo_mask])
                    grad_new_full[armijo_mask] = grad_new_partial

                    # Compute new directional derivative at x_new.
                    dir_deriv_new = torch.bmm(
                        grad_new_full.view(batch_size, 1, -1),
                        direction.view(batch_size, -1, 1),
                    ).squeeze(-1)
                    if dir_deriv_new.dim() > 1:
                        dir_deriv_new = dir_deriv_new.view(batch_size)

                    # Check the Wolfe curvature condition.
                    wolfe_satisfied = torch.abs(dir_deriv_new) <= torch.abs(
                        wolfe_threshold
                    )
                    # Combined success: both Armijo and Wolfe satisfied.
                    new_success = armijo_mask & wolfe_satisfied

                    best_alpha[new_success] = alpha[new_success]
                    best_x[new_success] = x_new[new_success]
                    best_grad[new_success] = grad_new_full[new_success]
                    success = success | new_success
                    if success.all():
                        break

                    # For examples that satisfy Armijo but not Wolfe, increase the step size.
                    increase_idx = armijo_mask & (~wolfe_satisfied) & (~success)
                    if increase_idx.any():
                        alpha[increase_idx] *= 2.0

                # For examples not satisfying Armijo, decrease the step size.
                decrease_idx = ~armijo_satisfied.view(batch_size) & (~success)
                if decrease_idx.any():
                    alpha[decrease_idx] *= 0.5
                    armijo_threshold[decrease_idx] = (
                        current_loss[decrease_idx]
                        + self.c1 * alpha[decrease_idx] * dir_deriv[decrease_idx]
                    )

        # For any examples that did not reach success, compute the gradient at the best found point.
        if not success.all():
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
        Run L-BFGS optimization to generate adversarial examples.

        The algorithm:
          1. Optionally adds random noise to initialize the adversarial example.
          2. Computes the gradient and loss at the current point.
          3. Uses the two-loop recursion to compute an approximate inverse Hessian–gradient product,
             giving a search direction.
          4. Performs a line search (using Armijo or strong Wolfe) to choose a step size.
          5. Updates the current point and stores the differences in positions (s_k) and gradients (y_k)
             to update the Hessian approximation.
          6. Optionally stops early if the adversarial criteria are met.

        Args:
            x_init: Initial point (e.g. clean images).
            gradient_fn: Function to compute the gradient.
            loss_fn: Function to compute the loss (must return per-example losses).
            success_fn: Function that returns a Boolean tensor indicating adversarial success.
            x_original: Original input images (for projection constraints).

        Returns:
            A tuple (x_adv, metrics) where x_adv are the optimized adversarial examples and metrics
            contains information such as iterations, gradient calls, total time, and success rate.
        """
        device = x_init.device
        batch_size = x_init.shape[0]

        # Initialize adversarial examples with optional random noise.
        if self.rand_init and self.init_std > 0:
            noise = torch.randn_like(x_init) * self.init_std
            x_adv = x_init + noise
            if x_original is not None:
                x_adv = project_adversarial_example(
                    x_adv, x_original, self.eps, self.norm
                )
        else:
            x_adv = x_init.clone()

        # Metrics for tracking progress.
        start_time = time.time()
        iterations = 0
        gradient_calls = 0

        # Check for early stopping criteria.
        if success_fn is not None:
            success = success_fn(x_adv)
            if self.verbose:
                success_rate = success.float().mean().item() * 100
                print(f"Initial success rate: {success_rate:.2f}%")
        else:
            success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Store the initial success state for metrics
        initial_success = success.clone()

        # Initialize history for L-BFGS updates.
        s_history: List[torch.Tensor] = []  # Differences in positions.
        y_history: List[torch.Tensor] = []  # Differences in gradients.
        rho_history: List[torch.Tensor] = []  # 1 / (s_k^T y_k).

        # Compute initial gradient and loss.
        grad = gradient_fn(x_adv)
        loss_current = loss_fn(x_adv)
        gradient_calls += 1

        # Main optimization loop.
        for t in range(self.n_iterations):
            iterations += 1

            # Skip examples that already meet the adversarial goal.
            if self.early_stopping and success.all():
                if self.verbose:
                    print(f"Stopping early at iteration {t}, all examples successful")
                break

            # Compute search direction: use steepest descent if no history is available.
            if len(s_history) == 0:
                direction = -grad
            else:
                direction = self._two_loop_recursion(
                    grad, s_history, y_history, rho_history
                )

            # Save current state for history update.
            x_old = x_adv.clone()
            grad_old = grad.clone()

            # Perform line search.
            if self.early_stopping:
                working_examples = ~success
                if working_examples.sum() == 0:
                    break

                # Create full tensors to hold step sizes and updates.
                alpha_full = torch.zeros(batch_size, device=device)
                x_new_full = x_adv.clone()
                grad_new_full = grad.clone()

                # Perform line search only on examples that are still being optimized.
                if (
                    working_examples.sum() > 0
                ):  # Only if at least one example needs optimization
                    # Create wrapped loss and gradient functions for the subset
                    def subset_loss_fn(x_subset):
                        # Create a full tensor to hold all examples
                        x_full = x_adv.clone()
                        # Update only working examples
                        x_full[working_examples] = x_subset
                        # Compute loss for all examples
                        full_loss = loss_fn(x_full)
                        # Return only losses for working examples
                        return full_loss[working_examples]

                    def subset_grad_fn(x_subset):
                        # Create a full tensor to hold all examples
                        x_full = x_adv.clone()
                        # Update only working examples
                        x_full[working_examples] = x_subset
                        # Compute gradients for all examples
                        full_grad = gradient_fn(x_full)
                        # Return only gradients for working examples
                        return full_grad[working_examples]

                    alpha_working, x_new_working, grad_new_working, _ = (
                        self._line_search(
                            x_adv[working_examples],
                            direction[working_examples],
                            loss_current[working_examples],
                            grad_old[working_examples],
                            (
                                x_original[working_examples]
                                if x_original is not None
                                else None
                            ),
                            subset_loss_fn,
                            subset_grad_fn,
                        )
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

            # Compute new loss.
            loss_current = loss_fn(x_adv)

            # Check for adversarial success.
            if success_fn is not None:
                # Update success status for all examples
                new_success = success_fn(x_adv)
                # Only update for examples that were not already successful
                if self.early_stopping:
                    success = success | new_success
                else:
                    success = new_success

                if self.verbose and (t + 1) % 5 == 0:
                    success_rate = success.float().mean().item() * 100
                    print(
                        f"Iteration {t+1}/{self.n_iterations}, Success rate: {success_rate:.2f}%"
                    )

            # Update the L-BFGS history.
            s_k = x_adv - x_old  # Change in positions.
            y_k = grad - grad_old  # Change in gradients.
            # Compute s_k^T y_k (using batch matrix multiplication) and avoid division by zero.
            s_dot_y = (
                torch.bmm(
                    s_k.view(batch_size, 1, -1), y_k.view(batch_size, -1, 1)
                ).squeeze()
                + 1e-10
            )
            rho_k = 1.0 / s_dot_y

            # Only update history for steps with meaningful curvature information.
            valid_update = torch.abs(s_dot_y) > 1e-10
            if valid_update.any():
                s_history.append(s_k)
                y_history.append(y_k)
                rho_history.append(rho_k)
                # Limit the history to the specified size.
                if len(s_history) > self.history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)

        # Compile optimization metrics.
        total_time = time.time() - start_time
        metrics = {
            "iterations": iterations,
            "gradient_calls": gradient_calls,
            "time": total_time,
            "success_rate": (
                success.float().mean().item() if success_fn is not None else 0.0
            ),
            "initial_success_rate": (
                initial_success.float().mean().item() if success_fn is not None else 0.0
            ),
        }

        return x_adv, metrics
