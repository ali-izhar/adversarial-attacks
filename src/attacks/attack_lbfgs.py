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

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack
from .optimize import LBFGSOptimizer


class LBFGS(BaseAttack):
    """
    Limited-memory BFGS (L-BFGS) adversarial attack.

    This attack uses the LBFGSOptimizer to generate adversarial examples. L-BFGS is a
    quasi-Newton method that approximates the BFGS algorithm using limited memory. It
    leverages historical gradient and position differences to estimate curvature information,
    which is then used to craft effective adversarial perturbations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
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
                Using 'armijo' is often more effective for adversarial attacks.
            max_line_search: Maximum number of iterations for the line search.
            initial_step: Initial step size for the line search.
                Higher values (3.0-5.0) can help the attack explore more aggressively.
            rand_init: Whether to initialize the attack with random noise.
            init_std: Standard deviation for random initialization.
                Higher values can help escape local minima.
            grad_scale: Factor to scale gradients for more aggressive updates.
            restart_freq: How often to restart from steepest descent direction.
            early_stopping: Stop early if adversarial criteria are met.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Initialize the base attack with model and common parameters.
        super().__init__(model, norm, eps, targeted, loss_fn, device, verbose)

        # Store the additional parameters
        self.grad_scale = grad_scale
        self.restart_freq = restart_freq

        # Scale epsilon for pixel space if normalization is being used
        eps_for_optimizer = eps
        if self.std is not None:
            # Store the unscaled epsilon for proper scaling during optimization
            self.unscaled_eps = eps
            # Will be used with proper scaling in the optimization loop
            if self.verbose:
                print(
                    f"Using normalized epsilon: {eps} (will be scaled during optimization)"
                )

        # Instantiate the L-BFGS optimizer with the specified parameters.
        self.optimizer = LBFGSOptimizer(
            norm=norm,
            eps=eps_for_optimizer,
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

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using the L-BFGS attack.

        The method moves the inputs and targets to the correct device, ensures batch sizes match,
        and defines helper functions to compute gradients, evaluate the loss, and determine if
        the adversarial goal is met. Finally, it calls the LBFGSOptimizer to perform the attack.

        Args:
            inputs: Input images to perturb.
            targets: Target labels for a targeted attack or true labels for an untargeted attack.

        Returns:
            A tuple (adversarial_examples, metrics), where metrics include iterations, gradient calls,
            total time, and success rate.
        """
        # Reset any previously stored metrics.
        self.reset_metrics()
        start_time = time.time()

        batch_size = inputs.shape[0]

        # Move inputs and targets to the attack device (e.g., GPU if available).
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions
        self.store_original_predictions(inputs)

        # Ensure that the number of targets matches the number of inputs.
        if inputs.size(0) != targets.size(0):
            # For targeted attacks with a single target class, expand the target tensor.
            if self.targeted and targets.size(0) == 1:
                targets = targets.expand(inputs.size(0))
            else:
                raise ValueError(
                    f"Input batch size {inputs.size(0)} doesn't match target batch size {targets.size(0)}"
                )

        # Store the targets to avoid passing them repeatedly in helper functions.
        self.original_targets = targets

        # Denormalize inputs to operate in original pixel space
        denormalized_inputs = self._denormalize(inputs)

        # For initialization, use random noise plus some epsilon-scale perturbation
        # to help quickly find adversarial regions, especially for Linf norm
        if self.optimizer.rand_init:
            if self.norm.lower() == "linf":
                # For Linf norm, add uniform random noise within epsilon bounds
                # This is more effective than Gaussian noise for Linf
                random_noise = (
                    torch.rand_like(denormalized_inputs) * 2 - 1
                )  # Uniform [-1, 1]
                if self.std is not None:
                    scaled_eps = self.unscaled_eps * self.std
                    random_noise = random_noise * scaled_eps
                else:
                    random_noise = random_noise * self.eps

                if self.verbose:
                    print(f"Adding Linf uniform random noise with magnitude {self.eps}")
            else:
                # For L2 norm, use Gaussian initialization but with larger scale
                random_noise = (
                    torch.randn_like(denormalized_inputs) * self.optimizer.init_std
                )
                if self.verbose:
                    print(
                        f"Adding L2 Gaussian random noise with std {self.optimizer.init_std}"
                    )

            # Apply the random noise
            init_perturbed = denormalized_inputs + random_noise

            # Project back to the allowed perturbation region if necessary
            init_perturbed = torch.clamp(init_perturbed, min=0.0, max=1.0)

            # Use this as the initial point for optimization
            x_init = init_perturbed
        else:
            x_init = denormalized_inputs.clone()

        # Scale epsilon by std if normalization is used
        if self.std is not None:
            # For each step of the optimization, we'll scale gradient by std
            # This ensures perturbation is applied in pixel space rather than normalized space
            self.optimizer.eps = self.unscaled_eps * self.std
            if self.verbose:
                print(f"Scaled epsilon to {self.optimizer.eps.mean().item()}")

        # Store iteration count for gradient scaling and restarts
        self.current_iteration = 0

        # Define a helper function to compute gradients of the loss with respect to inputs.
        def gradient_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # Use corresponding targets if x is a subset of the original inputs.
            if x_normalized.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            # Detach x to avoid modifying the original input and enable gradient computation.
            x_normalized = x_normalized.detach().clone()
            x_normalized.requires_grad_(True)

            # Forward pass through the model.
            outputs = self.model(x_normalized)
            # Compute the loss (using mean reduction for a single scalar per batch).
            loss = self._compute_loss(outputs, curr_targets, reduction="mean")

            # Backpropagate to compute the gradients.
            self.model.zero_grad()
            loss.backward()

            # Clone the computed gradient.
            grad = x_normalized.grad.clone()

            # The LBFGS optimizer assumes we're minimizing, so we need to adjust the gradient:
            # For untargeted attacks, we want to maximize the loss, so negate the gradient
            # For targeted attacks, we want to minimize the loss, so keep the gradient as is
            if not self.targeted:
                # For untargeted attacks (maximizing loss), use negative gradient for "minimization"
                grad = -grad
            else:
                # For targeted attacks, we're already minimizing, but apply stronger scaling
                # to increase effectiveness
                grad = grad * 1.5

            # Scale gradient if we're using normalization
            # This makes the gradient meaningful in the original pixel space
            if self.std is not None:
                grad = grad * self.std

            # Apply additional gradient scaling for more aggressive updates
            # Scale more strongly in later iterations, especially if we're not making progress
            if self.grad_scale != 1.0:
                # Add iteration-dependent scaling to make later updates more aggressive
                iter_factor = min(1.0 + (self.current_iteration / 20.0), 3.0)
                grad = grad * (self.grad_scale * iter_factor)

            # Scale even more for Linf norm to explore constraint boundary more aggressively
            if self.norm.lower() == "linf":
                # Optional additional scaling for Linf norm
                grad = grad * 2.0

                # Scale by sign to fully utilize the Linf constraint
                if self.current_iteration > 5:  # After just a few iterations
                    # With higher probability, use sign gradient for Linf norm
                    # This helps explore the constraint boundary more effectively
                    if (
                        torch.rand(1).item() < 0.5
                    ):  # 50% probability (increased from 30%)
                        if self.verbose and self.current_iteration % 10 == 0:
                            print("Using sign gradient for exploration")
                        grad = torch.sign(grad) * 0.1  # Scale down the sign gradient

            # Increment iteration counter for restart logic
            self.current_iteration += 1

            # Turn off gradient tracking for x.
            x_normalized.requires_grad_(False)
            return grad

        # Define a helper function to compute per-example losses for the optimizer.
        def loss_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # Use the appropriate targets for x.
            if x_normalized.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            # Evaluate the loss without tracking gradients.
            with torch.no_grad():
                outputs = self.model(x_normalized)
                # Use 'none' reduction to obtain individual losses for each example.
                loss = self._compute_loss(outputs, curr_targets, reduction="none")
            return loss

        # Define a helper function to check if the adversarial example is successful.
        def success_fn(x: torch.Tensor) -> torch.Tensor:
            # x is in the original pixel space, so we need to normalize it
            x_normalized = self._renormalize(x)

            # Retrieve the correct targets if x is a subset.
            if x_normalized.size(0) != self.original_targets.size(0):
                curr_targets = self.original_targets[: x_normalized.size(0)]
            else:
                curr_targets = self.original_targets

            with torch.no_grad():
                outputs = self.model(x_normalized)
                # _check_success should return a Boolean tensor for each example.
                success = self._check_success(outputs, curr_targets)

                # Provide detailed information in verbose mode
                if self.verbose and success.any() and self.current_iteration % 5 == 0:
                    successful_indices = torch.where(success)[0]
                    for idx in successful_indices:
                        print(
                            f"Found adversarial example at iteration {self.current_iteration}"
                        )
                        if idx < outputs.size(0) and idx < curr_targets.size(0):
                            pred_class = outputs[idx].argmax().item()
                            target_class = curr_targets[idx].item()
                            pred_conf = torch.softmax(outputs[idx], dim=0)[
                                pred_class
                            ].item()
                            print(
                                f"  Target: {target_class}, Prediction: {pred_class}, Confidence: {pred_conf:.4f}"
                            )

                return success

        # Add a custom two-loop recursion modifier to handle periodic restarts
        def custom_recursion_handler(iteration, direction, grad):
            # Every restart_freq iterations, reset to steepest descent direction
            if iteration > 0 and iteration % self.restart_freq == 0:
                if self.verbose:
                    print(f"Restarting to steepest descent at iteration {iteration}")
                return -grad
            return direction

        # Monkey patch the _two_loop_recursion method to include restarts
        original_method = self.optimizer._two_loop_recursion

        def patched_method(*args, **kwargs):
            direction = original_method(*args, **kwargs)
            if hasattr(self, "current_iteration"):
                return custom_recursion_handler(
                    self.current_iteration, direction, args[0]
                )
            return direction

        self.optimizer._two_loop_recursion = patched_method

        try:
            # Call the LBFGS optimizer using the helper functions.
            # The optimizer will work in original pixel space
            x_adv_denorm, opt_metrics = self.optimizer.optimize(
                x_init=x_init,  # Use our potentially randomly perturbed initialization
                gradient_fn=gradient_fn,
                loss_fn=loss_fn,
                success_fn=success_fn,
                x_original=denormalized_inputs,  # Use the original denormalized inputs as a reference for the perturbation constraint.
            )
        except Exception as e:
            if self.verbose:
                print(f"Error during LBFGS optimization: {e}")
                print("Falling back to basic gradient descent")

            # Simple fallback to gradient descent if LBFGS fails
            x_adv = x_init.clone()
            with torch.enable_grad():
                # More robust gradient descent implementation
                alpha = 0.01  # Step size
                best_adv = x_adv.clone()
                best_loss = float("inf")

                for i in range(50):  # Increased iterations for fallback
                    try:
                        # Compute gradient
                        grad = gradient_fn(x_adv)

                        # Take a step in the gradient direction
                        x_adv_new = x_adv + alpha * grad

                        # Project back to valid image range
                        x_adv_new = torch.clamp(x_adv_new, 0.0, 1.0)

                        # Project to epsilon ball
                        if self.norm.lower() == "l2":
                            # Manual L2 projection to avoid shape issues
                            delta = x_adv_new - denormalized_inputs
                            delta_flat = delta.reshape(delta.shape[0], -1)
                            l2_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
                            mask = l2_norm > self.eps
                            if mask.any():
                                scaling = self.eps / l2_norm
                                scaling[~mask] = 1.0
                                delta_flat_proj = delta_flat * scaling
                                delta_proj = delta_flat_proj.reshape(delta.shape)
                                x_adv_new = denormalized_inputs + delta_proj
                        else:
                            # Manual Linf projection
                            x_adv_new = torch.max(
                                torch.min(x_adv_new, denormalized_inputs + self.eps),
                                denormalized_inputs - self.eps,
                            )

                        # Evaluate current point
                        current_loss = loss_fn(x_adv_new).mean().item()

                        # Check if this is the best point so far
                        if current_loss < best_loss:
                            best_loss = current_loss
                            best_adv = x_adv_new.clone()

                        # Update for next iteration
                        x_adv = x_adv_new

                        # Check if adversarial criteria are met
                        if success_fn(x_adv).all():
                            if self.verbose:
                                print(f"Found adversarial example at iteration {i+1}")
                            break

                        # Decay step size over time
                        if (i + 1) % 10 == 0:
                            alpha *= 0.9

                    except Exception as inner_e:
                        if self.verbose:
                            print(f"Error in fallback iteration {i}: {inner_e}")
                        # Continue to next iteration if there's an error
                        continue

                x_adv = best_adv  # Use the best adversarial example found

                # Check success after optimization is complete
                success_rate = success_fn(x_adv).float().mean().item() * 100

            # Construct minimal metrics
            opt_metrics = {
                "iterations": i + 1,
                "gradient_calls": i + 1,
                "success_rate": success_rate,
            }
            x_adv_denorm = x_adv
        finally:
            # Restore the original _two_loop_recursion method
            self.optimizer._two_loop_recursion = original_method

        # Renormalize the adversarial examples before returning
        x_adv = self._renormalize(x_adv_denorm)

        # Update internal metrics with those returned by the optimizer.
        self.total_iterations = opt_metrics["iterations"]
        self.total_gradient_calls = opt_metrics["gradient_calls"]
        self.total_time = time.time() - start_time

        # Compile final metrics, converting the success rate to a percentage.
        metrics = {
            **self.get_metrics(),
            "success_rate": opt_metrics["success_rate"],
        }

        return x_adv, metrics
