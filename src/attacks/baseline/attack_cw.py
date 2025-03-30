"""Carlini & Wagner (C&W) L2 adversarial attack implementation.

This implements the C&W L2 attack from the paper 'Towards Evaluating the Robustness of Neural Networks'
[https://arxiv.org/abs/1608.04644]

The attack finds adversarial examples by solving the optimization problem:
    minimize ||δ||_2 subject to f(x+δ) ≠ f(x) (untargeted) or f(x+δ) = t (targeted)

By transforming it into an unconstrained form:
    minimize ||x' - x||_2 + c·f(x')

where:
    - x' = 1/2(tanh(w) + 1) is the adversarial example (ensured to be in [0,1])
    - f(x') measures how successfully the adversarial example fools the classifier
    - c is a hyperparameter that controls the tradeoff between distortion and attack success

Key features:
- Produces minimal L2 perturbation for misclassification
- Uses binary search to find optimal tradeoff between perturbation size and attack success
- Incorporates a confidence parameter to control robustness of adversarial examples
- Very effective but computationally more expensive than simpler methods
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from ..base import BaseAttack


class CW(BaseAttack):
    r"""
    Carlini & Wagner (C&W) L2 adversarial attack.

    This attack solves the optimization problem:

    .. math::
        \min_{\delta} \|\delta\|_2 \quad \text{subject to} \quad f(x+\delta) \neq y \text{ (untargeted) or } f(x+\delta) = t \text{ (targeted)}

    Using the change of variables :math:`x' = \frac{1}{2}(\tanh(w)+1)` to ensure valid pixel range [0,1],
    the problem becomes:

    .. math::
        \min_{w} \left\|\frac{1}{2}(\tanh(w)+1) - x\right\|_2^2 + c \cdot f\left(\frac{1}{2}(\tanh(w)+1)\right)

    where :math:`f(x')` is defined as:

    .. math::
        f(x') = \max(\max\{Z(x')_i: i \neq t\} - Z(x')_t, -\kappa) \quad \text{(targeted)}

    or:

    .. math::
        f(x') = \max(Z(x')_t - \max\{Z(x')_i: i \neq t\}, -\kappa) \quad \text{(untargeted)}

    This implementation includes binary search for the optimal c value as described in the original paper.

    Args:
        model: The neural network model to attack.
        confidence: Confidence parameter κ that controls the boundary between decision regions.
            Higher values produce more robust adversarial examples that transfer better.
        c_init: Initial value of the constant c that balances perturbation size and attack success.
        max_iter: Maximum number of optimization iterations.
        binary_search_steps: Number of binary search steps to find optimal c.
        learning_rate: Learning rate for the Adam optimizer.
        targeted: Whether to perform a targeted attack.
        abort_early: Whether to abort early if no improvement is found.
        clip_min: Minimum value for pixel clipping (default: 0.0).
        clip_max: Maximum value for pixel clipping (default: 1.0).
        verbose: Print progress updates.
        device: Device to run the attack on (e.g., CPU or GPU).

    Shape:
        - inputs: :math:`(N, C, H, W)` where `N = batch size`, `C = channels`, `H = height`, `W = width`.
          Input values should be in range [0, 1] or normalized according to model requirements.
        - targets: :math:`(N)` where each value is the target class index.
        - output: :math:`(N, C, H, W)` containing adversarial examples.

    Example:
        >>> attack = CW(model, confidence=0.0, c_init=0.01, max_iter=1000)
        >>> adv_images, metrics = attack.generate(images, labels)

    .. warning::
        Setting c_init too low might result in failed attacks. The binary search will adjust it,
        but starting with a reasonable value (like 0.1 or 1.0) may speed up convergence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        confidence: float = 0.0,
        c_init: float = 0.01,
        max_iter: int = 1000,
        binary_search_steps: int = 5,
        learning_rate: float = 0.01,
        targeted: bool = False,
        abort_early: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the C&W attack.

        Args:
            model: The model to attack.
            confidence: Confidence parameter κ (kappa) for the attack.
                Higher values produce more robust adversarial examples.
            c_init: Initial value of the constant c that balances the importance of
                perturbation size vs attack success.
            max_iter: Maximum number of optimization iterations.
            binary_search_steps: Number of binary search steps to find optimal c.
            learning_rate: Learning rate for the Adam optimizer.
            targeted: Whether to perform a targeted attack.
            abort_early: Whether to abort early if no improvement is found.
            clip_min: Minimum value for pixel clipping.
            clip_max: Maximum value for pixel clipping.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Note: C&W doesn't use eps or loss_fn from base class (it has its own custom loss)
        super().__init__(
            model,
            norm="L2",
            eps=float("inf"),  # C&W finds the minimal perturbation itself
            targeted=targeted,
            loss_fn="none",  # C&W uses a custom loss function
            device=device,
            verbose=verbose,
        )

        self.confidence = confidence
        self.c_init = c_init
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Number of iterations to check for early abort
        self.abort_check_interval = 100

        # C&W uses the tanh function to enforce box constraints
        self.boxplus = (clip_max - clip_min) / 2
        self.boxmul = (clip_max + clip_min) / 2

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from image space to tanh space for optimization.

        This applies the inverse of the tanh transformation:
        w = atanh((x - boxmul) / boxplus)

        The tanh space allows unconstrained optimization while ensuring
        the resulting image stays within valid pixel range.

        Args:
            x: Input tensor in image space [clip_min, clip_max]

        Returns:
            Tensor in tanh space
        """
        # If normalized, convert to original pixel space first
        if self.mean is not None and self.std is not None:
            x_denorm = self._denormalize(x)
        else:
            x_denorm = x

        # Normalize to [-1, 1]
        x_tanh = (x_denorm - self.boxmul) / self.boxplus
        # Apply arctanh (with clipping for numerical stability)
        return torch.atanh(torch.clamp(x_tanh, -0.99999, 0.99999))

    def _to_image_space(self, x_tanh: torch.Tensor) -> torch.Tensor:
        """
        Convert from tanh space back to image space.

        This applies the tanh transformation:
        x = tanh(w) * boxplus + boxmul

        This ensures pixel values remain within [clip_min, clip_max].

        Args:
            x_tanh: Tensor in tanh space

        Returns:
            Tensor in image space [clip_min, clip_max], normalized if needed
        """
        # Apply tanh to ensure range [-1, 1]
        x_t = torch.tanh(x_tanh)
        # Scale back to image space
        x_denorm = self.boxplus * x_t + self.boxmul

        # If normalized, convert back to normalized space for the model
        if self.mean is not None and self.std is not None:
            return self._renormalize(x_denorm)
        else:
            return x_denorm

    def _cw_loss(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
        w: torch.Tensor,
        x_origin: torch.Tensor,
        x_adv_tanh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the C&W loss function.

        The loss has two components:
        1. Classification loss (f) that encourages misclassification:
           f(x') = max(max{Z(x')_i: i≠t} - Z(x')_t, -κ) for targeted attacks
           f(x') = max(Z(x')_t - max{Z(x')_i: i≠t}, -κ) for untargeted attacks

        2. Distance loss (L2 norm) that minimizes the perturbation:
           d(x,x') = ||x - x'||_2^2

        The total loss is: L = d(x,x') + c·f(x')

        Args:
            logits: The model's output logits
            target_labels: The target class indices
            w: The weighting factor for the distance term
            x_origin: The original images
            x_adv_tanh: The adversarial examples in tanh space

        Returns:
            A tuple (total_loss, classification_loss, distance_loss)
        """
        # Convert adversarial examples from tanh space to image space (normalized if needed)
        x_adv = self._to_image_space(x_adv_tanh)

        # For L2 distance, we need to compute in denormalized space
        if self.mean is not None and self.std is not None:
            x_adv_denorm = self._denormalize(x_adv)
            x_origin_denorm = self._denormalize(x_origin)
            l2_dist = torch.norm(
                (x_adv_denorm - x_origin_denorm).view(x_origin.shape[0], -1), p=2, dim=1
            )
        else:
            # If not normalized, compute L2 distance directly
            l2_dist = torch.norm(
                (x_adv - x_origin).view(x_origin.shape[0], -1), p=2, dim=1
            )

        # Get the logits for the target class
        target_logits = torch.gather(logits, 1, target_labels.unsqueeze(1)).squeeze(1)

        # For each sample, get the largest logit that isn't the target class
        mask = torch.ones_like(logits).scatter_(1, target_labels.unsqueeze(1), 0)
        other_logits = torch.where(
            mask.bool(), logits, torch.tensor(-float("inf")).to(self.device)
        )
        max_other_logits = other_logits.max(dim=1)[0]

        # Compute classification loss
        if self.targeted:
            # For targeted attacks, we want target class to have higher logit
            # than all other classes (plus a confidence margin)
            cls_loss = torch.clamp(
                max_other_logits - target_logits + self.confidence, min=0
            )
        else:
            # For untargeted attacks, we want any class other than the original
            # to have a higher logit (plus a confidence margin)
            cls_loss = torch.clamp(
                target_logits - max_other_logits + self.confidence, min=0
            )

        # Compute total loss (weighted sum of classification and distance loss)
        total_loss = cls_loss + w * l2_dist

        return total_loss, cls_loss, l2_dist

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using C&W attack.

        The attack algorithm works as follows:
        1. Transform the problem to the tanh space to ensure valid pixel range
        2. Perform binary search to find optimal c value that balances attack success and distortion
        3. For each c, optimize the adversarial example using Adam over max_iter iterations
        4. Return the adversarial example with the smallest perturbation that successfully fools the model

        Args:
            inputs: Input images (clean samples) to perturb, of shape (N, C, H, W)
            targets: Target labels (for targeted attacks) or true labels (for untargeted attacks), of shape (N)

        Returns:
            A tuple (adversarial_examples, metrics):
            - adversarial_examples: Tensor of same shape as inputs containing the adversarial examples
            - metrics: Dictionary with performance metrics including success rate, L2 norm, iterations, etc.
        """
        # Reset metrics from previous runs
        self.reset_metrics()
        start_time = time.time()

        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions for evaluation
        original_predictions = self.store_original_predictions(inputs)

        batch_size = inputs.shape[0]

        # For untargeted attack, use original labels as targets (to move away from)
        if not self.targeted:
            with torch.no_grad():
                # Get true labels to use as targets for untargeted attack
                if targets is None or targets.shape[0] != inputs.shape[0]:
                    targets = self.model(inputs).argmax(dim=1)

        # Initialize variables for binary search
        c_lower = torch.zeros(batch_size, device=self.device)
        c_upper = torch.ones(batch_size, device=self.device) * 1e10
        c_current = torch.ones(batch_size, device=self.device) * self.c_init

        # Variables to track best adversarial examples and their properties
        best_adv = inputs.clone()
        best_l2 = torch.ones(batch_size, device=self.device) * float("inf")
        best_c = torch.zeros(batch_size, device=self.device)

        # Track attack success for each sample
        attack_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Track total iterations and gradient calls
        total_iterations = 0
        total_grad_calls = 0

        # Binary search to find the optimal c value
        for binary_step in range(self.binary_search_steps):
            if self.verbose:
                print(f"Binary search step {binary_step+1}/{self.binary_search_steps}")

            # Initialize variables for this binary search step
            x_adv_best = inputs.clone()
            best_loss = torch.ones(batch_size, device=self.device) * float("inf")
            found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # Convert original images to tanh space
            x_tanh = self._to_tanh_space(inputs)

            # Create a copy that requires gradients
            x_adv_tanh = x_tanh.clone().detach().requires_grad_(True)

            # Setup optimizer
            optimizer = torch.optim.Adam([x_adv_tanh], lr=self.learning_rate)

            # Main optimization loop for this binary search step
            for iteration in range(self.max_iter):
                # Forward pass - convert from tanh space to normalized image space for the model
                x_adv = self._to_image_space(x_adv_tanh)
                logits = self.model(x_adv)

                # Count this as a gradient call
                total_grad_calls += 1

                # Compute C&W loss
                loss, cls_loss, l2_dist = self._cw_loss(
                    logits, targets, c_current, inputs, x_adv_tanh
                )

                # Check if attack is successful for each sample
                if self.targeted:
                    success = logits.argmax(dim=1) == targets
                else:
                    success = logits.argmax(dim=1) != targets

                # Update best result for samples where attack is successful
                for i in range(batch_size):
                    if success[i] and l2_dist[i] < best_l2[i]:
                        best_l2[i] = l2_dist[i]
                        best_adv[i] = x_adv[i].detach()
                        best_c[i] = c_current[i]
                        attack_success[i] = True

                # Update the best loss seen so far
                loss_per_sample = loss.detach()
                for i in range(batch_size):
                    if loss_per_sample[i] < best_loss[i]:
                        best_loss[i] = loss_per_sample[i]
                        x_adv_best[i] = x_adv[i].detach()

                    # Also check if we've found a successful attack and update found flag
                    if success[i]:
                        found[i] = True

                # Early stop if we're making no progress
                if (
                    iteration % self.abort_check_interval == 0
                    and iteration > 0
                    and self.abort_early
                    and found.all()
                ):
                    break

                # Backward pass and update
                optimizer.zero_grad()
                (loss.sum()).backward()
                optimizer.step()

                # Count this iteration
                total_iterations += 1

            # Update c for the next binary search iteration
            for i in range(batch_size):
                if found[i]:
                    # Adjust c lower (to reduce perturbation)
                    c_upper[i] = c_current[i]
                else:
                    # Adjust c higher (to increase attack strength)
                    c_lower[i] = c_current[i]

                # Update current c
                c_current[i] = (c_lower[i] + c_upper[i]) / 2

        # Calculate success rate and other metrics
        success_rate = attack_success.float().mean().item() * 100
        avg_l2 = (
            best_l2[attack_success].mean().item()
            if attack_success.any()
            else float("inf")
        )

        # Update timing metrics
        self.total_time = time.time() - start_time
        self.total_iterations = total_iterations
        self.total_gradient_calls = total_grad_calls

        # Calculate L2 perturbation in denormalized space for proper reporting
        if self.mean is not None and self.std is not None:
            inputs_denorm = self._denormalize(inputs)
            best_adv_denorm = self._denormalize(best_adv)
            l2_norms = torch.norm(
                (best_adv_denorm - inputs_denorm).view(batch_size, -1), p=2, dim=1
            )
            avg_l2_denorm = (
                l2_norms[attack_success].mean().item()
                if attack_success.any()
                else float("inf")
            )

            # Add this to metrics
            metrics = {
                **self.get_metrics(),
                "success_rate": success_rate,
                "avg_l2_norm": avg_l2,
                "avg_l2_norm_denorm": avg_l2_denorm,
                "best_c": best_c.mean().item(),
                "iterations": total_iterations,
                "gradient_calls": total_grad_calls,
            }
        else:
            metrics = {
                **self.get_metrics(),
                "success_rate": success_rate,
                "avg_l2_norm": avg_l2,
                "best_c": best_c.mean().item(),
                "iterations": total_iterations,
                "gradient_calls": total_grad_calls,
            }

        return best_adv, metrics
