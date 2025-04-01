"""DeepFool adversarial attack implementation.

Implementation of the DeepFool attack from the paper:
'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
[https://arxiv.org/abs/1511.04599]

The attack iteratively finds the nearest decision boundary and perturbs the input
to cross that boundary. DeepFool works by linearizing the classifier around the
current point and finding the minimal perturbation to cross the linearized boundary.

Key features:
- Produces smaller perturbations compared to gradient-based methods
- Iteratively approximates the decision boundary
- Naturally produces more imperceptible adversarial examples
- Supports both L2 and Linf norms
- Does not require specifying an epsilon constraint (finds minimal perturbation automatically)
"""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..base import BaseAttack


class DeepFool(BaseAttack):
    r"""
    DeepFool adversarial attack.

    This attack implements the algorithm from the paper:
    "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks"
    [https://arxiv.org/abs/1511.04599]

    The attack finds adversarial examples by iteratively projecting the input onto the
    nearest decision boundary. For a classifier :math:`f`, DeepFool approximates the
    decision boundary as a hyperplane and computes the minimal perturbation to cross
    this linearized boundary.

    For a binary classifier, the algorithm computes:

    .. math::
        \min_{\delta} \|\delta\|_2 \quad \text{subject to} \quad f(x+\delta) \neq f(x)

    By linearizing f at each iteration:

    .. math::
        \delta_i = -\frac{f(x_i)}{||\nabla f(x_i)||_2^2} \nabla f(x_i)

    For multi-class classifiers, it finds the closest decision boundary by examining
    each class and selecting the perturbation with minimum distance.

    Args:
        model: The neural network model to attack.
        norm: Norm for the perturbation constraint ('L2' or 'Linf').
        num_classes: Number of classes in the model's output.
        overshoot: Parameter that controls how far beyond the decision boundary to perturb.
            Higher values (e.g., 0.02) make the attack more robust but increase perturbation size.
        max_iter: Maximum number of iterations to run.
        verbose: Print progress updates.
        device: Device to run the attack on (e.g., CPU or GPU).

    Shape:
        - inputs: :math:`(N, C, H, W)` where `N = batch size`, `C = channels`, `H = height`, `W = width`.
          Input values should be in range [0, 1] or normalized according to model requirements.
        - targets: :math:`(N)` where each value is the class index (not used directly by DeepFool).
        - output: :math:`(N, C, H, W)` containing adversarial examples.

    Example:
        >>> attack = DeepFool(model, norm='L2', overshoot=0.02, max_iter=50)
        >>> adv_images, metrics = attack.generate(images)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "L2",
        num_classes: int = 1000,
        overshoot: float = 0.02,
        max_iter: int = 50,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the DeepFool attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            num_classes: Number of classes in the model's output.
            overshoot: Parameter to overshoot the decision boundary (typically 0.02).
                Higher values make the attack more robust but increase perturbation size.
            max_iter: Maximum number of iterations to run.
            verbose: Print progress updates.
            device: Device to run the attack on (e.g., CPU or GPU).
        """
        # Note: DeepFool doesn't directly use epsilon, so we just pass a dummy value
        # and doesn't use targeted or loss_fn parameters
        super().__init__(
            model,
            norm,
            eps=float("inf"),  # DeepFool finds minimal perturbation automatically
            targeted=False,  # DeepFool is inherently untargeted
            loss_fn="none",  # DeepFool doesn't use a loss function
            device=device,
            verbose=verbose,
        )

        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter

    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples using DeepFool.

        The algorithm works by:
        1. Linearizing the decision boundary around the current point
        2. Finding the closest hyperplane (minimum distance to decision boundary)
        3. Moving the point just beyond this hyperplane (controlled by overshoot)
        4. Repeating until misclassification or max iterations reached

        This implementation applies the attack separately to each sample in the batch.

        Args:
            inputs: Input images (clean samples) to perturb, of shape (N, C, H, W)
            targets: Not used directly by DeepFool (can be None).

        Returns:
            A tuple (adversarial_examples, metrics):
            - adversarial_examples: Tensor of same shape as inputs containing the adversarial examples
            - metrics: Dictionary with performance metrics including success rate, iterations, perturbation norm
        """
        # Reset metrics from previous runs
        self.reset_metrics()
        start_time = time.time()

        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)

        # Get and store original predictions for evaluation
        self.store_original_predictions(inputs)

        # Initialize adversarial examples with the original inputs
        adv_inputs = inputs.clone().detach()

        # Get original predictions to measure success
        with torch.no_grad():
            original_outputs = self.model(inputs)
            original_preds = original_outputs.argmax(dim=1)

        batch_size = inputs.shape[0]

        # Variables to track attack success
        successful = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Perturbation norms for metrics
        pert_norms = torch.zeros(batch_size, device=self.device)

        # Track number of iterations for each sample
        iterations_per_sample = torch.zeros(
            batch_size, dtype=torch.int, device=self.device
        )

        # Validate number of classes early on
        if self.num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {self.num_classes}")

        # For each input in the batch, perform DeepFool attack separately
        for idx in range(batch_size):
            # We'll work with normalized inputs with the model
            x = inputs[idx : idx + 1].clone().detach()

            # Also track the adversarial example in normalized space initially
            adv_x = x.clone().detach().requires_grad_(True)

            # Original prediction for this sample
            with torch.no_grad():
                f_x = self.model(x)
                orig_pred = f_x.argmax(dim=1).item()
                current_pred = orig_pred

            if self.verbose:
                print(f"Sample {idx}: Original prediction class {orig_pred}")

            # Initialize perturbation
            total_perturbation = torch.zeros_like(x)

            # Iteration counter
            iteration = 0

            # Continue until misclassification or max iterations
            while (current_pred == orig_pred) and (iteration < self.max_iter):
                # Get logits and increase iteration count
                logits = self.model(adv_x)
                iteration += 1

                # Get current prediction
                current_pred = logits.argmax(dim=1).item()

                if current_pred != orig_pred:
                    break

                # Handling out-of-range predictions more gracefully
                if orig_pred >= self.num_classes:
                    if self.verbose:
                        print(
                            f"Original prediction {orig_pred} exceeds specified num_classes {self.num_classes}"
                        )
                    # Use actual model output size instead
                    actual_class_count = logits.shape[1]
                    target_classes = [
                        i for i in range(actual_class_count) if i != orig_pred
                    ]

                # Always consider all classes or use a smarter selection strategy
                target_classes = [i for i in range(self.num_classes) if i != orig_pred]

                # For very large number of classes, use dynamic selection based on model confidence
                if self.num_classes > 100:
                    values, indices = torch.topk(
                        logits[0], k=min(20, self.num_classes - 1)
                    )
                    high_conf_classes = [
                        i.item() for i in indices if i.item() != orig_pred
                    ]
                    # Add some low confidence classes to avoid missing important boundaries
                    low_conf_indices = torch.topk(
                        logits[0], k=self.num_classes, largest=False
                    )[1]
                    low_conf_classes = [
                        i.item() for i in low_conf_indices[:5] if i.item() != orig_pred
                    ]
                    target_classes = list(set(high_conf_classes + low_conf_classes))

                # Ensure adv_x requires gradients for backward pass
                if not adv_x.requires_grad:
                    adv_x = adv_x.detach().requires_grad_(True)

                # Initialize variables for finding closest boundary
                min_dist = float("inf")
                closest_class = None
                grad_closest = None

                # Reuse the same tensor for gradient computation to avoid memory fragmentation
                grad_placeholder = torch.zeros_like(adv_x)

                # For each target class:
                for k in target_classes:
                    # Create copies with gradient tracking
                    adv_x_k = adv_x.detach().clone().requires_grad_(True)
                    adv_x_orig = adv_x.detach().clone().requires_grad_(True)

                    # Forward passes
                    logits_k = self.model(adv_x_k)
                    logits_orig = self.model(adv_x_orig)

                    # Backward passes
                    score_k = logits_k[0, k]
                    score_k.backward()

                    score_orig = logits_orig[0, orig_pred]
                    score_orig.backward()

                    # Get gradients
                    grad_k = adv_x_k.grad
                    grad_orig = adv_x_orig.grad

                    # Make sure gradients have correct shape
                    if grad_k is None or grad_orig is None:
                        continue

                    # Compute gradient difference - ensure correct shape
                    grad_diff = grad_orig - grad_k

                    # Compute score difference - use scalar values
                    score_diff = float(logits_orig[0, orig_pred] - logits_k[0, k])

                    # Compute distance to boundary for this class
                    if self.norm.lower() == "l2":
                        # Avoid division by zero
                        norm_grad_diff = torch.norm(grad_diff.flatten(), p=2)
                        if norm_grad_diff < 1e-7:
                            continue

                        dist = abs(score_diff) / norm_grad_diff
                    else:  # Linf
                        norm_grad_diff = torch.norm(grad_diff.flatten(), p=float("inf"))
                        if norm_grad_diff < 1e-7:
                            continue

                        dist = abs(score_diff) / norm_grad_diff

                    # Check if this is the closest boundary so far
                    if dist < min_dist:
                        min_dist = dist
                        closest_class = k
                        grad_closest = grad_diff

                # If we found a class to perturb toward
                if closest_class is not None:
                    # Compute perturbation direction (ensure correct shape)
                    if self.norm.lower() == "l2":
                        grad_norm = torch.norm(grad_closest.flatten(), p=2)
                        # Ensure no dimension issues in perturbation calculation
                        perturbation = min_dist * (grad_closest / grad_norm)
                    else:  # Linf
                        perturbation = min_dist * torch.sign(grad_closest)

                    # Make sure perturbation has same shape as adv_x before adding
                    if perturbation.shape != adv_x.shape:
                        perturbation = perturbation.reshape(adv_x.shape)

                    # Apply perturbation
                    adv_x = adv_x + perturbation
                    total_perturbation += perturbation

                    # Ensure valid image range - this needs to happen in normalized space
                    # to ensure the model can process it correctly
                    if self.mean is not None and self.std is not None:
                        # Clone first to avoid modifying the original
                        adv_x_denorm = adv_x.clone()

                        # Extract the standard deviation and mean for each channel
                        std_c = self.std.squeeze()  # Convert from [1,C,1,1] to [C]
                        mean_c = self.mean.squeeze()

                        # Apply denormalization properly with broadcasting
                        for c in range(adv_x.shape[1]):
                            # Properly apply the normalization with proper dimensions
                            adv_x_denorm[0, c] = adv_x[0, c] * std_c[c] + mean_c[c]

                        # Clip the values
                        adv_x_denorm = torch.clamp(adv_x_denorm, 0.0, 1.0)

                        # Re-normalize back
                        for c in range(adv_x.shape[1]):
                            adv_x[0, c] = (adv_x_denorm[0, c] - mean_c[c]) / std_c[c]
                    else:
                        # Without normalization, just use 0-1 range
                        adv_x = torch.clamp(adv_x, 0.0, 1.0)

                    # Detach and require gradients again
                    adv_x = adv_x.detach().requires_grad_(True)
                else:
                    # No suitable class found, break the loop
                    break

            # Apply overshoot to ensure crossing the boundary
            total_perturbation = (1 + self.overshoot) * total_perturbation

            # Apply the final perturbation to the normalized input
            adv_x_final = torch.clamp(x + total_perturbation, 0.0, 1.0)

            # Record the final adversarial example in the normalized space
            adv_inputs[idx] = adv_x_final.detach()

            # Record metrics for this sample
            iterations_per_sample[idx] = iteration

            # Check if attack was successful
            with torch.no_grad():
                final_pred = self.model(adv_x_final).argmax(dim=1).item()
                successful[idx] = final_pred != orig_pred

            # Record perturbation norm
            if self.norm.lower() == "l2":
                pert_norms[idx] = torch.norm(total_perturbation.flatten(), p=2)
            else:  # Linf
                pert_norms[idx] = torch.norm(
                    total_perturbation.flatten(), p=float("inf")
                )

            # Early stopping based on perturbation norm threshold
            max_perturbation_threshold = 0.2  # Or any other appropriate threshold
            current_pert_norm = torch.norm(
                total_perturbation.flatten(),
                p=2 if self.norm.lower() == "l2" else float("inf"),
            )
            if current_pert_norm > max_perturbation_threshold:
                if self.verbose:
                    print(
                        f"Stopping early: perturbation norm {current_pert_norm:.4f} exceeds threshold {max_perturbation_threshold}"
                    )
                break

        # Calculate success rate and other metrics
        success_rate = successful.float().mean().item() * 100
        avg_iterations = iterations_per_sample.float().mean().item()

        # Update total time and iterations
        self.total_time = time.time() - start_time
        self.total_iterations = int(iterations_per_sample.sum().item())

        # Compile metrics
        metrics = {
            **self.get_metrics(),
            "success_rate": success_rate,
            "avg_iterations": avg_iterations,
            "avg_perturbation_norm": pert_norms.mean().item(),
        }

        return adv_inputs, metrics
