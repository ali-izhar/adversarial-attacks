"""DeepFool adversarial attack implementation.

This file implements the DeepFool attack, which is an iterative method that efficiently
finds the minimal perturbation needed to cross the decision boundary. The algorithm
works by approximating the classifier with a linear model at each iteration and finding
the closest decision boundary to move the input across.

Key features:
- Produces minimal perturbations compared to simple gradient methods
- Iteratively approximates the decision boundary
- Naturally produces more imperceptible adversarial examples
- Supports both L2 and Linf norms
- Focuses on finding the closest decision boundary

Expected inputs:
- A neural network model to attack
- Input samples (images) to perturb
- Number of classes for the classification task
- Maximum number of iterations

Expected outputs:
- Adversarial examples that aim to fool the model
- Performance metrics including success rate
"""

import torch
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack


class DeepFool(BaseAttack):
    """
    DeepFool adversarial attack.

    This attack iteratively finds the nearest decision boundary and pushes the
    input just enough to cross that boundary. It produces more minimal perturbations
    compared to simpler methods like FGSM or PGD.
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
            overshoot: Parameter to overshoot the decision boundary.
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

        The attack performs the following steps for each input image:
          1. For each iteration:
             a. Compute the gradient of each output class with respect to the input.
             b. Find the closest decision boundary.
             c. Move the input towards that boundary with minimal perturbation.
          2. Apply the perturbation with a small overshoot to ensure crossing the boundary.
          3. Return the adversarial examples along with metrics.

        Args:
            inputs: Input images (clean samples) to perturb.
            targets: Not used by DeepFool (can be None).

        Returns:
            A tuple (adversarial_examples, metrics).
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

        # For each input in the batch, perform DeepFool attack separately
        for idx in range(batch_size):
            x = inputs[idx : idx + 1].clone().detach().requires_grad_(True)
            adv_x = x.clone()

            # Original prediction for this sample
            with torch.no_grad():
                f_x = self.model(x)
                orig_pred = f_x.argmax(dim=1).item()
                current_pred = orig_pred

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

                # Get all classes except the original prediction
                other_classes = list(range(self.num_classes))
                other_classes.remove(orig_pred)

                # For limited number of classes, use all; otherwise sample subset
                if self.num_classes <= 10:
                    target_classes = other_classes
                else:
                    # Choose the top k classes (closest decision boundaries)
                    # that are not the original class
                    values, indices = torch.topk(logits[0], k=10)
                    target_classes = [
                        i.item() for i in indices if i.item() != orig_pred
                    ]
                    # If we have fewer than 9 classes, add some random ones
                    if len(target_classes) < 9:
                        additional = np.random.choice(
                            other_classes,
                            size=min(9 - len(target_classes), len(other_classes)),
                            replace=False,
                        )
                        target_classes.extend(additional)

                # Initialize variables for finding closest boundary
                min_dist = float("inf")
                closest_class = None
                grad_closest = None

                # For each target class
                for k in target_classes:
                    # Zero gradient
                    self.model.zero_grad()
                    if adv_x.grad is not None:
                        adv_x.grad.zero_()

                    # Get score for class k
                    score_k = logits[0, k]

                    # Compute gradient of score_k with respect to input
                    score_k.backward(retain_graph=True)
                    grad_k = adv_x.grad.clone()

                    # Get score for original class
                    score_orig = logits[0, orig_pred]

                    # Compute gradient of score_orig with respect to input
                    self.model.zero_grad()
                    adv_x.grad.zero_()
                    score_orig.backward(retain_graph=True)
                    grad_orig = adv_x.grad.clone()

                    # Compute difference in gradients
                    grad_diff = grad_orig - grad_k

                    # Compute difference in scores
                    score_diff = score_orig - score_k

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
                    # Compute perturbation direction
                    if self.norm.lower() == "l2":
                        grad_norm = torch.norm(grad_closest.flatten(), p=2)
                        perturbation = (min_dist / grad_norm) * grad_closest
                    else:  # Linf
                        perturbation = min_dist * torch.sign(grad_closest)

                    # Apply perturbation
                    adv_x = adv_x + perturbation
                    total_perturbation += perturbation

                    # Ensure valid image range
                    adv_x = torch.clamp(adv_x, 0.0, 1.0)

                    # Detach and require gradients again
                    adv_x = adv_x.detach().requires_grad_(True)
                else:
                    # No suitable class found, break the loop
                    break

            # Apply overshoot to ensure crossing the boundary
            total_perturbation = (1 + self.overshoot) * total_perturbation
            adv_x = torch.clamp(x + total_perturbation, 0.0, 1.0)

            # Record the final adversarial example
            adv_inputs[idx] = adv_x.detach()

            # Record metrics for this sample
            iterations_per_sample[idx] = iteration

            # Check if attack was successful
            with torch.no_grad():
                final_pred = self.model(adv_x).argmax(dim=1).item()
                successful[idx] = final_pred != orig_pred

            # Record perturbation norm
            if self.norm.lower() == "l2":
                pert_norms[idx] = torch.norm(total_perturbation.flatten(), p=2)
            else:  # Linf
                pert_norms[idx] = torch.norm(
                    total_perturbation.flatten(), p=float("inf")
                )

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
