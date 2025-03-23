"""Carlini & Wagner (C&W) L2 adversarial attack implementation.

This file implements the C&W L2 attack, which is an optimization-based method that
explicitly finds the minimal L2 perturbation needed to misclassify an input. It uses
a carefully designed loss function with box constraints and produces perturbations
that are minimal in L2 norm while being highly effective.

Key features:
- Produces minimal L2 perturbation for misclassification
- Uses binary search to find optimal tradeoff between perturbation size and attack success
- Incorporates a confidence parameter to control robustness of adversarial examples
- Very effective but computationally more expensive than simpler methods
- Often produces state-of-the-art imperceptible adversarial examples

Expected inputs:
- A neural network model to attack
- Input samples (images) to perturb
- Target labels (for targeted attacks) or true labels (for untargeted attacks)
- Confidence parameter to control attack strength

Expected outputs:
- Adversarial examples that aim to fool the model
- Performance metrics including success rate and perturbation magnitude
"""

import torch
import time
from typing import Tuple, Dict, Any, Optional

from .base import BaseAttack


class CW(BaseAttack):
    """
    Carlini & Wagner (C&W) L2 adversarial attack.

    This attack uses an optimization-based approach to find minimal perturbations
    that fool the target model. It's considered one of the most effective attacks
    for generating imperceptible adversarial examples, but is computationally more
    expensive than simpler gradient-based methods.
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
            confidence: Confidence parameter for adversarial examples.
            c_init: Initial value of the constant c.
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
        """Convert from image space to tanh space for optimization."""
        # Normalize to [-1, 1]
        x_tanh = (x - self.boxmul) / self.boxplus
        # Apply arctanh (with clipping for numerical stability)
        return torch.atanh(torch.clamp(x_tanh, -0.99999, 0.99999))

    def _to_image_space(self, x_tanh: torch.Tensor) -> torch.Tensor:
        """Convert from tanh space back to image space."""
        # Apply tanh to ensure range [-1, 1]
        x_t = torch.tanh(x_tanh)
        # Scale back to image space
        return self.boxplus * x_t + self.boxmul

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

        The loss combines two terms:
        1. The classification loss that encourages misclassification
        2. The distance loss that minimizes the L2 perturbation

        Args:
            logits: The model's output logits
            target_labels: The target class indices
            w: The weighting factor for the distance term
            x_origin: The original images
            x_adv_tanh: The adversarial examples in tanh space

        Returns:
            A tuple (total_loss, classification_loss, distance_loss)
        """
        # Convert adversarial examples from tanh space to image space
        x_adv = self._to_image_space(x_adv_tanh)

        # Compute L2 distance between original and adversarial examples
        l2_dist = torch.norm((x_adv - x_origin).view(x_origin.shape[0], -1), p=2, dim=1)

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

        The attack performs the following steps:
          1. Use binary search to find the optimal c value (balancing attack success and perturbation size).
          2. For each c, optimize the adversarial example using Adam over the tanh-transformed space.
          3. Return the adversarial example with the smallest perturbation that successfully fools the model.

        Args:
            inputs: Input images (clean samples) to perturb.
            targets: Target labels (for targeted attacks) or true labels (for untargeted attacks).

        Returns:
            A tuple (adversarial_examples, metrics).
        """
        # Reset metrics from previous runs
        self.reset_metrics()
        start_time = time.time()

        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get and store original predictions for evaluation
        self.store_original_predictions(inputs)

        batch_size = inputs.shape[0]

        # For untargeted attack, use original labels as targets (to move away from)
        if not self.targeted:
            with torch.no_grad():
                # Get true labels to use as targets for untargeted attack
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
                # Forward pass
                x_adv = self._to_image_space(x_adv_tanh)
                logits = self.model(x_adv)

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
        self.total_iterations = self.max_iter * self.binary_search_steps

        # Compile metrics
        metrics = {
            **self.get_metrics(),
            "success_rate": success_rate,
            "avg_l2_norm": avg_l2,
            "best_c": best_c.mean().item(),
        }

        return best_adv, metrics
