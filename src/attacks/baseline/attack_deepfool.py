#!/usr/bin/env python

"""DeepFool adversarial attack implementation.

Some code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import time
import torch

from .attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 30)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        early_stopping (bool): whether to stop early when all samples are fooled. (Default: True)
        top_k_classes (int): number of top classes to consider when finding closest boundary. (Default: 5)
    Shape:
        - images: :math:`(N, C, H, W)` normalized images with ImageNet mean/std
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)` normalized adversarial images.
    Examples::
        >>> attack = DeepFool(model, steps=30, overshoot=0.02, early_stopping=True, top_k_classes=5)
        >>> adv_images = attack(images, labels)

    Note:
        The attack operates in normalized input space. All perturbations and boundary
        calculations are performed in the normalized domain, and results are properly
        bounded to valid normalized image values.
    """

    def __init__(
        self, model, steps=30, overshoot=0.02, early_stopping=True, top_k_classes=5
    ):
        """Initialize DeepFool attack.

        Args:
            model: Target model to attack
            steps: Maximum number of iterations for finding adversarial examples
            overshoot: Parameter to enhance the noise (default: 0.02)
            early_stopping: Whether to stop early when all samples are fooled (default: True)
            top_k_classes: Number of top classes to consider when finding closest boundary (default: 5)
        """
        super().__init__("DeepFool", model)
        self.steps = steps  # Maximum number of iterations
        self.overshoot = overshoot  # Parameter to enhance the noise
        # Whether to stop early when all samples are fooled
        self.early_stopping = early_stopping
        # Only consider top-k classes for efficiency
        self.top_k_classes = top_k_classes
        # DeepFool only supports untargeted attacks
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""Overridden."""
        # Call the main implementation and return only the adversarial images
        adv_images, _ = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        r"""Optimized batch implementation of DeepFool."""
        # Start timer for performance tracking
        start_time = time.time()

        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.size(0)

        if self.targeted:
            return "Targeted attacks are not supported for DeepFool"

        # Calculate normalized min/max bounds for valid pixel values
        # This ensures perturbations keep the image within valid range
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Process each sample individually to avoid inplace operation issues
        # Practical consideration: batch processing with individual handling
        # avoids gradient accumulation issues in PyTorch
        adv_images = []
        sample_iterations = torch.zeros(batch_size, device=self.device)
        target_preds = torch.zeros_like(labels)

        # Reset gradient call counter for accurate tracking
        prev_grad_calls = self.total_gradient_calls

        # Process each image individually
        for i in range(batch_size):
            x = images[i : i + 1].clone()
            y = labels[i : i + 1]

            # Use the individual implementation for each sample
            fooled, iters, adv_x, target_class = self._attack_single_sample(
                x, y, self.steps, min_bound, max_bound
            )

            # Update metrics
            sample_iterations[i] = iters
            target_preds[i] = target_class
            adv_images.append(adv_x)

            # Properly track iterations in the main counter
            self.total_iterations += iters

        # Concatenate results
        adv_batch = torch.cat(adv_images, dim=0)

        # Calculate number of fooled samples
        with torch.no_grad():
            preds = self.get_logits(adv_batch).argmax(dim=1)
            fooled_mask = preds != labels
            fooled_count = fooled_mask.sum().item()

        # Calculate perturbation metrics only for successful attacks
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_batch, fooled_mask
        )

        # Record metrics
        avg_iters = sample_iterations.float().mean().item()
        grad_calls = self.total_gradient_calls - prev_grad_calls
        avg_grad_calls = grad_calls / batch_size

        # Print stats
        print(
            f"DeepFool stats: avg iterations={avg_iters:.2f}, fooled {fooled_count}/{batch_size} samples, avg gradient calls={avg_grad_calls:.2f}"
        )
        print(
            f"DeepFool metrics - L2: {perturbation_metrics['l2_norm']:.4f}, L∞: {perturbation_metrics['linf_norm']:.4f}, SSIM: {perturbation_metrics['ssim']:.4f}"
        )

        # Measure and record time taken
        end_time = time.time()
        self.total_time += end_time - start_time

        return adv_batch, target_preds

    def _attack_single_sample(self, x, y, steps, min_bound, max_bound):
        """Attack a single sample with DeepFool algorithm.

        Args:
            x: Single input image (1,C,H,W)
            y: True label
            steps: Maximum iterations
            min_bound: Minimum valid pixel values
            max_bound: Maximum valid pixel values

        Returns:
            (success, iterations, adv_image, target_class)
        """
        # Get true label as scalar
        true_label = y.item()

        # Initialize
        adv_x = x.clone()  # Current adversarial example x_k
        iterations = 0
        current_pred = None
        target_class = -1
        fooled = False

        # Track perturbation - this accumulates the total perturbation δ
        total_perturbation = torch.zeros_like(x)

        # Main iteration loop - implements the iterative DeepFool algorithm
        for i in range(steps):
            # If already fooled, exit - practical early stopping
            if fooled:
                break

            # Create input that requires gradient
            adv_x_i = adv_x.clone().detach().requires_grad_(True)

            # Forward pass to get logits f(x_k)
            # NOTE: We don't call self.get_logits() to avoid duplicate gradient call counting
            # since we're manually tracking gradients below
            logits = self.model(adv_x_i)

            # Check current prediction
            current_pred = logits[0].argmax().item()

            # If already misclassified, we're done
            # This corresponds to checking k̂(x + δ) ≠ k̂(x) in the paper
            if current_pred != true_label:
                fooled = True
                target_class = current_pred
                break

            # Get top-k classes (exclude true class)
            # Practical optimization: only check the most likely classes
            # instead of all possible classes (important for ImageNet)
            values, indices = torch.topk(logits[0], k=self.top_k_classes + 1)

            # Filter out true class
            other_classes = []
            for idx in indices:
                if idx.item() != true_label and len(other_classes) < self.top_k_classes:
                    other_classes.append(idx.item())

            # Ensure we have at least one class
            if not other_classes:
                # Use the second highest class if true_label is the highest
                if indices[0].item() == true_label:
                    other_classes = [indices[1].item()]
                else:
                    other_classes = [indices[0].item()]

            # Get gradient of true class - ∇f_{k̂(x)}(x)
            true_logit = logits[0, true_label]
            adv_x_i.grad = None
            true_logit.backward(retain_graph=True)
            # Count as one gradient call
            self.track_gradient_calls(1)
            grad_true = adv_x_i.grad.clone()

            # Find closest boundary - implements the minimization part of:
            # r_k = (min_{j≠k̂(x)} |f_j(x) - f_{k̂(x)}(x)| / ‖∇f_j(x) - ∇f_{k̂(x)}(x)‖²) · (∇f_{j*}(x) - ∇f_{k̂(x)}(x))
            min_distance = float("inf")
            closest_grad = None
            closest_diff = None

            # Check each class - finding j* that minimizes the distance
            for k_idx, k in enumerate(other_classes):
                adv_x_i.grad = None
                # Only retain graph if not the last class
                retain_graph = k_idx < len(other_classes) - 1
                logits[0, k].backward(retain_graph=retain_graph)
                # Count as one gradient call
                self.track_gradient_calls(1)

                # Compute w_k (difference in gradients) - ∇f_j(x) - ∇f_{k̂(x)}(x)
                grad_k = adv_x_i.grad.clone()
                grad_diff = grad_k - grad_true

                # Compute f_k (difference in logits) - f_j(x) - f_{k̂(x)}(x)
                f_diff = logits[0, k].item() - true_logit.item()

                # Compute distance to boundary - |f_j(x) - f_{k̂(x)}(x)| / ‖∇f_j(x) - ∇f_{k̂(x)}(x)‖
                grad_norm = torch.norm(grad_diff.flatten())
                if grad_norm < 1e-6:  # Numerical stability check
                    continue

                dist = abs(f_diff) / grad_norm

                # Update if this is the closest boundary - finding j*
                if dist < min_distance:
                    min_distance = dist
                    closest_grad = grad_diff
                    closest_diff = f_diff

            # If no valid boundary found, break
            # Practical consideration for numerical stability
            if closest_grad is None:
                break

            # Compute perturbation - this is r_k in the paper
            # r_k = (|f_j(x) - f_{k̂(x)}(x)| / ‖∇f_j(x) - ∇f_{k̂(x)}(x)‖²) · (∇f_j(x) - ∇f_{k̂(x)}(x))
            pert_norm = torch.norm(closest_grad.flatten())
            perturbation = (
                abs(closest_diff) / (pert_norm**2 + 1e-8) * closest_grad
            )  # 1e-8 for stability

            # Add perturbation - implements x_{k+1} = x_k + r_k
            adv_x = torch.clamp(adv_x + perturbation, min=min_bound, max=max_bound)
            total_perturbation = adv_x - x

            # Update iteration count
            iterations += 1

        # Apply overshoot to final perturbation
        # Practical enhancement from the paper: overshoot slightly beyond the boundary
        # to ensure successful misclassification (helps with non-linear boundaries)
        adv_x = torch.clamp(
            x + (1 + self.overshoot) * total_perturbation, min=min_bound, max=max_bound
        )

        # Return results
        return fooled, iterations, adv_x, target_class
