"""Projected Gradient Descent (PGD) adversarial attack implementation."""

import time
import torch
import torch.nn as nn
import numpy as np

from .baseline.attack import Attack
from .optimize.pgd import PGDOptimizer


class PGD(Attack):
    """
    Simplified Projected Gradient Descent (PGD) adversarial attack.

    This attack uses the PGDOptimizer to generate adversarial examples by iteratively taking
    steps in the gradient direction and then projecting the perturbed input back onto a
    constraint set defined by a norm ball (L2 or Linf).
    """

    def __init__(
        self,
        model,
        norm: str = "L2",  # As described in the paper, PGD supports both L2 and Linf norms
        eps: float = 0.5,  # The perturbation budget (epsilon) as described in the paper
        n_iterations: int = 100,  # Maximum iterations T in the algorithm
        step_size: float = 0.1,  # Step size alpha in the gradient update equations
        loss_fn: str = "cross_entropy",  # Loss function to maximize/minimize
        rand_init: bool = True,  # Random initialization as described in the algorithm
        early_stopping: bool = True,  # Implements the early stopping condition from the algorithm
        verbose: bool = False,
    ):
        """
        Initialize the PGD attack with minimal parameters.

        Args:
            model: The model to attack
            norm: Norm for the perturbation constraint ('L2' or 'Linf')
            eps: Maximum allowed perturbation magnitude
            n_iterations: Maximum number of iterations to run
            step_size: Step size for gradient updates
            loss_fn: Loss function to use ('cross_entropy', 'margin', or 'carlini_wagner')
            rand_init: Whether to initialize with random noise
            early_stopping: Whether to stop when adversarial examples are found
            verbose: Whether to print progress information
        """
        # Initialize the base Attack class
        super().__init__("PGD", model)

        # Record attack parameters
        self.norm = norm
        self.orig_eps = eps  # Store original epsilon specified in [0,1] space

        # Scale epsilon to normalized space by dividing by ImageNet std
        mean_std = self.std.clone().detach().mean().item()
        self.eps = (
            eps / mean_std
        )  # Scaling to normalized space for consistent perturbation

        self.n_iterations = n_iterations

        # Also scale step size for consistent gradient updates in normalized space
        self.step_size = step_size / mean_std

        self.loss_fn_type = loss_fn
        self.rand_init = (
            rand_init  # Corresponds to δ₀ ~ U(-0.01,0.01)ⁿ in the algorithm
        )
        self.early_stopping = early_stopping  # Implements the early stopping condition
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer for PGD - this will handle the core optimization algorithm
        self.optimizer = PGDOptimizer(
            norm=norm,  # Corresponds to L2 or Linf norm as shown in paper
            eps=self.eps,  # Pass the scaled epsilon for normalized space
            n_iterations=n_iterations,  # Corresponds to T in the algorithm
            step_size=self.step_size,  # Pass the scaled step size
            rand_init=rand_init,  # Implements random initialization of δ₀
            early_stopping=early_stopping,  # Implements early stopping condition
            verbose=verbose,
            maximize=True,  # Default for untargeted attacks, will be set in forward()
        )

    def compute_skimage_ssim(self, img1, img2):
        """
        Compute SSIM using scikit-image implementation.

        Args:
            img1: First tensor of images [N,C,H,W]
            img2: Second tensor of images [N,C,H,W]

        Returns:
            Average SSIM value across the batch
        """
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        # Move tensors to CPU and convert to numpy
        img1_np = img1.detach().cpu().permute(0, 2, 3, 1).numpy()
        img2_np = img2.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(img1_np.shape[0]):
            # Convert to [0,1] range - using shared scaling to preserve differences
            # First find global min/max across both images
            min_val = min(img1_np[i].min(), img2_np[i].min())
            max_val = max(img1_np[i].max(), img2_np[i].max())

            # Apply same normalization to both images to maintain relative differences
            scale = max_val - min_val
            if scale < 1e-7:  # Avoid division by zero
                scale = 1.0

            img1_norm = (img1_np[i] - min_val) / scale
            img2_norm = (img2_np[i] - min_val) / scale

            # Multi-channel SSIM - compute for each channel separately to be more sensitive
            ssim_val = 0
            for c in range(img1_norm.shape[2]):
                ssim_val += ssim(
                    img1_norm[:, :, c],
                    img2_norm[:, :, c],
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )

            # Average across channels
            ssim_val /= img1_norm.shape[2]
            ssim_values.append(ssim_val)

        return np.mean(ssim_values)

    def refine_perturbation(
        self, original_images, adv_images, labels, refinement_steps=10
    ):
        """
        Refine adversarial examples to minimize perturbation while maintaining misclassification.

        Args:
            original_images: Original clean images
            adv_images: Adversarial examples to refine
            labels: True labels for untargeted attacks, target labels for targeted
            refinement_steps: Number of binary search steps for refinement

        Returns:
            Refined adversarial examples with smaller perturbation
        """
        batch_size = original_images.size(0)
        refined_images = adv_images.clone()

        # For targeted attacks, get target labels
        if self.targeted:
            target_labels = self.get_target_label(original_images, labels)
        else:
            target_labels = labels

        # Initialize alpha for each image (controls interpolation between original and adversarial)
        # alpha=0 means pure adversarial, alpha=1 means pure original
        alphas = torch.zeros(batch_size, device=self.device)

        # Identify initially successful adversarial examples
        with torch.no_grad():
            outputs = self.get_logits(adv_images)
            if self.targeted:
                # For targeted attacks, success means predicting the target class
                initial_success = outputs.argmax(dim=1) == target_labels
            else:
                # For untargeted attacks, success means not predicting the true class
                initial_success = outputs.argmax(dim=1) != labels

        # Only refine successful examples
        successful_indices = torch.where(initial_success)[0]

        if len(successful_indices) == 0:
            if self.verbose:
                print("No successful adversarial examples to refine")
            return refined_images

        if self.verbose:
            print(
                f"Refining {len(successful_indices)} successful adversarial examples..."
            )

        # Binary search to find the minimal perturbation for each example
        for step in range(refinement_steps):
            # Create interpolated images
            current_alphas = alphas.view(-1, 1, 1, 1)
            interpolated = (
                current_alphas * original_images + (1 - current_alphas) * refined_images
            )

            # Check if still adversarial
            with torch.no_grad():
                outputs = self.get_logits(interpolated)
                if self.targeted:
                    # For targeted attacks, success means predicting the target class
                    still_successful = outputs.argmax(dim=1) == target_labels
                else:
                    # For untargeted attacks, success means not predicting the true class
                    still_successful = outputs.argmax(dim=1) != labels

            # Update alphas with binary search
            for i in successful_indices:
                if still_successful[i]:
                    # If still successful, try moving closer to original
                    alphas[i] = alphas[i] + (1 - alphas[i]) / 2
                    # Update the refined image with this better version
                    refined_images[i] = interpolated[i]
                else:
                    # If not successful, move back toward adversarial
                    alphas[i] = alphas[i] / 2

        # Calculate improvement in perturbation metrics
        orig_l2 = (
            torch.norm((adv_images - original_images).view(batch_size, -1), dim=1)
            .mean()
            .item()
        )
        refined_l2 = (
            torch.norm((refined_images - original_images).view(batch_size, -1), dim=1)
            .mean()
            .item()
        )

        if self.verbose:
            l2_reduction = (orig_l2 - refined_l2) / orig_l2 * 100
            print(f"Refinement reduced L2 perturbation by {l2_reduction:.2f}%")

        return refined_images

    def forward(self, images, labels):
        """
        Simplified PGD attack implementation with binary search for minimal perturbation.

        Args:
            images: Input images to perturb
            labels: Target labels (true labels for untargeted, target labels for targeted)

        Returns:
            Adversarial examples
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Note start time for performance tracking
        start_time = time.time()

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # Set optimize direction based on attack mode
        # For untargeted attacks (maximize=True), we want to maximize the loss to move away from true class
        # For targeted attacks (maximize=False), we want to minimize the loss to move toward target class
        self.optimizer.maximize = not self.targeted

        # Calculate normalized min/max bounds for valid pixel ranges after denormalization
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Select appropriate loss function based on configuration
        if self.loss_fn_type == "cross_entropy":
            # Cross-entropy loss function as described in the paper section on optimization formulation
            ce_loss = nn.CrossEntropyLoss(reduction="none")

            def loss_fn(outputs, targets, targeted=False):
                if targeted:
                    return -ce_loss(outputs, targets)  # Minimize for targeted attacks
                else:
                    return ce_loss(outputs, targets)  # Maximize for untargeted attacks

        elif self.loss_fn_type == "margin":
            # Margin loss focuses on the difference between the logit of the correct class
            # and the logit of the most likely incorrect class, as discussed in the paper

            def loss_fn(outputs, targets, targeted=False):
                if targeted:
                    # For targeted attacks, maximize the margin between target class and others
                    target_logits = outputs.gather(1, targets.view(-1, 1)).squeeze(1)
                    max_other_logits = outputs.clone()
                    max_other_logits.scatter_(1, targets.view(-1, 1), float("-inf"))
                    max_other_logits = max_other_logits.max(1)[0]
                    margin = max_other_logits - target_logits
                    return margin  # For targeted, this will be minimized
                else:
                    # For untargeted attacks, minimize the margin between true class and others
                    correct_logits = outputs.gather(1, targets.view(-1, 1)).squeeze(1)
                    max_other_logits = outputs.clone()
                    max_other_logits.scatter_(1, targets.view(-1, 1), float("-inf"))
                    max_other_logits = max_other_logits.max(1)[0]
                    margin = correct_logits - max_other_logits
                    return -margin  # For untargeted, this will be maximized

        elif self.loss_fn_type == "carlini_wagner":
            # Carlini-Wagner loss function with confidence parameter κ, similar to the targeted loss
            # formulation in the paper's optimization section
            confidence = (
                0.0  # Corresponds to κ in the paper's targeted attack formulation
            )

            def loss_fn(outputs, targets, targeted=False):
                if targeted:
                    # Target class should have larger logit than all others
                    target_logits = outputs.gather(1, targets.view(-1, 1)).squeeze(1)
                    max_other_logits = outputs.clone()
                    max_other_logits.scatter_(1, targets.view(-1, 1), float("-inf"))
                    max_other_logits = max_other_logits.max(1)[0]
                    cw_loss = torch.clamp(
                        max_other_logits - target_logits + confidence, min=0
                    )
                    return cw_loss
                else:
                    # True class should have smaller logit than others
                    correct_logits = outputs.gather(1, targets.view(-1, 1)).squeeze(1)
                    max_other_logits = outputs.clone()
                    max_other_logits.scatter_(1, targets.view(-1, 1), float("-inf"))
                    max_other_logits = max_other_logits.max(1)[0]
                    cw_loss = -torch.clamp(
                        correct_logits - max_other_logits + confidence, min=0
                    )
                    return cw_loss

        # Define the gradient function for the optimizer - computes ∇_δ L(f(x + δ_t), y)
        # This corresponds to the gradient computation step in the algorithm
        def gradient_fn(x):
            x.requires_grad_(True)
            outputs = self.get_logits(x)
            curr_labels = target_labels[: x.size(0)]
            loss_values = loss_fn(outputs, curr_labels, self.targeted)
            mean_loss = loss_values.mean()
            grad = torch.autograd.grad(mean_loss, x)[0]  # Compute gradient w.r.t. input
            return grad

        # Define success function for early stopping - checks if attack has succeeded
        # This implements the early stopping condition from the algorithm
        def success_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)

                # Get appropriate labels for the current batch
                curr_labels = (
                    target_labels[: x.size(0)] if self.targeted else labels[: x.size(0)]
                )

                if self.targeted:
                    # Attack succeeds if model predicts target class
                    return outputs.argmax(dim=1) == curr_labels
                else:
                    # Attack succeeds if model predicts any class other than the true label
                    return outputs.argmax(dim=1) != curr_labels

        # Run binary search to find the minimal perturbation that still succeeds
        # Store the original epsilon to restore it later
        original_eps = self.eps

        # First run with full epsilon to see if the attack succeeds
        self.optimizer.eps = original_eps

        # Override the PGDOptimizer's clamping with normalized bounds
        # We'll use a modified version of optimize that respects these bounds
        adv_images, initial_metrics = self.optimize_with_normalized_bounds(
            x_init=images,
            gradient_fn=gradient_fn,
            success_fn=success_fn,
            x_original=images,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Evaluate success of initial attack
        initial_success_rate, initial_success_mask, _ = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # If attack is successful and we have enough successful examples, try binary search
        # to find minimal perturbation
        if initial_success_rate > 50 and torch.sum(initial_success_mask) >= 2:
            if self.verbose:
                print(
                    f"Initial attack success rate: {initial_success_rate:.2f}%. Refining perturbation..."
                )

            # Binary search parameters
            low_eps = original_eps * 0.01  # Start with 1% of original epsilon
            high_eps = original_eps
            best_eps = high_eps
            best_adv_images = adv_images.clone()

            # Track successful indices to only optimize those
            successful_indices = torch.where(initial_success_mask)[0]

            # Binary search for minimal epsilon
            for search_step in range(5):  # 5 binary search steps
                # Try middle epsilon
                mid_eps = (low_eps + high_eps) / 2
                self.optimizer.eps = mid_eps

                # Only attack already successful examples
                if successful_indices.numel() > 0:
                    curr_images = images[successful_indices]
                    curr_labels = labels[successful_indices]
                    curr_target_labels = (
                        target_labels[successful_indices]
                        if self.targeted
                        else curr_labels
                    )

                    # Run attack with current epsilon
                    curr_adv_images, _ = self.optimize_with_normalized_bounds(
                        x_init=curr_images,
                        gradient_fn=gradient_fn,
                        success_fn=success_fn,
                        x_original=curr_images,
                        min_bound=min_bound,
                        max_bound=max_bound,
                    )

                    # Check success on these examples
                    curr_success_rate, curr_success_mask, _ = (
                        self.evaluate_attack_success(
                            curr_images, curr_adv_images, curr_labels
                        )
                    )

                    # If still successful for most examples, try smaller epsilon
                    if curr_success_rate >= 80:
                        high_eps = mid_eps
                        best_eps = mid_eps

                        # Update best adversarial examples for successful indices
                        for i, orig_idx in enumerate(successful_indices):
                            if curr_success_mask[i]:
                                best_adv_images[orig_idx] = curr_adv_images[i]
                    else:
                        # If not successful enough, try larger epsilon
                        low_eps = mid_eps

            # Set epsilon back to the best value found
            self.eps = best_eps
            self.optimizer.eps = best_eps
            adv_images = best_adv_images

            if self.verbose:
                print(
                    f"Binary search found optimal epsilon: {best_eps:.6f} (original: {original_eps:.6f})"
                )
        else:
            # If initial attack wasn't successful enough, keep the original result
            self.eps = original_eps
            self.optimizer.eps = original_eps

        # Apply post-processing refinement to further reduce perturbation with a focus on
        # maintaining high SSIM values (visual similarity)
        refined_adv_images = self.refine_perturbation_with_bounds(
            original_images=images,
            adv_images=adv_images,
            labels=labels,
            refinement_steps=10,  # Fewer refinement steps to avoid over-processing
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Check if refinement maintained attack success
        refined_success_rate, refined_success_mask, _ = self.evaluate_attack_success(
            images, refined_adv_images, labels
        )

        # Calculate SSIM improvement
        original_ssim = self.compute_skimage_ssim(images, adv_images)
        refined_ssim = self.compute_skimage_ssim(images, refined_adv_images)

        # Use refined images only if they maintain a reasonable success rate
        # (at least 95% of original success) AND improve SSIM
        if (
            refined_success_rate >= 0.95 * initial_success_rate
            and refined_ssim > original_ssim
        ):
            adv_images = refined_adv_images
            if self.verbose:
                print(
                    f"Using refined adversarial examples with {refined_success_rate:.2f}% success rate"
                    f" and SSIM {refined_ssim:.4f} (original: {original_ssim:.4f})"
                )
        elif self.verbose:
            print(
                f"Refinement didn't improve overall quality (success rate: {refined_success_rate:.2f}% vs {initial_success_rate:.2f}%, "
                f"SSIM: {refined_ssim:.4f} vs {original_ssim:.4f}), using original adversarial examples"
            )

        # Update attack metrics
        self.total_iterations += initial_metrics["iterations"] * images.size(0)

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - start_time

        # Ensure outputs are properly clamped to valid normalized range
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, _ = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Compute perturbation metrics - L2 and L∞ as described in the paper
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, success_mask
        )

        # Update attack_success_count for metrics
        self.attack_success_count += success_mask.sum().item()
        self.total_samples += images.size(0)

        # Log metrics if verbose
        if self.verbose:
            print(f"Attack complete: {success_rate:.2f}% success rate")
            print(
                f"Metrics: L2={perturbation_metrics['l2_norm']:.6f}, "
                f"L∞={perturbation_metrics['linf_norm']:.6f}, "
                f"SSIM={perturbation_metrics['ssim']:.4f}"
            )
            print(f"Iterations: {initial_metrics['iterations']}")
            print(f"Time per sample: {initial_metrics['time_per_sample']*1000:.2f}ms")

        return adv_images

    def optimize_with_normalized_bounds(
        self,
        x_init,
        gradient_fn,
        success_fn=None,
        x_original=None,
        min_bound=None,
        max_bound=None,
    ):
        """
        Modified version of PGDOptimizer.optimize that uses normalized bounds.

        This wraps the optimizer.optimize method to apply proper normalized bounds
        at each iteration rather than simple [0,1] clamping.
        """

        # Run the original optimizer with our normalized bounds
        adv_images, metrics = self.optimizer.optimize(
            x_init=x_init,
            gradient_fn=gradient_fn,
            success_fn=success_fn,
            x_original=x_original,
            min_bound=min_bound,  # Pass the normalized bounds to the optimizer
            max_bound=max_bound,  # Pass the normalized bounds to the optimizer
        )

        # Ensure final output is properly clamped
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound)

        return adv_images, metrics

    def refine_perturbation_with_bounds(
        self,
        original_images,
        adv_images,
        labels,
        refinement_steps=10,
        min_bound=None,
        max_bound=None,
    ):
        """
        Refine adversarial examples to minimize perturbation while maintaining misclassification.
        Uses normalized bounds for proper clamping.

        Args:
            original_images: Original clean images
            adv_images: Adversarial examples to refine
            labels: True labels for untargeted attacks, target labels for targeted
            refinement_steps: Number of binary search steps for refinement
            min_bound: Minimum valid normalized values (per channel)
            max_bound: Maximum valid normalized values (per channel)

        Returns:
            Refined adversarial examples with smaller perturbation
        """
        batch_size = original_images.size(0)
        refined_images = adv_images.clone()

        # For targeted attacks, get target labels
        if self.targeted:
            target_labels = self.get_target_label(original_images, labels)
        else:
            target_labels = labels

        # Initialize alpha for each image (controls interpolation between original and adversarial)
        # alpha=0 means pure adversarial, alpha=1 means pure original
        # Start with a smaller value to maintain visual quality
        alphas = torch.zeros(batch_size, device=self.device)

        # Identify initially successful adversarial examples
        with torch.no_grad():
            outputs = self.get_logits(adv_images)
            if self.targeted:
                # For targeted attacks, success means predicting the target class
                initial_success = outputs.argmax(dim=1) == target_labels
            else:
                # For untargeted attacks, success means not predicting the true class
                initial_success = outputs.argmax(dim=1) != labels

        # Only refine successful examples
        successful_indices = torch.where(initial_success)[0]

        if len(successful_indices) == 0:
            if self.verbose:
                print("No successful adversarial examples to refine")
            return refined_images

        if self.verbose:
            print(
                f"Refining {len(successful_indices)} successful adversarial examples..."
            )

        # Use a more conservative step size for better SSIM
        refinement_step_size = 0.1  # More conservative binary search step

        # Binary search to find the minimal perturbation for each example
        for step in range(refinement_steps):
            # Create interpolated images
            current_alphas = alphas.view(-1, 1, 1, 1)
            interpolated = (
                current_alphas * original_images + (1 - current_alphas) * refined_images
            )

            # Apply proper clamping to interpolated images
            interpolated = torch.clamp(interpolated, min=min_bound, max=max_bound)

            # Check if still adversarial
            with torch.no_grad():
                outputs = self.get_logits(interpolated)
                if self.targeted:
                    # For targeted attacks, success means predicting the target class
                    still_successful = outputs.argmax(dim=1) == target_labels
                else:
                    # For untargeted attacks, success means not predicting the true class
                    still_successful = outputs.argmax(dim=1) != labels

            # Update alphas with binary search - use more conservative steps
            for i in successful_indices:
                if still_successful[i]:
                    # If still successful, try moving closer to original
                    # Use a more conservative step to maintain visual quality
                    alphas[i] = alphas[i] + (1 - alphas[i]) * refinement_step_size
                    # Update the refined image with this better version
                    refined_images[i] = interpolated[i]
                else:
                    # If not successful, move back toward adversarial
                    # Use smaller steps when backtracking to avoid large visual changes
                    alphas[i] = alphas[i] * (1 - refinement_step_size)

            # Recompute the refined images explicitly to ensure proper bounds
            for i in successful_indices:
                # Linearly interpolate between original and adversarial
                alpha_i = alphas[i].view(1, 1, 1)
                refined_images[i] = (
                    alpha_i * original_images[i] + (1 - alpha_i) * adv_images[i]
                )
                # Apply clipping to ensure valid pixel values
                refined_images[i] = torch.clamp(
                    refined_images[i], min=min_bound, max=max_bound
                )

            # Adaptive refinement: reduce step size as we progress for finer control
            if step > refinement_steps // 2:
                refinement_step_size *= 0.9

        # Calculate improvement in perturbation metrics
        orig_l2 = (
            torch.norm((adv_images - original_images).view(batch_size, -1), dim=1)
            .mean()
            .item()
        )
        refined_l2 = (
            torch.norm((refined_images - original_images).view(batch_size, -1), dim=1)
            .mean()
            .item()
        )

        # Calculate SSIM improvement
        orig_ssim = self.compute_skimage_ssim(original_images, adv_images)
        refined_ssim = self.compute_skimage_ssim(original_images, refined_images)

        if self.verbose:
            l2_reduction = (orig_l2 - refined_l2) / orig_l2 * 100
            ssim_improvement = (refined_ssim - orig_ssim) / (1 - orig_ssim) * 100
            print(f"Refinement reduced L2 perturbation by {l2_reduction:.2f}%")
            print(
                f"Refinement improved SSIM from {orig_ssim:.4f} to {refined_ssim:.4f} ({ssim_improvement:.2f}% improvement)"
            )

        return refined_images

    def get_metrics(self):
        """
        Override base class method to ensure metrics are reported correctly
        even when success rate is 0%.

        Returns:
            Dictionary of attack metrics
        """
        # Get base metrics from parent class
        metrics = super().get_metrics()

        # If success rate is 0%, metrics might be zeroed out
        # Instead, report non-zero metrics if we have them
        if (
            metrics["success_rate"] == 0
            and self.l2_norms
            and self.linf_norms
            and self.ssim_values
        ):
            # Use the actual calculated values instead of zeros
            metrics["l2_norm"] = np.mean(self.l2_norms)
            metrics["linf_norm"] = np.mean(self.linf_norms)
            metrics["ssim"] = np.mean(self.ssim_values)

        return metrics
