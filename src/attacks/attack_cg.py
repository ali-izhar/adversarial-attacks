"""Conjugate Gradient (CG) adversarial attack implementation.

This attack implements the Conjugate Gradient method for generating adversarial examples.
It uses the conjugate gradient optimizer to efficiently find perturbations that cause
misclassification with minimal visual impact.

The implementation follows the same interface as other baseline attacks and integrates
with the Attack base class for consistent evaluation and reporting.
"""

import time
import torch
import torch.nn as nn
import numpy as np

from .baseline.attack import Attack
from .optimize.cg import ConjugateGradientOptimizer
from .optimize.projections import project_adversarial_example


class CG(Attack):
    r"""
    Enhanced Conjugate Gradient Attack with perceptual constraints.

    # Paper: "For adversarial attacks, we formulate the optimization problem as:
    # min_δ L(f(x + δ), y) subject to ||δ||_p ≤ ε"

    This attack creates adversarial examples by optimizing the input using
    conjugate gradient descent to find minimal perturbations that cause
    misclassification while maintaining visual similarity. The improved implementation
    uses proper conjugate gradient formulas for better convergence.

    Arguments:
        model (nn.Module): Model to attack.
        norm (str): Norm of the attack ('L2' or 'Linf').
        eps (float): Maximum perturbation.
        n_iter (int): Number of iterations.
        beta_method (str): Formula to use for conjugate updates ('FR', 'PR', or 'HS').
        restart_interval (int): Interval for restarting conjugate gradient updates.
        tv_lambda (float): Total variation regularization weight for smoothness.
        color_lambda (float): Color regularization weight for perceptual quality.
        perceptual_lambda (float): Weight for frequency domain perceptual loss.
        rand_init (bool): If True, use random initialization.
        fgsm_init (bool): If True, use FGSM initialization for better starting point.
        adaptive_restart (bool): If True, dynamically restart based on conjugacy loss.
        early_stopping (bool): If True, stop when adversarial goal is achieved.
        verbose (bool): If True, print progress updates.
        strict_epsilon_constraint (bool): If True, strictly enforce the epsilon constraint.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iter: int = 50,
        beta_method: str = "HS",
        restart_interval: int = 10,
        tv_lambda: float = 0.05,
        color_lambda: float = 0.05,
        perceptual_lambda: float = 0.05,
        rand_init: bool = True,
        fgsm_init: bool = True,
        adaptive_restart: bool = True,
        early_stopping: bool = True,
        verbose: bool = False,
        strict_epsilon_constraint: bool = True,
    ):
        # Initialize the Attack base class
        super().__init__("CG", model)

        # Record attack parameters
        self.norm = norm
        self.orig_eps = eps  # Store original epsilon specified in [0,1] space

        # Scale epsilon to normalized space by dividing by ImageNet std
        mean_std = self.std.clone().detach().mean().item()
        self.eps = (
            eps / mean_std
        )  # Scaling to normalized space for consistent perturbation

        self.n_iter = n_iter
        self.beta_method = beta_method
        self.restart_interval = restart_interval
        self.tv_lambda = tv_lambda
        self.color_lambda = color_lambda
        self.perceptual_lambda = perceptual_lambda
        self.rand_init = rand_init
        self.fgsm_init = fgsm_init
        self.adaptive_restart = adaptive_restart
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.strict_epsilon_constraint = strict_epsilon_constraint

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer with correct parameters
        self.optimizer = ConjugateGradientOptimizer(
            norm=norm,
            eps=self.eps,  # Pass the scaled epsilon for normalized space
            n_iterations=n_iter,
            beta_method=beta_method,
            restart_interval=restart_interval,
            rand_init=rand_init,
            early_stopping=early_stopping,
            verbose=verbose,
            fgsm_init=fgsm_init,
            adaptive_restart=adaptive_restart,
            momentum=0.0,  # Don't use momentum for proper CG
            line_search_factor=0.5,
            sufficient_decrease=1e-4,
        )

    def total_variation_loss(self, delta):
        """Calculate enhanced total variation loss for smoothness while preserving edges.

        This version uses a weighted approach that penalizes changes more in smooth areas
        and less along existing edges in the image.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)

        Returns:
            Total variation loss normalized by image size (per-sample)
        """
        batch_size = delta.size(0)
        n_channels = delta.size(1)

        # Compute differences in x and y directions
        diff_h = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        diff_w = delta[:, :, :, 1:] - delta[:, :, :, :-1]

        # Use L1 norm (absolute differences) which tends to preserve edges better than L2
        tv_h = torch.abs(diff_h).sum(dim=(1, 2, 3))
        tv_w = torch.abs(diff_w).sum(dim=(1, 2, 3))

        # Normalize by image size
        n_pixels = delta.size(2) * delta.size(3)

        return (tv_h + tv_w) / (n_pixels * n_channels)

    def color_regularization(self, delta):
        """Calculate enhanced color regularization to penalize visible changes.

        This uses a YUV-style weighting approach where luminance (Y) changes are
        penalized more than chrominance (U,V) changes, as the human eye is more
        sensitive to luminance differences.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)

        Returns:
            Color regularization loss (per-sample)
        """
        batch_size = delta.size(0)

        # RGB to YUV-approximate weights
        # Y = 0.299*R + 0.587*G + 0.114*B (luminance - most sensitive)
        # U and V are chrominance components (less sensitive)
        y_weights = torch.tensor([0.299, 0.587, 0.114], device=delta.device).view(
            1, 3, 1, 1
        )

        # Calculate luminance component of perturbation
        luma_delta = (delta * y_weights).sum(dim=1, keepdim=True)

        # Penalize luminance changes more heavily (5x weight)
        luma_penalty = 5.0 * (luma_delta**2).mean(dim=(1, 2, 3))

        # Standard color penalty with channel-specific weights
        # Red and Green changes are more noticeable than Blue
        channel_weights = torch.tensor([1.2, 1.0, 0.8], device=delta.device).view(
            1, 3, 1, 1
        )
        color_penalty = ((delta * channel_weights) ** 2).mean(dim=(1, 2, 3))

        # Combined penalty
        return luma_penalty + color_penalty

    def perceptual_loss(self, delta, input_images):
        """Calculate enhanced perceptual loss to preserve image structure.

        This enhanced version focuses more heavily on the low and mid frequency
        components that are most important for visual quality, with special emphasis
        on preserving image structure.

        Args:
            delta: Perturbation tensor of shape (B, C, H, W)
            input_images: Original images of shape (B, C, H, W)

        Returns:
            Perceptual loss value (per-sample)
        """
        # Compute frequency domain representation
        fft_orig = torch.fft.fft2(input_images, dim=(2, 3))
        fft_pert = torch.fft.fft2(input_images + delta, dim=(2, 3))

        # Compute magnitude difference in frequency domain
        mag_orig = torch.abs(fft_orig)
        mag_pert = torch.abs(fft_pert)

        # Create emphasis on lower frequencies (more critical for image quality)
        batch_size, channels, height, width = input_images.shape

        # Create a frequency weighting mask
        y_coords = torch.arange(height, device=input_images.device).view(1, 1, -1, 1)
        x_coords = torch.arange(width, device=input_images.device).view(1, 1, 1, -1)

        # Compute distance from center (DC component)
        center_y, center_x = height // 2, width // 2
        dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_dist = torch.sqrt(
            torch.tensor(center_y**2 + center_x**2, device=dist.device)
        )

        # Create a stronger weighting for low and mid frequencies
        # Low frequencies (DC) - highest weight
        low_freq_mask = torch.exp(-dist / (max_dist * 0.1))
        # Mid frequencies - medium weight
        mid_freq_mask = torch.exp(-(((dist - max_dist * 0.2) / (max_dist * 0.2)) ** 2))
        # Combined mask - emphasize both low and mid frequencies
        weight_mask = low_freq_mask + 0.5 * mid_freq_mask
        weight_mask = weight_mask.expand(batch_size, channels, -1, -1)

        # Apply weighting and compute difference
        weighted_diff = ((mag_orig - mag_pert).abs() * weight_mask).sum(dim=(1, 2, 3))

        # Normalize by image size
        return weighted_diff / (channels * height * width)

    def forward(self, images, labels):
        r"""
        Overridden method for generating adversarial examples.

        # Paper: "min_δ L(f(x + δ), y) subject to ||δ||_p ≤ ε"
        # This is the main attack method implementing the paper's approach

        Arguments:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): Labels.
                - If self.targeted is False, labels are the actual labels
                - If self.targeted is True, labels are the target labels

        Returns:
            adversarial_images (torch.Tensor): Adversarial examples.
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Note start time for performance tracking
        self.start_time = time.time()

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        # Calculate normalized min/max bounds for valid pixel ranges after denormalization
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Use cross-entropy loss for classification
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Define the gradient function for the optimizer
        def gradient_fn(x):
            # Ensure gradients are enabled
            x.requires_grad_(True)

            # Forward pass
            outputs = self.get_logits(x)

            # Get appropriate labels for the current batch
            # This handles cases when we're only processing a subset of samples
            if self.targeted:
                # For targeted, use target labels matching the current batch size
                curr_labels = target_labels[: x.size(0)]
            else:
                # For untargeted, use true labels matching the current batch size
                curr_labels = labels[: x.size(0)]

            # Classification loss
            if self.targeted:
                # For targeted attacks, minimize negative CE loss to target class
                class_loss = -ce_loss(outputs, curr_labels)
            else:
                # For untargeted attacks, maximize CE loss to true class
                class_loss = ce_loss(outputs, curr_labels)

            # Compute current perturbation
            delta = x - images[: x.size(0)]

            # Add regularization losses - all should return per-sample values
            # Note: These regularization terms are enhancements beyond the paper
            tv_loss = self.total_variation_loss(delta)
            color_loss = self.color_regularization(delta)
            percept_loss = self.perceptual_loss(delta, images[: x.size(0)])

            # Combine losses with weights - keeping per-sample values
            total_loss = (
                class_loss
                + self.tv_lambda * tv_loss
                + self.color_lambda * color_loss
                + self.perceptual_lambda * percept_loss
            )

            # Compute gradients for each sample
            # Paper: "g_t = ∇_δ L(f(x + δ_t), y)" - computing the gradient
            grads = []
            for i in range(x.size(0)):
                grad = torch.autograd.grad(
                    total_loss[i], x, retain_graph=(i < x.size(0) - 1)
                )[0][i : i + 1]
                grads.append(grad)

            # Combine gradients
            full_grad = torch.cat(grads, dim=0)
            return full_grad

        # Define the loss function that returns per-sample loss values
        def loss_fn(x):
            with torch.no_grad():
                # Forward pass
                outputs = self.get_logits(x)

                # Get matching labels for current batch size
                if self.targeted:
                    curr_labels = target_labels[: x.size(0)]
                else:
                    curr_labels = labels[: x.size(0)]

                # Classification loss
                if self.targeted:
                    class_loss = -ce_loss(outputs, curr_labels)
                else:
                    class_loss = ce_loss(outputs, curr_labels)

                # Compute current perturbation
                delta = x - images[: x.size(0)]

                # Add regularization losses
                tv_loss = self.total_variation_loss(delta)
                color_loss = self.color_regularization(delta)
                percept_loss = self.perceptual_loss(delta, images[: x.size(0)])

                # Combine losses with weights
                total_loss = (
                    class_loss
                    + self.tv_lambda * tv_loss
                    + self.color_lambda * color_loss
                    + self.perceptual_lambda * percept_loss
                )

                return total_loss

        # Define success function for early stopping
        def success_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)

                # Get matching labels for the current batch
                if self.targeted:
                    curr_labels = target_labels[: x.size(0)]
                else:
                    curr_labels = labels[: x.size(0)]

                if self.targeted:
                    # Attack succeeds if model predicts target class
                    return outputs.argmax(dim=1) == curr_labels
                else:
                    # Attack succeeds if model predicts any class other than the true label
                    # Paper: "break if ||r_{k+1}|| < tol or arg max_j f_j(x + δ_{k+1}) ≠ y_true"
                    # This is the second condition for early stopping
                    return outputs.argmax(dim=1) != curr_labels

        # Run the optimizer to generate adversarial examples
        # This executes the Algorithm 1 from the paper (Efficient Conjugate Gradient Attack)
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
            targeted=self.targeted,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += int(metrics["iterations"] * images.size(0))
        self.total_gradient_calls += int(metrics["gradient_calls"] * images.size(0))

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time

        # Strictly enforce epsilon constraint if enabled, but ensure minimal perturbation size
        if self.strict_epsilon_constraint:
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Calculate current perturbation magnitude
                delta = adv_images[i : i + 1] - images[i : i + 1]
                if self.norm.lower() == "l2":
                    delta_norm = torch.norm(delta.flatten(), p=2).item()
                else:  # Linf
                    delta_norm = torch.norm(delta.flatten(), p=float("inf")).item()

                # Only project if perturbation exceeds epsilon
                if delta_norm > self.eps:
                    adv_images[i : i + 1] = project_adversarial_example(
                        adv_images[i : i + 1],
                        images[i : i + 1],
                        self.eps,
                        self.norm,
                        min_bound=min_bound,
                        max_bound=max_bound,
                    )

        # Ensure outputs are properly clamped to valid input range
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Apply post-processing refinement to improve SSIM and reduce perturbation
        if metrics["success_rate"] > 0:  # Only refine if we have successful examples
            adv_images = self.refine_perturbation(
                images,
                adv_images,
                labels,
                refinement_steps=10,  # Reduced from 15 to 10 for less aggressive refinement
                min_bound=min_bound,
                max_bound=max_bound,
            )

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Compute perturbation metrics on all samples, not just successful ones
        perturbation_metrics = self.compute_perturbation_metrics(
            images, adv_images, None  # Pass None instead of success_mask
        )

        # Update attack_success_count for metrics
        self.attack_success_count += success_mask.sum().item()
        self.total_samples += images.size(0)

        # Log metrics if verbose
        if self.verbose:
            orig_preds, adv_preds = pred_info
            print(f"Attack complete: {success_rate:.2f}% success rate")
            print(
                f"Metrics: L2={perturbation_metrics['l2_norm']:.6f}, "
                f"L∞={perturbation_metrics['linf_norm']:.6f}, "
                f"SSIM={perturbation_metrics['ssim']:.4f}"
            )
            print(
                f"Iterations: {metrics['iterations']:.1f}, Gradient calls: {metrics['gradient_calls']:.1f}"
            )
            print(f"Time per sample: {metrics['time_per_sample']*1000:.2f}ms")

        return adv_images

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

    def compute_perturbation_metrics(self, original, perturbed, success_mask=None):
        """
        Compute metrics to evaluate the perturbation quality.

        This overrides the base class method to use scikit-image SSIM.

        Args:
            original: Original clean images
            perturbed: Perturbed adversarial images
            success_mask: Optional mask indicating successful perturbations

        Returns:
            Dictionary with perturbation metrics
        """
        # Get base metrics from parent class
        metrics = super().compute_perturbation_metrics(
            original, perturbed, success_mask
        )

        # Override SSIM calculation with scikit-image implementation
        metrics["ssim"] = self.compute_skimage_ssim(original, perturbed)

        return metrics

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

    def refine_perturbation(
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

        Args:
            original_images: Original clean images
            adv_images: Adversarial examples to refine
            labels: True labels for untargeted attacks, target labels for targeted
            refinement_steps: Number of binary search steps for refinement
            min_bound: Minimum bound for valid pixel ranges after denormalization
            max_bound: Maximum bound for valid pixel ranges after denormalization

        Returns:
            Refined adversarial examples with smaller perturbation
        """
        # Use default bounds if not provided
        if min_bound is None:
            min_bound = 0.0
        if max_bound is None:
            max_bound = 1.0

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

            # Ensure interpolated images are within valid bounds
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
