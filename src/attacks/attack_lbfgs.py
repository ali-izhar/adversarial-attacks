"""Simplified Limited-memory BFGS (L-BFGS) adversarial attack implementation.

This file implements the L-BFGS attack, which leverages scipy's L-BFGS-B optimizer
to create adversarial examples with minimal perturbation. The L-BFGS method
approximates second-order curvature information without explicitly forming the Hessian matrix.

Key features:
- Efficient quasi-Newton optimization for adversarial examples
- Binary search for optimal trade-off between perturbation size and attack success
- Supports both targeted and untargeted attacks
- Compatible with different loss functions
- Constraint handling for both L2 and Linf perturbation norms
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

import time
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from .baseline.attack import Attack
from .optimize.lbfgs import LBFGSOptimizer


class LBFGS(Attack):
    """
    Simplified L-BFGS adversarial attack.

    This attack uses scipy's L-BFGS-B optimizer to generate adversarial examples
    by efficiently searching for minimal perturbations that cause misclassification.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,
        n_iterations: int = 100,
        history_size: int = 20,
        initial_const: float = 1.0,
        binary_search_steps: int = 10,
        const_factor: float = 15.0,
        repeat_search: bool = True,
        rand_init: bool = True,
        init_std: float = 0.05,
        early_stopping: bool = True,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the L-BFGS attack.

        Args:
            model: The model to attack
            norm: Norm for the perturbation constraint ('L2' or 'Linf')
            eps: Maximum allowed perturbation magnitude
            n_iterations: Maximum number of iterations for L-BFGS
            history_size: Size of history used for Hessian approximation
            initial_const: Initial trade-off constant between perturbation and loss
            binary_search_steps: Number of binary search steps to find optimal constant
            const_factor: Factor to multiply constant by when no solution is found
            repeat_search: Whether to repeat search with upper bound on last step
            rand_init: Whether to initialize with random noise
            init_std: Standard deviation for random initialization
            early_stopping: Whether to stop when adversarial criteria are met
            verbose: Whether to print progress information
            device: Device to run the attack on
        """
        # Initialize the base Attack class
        super().__init__("LBFGS", model)

        # Record attack parameters
        self.norm = norm
        self.orig_eps = eps  # Store original epsilon specified in [0,1] space

        # Scale epsilon to normalized space by dividing by ImageNet std
        mean_std = self.std.clone().detach().mean().item()
        self.eps = (
            eps / mean_std
        )  # Scaling to normalized space for consistent perturbation

        self.n_iterations = n_iterations
        self.history_size = history_size
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.const_factor = const_factor
        self.repeat_search = repeat_search
        self.rand_init = rand_init
        self.init_std = init_std
        self.early_stopping = early_stopping
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer
        self.optimizer = LBFGSOptimizer(
            norm=norm,
            eps=self.eps,  # Pass the scaled epsilon for normalized space
            n_iterations=n_iterations,
            history_size=history_size,
            initial_const=initial_const,
            binary_search_steps=binary_search_steps,
            const_factor=const_factor,
            repeat_search=repeat_search,
            rand_init=rand_init,
            init_std=init_std,
            early_stopping=early_stopping,
            verbose=verbose,
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

    def forward(self, images, labels):
        """
        Generate adversarial examples using L-BFGS optimization.

        Args:
            images: Input images to perturb
            labels: Target labels

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

        # Calculate normalized min/max bounds for valid pixel ranges after denormalization
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)
        max_bound = max_bound.view(1, 3, 1, 1)

        # Define the loss function
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Define the gradient function
        def gradient_fn(x):
            x.requires_grad_(True)
            outputs = self.get_logits(x)

            # Get appropriate labels for the current batch
            curr_labels = target_labels[: x.size(0)]

            if self.targeted:
                # Get the target class logit
                target_logits = outputs.gather(1, curr_labels.unsqueeze(1)).squeeze(1)

                # Get the highest logit of other classes
                other_logits = outputs.clone()
                other_logits.scatter_(1, curr_labels.unsqueeze(1), float("-inf"))
                highest_other_logits = other_logits.max(1)[0]

                # Margin loss: ensuring target class has higher logit by at least 'confidence'
                margin = highest_other_logits - target_logits + confidence
                # For targeted attacks, we want to minimize this margin
                loss = torch.clamp(margin, min=0).mean()
            else:
                # Get the true class logit
                true_logits = outputs.gather(1, curr_labels.unsqueeze(1)).squeeze(1)

                # Get the highest logit of other classes
                other_logits = outputs.clone()
                other_logits.scatter_(1, curr_labels.unsqueeze(1), float("-inf"))
                highest_other_logits = other_logits.max(1)[0]

                # Margin loss: ensuring true class has lower logit by at least 'confidence'
                margin = true_logits - highest_other_logits + confidence
                # For untargeted attacks, we want to maximize this margin
                loss = -torch.clamp(margin, min=0).mean()

            # Compute gradient
            loss.backward()
            grad = x.grad.clone()
            x.grad = None

            return grad

        # Define the loss function that returns per-sample loss values
        confidence = 10.0  # Confidence parameter for stronger adversarial examples

        def loss_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)
                curr_labels = target_labels[: x.size(0)]

                if self.targeted:
                    # Get the target class logit
                    target_logits = outputs.gather(1, curr_labels.unsqueeze(1)).squeeze(
                        1
                    )

                    # Get the highest logit of other classes
                    other_logits = outputs.clone()
                    other_logits.scatter_(1, curr_labels.unsqueeze(1), float("-inf"))
                    highest_other_logits = other_logits.max(1)[0]

                    # Margin loss: ensuring target class has higher logit by at least 'confidence'
                    margin = highest_other_logits - target_logits + confidence
                    # For targeted attacks, we want to minimize this margin
                    loss = torch.clamp(margin, min=0)
                else:
                    # Get the true class logit
                    true_logits = outputs.gather(1, curr_labels.unsqueeze(1)).squeeze(1)

                    # Get the highest logit of other classes
                    other_logits = outputs.clone()
                    other_logits.scatter_(1, curr_labels.unsqueeze(1), float("-inf"))
                    highest_other_logits = other_logits.max(1)[0]

                    # Margin loss: ensuring true class has lower logit by at least 'confidence'
                    margin = true_logits - highest_other_logits + confidence
                    # For untargeted attacks, we want to maximize this margin
                    loss = -torch.clamp(margin, min=0)

                return loss

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
                    return outputs.argmax(dim=1) != curr_labels

        # Run the optimizer to generate adversarial examples
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            loss_fn=loss_fn,
            success_fn=success_fn,
            x_original=images,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Update attack metrics from optimizer results
        self.total_iterations += metrics["iterations"]
        self.total_gradient_calls += metrics["gradient_calls"]

        # End timing
        end_time = time.time()
        self.total_time += end_time - start_time

        # Ensure outputs are properly clamped
        adv_images = torch.clamp(adv_images, min=min_bound, max=max_bound).detach()

        # Evaluate success and compute perturbation metrics
        success_rate, success_mask, pred_info = self.evaluate_attack_success(
            images, adv_images, labels
        )

        # Compute perturbation metrics - always include all samples, not just successful ones
        # This ensures metrics are reported even with 0% success rate
        perturbation_metrics = self.compute_perturbation_metrics(
            images,
            adv_images,
            None,  # Pass None instead of success_mask to include all samples
        )

        # Add mean L2 distortion of successful examples from optimizer metrics
        if "mean_successful_l2" in metrics and metrics["mean_successful_l2"] > 0:
            perturbation_metrics["mean_successful_l2"] = metrics["mean_successful_l2"]

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

            if "mean_successful_l2" in perturbation_metrics:
                print(
                    f"Mean L2 of successful examples: {perturbation_metrics['mean_successful_l2']:.6f}"
                )

            print(
                f"Iterations: {metrics['iterations']}, Gradient calls: {metrics['gradient_calls']}"
            )
            print(f"Time per sample: {metrics['time_per_sample']*1000:.2f}ms")

        return adv_images

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
