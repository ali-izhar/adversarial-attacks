"""Projected Gradient Descent (PGD) adversarial attack implementation."""

import time
import torch
import torch.nn as nn

from .baseline.attack import Attack
from .optimize.pgd import PGDOptimizer


class PGD(Attack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    This attack uses the PGDOptimizer to generate adversarial examples by iteratively taking
    steps in the gradient direction and then projecting the perturbed input back onto a
    constraint set defined by a norm ball (L2 or Linf).
    This implementation includes optional binary search for minimal epsilon (targeted)
    and perturbation refinement.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 0.5,  # Epsilon in NORMALIZED space
        n_iterations: int = 40,  # Tuned default steps
        step_size: float = 0.05,  # Step size in NORMALIZED space (tuned for L2)
        loss_fn: str = "cross_entropy",
        rand_init: bool = True,
        early_stopping: bool = True,  # Use early stopping in optimizer
        refine_steps: int = 10,  # Steps for binary search refinement
        use_binary_search_eps: bool = True,  # Use binary search for targeted attacks
        verbose: bool = False,
    ):
        """
        Initialize the PGD attack.

        Args:
            model: The model to attack.
            norm: Norm for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude in NORMALIZED space.
            n_iterations: Maximum number of iterations for the optimizer.
            step_size: Step size for gradient updates in NORMALIZED space.
            loss_fn: Loss function ('cross_entropy', 'margin', 'carlini_wagner').
            rand_init: Whether to initialize with random noise.
            early_stopping: Whether the optimizer should stop early if successful.
            refine_steps: Number of refinement steps (binary search on alpha). Default 10.
            use_binary_search_eps: Whether to perform binary search on epsilon for targeted attacks.
            verbose: Whether to print progress information.
        """
        super().__init__("PGD", model)

        if norm.upper() not in ["L2", "LINF"]:
            raise ValueError(f"Norm {norm} not supported. Use 'L2' or 'Linf'.")
        self.norm = norm.upper()
        self.orig_eps = eps  # Store original epsilon in normalized space
        self.eps = eps  # Current epsilon (may change during binary search)

        # Note: No scaling needed for eps/step_size as they are already in normalized space
        self.n_iterations = n_iterations
        self.step_size = step_size

        self.loss_fn_type = loss_fn
        self.rand_init = rand_init
        self.early_stopping = early_stopping  # Passed to optimizer
        self.refine_steps = refine_steps
        self.use_binary_search_eps = use_binary_search_eps
        self.verbose = verbose

        # Set up supported modes
        self.supported_mode = ["default", "targeted"]

        # Create the optimizer for PGD
        # Maximize flag will be set in forward() based on targeted mode
        self.optimizer = PGDOptimizer(
            norm=self.norm,
            eps=self.eps,  # Will be updated if binary search is used
            n_iterations=self.n_iterations,
            step_size=self.step_size,
            rand_init=self.rand_init,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            maximize=True,  # Default, overridden in forward
        )

    # Removed compute_skimage_ssim - use base class compute_perturbation_metrics

    # Refine perturbation method adapted for normalized bounds
    def refine_perturbation_with_bounds(
        self,
        original_images,
        adv_images,
        labels,  # True labels needed for untargeted checks
        refinement_steps=10,
        min_bound=None,
        max_bound=None,
    ):
        """
        Refine adversarial examples to minimize perturbation while maintaining misclassification,
        respecting normalized image bounds. Uses binary search on interpolation factor alpha.

        Args:
            original_images: Original clean images (normalized).
            adv_images: Adversarial examples to refine (normalized).
            labels: True labels for untargeted, target labels for targeted.
            refinement_steps: Number of binary search steps.
            min_bound: Minimum valid normalized values.
            max_bound: Maximum valid normalized values.

        Returns:
            Refined adversarial examples (normalized).
        """
        batch_size = original_images.size(0)
        refined_images = adv_images.clone().detach()  # Start with current best adv

        # Determine target labels for checking success
        if self.targeted:
            eval_labels = self.get_target_label(
                original_images, labels
            )  # Target labels
        else:
            eval_labels = labels  # True labels

        # Alpha=0: pure adversarial, Alpha=1: pure original
        # low_alpha represents the best known alpha that maintains success
        # high_alpha represents the last alpha tried that failed (or initial adv alpha=0)
        low_alpha = torch.zeros(
            batch_size, device=self.device
        )  # Best successful alpha found so far
        high_alpha = torch.zeros(
            batch_size, device=self.device
        )  # Tracks failed attempts if any

        # Identify initially successful adversarial examples based on eval_labels
        with torch.no_grad():
            outputs = self.get_logits(adv_images)  # Tracks grad calls via base method
            if self.targeted:
                initial_success = outputs.argmax(dim=1) == eval_labels
            else:
                initial_success = outputs.argmax(dim=1) != eval_labels

        # Only refine successful examples
        successful_indices = torch.where(initial_success)[0]

        if len(successful_indices) == 0:
            if self.verbose:
                print("Refinement: No successful examples to refine.")
            return refined_images  # Return original adv if none were successful

        # Keep track of the best refined image found FOR EACH sample
        best_refined_for_sample = refined_images.clone()

        if self.verbose:
            print(
                f"Refinement: Refining {len(successful_indices)} successful examples..."
            )

        # Binary search for minimal perturbation (maximal alpha) for each successful sample
        for step in range(refinement_steps):
            # Alpha to test in this step (midpoint for active search)
            # Test alpha = low_alpha + (1 - low_alpha) / 2 # Try halfway to original
            test_alpha = (
                low_alpha + 1.0
            ) / 2.0  # Explore range [0, 1] more evenly initially

            # Create interpolated images ONLY for successful indices
            current_alphas_view = test_alpha[successful_indices].view(-1, 1, 1, 1)
            orig_subset = original_images[successful_indices]
            adv_subset = refined_images[
                successful_indices
            ]  # Use current refined as starting point

            interpolated_subset = (
                current_alphas_view * orig_subset
                + (1 - current_alphas_view) * adv_subset
            )

            # Ensure interpolated images are within valid bounds
            interpolated_subset = torch.clamp(
                interpolated_subset, min=min_bound, max=max_bound
            )

            # Check if still adversarial
            with torch.no_grad():
                outputs_subset = self.get_logits(
                    interpolated_subset
                )  # Tracks grad calls
                if self.targeted:
                    still_successful_subset = (
                        outputs_subset.argmax(dim=1) == eval_labels[successful_indices]
                    )
                else:
                    still_successful_subset = (
                        outputs_subset.argmax(dim=1) != eval_labels[successful_indices]
                    )

            # Update alphas based on success for each sample being refined
            for i, idx in enumerate(successful_indices):
                if still_successful_subset[i]:
                    # Success! This alpha works. It becomes the new lower bound.
                    low_alpha[idx] = test_alpha[idx]
                    # Store this successful interpolated image as potentially the best refined one
                    best_refined_for_sample[idx] = interpolated_subset[i]
                # else:
                # Failure. This alpha is too large. The previous low_alpha remains the best.
                # We don't need to explicitly update high_alpha here for this search strategy.

        # Final refined images are the best ones found during the search
        refined_images = best_refined_for_sample

        if self.verbose:
            # Calculate perturbation reduction (optional)
            orig_pert = adv_images - original_images
            refined_pert = refined_images - original_images
            if self.norm == "L2":
                orig_norm = (
                    torch.norm(orig_pert.view(batch_size, -1), p=2, dim=1)[
                        successful_indices
                    ]
                    .mean()
                    .item()
                )
                refined_norm = (
                    torch.norm(refined_pert.view(batch_size, -1), p=2, dim=1)[
                        successful_indices
                    ]
                    .mean()
                    .item()
                )
                reduction = (
                    (orig_norm - refined_norm) / orig_norm * 100
                    if orig_norm > 1e-9
                    else 0
                )
                print(
                    f"Refinement: Reduced L2 perturbation by {reduction:.2f}% on average for refined samples."
                )
            elif self.norm == "LINF":
                orig_norm = (
                    torch.norm(orig_pert.view(batch_size, -1), p=float("inf"), dim=1)[
                        successful_indices
                    ]
                    .mean()
                    .item()
                )
                refined_norm = (
                    torch.norm(
                        refined_pert.view(batch_size, -1), p=float("inf"), dim=1
                    )[successful_indices]
                    .mean()
                    .item()
                )
                reduction = (
                    (orig_norm - refined_norm) / orig_norm * 100
                    if orig_norm > 1e-9
                    else 0
                )
                print(
                    f"Refinement: Reduced Linf perturbation by {reduction:.2f}% on average for refined samples."
                )

        return refined_images.detach()

    def forward(self, images, labels):
        """
        Perform the PGD attack.

        Args:
            images: Input images (normalized).
            labels: True labels.

        Returns:
            Adversarial examples (normalized).
        """
        start_time = time.time()  # Track time for metrics

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.size(0)

        # Get target labels if needed
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
            eval_labels = target_labels  # Labels to check success against
            self.optimizer.maximize = False  # Minimize loss for targeted
        else:
            target_labels = labels  # Labels for loss calculation
            eval_labels = labels  # Labels to check success against (original labels)
            self.optimizer.maximize = True  # Maximize loss for untargeted

        # Calculate normalized min/max bounds once
        min_bound = (-self.mean / self.std).to(device=images.device, dtype=images.dtype)
        max_bound = ((1 - self.mean) / self.std).to(
            device=images.device, dtype=images.dtype
        )
        min_bound = min_bound.view(1, 3, 1, 1)  # Reshape for broadcasting
        max_bound = max_bound.view(1, 3, 1, 1)  # Reshape for broadcasting

        # --- Define Loss Function ---
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        if self.loss_fn_type == "cross_entropy":

            def loss_fn(outputs, current_labels):
                loss_val = ce_loss(outputs, current_labels)
                # Return positive loss if maximizing (untargeted), negative if minimizing (targeted)
                return loss_val if self.optimizer.maximize else -loss_val

        elif self.loss_fn_type == "margin":

            def loss_fn(outputs, current_labels):
                target_logits = outputs.gather(1, current_labels.view(-1, 1)).squeeze(1)
                max_other_logits = outputs.clone()
                max_other_logits.scatter_(1, current_labels.view(-1, 1), float("-inf"))
                max_other_logits = max_other_logits.max(1)[0]
                # Margin: target_logit - max_other_logit
                margin = target_logits - max_other_logits
                # If targeted (minimize loss -> maximize margin), return -margin
                # If untargeted (maximize loss -> minimize margin), return margin
                return -margin if not self.optimizer.maximize else margin

        elif self.loss_fn_type == "carlini_wagner":
            confidence = 0.0

            def loss_fn(outputs, current_labels):
                target_logits = outputs.gather(1, current_labels.view(-1, 1)).squeeze(1)
                max_other_logits = outputs.clone()
                max_other_logits.scatter_(1, current_labels.view(-1, 1), float("-inf"))
                max_other_logits = max_other_logits.max(1)[0]
                # CW loss: max(max_other - target + conf, 0)
                cw_loss = torch.clamp(
                    max_other_logits - target_logits + confidence, min=0
                )
                # If targeted (minimize loss), return cw_loss
                # If untargeted (maximize loss), return -cw_loss (encourage target < max_other)
                # Note: The untargeted CW version might need adjustment based on literature.
                # A common untargeted form is max(target - max_other + conf, 0), which we want to maximize.
                # Let's stick to the structure: return positive loss if maximizing, negative if minimizing.
                if self.optimizer.maximize:  # Untargeted
                    # Maximize max(target - max_other + conf, 0)
                    untargeted_cw = torch.clamp(
                        target_logits - max_other_logits + confidence, min=0
                    )
                    return untargeted_cw
                else:  # Targeted
                    # Minimize max(max_other - target + conf, 0)
                    return cw_loss

        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_type}")

        # --- Define Gradient and Success Functions for Optimizer ---
        def gradient_fn(x):
            # Ensure grad calculation is enabled for the input
            x_req_grad = x.detach().requires_grad_(True)
            outputs = self.get_logits(x_req_grad)  # Base method tracks grad calls
            # Use appropriate labels for loss calculation (target_labels)
            current_batch_labels = target_labels[: x_req_grad.size(0)]
            loss_values = loss_fn(outputs, current_batch_labels)
            mean_loss = loss_values.mean()
            grad = torch.autograd.grad(mean_loss, x_req_grad)[0]
            return grad.detach()  # Return detached grad

        def success_fn(x):
            with torch.no_grad():
                outputs = self.get_logits(x)  # Base method tracks grad calls
                # Use appropriate labels for success evaluation (eval_labels)
                current_batch_labels = eval_labels[: x.size(0)]
                if self.targeted:
                    return outputs.argmax(dim=1) == current_batch_labels
                else:
                    return outputs.argmax(dim=1) != current_batch_labels

        # --- Initial Attack Run ---
        # Store original optimizer settings
        original_eps = self.optimizer.eps
        original_iterations = self.optimizer.n_iterations
        total_optimizer_iterations = 0
        total_optimizer_grad_calls = 0

        # Set current epsilon for the optimizer
        self.optimizer.eps = self.eps

        if self.verbose:
            print(
                f"Running PGD with eps={self.optimizer.eps:.6f}, steps={self.optimizer.n_iterations}"
            )

        # Run the optimizer
        adv_images, run_metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            success_fn=success_fn if self.early_stopping else None,
            x_original=images,
            min_bound=min_bound,
            max_bound=max_bound,
        )

        # Accumulate metrics from this run
        total_optimizer_iterations += run_metrics["iterations"] * batch_size
        total_optimizer_grad_calls += run_metrics.get("gradient_calls", 0) * batch_size

        # --- Optional: Binary Search for Epsilon (Targeted Attacks Only) ---
        best_adv_images = adv_images.clone()  # Keep track of the best images found
        final_eps = self.eps

        if self.targeted and self.use_binary_search_eps:
            # Evaluate success of initial attack run
            with torch.no_grad():
                initial_outputs = self.get_logits(adv_images)
                initial_success_mask = initial_outputs.argmax(dim=1) == target_labels
                initial_success_rate = 100 * initial_success_mask.float().mean().item()

            # Perform binary search only if initial attack was reasonably successful
            if initial_success_rate > 50 and initial_success_mask.sum() >= 2:
                if self.verbose:
                    print(
                        f"Initial success rate {initial_success_rate:.2f}%. Starting binary search for epsilon..."
                    )

                # Binary search parameters
                low_eps = 0.0
                high_eps = self.eps  # Start search from initial epsilon down to 0
                best_eps_found = high_eps  # Keep track of the best epsilon that worked

                # Only optimize previously successful examples during search
                successful_indices = torch.where(initial_success_mask)[0]

                if successful_indices.numel() > 0:
                    search_steps = 7  # Number of binary search steps for epsilon
                    for search_step in range(search_steps):
                        mid_eps = (low_eps + high_eps) / 2
                        self.optimizer.eps = mid_eps

                        if self.verbose:
                            print(
                                f"  Binary search step {search_step+1}: Testing eps={mid_eps:.6f}"
                            )

                        # Run optimizer only on the subset of images that were successful initially
                        subset_images = images[successful_indices]
                        subset_labels = target_labels[
                            successful_indices
                        ]  # Use target labels for loss/success

                        # Redefine loss/grad/success functions for the subset if necessary,
                        # or ensure the existing ones handle subset indexing correctly.
                        # Assuming existing functions work with subset size.

                        subset_adv_images, search_run_metrics = self.optimizer.optimize(
                            x_init=subset_images,  # Start from clean images in the subset
                            gradient_fn=gradient_fn,  # Should work with subset size
                            success_fn=(
                                success_fn if self.early_stopping else None
                            ),  # Should work with subset size
                            x_original=subset_images,
                            min_bound=min_bound,
                            max_bound=max_bound,
                        )

                        # Accumulate metrics from this search run
                        total_optimizer_iterations += search_run_metrics[
                            "iterations"
                        ] * subset_images.size(0)
                        total_optimizer_grad_calls += search_run_metrics.get(
                            "gradient_calls", 0
                        ) * subset_images.size(0)

                        # Evaluate success on this subset with the current mid_eps
                        with torch.no_grad():
                            subset_outputs = self.get_logits(subset_adv_images)
                            subset_success_mask = (
                                subset_outputs.argmax(dim=1) == subset_labels
                            )
                            subset_success_rate = (
                                100 * subset_success_mask.float().mean().item()
                            )

                        if self.verbose:
                            print(
                                f"    Success rate with eps={mid_eps:.6f}: {subset_success_rate:.2f}%"
                            )

                        # If still successful for a high percentage, try smaller epsilon
                        success_threshold = (
                            90  # Require high success rate to reduce epsilon
                        )
                        if subset_success_rate >= success_threshold:
                            high_eps = mid_eps  # This epsilon worked, try lower
                            best_eps_found = mid_eps
                            # Update the best adversarial examples for the successfully attacked subset
                            for i, orig_idx in enumerate(successful_indices):
                                if subset_success_mask[i]:
                                    best_adv_images[orig_idx] = subset_adv_images[i]
                        else:
                            low_eps = mid_eps  # This epsilon failed, need higher

                    final_eps = best_eps_found  # Use the best epsilon found
                    if self.verbose:
                        print(
                            f"Binary search finished. Optimal epsilon found: {final_eps:.6f}"
                        )
                else:
                    if self.verbose:
                        print(
                            "Binary search skipped: Not enough successful examples initially."
                        )
            else:
                if self.verbose:
                    print(
                        f"Binary search skipped: Initial success rate {initial_success_rate:.2f}% too low."
                    )
        elif self.targeted:
            if self.verbose:
                print("Binary search for epsilon disabled.")

        # Restore original optimizer settings if they were changed
        self.eps = self.orig_eps
        self.optimizer.eps = original_eps
        self.optimizer.n_iterations = original_iterations

        # Use the best images found (either from initial run or binary search)
        adv_images = best_adv_images

        # --- Optional: Refinement Step ---
        if self.refine_steps > 0:
            if self.verbose:
                print("Applying perturbation refinement...")
            refined_adv_images = self.refine_perturbation_with_bounds(
                original_images=images,
                adv_images=adv_images,  # Refine the current best adv images
                labels=labels,  # Pass original labels (needed for untargeted checks within refine)
                refinement_steps=self.refine_steps,
                min_bound=min_bound,
                max_bound=max_bound,
            )

            # Evaluate success of refined images
            with torch.no_grad():
                refined_outputs = self.get_logits(refined_adv_images)
                if self.targeted:
                    refined_success_mask = (
                        refined_outputs.argmax(dim=1) == target_labels
                    )
                else:
                    refined_success_mask = refined_outputs.argmax(dim=1) != labels
                refined_success_rate = 100 * refined_success_mask.float().mean().item()

            # Optionally, only use refined images if they maintain high success rate
            # Or compare perturbation norms/SSIM (requires computing metrics here)
            # Simple approach: use refined if success is still high
            # Evaluate success of original adv images for comparison
            with torch.no_grad():
                original_outputs = self.get_logits(adv_images)
                if self.targeted:
                    original_success_mask = (
                        original_outputs.argmax(dim=1) == target_labels
                    )
                else:
                    original_success_mask = original_outputs.argmax(dim=1) != labels
                original_success_rate = (
                    100 * original_success_mask.float().mean().item()
                )

            # Use refined images if success rate doesn't drop too much (e.g., by > 5%)
            if refined_success_rate >= original_success_rate - 5.0:
                if self.verbose:
                    print(
                        f"Using refined images (Success: {refined_success_rate:.2f}%)"
                    )
                adv_images = refined_adv_images
            else:
                if self.verbose:
                    print(
                        f"Keeping original adv images (Success: {original_success_rate:.2f}%), refinement dropped success to {refined_success_rate:.2f}%"
                    )
                # Keep the adv_images from before refinement
        else:
            if self.verbose:
                print("Refinement step skipped.")

        # --- Finalization and Metric Collection ---
        end_time = time.time()
        self.total_time += end_time - start_time

        # Update final iteration and gradient counts based on optimizer runs
        self.total_iterations += total_optimizer_iterations
        # Track gradient calls accumulated during optimizer runs
        self.total_gradient_calls += total_optimizer_grad_calls
        # Note: get_logits calls during refinement also contribute via base class

        # Ensure final images are detached and clamped
        final_adv_images = torch.clamp(
            adv_images, min=min_bound, max=max_bound
        ).detach()

        # Evaluate success and compute metrics using BASE CLASS methods on FINAL images
        _success_rate, success_mask, _ = self.evaluate_attack_success(
            images, final_adv_images, labels  # Use true labels for evaluation context
        )
        perturbation_metrics = self.compute_perturbation_metrics(
            images, final_adv_images, success_mask
        )

        if self.verbose:
            print(f"PGD Attack Complete. Final Success Rate: {_success_rate:.2f}%")
            print(f"  Avg L2 Pert: {perturbation_metrics['l2_norm']:.6f}")
            print(f"  Avg Linf Pert: {perturbation_metrics['linf_norm']:.6f}")
            print(f"  Avg SSIM: {perturbation_metrics['ssim']:.4f}")
            if self.total_samples > 0:
                avg_iters = self.total_iterations / self.total_samples
                avg_grads = self.total_gradient_calls / self.total_samples
                print(f"  Avg Iterations per sample: {avg_iters:.2f}")
                print(f"  Avg Grad Calls per sample: {avg_grads:.2f}")

        return final_adv_images
