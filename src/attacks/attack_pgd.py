"""Projected Gradient Descent (PGD) adversarial attack implementation."""

import time
import torch
import torch.nn as nn

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
        self.eps = eps
        self.n_iterations = n_iterations
        self.step_size = step_size
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
            eps=eps,  # Corresponds to ε in the perturbation constraint ||δ||_p ≤ ε
            n_iterations=n_iterations,  # Corresponds to T in the algorithm
            step_size=step_size,  # Corresponds to α in the update equations
            rand_init=rand_init,  # Implements random initialization of δ₀
            early_stopping=early_stopping,  # Implements early stopping condition
            verbose=verbose,
            maximize=True,  # Default for untargeted attacks, will be set in forward()
        )

    def forward(self, images, labels):
        """
        Simplified PGD attack implementation.

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

        # Run the optimizer to generate adversarial examples
        # This is where the actual PGD iteration happens (inside the PGDOptimizer)
        # The optimizer will handle:
        # 1. Initialization (random or zero)
        # 2. Computing gradients at each step
        # 3. Normalizing gradients (sign for L∞, unit vector for L2)
        # 4. Taking gradient steps (α ⋅ normalized_gradient)
        # 5. Projecting to the ε-constraint ball (L2 or L∞)
        # 6. Clipping to valid image range [0,1]
        # 7. Checking early stopping condition
        adv_images, metrics = self.optimizer.optimize(
            x_init=images,
            gradient_fn=gradient_fn,
            success_fn=success_fn,
            x_original=images,
        )

        # Update attack metrics
        self.total_iterations += metrics["iterations"] * images.size(0)

        # End timing
        self.end_time = time.time()
        self.total_time += self.end_time - start_time

        # Ensure outputs are properly clamped to [0,1] - this matches the constraint in the paper
        # that requires x + δ ∈ [0,1]^n (valid image range)
        adv_images = torch.clamp(adv_images, 0, 1).detach()

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
            print(f"Iterations: {metrics['iterations']}")
            print(f"Time per sample: {metrics['time_per_sample']*1000:.2f}ms")

        return adv_images
