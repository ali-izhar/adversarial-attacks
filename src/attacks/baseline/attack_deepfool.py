"""DeepFool adversarial attack implementation.

Code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import torch
import torch.nn as nn
import time

from .attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
            `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, steps=50, overshoot=0.02):
        """Initialize DeepFool attack.

        Args:
            model: Target model to attack
            steps: Maximum number of iterations for finding adversarial examples
            overshoot: Parameter to enhance the noise (default: 0.02)
        """
        super().__init__("DeepFool", model)
        self.steps = steps  # Maximum number of iterations
        self.overshoot = overshoot  # Parameter to enhance the noise
        # DeepFool only supports untargeted attacks
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # Call the main implementation and return only the adversarial images
        adv_images, _ = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        r"""
        Optimized batch implementation of DeepFool.
        """
        # Start timer for performance tracking
        start_time = time.time()

        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        original_images = images.clone()
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)

        # Get initial predictions and prepare tensors
        with torch.no_grad():
            logits = self.get_logits(images)
            _, initial_preds = torch.max(logits, dim=1)

        # Keep track of fooled samples
        fooled = initial_preds != labels
        target_labels = initial_preds.clone()

        # Initialize perturbations with zeros
        r_tot = torch.zeros_like(images)
        curr_steps = 0

        # Store per-sample iterations for metrics
        sample_iterations = torch.zeros(batch_size, device=self.device)
        active_samples = torch.logical_not(fooled)

        # Get the number of classes
        num_classes = logits.shape[1]

        # Main iteration loop
        while active_samples.any() and curr_steps < self.steps:
            # Keep only active samples (not yet fooled)
            curr_images = torch.clamp(images + r_tot, 0, 1)
            curr_images.requires_grad_(True)

            # Get current logits
            fs = self.get_logits(curr_images)

            # Get predictions
            _, curr_preds = torch.max(fs, dim=1)

            # Update fooled status
            newly_fooled = (curr_preds != labels) & active_samples
            if newly_fooled.any():
                target_labels[newly_fooled] = curr_preds[newly_fooled]
                active_samples = torch.logical_not(newly_fooled) & active_samples

                # Exit early if all samples are fooled
                if not active_samples.any():
                    break

            # Initialize perturbation for this step
            w_norm_list = []
            f_k_list = []
            grad_list = []
            k_list = []

            # For active samples, compute gradients for all classes in parallel
            pert = torch.ones_like(curr_images) * float("inf")

            # Process only active samples
            active_idx = torch.where(active_samples)[0]
            if len(active_idx) == 0:
                break

            # For each active sample, find closest boundary
            for i in active_idx:
                # Current sample and true label
                x_i = curr_images[i : i + 1].detach().requires_grad_(True)
                true_label = labels[i].item()

                # Get current prediction and logits
                fs_i = self.get_logits(x_i)[0]
                f_0 = fs_i[true_label]

                # Find gradient for true class
                if x_i.grad is not None:
                    x_i.grad.zero_()
                f_0.backward(retain_graph=True)
                self.total_gradient_calls += 1
                grad_0 = x_i.grad.clone()

                # Find the closest decision boundary
                min_dist = float("inf")
                closest_k = -1
                w_k_best = None
                f_k_best = None

                # Compute gradients for other classes
                for k in range(num_classes):
                    if k == true_label:
                        continue

                    # Reset gradients
                    if x_i.grad is not None:
                        x_i.grad.zero_()

                    # Compute gradient for class k
                    fs_i[k].backward(retain_graph=(k < num_classes - 1))
                    self.total_gradient_calls += 1

                    # Compute decision boundary parameters
                    w_k = x_i.grad.clone() - grad_0
                    f_k = fs_i[k] - f_0

                    # Skip if gradient is too small
                    w_k_norm = torch.norm(w_k.flatten())
                    if w_k_norm < 1e-8:
                        continue

                    # Compute distance to boundary
                    dist_k = torch.abs(f_k) / w_k_norm

                    # Update if this boundary is closer
                    if dist_k < min_dist:
                        min_dist = dist_k
                        closest_k = k
                        w_k_best = w_k
                        f_k_best = f_k

                # Compute perturbation
                if closest_k >= 0:
                    w_k_norm = torch.norm(w_k_best.flatten())
                    r_i = torch.abs(f_k_best) / (w_k_norm**2 + 1e-10) * w_k_best
                    pert[i : i + 1] = r_i

                # Update sample iterations count
                sample_iterations[i] += 1

            # Apply perturbation with overshoot
            r_tot = r_tot + (1 + self.overshoot) * pert

            # Ensure the perturbed images are within valid range
            perturbed_images = torch.clamp(original_images + r_tot, 0, 1)
            r_tot = perturbed_images - original_images

            # Update images for next iteration (no need to clone since we don't modify)
            images = original_images

            # Count iterations for all active samples
            self.total_iterations += active_samples.sum().item()
            curr_steps += 1

        # Final perturbed images
        adv_images = torch.clamp(original_images + r_tot, 0, 1).detach()

        # Calculate perturbation metrics
        perturbation_metrics = self.compute_perturbation_metrics(
            original_images, adv_images
        )

        # Print metrics for reporting
        avg_iters = sample_iterations.sum().item() / batch_size
        print(
            f"DeepFool stats: avg iterations={avg_iters:.2f}, fooled {batch_size - active_samples.sum().item()}/{batch_size} samples"
        )
        print(
            f"DeepFool metrics - L2: {perturbation_metrics['l2_norm']:.4f}, L∞: {perturbation_metrics['linf_norm']:.4f}, SSIM: {perturbation_metrics['ssim']:.4f}"
        )

        # Measure and record time taken
        end_time = time.time()
        self.total_time += end_time - start_time

        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        """Process a single image in DeepFool algorithm.

        Args:
            image: Input image to process
            label: True label of the image

        Returns:
            early_stop: Whether to stop processing this image
            pre: Predicted label
            perturbation: Perturbation to add to the image
        """
        # Get a copy of the image for gradient computation
        image_copy = image.clone().detach().requires_grad_(True)

        # Get model predictions
        fs = self.get_logits(image_copy)[0]
        _, pre = torch.max(fs, dim=0)

        # If already misclassified, return early with no perturbation
        if pre != label:
            return (True, pre, torch.zeros_like(image))

        # Get the score for the true class
        f_0 = fs[label]

        # Get scores for all other classes
        wrong_classes = [i for i in range(len(fs)) if i != label]

        # If no wrong classes (unlikely in practice), return early
        if len(wrong_classes) == 0:
            return (True, pre, torch.zeros_like(image))

        f_k = fs[wrong_classes]

        # Calculate gradients for each wrong class
        w_k = []
        for k in wrong_classes:
            if image_copy.grad is not None:
                image_copy.grad.zero_()
            fs[k].backward(retain_graph=True)
            # Each backward pass counts as a gradient call
            self.total_gradient_calls += 1
            w_k.append(image_copy.grad.clone().detach())

        # Calculate gradients for true class
        if image_copy.grad is not None:
            image_copy.grad.zero_()
        f_0.backward(retain_graph=False)
        # This backward pass is another gradient call
        self.total_gradient_calls += 1
        w_0 = image_copy.grad.clone().detach()

        # Calculate the difference in scores and gradients
        f_prime = f_k - f_0
        w_prime = torch.stack(w_k) - w_0

        # Calculate the distance to decision boundary for each class
        # Add small epsilon to avoid division by zero
        norm_values = torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        value = torch.abs(f_prime) / (norm_values + 1e-10)

        # Find the closest decision boundary
        _, hat_L = torch.min(value, 0)

        # Calculate the perturbation vector
        norm_squared = torch.norm(w_prime[hat_L], p=2) ** 2
        # Add small epsilon to avoid division by zero
        delta = torch.abs(f_prime[hat_L]) * w_prime[hat_L] / (norm_squared + 1e-10)

        # Determine the target label
        target_label = wrong_classes[hat_L]

        # Return the perturbation with overshoot
        perturbation = (1 + self.overshoot) * delta

        # Make sure the perturbation respects image bounds when applied
        perturbed_image = image + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbation = perturbed_image - image

        return (False, target_label, perturbation)

    def _construct_jacobian(self, y, x):
        """Construct the Jacobian matrix of model outputs with respect to inputs.

        Args:
            y: Model outputs
            x: Input tensor

        Returns:
            Jacobian matrix
        """
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            # Compute gradients for each output element
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            # Count each backward pass as a gradient call
            self.total_gradient_calls += 1
            x_grads.append(x.grad.clone().detach())
        # Stack all gradients to form the Jacobian matrix
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
