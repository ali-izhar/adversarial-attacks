"""
Implementation of Momentum Iterative Fast Gradient Sign Method (MI-FGSM).

This attack was introduced in the paper:
"Boosting Adversarial Attacks with Momentum"
by Dong et al., CVPR 2018.

MI-FGSM incorporates momentum into the iterative process to stabilize
update directions and escape from poor local maxima, which significantly
improves the attack success rates.
"""

import torch
from .attack import Attack


class MIFGSM(Attack):
    """
    Momentum Iterative Fast Gradient Sign Method (MI-FGSM).

    This is a stronger iterative variant of FGSM that incorporates momentum
    to stabilize the update directions and escape poor local maxima.

    Paper: https://arxiv.org/abs/1710.06081

    Args:
        model: Model to attack
        eps: Maximum perturbation (Default: 8/255)
        steps: Number of attack iterations (Default: 10)
        alpha: Step size per iteration (Default: 2/255)
        decay_factor: Decay factor for momentum (Default: 1.0)
        targeted: If True, targeted attack; if False, untargeted attack
        norm_type: Type of norm constraint ('Linf' or 'L2')
    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        steps=10,
        alpha=2 / 255,
        decay_factor=1.0,
        targeted=False,
        norm_type="Linf",
    ):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay_factor = decay_factor
        self.targeted = targeted
        self.norm_type = norm_type
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Attack metrics for evaluation
        self.reset_metrics()

    def reset_metrics(self):
        """Reset attack metrics."""
        self.iterations = 0
        self.gradient_calls = 0
        self.success_rate = 0.0
        self.l2_norm = 0.0
        self.linf_norm = 0.0
        self.ssim = 0.0
        self.time_per_sample = 0.0
        self.successful_examples = []

    def set_mode_targeted(self):
        """Set attack to targeted mode."""
        self.targeted = True

    def set_mode_default(self):
        """Set attack to default (untargeted) mode."""
        self.targeted = False

    def set_mode_targeted_least_likely(self):
        """Set attack to target least likely class."""
        self.targeted = True

    def __call__(self, x, y):
        """
        Perform MI-FGSM attack on a batch of images.

        Args:
            x: Input images, batch of tensors with shape [B, C, H, W]
            y: Target labels for targeted attack, or true labels for untargeted attack

        Returns:
            Adversarial examples
        """
        images = x.clone().detach().to(self.device)
        labels = y.clone().detach().to(self.device)

        # Initialize adversarial examples
        adv_images = images.clone().detach()

        # Initialize momentum accumulator
        momentum = torch.zeros_like(images).detach().to(self.device)

        # Metrics initialization
        self.iterations = self.steps
        self.gradient_calls = self.steps

        # Perform attack iterations
        for i in range(self.steps):
            adv_images.requires_grad = True

            # Forward pass
            outputs = self.model(adv_images)

            # If targeted, maximize the probability of target class
            # If untargeted, minimize the probability of true class
            if self.targeted:
                # Targeted attack aims to maximize the loss for the target label
                loss = -self.loss_fn(outputs, labels)
            else:
                # Untargeted attack aims to minimize the loss for the true label
                loss = self.loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Update metrics
            self.gradient_calls += 1

            # Get gradients
            grad = adv_images.grad.data

            # Normalize gradients using mean absolute value (more stable than simple L1 norm)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

            # Update momentum with current gradient
            grad = grad + momentum * self.decay_factor
            momentum = grad

            # Create perturbation based on norm type
            if self.norm_type == "Linf":
                # Linf norm uses sign of gradient
                adv_images = adv_images.detach() + self.alpha * grad.sign()

                # Project back to epsilon ball
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            else:  # L2 norm
                # L2 norm constrains the magnitude of the perturbation
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(
                    -1, 1, 1, 1
                )
                grad_normalized = grad / (grad_norm + 1e-10)

                # Apply perturbation
                adv_images = adv_images.detach() + self.alpha * grad_normalized

                # Project back to epsilon ball
                delta = adv_images - images
                delta_norm = torch.norm(
                    delta.view(delta.shape[0], -1), p=2, dim=1
                ).view(-1, 1, 1, 1)
                factor = self.eps / (delta_norm + 1e-10)
                factor = torch.min(factor, torch.ones_like(factor))
                delta = delta * factor

                # Apply clamping
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        # Calculate attack metrics
        with torch.no_grad():
            # Calculate perturbation norms
            perturbation = adv_images.detach() - images
            self.l2_norm = (
                torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1)
                .mean()
                .item()
            )
            self.linf_norm = (
                torch.norm(
                    perturbation.view(perturbation.shape[0], -1), p=float("inf"), dim=1
                )
                .mean()
                .item()
            )

            # Check attack success
            final_outputs = self.model(adv_images)
            final_predictions = final_outputs.argmax(dim=1)

            if self.targeted:
                success_mask = final_predictions == labels
            else:
                original_predictions = self.model(images).argmax(dim=1)
                success_mask = final_predictions != original_predictions

            success_count = success_mask.sum().item()
            self.success_rate = 100 * (success_count / len(labels))

        return adv_images.detach()

    def get_metrics(self):
        """Return attack metrics."""
        return {
            "success_rate": self.success_rate,
            "l2_norm": self.l2_norm,
            "linf_norm": self.linf_norm,
            "ssim": self.ssim,
            "iterations": self.iterations,
            "gradient_calls": self.gradient_calls,
            "time_per_sample": self.time_per_sample,
        }

    def compute_perturbation_metrics(self, original_images, adversarial_images):
        """Compute metrics for the perturbation."""
        with torch.no_grad():
            batch_size = original_images.shape[0]
            perturbation = adversarial_images.detach() - original_images

            l2_norm = (
                torch.norm(perturbation.view(batch_size, -1), p=2, dim=1).mean().item()
            )
            linf_norm = (
                torch.norm(perturbation.view(batch_size, -1), p=float("inf"), dim=1)
                .mean()
                .item()
            )

            # SSIM calculation would go here, but we'll use a placeholder for now
            ssim = 0.0

        return {"l2_norm": l2_norm, "linf_norm": linf_norm, "ssim": ssim}

    def evaluate_attack_success(self, original_images, adversarial_images, true_labels):
        """
        Evaluate if the attack was successful for each image in the batch.

        Args:
            original_images: Original images
            adversarial_images: Adversarial images
            true_labels: True class labels

        Returns:
            Tuple of (success_rate, success_mask, (original_preds, adversarial_preds))
        """
        with torch.no_grad():
            # Get predictions for original and adversarial images
            original_outputs = self.model(original_images)
            original_preds = original_outputs.argmax(dim=1)

            adversarial_outputs = self.model(adversarial_images)
            adversarial_preds = adversarial_outputs.argmax(dim=1)

            # Determine success based on attack mode
            if self.targeted:
                # Targeted: Success if prediction matches target
                success_mask = adversarial_preds == true_labels
            else:
                # Untargeted: Success if prediction differs from original prediction
                success_mask = adversarial_preds != original_preds

            # Calculate success rate
            success_rate = 100 * (success_mask.sum().item() / len(true_labels))

        return success_rate, success_mask, (original_preds, adversarial_preds)
