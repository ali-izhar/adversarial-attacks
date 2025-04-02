"""Fast Gradient Sign Method (FGSM) adversarial attack implementation.

Code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import torch
import torch.nn as nn

from .attack import Attack


class FFGSM(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 6/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
            `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=6/255)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, alpha=6 / 255):
        """Initialize FFGSM attack.

        Args:
            model: Target model to attack
            eps: Maximum perturbation size (default: 8/255)
            alpha: Step size for gradient update (default: 6/255)

        Raises:
            ValueError: If alpha is greater than epsilon
        """
        super().__init__("FFGSM", model)
        self.eps = eps  # Maximum perturbation size
        self.alpha = alpha  # Step size for gradient update

        # Validate that alpha is less than or equal to epsilon
        if self.alpha > self.eps:
            raise ValueError(
                f"alpha ({self.alpha}) must be less than or equal to eps ({self.eps})"
            )

        # FFGSM supports both untargeted and targeted attacks
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # Use cross-entropy loss for classification tasks
        loss = nn.CrossEntropyLoss()

        # Initialize adversarial images with random noise within epsilon bound
        # This is the key difference from standard FGSM - starting from a random point
        adv_images = images + torch.randn_like(images).uniform_(
            -self.eps, self.eps
        )  # nopep8
        # Ensure the initial perturbation is within valid image range
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        # Enable gradient computation for the perturbed images
        adv_images.requires_grad = True

        # Get model predictions for the randomly perturbed images
        outputs = self.get_logits(adv_images)

        # Calculate loss based on attack mode
        if self.targeted:
            # For targeted attacks, minimize loss with respect to target labels
            cost = -loss(outputs, target_labels)
        else:
            # For untargeted attacks, maximize loss with respect to true labels
            cost = loss(outputs, labels)

        # Compute gradients of loss with respect to perturbed images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        # FFGSM update: take a step in the direction of signed gradients
        # This is similar to FGSM but with a different step size (alpha)
        adv_images = adv_images + self.alpha * grad.sign()

        # Project the perturbation back to epsilon-ball around original images
        # This ensures the total perturbation doesn't exceed epsilon
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        # Final adversarial images are original images plus bounded perturbation
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
