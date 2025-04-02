"""Fast Gradient Sign Method (FGSM) adversarial attack implementation.

Code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import torch
import torch.nn as nn
import time

from .attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
            `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        """Initialize FGSM attack.

        Args:
            model: Target model to attack
            eps: Maximum perturbation size (default: 8/255)
        """
        super().__init__("FGSM", model)
        self.eps = eps  # Maximum perturbation size
        # FGSM supports both untargeted and targeted attacks
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Note start time for performance tracking
        start_time = time.time()

        # For targeted attacks, get the target labels
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # Use cross-entropy loss for classification tasks
        loss = nn.CrossEntropyLoss()

        # Enable gradient computation for input images
        images.requires_grad = True

        # Get model predictions
        outputs = self.get_logits(images)

        # FGSM is a single-step method, so we count this as one iteration per sample
        self.total_iterations += images.size(0)

        # Calculate loss based on attack mode
        if self.targeted:
            # For targeted attacks, minimize loss with respect to target labels
            cost = -loss(outputs, target_labels)
        else:
            # For untargeted attacks, maximize loss with respect to true labels
            cost = loss(outputs, labels)

        # Compute gradients of loss with respect to input images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        # FGSM update: add signed gradients scaled by epsilon
        # The sign of the gradient indicates the direction of steepest ascent
        adv_images = images + self.eps * grad.sign()

        # Clamp values to valid image range [0,1]
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # Track how long the attack took
        end_time = time.time()
        self.total_time += end_time - start_time

        return adv_images
