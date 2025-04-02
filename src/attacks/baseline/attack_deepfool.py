"""DeepFool adversarial attack implementation.

Code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
"""

import torch
import torch.nn as nn

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
        adv_images, target_labels = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        r"""
        Overridden.
        """
        # Clone and detach input images to avoid modifying the original data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        # Track which images still need to be fooled
        correct = torch.tensor([True] * batch_size)
        # Store the target labels that the model predicts
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        # Initialize adversarial images list
        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)

        # Main DeepFool loop: continue until all images are fooled or max steps reached
        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                # Process each image individually
                early_stop, pre, adv_image = self._forward_indiv(
                    adv_images[idx], labels[idx]
                )
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        # Concatenate all processed images and return
        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        """Process a single image in DeepFool algorithm.

        Args:
            image: Input image to process
            label: True label of the image

        Returns:
            early_stop: Whether to stop processing this image
            pre: Predicted label
            adv_image: Adversarial image
        """
        # Enable gradient computation for the image
        image.requires_grad = True
        # Get model predictions
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        # If already misclassified, return early
        if pre != label:
            return (True, pre, image)

        # Get the score for the true class
        f_0 = fs[label]

        # Get scores and gradients for all other classes
        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]

        # Calculate gradients for each wrong class
        w_k = []
        for k in wrong_classes:
            if image.grad is not None:
                image.grad.zero_()
            fs[k].backward(retain_graph=True)
            w_k.append(image.grad.clone().detach())

        # Calculate gradients for true class
        if image.grad is not None:
            image.grad.zero_()
        f_0.backward(retain_graph=False)
        w_0 = image.grad.clone().detach()

        # Calculate the difference in scores and gradients
        f_prime = f_k - f_0
        w_prime = torch.stack(w_k) - w_0

        # Calculate the distance to decision boundary for each class
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        # Find the closest decision boundary
        _, hat_L = torch.min(value, 0)

        # Calculate the perturbation vector
        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        # Determine the target label
        target_label = wrong_classes[hat_L]

        # Apply the perturbation with overshoot
        adv_image = image + (1 + self.overshoot) * delta
        # Ensure the adversarial image is within valid range
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

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
            x_grads.append(x.grad.clone().detach())
        # Stack all gradients to form the Jacobian matrix
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
