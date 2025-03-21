import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """
    Base class for all adversarial attack implementations.

    Parameters:
    -----------
    model : torch.nn.Module
        The target model to attack
    epsilon : float
        Maximum perturbation magnitude
    norm : str
        Norm type for constraint ('L2' or 'Linf')
    targeted : bool
        Whether to perform a targeted attack
    """

    def __init__(self, model, epsilon=0.3, norm="L2", targeted=False):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.targeted = targeted

        # Set device
        self.device = next(model.parameters()).device

        # Set loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Ensure model is in eval mode
        self.model.eval()

    def _project(self, x, x_orig):
        """
        Project perturbation according to specified norm constraints.

        Parameters:
        -----------
        x : torch.Tensor
            Current perturbed image
        x_orig : torch.Tensor
            Original unperturbed image

        Returns:
        --------
        torch.Tensor
            Projected image that satisfies constraints
        """
        # Calculate perturbation
        delta = x - x_orig

        # Project based on norm type
        if self.norm == "L2":
            # L2 projection
            norm = torch.norm(delta.view(delta.shape[0], -1), dim=1)
            mask = norm > self.epsilon
            if mask.any():
                delta[mask] = delta[mask] / norm[mask].view(-1, 1, 1, 1) * self.epsilon
        elif self.norm == "Linf":
            # L∞ projection
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        # Ensure valid image range [0, 1]
        return torch.clamp(x_orig + delta, 0, 1)

    @abstractmethod
    def generate(self, x, y):
        """
        Generate adversarial examples.

        Parameters:
        -----------
        x : torch.Tensor
            Original input images
        y : torch.Tensor
            True labels for untargeted attack, target labels for targeted attack

        Returns:
        --------
        torch.Tensor
            Adversarial examples
        """
        pass

    def _compute_loss(self, x, y):
        """
        Compute the loss for optimization.

        Parameters:
        -----------
        x : torch.Tensor
            Current input
        y : torch.Tensor
            Target labels

        Returns:
        --------
        torch.Tensor
            Loss value
        """
        outputs = self.model(x)

        if self.targeted:
            # For targeted attacks, minimize loss to target
            return -self.loss_fn(outputs, y)
        else:
            # For untargeted attacks, maximize loss from true class
            return self.loss_fn(outputs, y)

    def success_rate(self, x_adv, x_orig, y):
        """
        Calculate attack success rate.

        Parameters:
        -----------
        x_adv : torch.Tensor
            Adversarial examples
        x_orig : torch.Tensor
            Original inputs
        y : torch.Tensor
            True labels

        Returns:
        --------
        float
            Success rate (percentage of successful attacks)
        """
        with torch.no_grad():
            orig_preds = self.model(x_orig).argmax(dim=1)
            adv_preds = self.model(x_adv).argmax(dim=1)

            if self.targeted:
                # For targeted attacks, success means prediction matches target
                success = (adv_preds == y).float().mean().item()
            else:
                # For untargeted attacks, success means prediction differs from original
                success = (adv_preds != y).float().mean().item()

        return success * 100  # Return as percentage
