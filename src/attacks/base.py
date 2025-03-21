"""Base class for adversarial attacks."""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union


class BaseAttack(ABC):
    """
    Base class for adversarial attacks.

    This defines the interface that all optimization-based attacks should implement.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm: str = "L2",
        eps: float = 0.5,
        targeted: bool = False,
        loss_fn: str = "cross_entropy",
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        """
        Initialize the attack.

        Args:
            model: The model to attack
            norm: The norm to use for the perturbation constraint ('L2' or 'Linf')
            eps: The maximum perturbation size
            targeted: Whether to perform a targeted attack
            loss_fn: The loss function to use ('cross_entropy' or 'margin')
            device: The device to use (CPU or GPU)
            verbose: Whether to print progress information
        """
        self.model = model
        self.norm = norm
        self.eps = eps
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose

        # Performance tracking
        self.total_iterations = 0
        self.total_gradient_calls = 0
        self.total_time = 0

        # Put model in evaluation mode
        self.model.eval()

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute the loss function.

        Args:
            outputs: The model outputs
            targets: The target labels
            reduction: How to reduce the loss ('none', 'mean', 'sum')

        Returns:
            The loss value (per-example losses if reduction='none')
        """
        if self.loss_fn == "cross_entropy":
            return torch.nn.functional.cross_entropy(
                outputs, targets, reduction=reduction
            )
        elif self.loss_fn == "margin":
            # Margin loss maximizes the difference between the target class and all other classes
            if self.targeted:
                # For targeted attacks, we want to maximize the target class
                target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Return per-example losses
                batch_loss = other_logits - target_logits
            else:
                # For untargeted attacks, we want to minimize the true class
                true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Return per-example losses
                batch_loss = true_logits - other_logits

            # Apply reduction if requested
            if reduction == "mean":
                return batch_loss.mean()
            elif reduction == "sum":
                return batch_loss.sum()
            elif reduction == "none":
                return batch_loss
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    def _compute_gradient(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the loss with respect to the inputs.

        Args:
            inputs: The input data
            targets: The target labels

        Returns:
            The gradient of the loss with respect to the inputs
        """
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        # Use mean reduction for backward pass to get proper gradients
        loss = self._compute_loss(outputs, targets, reduction="mean")
        self.model.zero_grad()
        loss.backward()
        grad = inputs.grad.clone()
        inputs.requires_grad_(False)
        self.total_gradient_calls += 1
        return grad

    def _check_success(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if the attack was successful.

        Args:
            outputs: The model outputs
            targets: The target labels

        Returns:
            A boolean tensor indicating success for each input
        """
        if self.targeted:
            return outputs.argmax(dim=1) == targets
        else:
            return outputs.argmax(dim=1) != targets

    @abstractmethod
    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples.

        Args:
            inputs: The input data
            targets: The true labels for untargeted attacks, target labels for targeted attacks

        Returns:
            A tuple of (adversarial_examples, metrics)

        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def reset_metrics(self) -> None:
        """Reset the performance tracking metrics."""
        self.total_iterations = 0
        self.total_gradient_calls = 0
        self.total_time = 0

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get the performance metrics."""
        return {
            "iterations": self.total_iterations,
            "gradient_calls": self.total_gradient_calls,
            "time": self.total_time,
        }
