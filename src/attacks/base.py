"""Base class for adversarial attacks."""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union


class BaseAttack(ABC):
    """
    Base class for adversarial attacks.

    This class defines the common interface and utility functions that
    all optimization-based attacks should implement, including loss and
    gradient computations, performance tracking, and success checking.
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
            model: The neural network model to attack.
            norm: The norm used for the perturbation constraint ('L2' or 'Linf').
            eps: Maximum allowed perturbation magnitude.
            targeted: Whether the attack is targeted (aim for a specific target) or untargeted.
            loss_fn: The loss function to use ('cross_entropy' or 'margin').
            device: The device to run the attack on (CPU or GPU).
            verbose: Whether to print progress and debug information.
        """
        self.model = model
        self.norm = norm
        self.eps = eps
        self.targeted = targeted
        self.loss_fn = loss_fn
        # Use the provided device, or default to CUDA if available.
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose

        # Performance tracking metrics
        self.total_iterations = 0
        self.total_gradient_calls = 0
        self.total_time = 0

        # Ensure the model is in evaluation mode (important for consistency in attacks).
        self.model.eval()

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute the loss function.

        This method computes the loss used in the attack. It supports both cross-entropy loss,
        which is standard for classification, and margin loss, which is useful for certain attack
        objectives.

        Args:
            outputs: The model outputs (logits).
            targets: The target labels.
            reduction: Specifies the reduction to apply ('none', 'mean', or 'sum').

        Returns:
            The computed loss value. When reduction='none', returns per-example losses.
        """
        if self.loss_fn == "cross_entropy":
            # Standard cross-entropy loss
            return torch.nn.functional.cross_entropy(
                outputs, targets, reduction=reduction
            )
        elif self.loss_fn == "margin":
            # Margin loss aims to maximize the difference between the target class and the next highest logit.
            if self.targeted:
                # For targeted attacks, the goal is to maximize the target class logit
                target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                # Exclude the target class from the max search
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Loss per example: difference between the highest non-target and target logits
                batch_loss = other_logits - target_logits
            else:
                # For untargeted attacks, the goal is to minimize the true class logit relative to others.
                true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Loss per example: difference between true class and the highest non-true logit.
                batch_loss = true_logits - other_logits

            # Apply the requested reduction.
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

        This is a convenience method that computes the gradient by enabling gradient tracking on the input,
        performing a forward pass, computing the loss, and backpropagating.

        Args:
            inputs: The input data for which the gradient is computed.
            targets: The corresponding target labels.

        Returns:
            The gradient of the loss with respect to the inputs.
        """
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        # Use mean reduction to obtain a scalar loss.
        loss = self._compute_loss(outputs, targets, reduction="mean")
        self.model.zero_grad()
        loss.backward()
        # Clone the gradient so that further operations do not affect it.
        grad = inputs.grad.clone()
        inputs.requires_grad_(False)
        self.total_gradient_calls += 1
        return grad

    def _check_success(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if the attack was successful.

        For targeted attacks, success means the model's prediction matches the target label.
        For untargeted attacks, success is when the model misclassifies the input.

        Args:
            outputs: The model outputs (logits).
            targets: The target labels.

        Returns:
            A boolean tensor indicating success for each input.
        """
        if self.targeted:
            # For targeted attacks, the predicted class should match the target.
            return outputs.argmax(dim=1) == targets
        else:
            # For untargeted attacks, the predicted class should differ from the true label.
            return outputs.argmax(dim=1) != targets

    @abstractmethod
    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples.

        This abstract method should be implemented by subclasses to generate adversarial
        examples using a specific optimization method.

        Args:
            inputs: The input data.
            targets: The true labels for untargeted attacks or target labels for targeted attacks.

        Returns:
            A tuple (adversarial_examples, metrics), where metrics includes performance
            details such as iterations, gradient calls, and time.
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
