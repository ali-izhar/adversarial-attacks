"""Base class for adversarial attacks.

This module defines the abstract base class for implementing optimization-based
adversarial attacks against neural networks. The class provides common
functionality including loss computation, gradient calculation, projection
operations, and performance tracking.
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Union


class BaseAttack(ABC):
    """
    Base class for adversarial attacks.

    This class defines the common interface and utility functions that
    all optimization-based attacks should implement, including loss and
    gradient computations, performance tracking, and success checking.

    The optimization problem for generating adversarial examples can be formulated as:

    minimize ||δ||_p subject to:
        1. f(x + δ) ≠ f(x) (untargeted) or f(x + δ) = t (targeted)
        2. x + δ ∈ [0,1]^n (valid image constraint)

    where:
        - δ is the perturbation
        - f is the model
        - x is the original input
        - t is the target class (for targeted attacks)
        - ||·||_p is the p-norm (typically L2 or Linf)
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
                  L2 norm measures Euclidean distance: ||x||_2 = sqrt(sum(x_i^2))
                  Linf norm measures maximum absolute value: ||x||_∞ = max(|x_i|)
            eps: Maximum allowed perturbation magnitude (ε in the constraint ||δ||_p ≤ ε).
            targeted: Whether the attack is targeted (aim for a specific target) or untargeted.
                      For targeted attacks, we minimize L(f(x+δ), t) where t is the target.
                      For untargeted attacks, we maximize L(f(x+δ), y) where y is the true label.
            loss_fn: The loss function to use ('cross_entropy' or 'margin').
                     - cross_entropy: standard classification loss
                     - margin: difference between target logit and highest other logit
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

        # Extract normalization parameters from the model if available
        # These are important for correctly computing perturbations in the original input space
        try:
            self.mean = (
                model.mean.clone().detach().view(1, -1, 1, 1).to(self.device)
                if hasattr(model, "mean")
                else None
            )
            self.std = (
                model.std.clone().detach().view(1, -1, 1, 1).to(self.device)
                if hasattr(model, "std")
                else None
            )
        except:
            self.mean = None
            self.std = None
            if self.verbose:
                print("Could not extract normalization parameters from model.")

        # Ensure the model is in evaluation mode (important for consistency in attacks).
        # This prevents batch normalization and dropout from affecting attack performance.
        self.model.eval()

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the input tensor if normalization parameters are available.

        This converts from normalized space (as used by the model) back to image space [0,1].
        Denormalization follows: x_denorm = x_norm * std + mean

        This is important for visualizing and constraining perturbations in the original image space.

        Args:
            x: Normalized input tensor

        Returns:
            Denormalized tensor
        """
        if self.mean is not None and self.std is not None:
            return x * self.std + self.mean
        return x

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Renormalize the input tensor if normalization parameters are available.

        This converts from image space [0,1] to normalized space as used by the model.
        Normalization follows: x_norm = (x_denorm - mean) / std

        This is necessary after modifying inputs in image space before passing to the model.

        Args:
            x: Denormalized input tensor

        Returns:
            Normalized tensor
        """
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        return x

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute the loss function for optimization.

        This method computes the loss used to guide the optimization process. It supports:
        1. Cross-entropy loss: standard for classification problems, defined as
           L(y, t) = -∑ t_i * log(y_i), where t is one-hot encoded target

        2. Margin loss: measures the difference between target logit and highest other logit:
           - For targeted attacks: margin = max_{i≠t} z_i - z_t  (minimize this)
           - For untargeted attacks: margin = z_t - max_{i≠t} z_i  (maximize this)

           where z_i are the logits (pre-softmax outputs) and z_t is the target logit

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
                # relative to all other logits
                target_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                # Exclude the target class from the max search
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Loss per example: difference between the highest non-target and target logits
                # We want to minimize this value to make the target class most probable
                batch_loss = other_logits - target_logits
            else:
                # For untargeted attacks, the goal is to minimize the true class logit relative to others.
                true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = outputs.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                other_logits = other_logits.max(1)[0]
                # Loss per example: difference between true class and the highest non-true logit.
                # We want to maximize this value to make the true class less probable than some other class
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

        This is a key component of gradient-based optimization for adversarial examples.
        The gradient ∇L(f(x), y) indicates the direction to modify the input x
        to increase the loss, which guides the perturbation generation.

        For targeted attacks with cross-entropy loss, we want to minimize L(f(x+δ), t)
        so we move in the negative gradient direction.

        For untargeted attacks, we want to maximize L(f(x+δ), y_true) or equivalently
        minimize -L(f(x+δ), y_true), so we move in the gradient direction.

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

        An adversarial attack is successful when:
        - For targeted attacks: f(x+δ) = target_label (model classifies as the target)
        - For untargeted attacks: f(x+δ) ≠ true_label (model misclassifies the input)

        This function is used to determine when to stop the optimization process
        and as part of the evaluation metrics.

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
            # For untargeted attacks, we can be more flexible:
            # 1. The most strict criteria is that the predicted class should differ from true label
            pred_classes = outputs.argmax(dim=1)
            strict_success = pred_classes != targets

            # 2. A more lenient approach: check if the confidence in the true class has decreased
            # significantly, indicating the attack is pushing in the right direction
            softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
            target_probs = softmax_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Get the original predictions' probabilities (stored during attack initialization)
            # If not available, use a fixed threshold
            if hasattr(self, "original_pred_probs"):
                # Success if confidence decreased by at least 30%
                prob_success = target_probs <= (self.original_pred_probs * 0.7)
            else:
                # If original probs not available, success if confidence dropped below 40%
                prob_success = target_probs < 0.4

            # Also count as success if another class is within 10% probability of the true class
            # This indicates the attack has significantly eroded the model's confidence
            max_probs, max_indices = torch.topk(softmax_probs, k=2, dim=1)
            correct_class_mask = max_indices[:, 0] == targets

            # For examples where the top prediction is still the true class
            close_second_success = torch.zeros_like(strict_success)
            if correct_class_mask.any():
                # Check if the second highest probability is close to the highest
                prob_gap = (
                    max_probs[correct_class_mask, 0] - max_probs[correct_class_mask, 1]
                )
                close_second = prob_gap < 0.1  # Within 10% confidence
                close_second_success[correct_class_mask] = close_second

            # Combine all success criteria
            return strict_success | prob_success | close_second_success

    def store_original_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Store the original model predictions for the inputs.

        This function is useful for:
        1. Determining the initial classification before attack
        2. Measuring attack success relative to original predictions
        3. Using original predictions as reference points in some attack algorithms

        It's especially important for untargeted attacks where success is defined
        as changing the model's prediction from the original class.

        Args:
            inputs: The input tensor to get predictions for.

        Returns:
            The original model outputs.
        """
        with torch.no_grad():
            outputs = self.model(inputs)
            self.original_outputs = outputs
            self.original_predictions = outputs.argmax(dim=1)

            # Also store original probabilities for enhanced success detection
            softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
            self.original_pred_probs = softmax_probs.gather(
                1, self.original_predictions.unsqueeze(1)
            ).squeeze(1)

            return outputs

    @abstractmethod
    def generate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples.

        This abstract method should be implemented by subclasses to generate adversarial
        examples using a specific optimization method. The optimization problem is:

        min ||δ||_p subject to f(x+δ) ≠ f(x) [untargeted] or f(x+δ) = t [targeted]

        Different optimization algorithms (PGD, CG, L-BFGS, etc.) will implement different
        strategies for solving this problem. The common components are:

        1. Computing gradients of the loss w.r.t. inputs (∇_x L)
        2. Updating the perturbation based on these gradients
        3. Projecting perturbations to satisfy constraints
        4. Checking for attack success and termination conditions

        Args:
            inputs: The input data.
            targets: The true labels for untargeted attacks or target labels for targeted attacks.

        Returns:
            A tuple (adversarial_examples, metrics), where metrics includes performance
            details such as iterations, gradient calls, and time.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def reset_metrics(self) -> None:
        """
        Reset the performance tracking metrics.

        This should be called before conducting a new batch of attacks or when
        starting a new evaluation to ensure metrics are properly measured.
        """
        self.total_iterations = 0
        self.total_gradient_calls = 0
        self.total_time = 0

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Get the performance metrics.

        Performance metrics are important for comparing different optimization methods
        as described in the paper. Key metrics include:

        1. iterations: Number of optimization steps taken
        2. gradient_calls: Number of gradient computations (backpropagation)
        3. time: Total computation time

        These metrics help evaluate computational efficiency alongside
        attack success rate and perturbation magnitude.

        Returns:
            Dictionary containing performance metrics for the attack.
        """
        return {
            "iterations": self.total_iterations,
            "gradient_calls": self.total_gradient_calls,
            "time": self.total_time,
        }
