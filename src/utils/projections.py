"""Projection utilities for constraint handling in adversarial attacks."""

import torch


def project_box(
    x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0
) -> torch.Tensor:
    """
    Project tensor values to the specified range [min_val, max_val].

    Args:
        x: Input tensor
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Tensor with values clamped to the specified range
    """
    return torch.clamp(x, min_val, max_val)


def project_l2_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Project perturbation tensor onto the L2 ball of radius epsilon.

    Args:
        delta: Perturbation tensor
        epsilon: Radius of the L2 ball

    Returns:
        Projected perturbation tensor
    """
    # Store original shape for proper reshaping
    original_shape = delta.shape

    # Flatten the tensor for norm calculation
    flat_delta = delta.reshape(delta.shape[0], -1)

    # Calculate the L2 norm of each perturbation
    l2_norm = torch.norm(flat_delta, p=2, dim=1, keepdim=True)

    # Identify perturbations that exceed the radius
    mask = l2_norm > epsilon

    # Project those perturbations back to the epsilon-ball surface
    if mask.any():
        scaling = epsilon / l2_norm
        scaling[~mask] = 1.0  # Only scale perturbations that exceed epsilon
        scaled_delta = flat_delta * scaling

        # Reshape back to original shape using stored shape
        return scaled_delta.reshape(original_shape)

    return delta


def project_linf_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Project perturbation tensor onto the L∞ ball of radius epsilon.

    Args:
        delta: Perturbation tensor
        epsilon: Radius of the L∞ ball

    Returns:
        Projected perturbation tensor
    """
    return torch.clamp(delta, -epsilon, epsilon)


def project_perturbation(
    delta: torch.Tensor, epsilon: float, norm: str
) -> torch.Tensor:
    """
    Project perturbation according to specified norm constraints.

    Args:
        delta: Perturbation tensor
        epsilon: Maximum perturbation size
        norm: Type of norm constraint ('L2' or 'Linf')

    Returns:
        Projected perturbation tensor
    """
    if norm.lower() == "l2":
        return project_l2_ball(delta, epsilon)
    elif norm.lower() in ["linf", "l∞", "l_inf"]:
        return project_linf_ball(delta, epsilon)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


def project_adversarial_example(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    epsilon: float,
    norm: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Project adversarial examples to satisfy both norm constraints and box constraints.

    Args:
        x_adv: Current adversarial examples
        x_orig: Original clean inputs
        epsilon: Maximum perturbation size
        norm: Type of norm constraint ('L2' or 'Linf')
        min_val: Minimum allowed pixel value
        max_val: Maximum allowed pixel value

    Returns:
        Projected adversarial examples
    """
    # Calculate perturbation
    delta = x_adv - x_orig

    # Project perturbation according to norm constraint
    delta = project_perturbation(delta, epsilon, norm)

    # Apply perturbation and ensure valid image range
    return project_box(x_orig + delta, min_val, max_val)
