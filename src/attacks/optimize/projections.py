"""Projection utilities for constraint handling in adversarial attacks."""

import torch


def project_box(
    x: torch.Tensor, min_val: torch.Tensor = 0.0, max_val: torch.Tensor = 1.0
) -> torch.Tensor:
    """
    Project tensor values to the specified range [min_val, max_val].

    Args:
        x: Input tensor
        min_val: Minimum allowed value (can be a tensor for per-channel bounds)
        max_val: Maximum allowed value (can be a tensor for per-channel bounds)

    Returns:
        Tensor with values clamped to the specified range
    """
    if isinstance(min_val, torch.Tensor):
        min_val = min_val.to(device=x.device, dtype=x.dtype)
    if isinstance(max_val, torch.Tensor):
        max_val = max_val.to(device=x.device, dtype=x.dtype)

    return torch.clamp(x, min_val, max_val)


def project_l2_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Project perturbation tensor onto the L2 ball of radius epsilon.

    Instead of only projecting when over the limit, this implementation
    forces all perturbations to have exactly epsilon magnitude when
    they're close to it, similar to how PGD works.

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

    # Avoid division by zero
    eps_tensor = torch.ones_like(l2_norm) * epsilon
    l2_norm = torch.where(l2_norm > 1e-8, l2_norm, torch.ones_like(l2_norm) * 1e-8)

    # Calculate scaling factor
    scaling = torch.min(torch.ones_like(l2_norm), eps_tensor / l2_norm)

    # Scale the perturbation to have at most epsilon magnitude
    scaled_delta = flat_delta * scaling

    # Reshape back to original shape
    return scaled_delta.reshape(original_shape)


def project_linf_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Project perturbation tensor onto the L∞ ball of radius epsilon.

    Args:
        delta: Perturbation tensor
        epsilon: Radius of the L∞ ball

    Returns:
        Projected perturbation tensor
    """
    # Simple element-wise clamping to [-epsilon, epsilon]
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
    min_val: float = None,
    max_val: float = None,
) -> torch.Tensor:
    """
    Project adversarial examples to satisfy both norm constraints and box constraints.

    Args:
        x_adv: Current adversarial examples
        x_orig: Original clean inputs
        epsilon: Maximum perturbation size
        norm: Type of norm constraint ('L2' or 'Linf')
        min_val: Minimum allowed pixel value (default: None, will be calculated from ImageNet stats)
        max_val: Maximum allowed pixel value (default: None, will be calculated from ImageNet stats)

    Returns:
        Projected adversarial examples
    """
    # If min_val and max_val are not provided, we're working in normalized space
    # and should use appropriate bounds for ImageNet normalization
    if min_val is None or max_val is None:
        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406], device=x_adv.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x_adv.device).view(1, 3, 1, 1)

        # Calculate normalized min/max bounds
        min_val = (-mean / std).to(x_adv.device, x_adv.dtype)
        max_val = ((1 - mean) / std).to(x_adv.device, x_adv.dtype)

    # Calculate perturbation
    delta = x_adv - x_orig

    # Project perturbation according to norm constraint
    delta = project_perturbation(delta, epsilon, norm)

    # Apply perturbation and ensure valid image range
    projected = x_orig + delta

    # Clamp to valid normalized range
    return project_box(projected, min_val, max_val)
