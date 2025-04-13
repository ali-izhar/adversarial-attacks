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
    projected = torch.clamp(delta, -epsilon, epsilon)

    # Verify that the projection worked correctly - helpful for debugging
    max_abs_val = projected.abs().max().item()
    if max_abs_val > epsilon + 1e-5:
        print(
            f"WARNING: Linf projection failed! Max abs value: {max_abs_val:.6f} > epsilon: {epsilon:.6f}"
        )

    return projected


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
    x_original: torch.Tensor,
    eps: float,
    norm: str,
    min_bound: torch.Tensor = None,
    max_bound: torch.Tensor = None,
) -> torch.Tensor:
    """
    Project adversarial examples onto the epsilon ball around original images
    and ensure they are valid images.

    Args:
        x_adv: Adversarial examples
        x_original: Original images
        eps: Perturbation budget (epsilon)
        norm: Norm type ('l2' or 'linf')
        min_bound: Minimum valid pixel values for normalized images
        max_bound: Maximum valid pixel values for normalized images

    Returns:
        Projected adversarial examples
    """
    # Use default bounds if not provided
    if min_bound is None:
        min_bound = 0.0
    if max_bound is None:
        max_bound = 1.0

    # Ensure tensors are on the same device
    device = x_adv.device

    # If min_bound and max_bound are not tensors, convert them
    if not isinstance(min_bound, torch.Tensor):
        min_bound = torch.tensor(min_bound, device=device)
    if not isinstance(max_bound, torch.Tensor):
        max_bound = torch.tensor(max_bound, device=device)

    # Ensure bounds are properly shaped for broadcasting
    if min_bound.dim() == 0:
        min_bound = min_bound.view(1, 1, 1, 1)
    if max_bound.dim() == 0:
        max_bound = max_bound.view(1, 1, 1, 1)

    # Make a copy to avoid modifying the input
    x_adv_projected = x_adv.clone()

    # Compute the perturbation
    delta = x_adv_projected - x_original

    # Project perturbation onto the epsilon ball
    if norm.lower() == "l2":
        # L2 projection
        batch_size = delta.shape[0]
        delta_flat = delta.view(batch_size, -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)

        # Project all perturbations - apply scaling factor
        # This avoids the index error with masks
        factor = torch.ones_like(delta_norm)

        # Only modify factor where norm exceeds epsilon
        too_large = delta_norm > eps
        factor[too_large] = eps / delta_norm[too_large]

        # Apply the scaling factor to each perturbation
        delta_flat = delta_flat * factor

        # Reshape back to original shape
        delta = delta_flat.view(delta.shape)
    else:  # Linf projection
        # Element-wise clipping for Linf
        delta = torch.clamp(delta, min=-eps, max=eps)

    # Apply projected perturbation
    x_adv_projected = x_original + delta

    # Ensure valid image bounds
    x_adv_projected = torch.clamp(x_adv_projected, min=min_bound, max=max_bound)

    return x_adv_projected
