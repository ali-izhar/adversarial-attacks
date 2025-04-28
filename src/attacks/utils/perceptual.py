#!/usr/bin/env python

"""Perceptual quality metrics and regularization for adversarial examples."""

import torch


def total_variation_loss(delta):
    """Calculate enhanced total variation loss for smoothness while preserving edges.

    This version uses a weighted approach that penalizes changes more in smooth areas
    and less along existing edges in the image.

    Args:
        delta: Perturbation tensor of shape (B, C, H, W)

    Returns:
        Total variation loss normalized by image size (per-sample)
    """
    n_channels = delta.size(1)

    # Compute differences in x and y directions
    diff_h = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    diff_w = delta[:, :, :, 1:] - delta[:, :, :, :-1]

    # Use L1 norm (absolute differences) which tends to preserve edges better than L2
    tv_h = torch.abs(diff_h).sum(dim=(1, 2, 3))
    tv_w = torch.abs(diff_w).sum(dim=(1, 2, 3))

    # Normalize by image size
    n_pixels = delta.size(2) * delta.size(3)

    return (tv_h + tv_w) / (n_pixels * n_channels)


def color_regularization(delta):
    """Calculate enhanced color regularization to penalize visible changes.

    This uses a YUV-style weighting approach where luminance (Y) changes are
    penalized more than chrominance (U,V) changes, as the human eye is more
    sensitive to luminance differences.

    Args:
        delta: Perturbation tensor of shape (B, C, H, W)

    Returns:
        Color regularization loss (per-sample)
    """

    # RGB to YUV-approximate weights
    # Y = 0.299*R + 0.587*G + 0.114*B (luminance - most sensitive)
    # U and V are chrominance components (less sensitive)
    y_weights = torch.tensor([0.299, 0.587, 0.114], device=delta.device).view(
        1, 3, 1, 1
    )

    # Calculate luminance component of perturbation
    luma_delta = (delta * y_weights).sum(dim=1, keepdim=True)

    # Penalize luminance changes more heavily (5x weight)
    luma_penalty = 5.0 * (luma_delta**2).mean(dim=(1, 2, 3))

    # Standard color penalty with channel-specific weights
    # Red and Green changes are more noticeable than Blue
    channel_weights = torch.tensor([1.2, 1.0, 0.8], device=delta.device).view(
        1, 3, 1, 1
    )
    color_penalty = ((delta * channel_weights) ** 2).mean(dim=(1, 2, 3))

    # Combined penalty
    return luma_penalty + color_penalty


def perceptual_loss(delta, input_images):
    """Calculate enhanced perceptual loss to preserve image structure.

    This enhanced version focuses more heavily on the low and mid frequency
    components that are most important for visual quality, with special emphasis
    on preserving image structure.

    Args:
        delta: Perturbation tensor of shape (B, C, H, W)
        input_images: Original images of shape (B, C, H, W)

    Returns:
        Perceptual loss value (per-sample)
    """
    # Compute frequency domain representation
    fft_orig = torch.fft.fft2(input_images, dim=(2, 3))
    fft_pert = torch.fft.fft2(input_images + delta, dim=(2, 3))

    # Compute magnitude difference in frequency domain
    mag_orig = torch.abs(fft_orig)
    mag_pert = torch.abs(fft_pert)

    # Create emphasis on lower frequencies (more critical for image quality)
    batch_size, channels, height, width = input_images.shape

    # Create a frequency weighting mask
    y_coords = torch.arange(height, device=input_images.device).view(1, 1, -1, 1)
    x_coords = torch.arange(width, device=input_images.device).view(1, 1, 1, -1)

    # Compute distance from center (DC component)
    center_y, center_x = height // 2, width // 2
    dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = torch.sqrt(torch.tensor(center_y**2 + center_x**2, device=dist.device))

    # Create a stronger weighting for low and mid frequencies
    # Low frequencies (DC) - highest weight
    low_freq_mask = torch.exp(-dist / (max_dist * 0.1))
    # Mid frequencies - medium weight
    mid_freq_mask = torch.exp(-(((dist - max_dist * 0.2) / (max_dist * 0.2)) ** 2))
    # Combined mask - emphasize both low and mid frequencies
    weight_mask = low_freq_mask + 0.5 * mid_freq_mask
    weight_mask = weight_mask.expand(batch_size, channels, -1, -1)

    # Apply weighting and compute difference
    weighted_diff = ((mag_orig - mag_pert).abs() * weight_mask).sum(dim=(1, 2, 3))

    # Normalize by image size
    return weighted_diff / (channels * height * width)


def refine_perturbation(
    model,
    original_images,
    adv_images,
    labels,
    targeted=False,
    get_target_label_fn=None,
    refinement_steps=10,
    min_bound=None,
    max_bound=None,
    device=None,
    verbose=False,
):
    """
    Refine adversarial examples to minimize perturbation while maintaining misclassification.

    Args:
        model: The model used to evaluate examples
        original_images: Original clean images
        adv_images: Adversarial examples to refine
        labels: True labels for untargeted attacks, target labels for targeted
        targeted: Whether this is a targeted attack (True) or untargeted (False)
        get_target_label_fn: Function to get target labels for targeted attacks
        refinement_steps: Number of binary search steps for refinement
        min_bound: Minimum bound for valid pixel ranges
        max_bound: Maximum bound for valid pixel ranges
        device: Device to run computations on
        verbose: Whether to print progress information

    Returns:
        Refined adversarial examples with smaller perturbation
    """
    # Use default bounds if not provided
    if min_bound is None:
        min_bound = 0.0
    if max_bound is None:
        max_bound = 1.0

    # Set device
    if device is None:
        device = next(model.parameters()).device

    batch_size = original_images.size(0)
    refined_images = adv_images.clone()

    # For targeted attacks, get target labels
    if targeted:
        if get_target_label_fn is not None:
            target_labels = get_target_label_fn(original_images, labels)
        else:
            target_labels = labels
    else:
        target_labels = labels

    # Initialize alpha for each image (controls interpolation between original and adversarial)
    # alpha=0 means pure adversarial, alpha=1 means pure original
    alphas = torch.zeros(batch_size, device=device)

    # Identify initially successful adversarial examples
    with torch.no_grad():
        outputs = model(adv_images)
        if targeted:
            # For targeted attacks, success means predicting the target class
            initial_success = outputs.argmax(dim=1) == target_labels
        else:
            # For untargeted attacks, success means not predicting the true class
            initial_success = outputs.argmax(dim=1) != labels

    # Only refine successful examples
    successful_indices = torch.where(initial_success)[0]

    if len(successful_indices) == 0:
        if verbose:
            print("No successful adversarial examples to refine")
        return refined_images

    if verbose:
        print(f"Refining {len(successful_indices)} successful adversarial examples...")

    # Binary search to find the minimal perturbation for each example
    for step in range(refinement_steps):
        # Create interpolated images
        current_alphas = alphas.view(-1, 1, 1, 1)
        interpolated = (
            current_alphas * original_images + (1 - current_alphas) * refined_images
        )

        # Ensure interpolated images are within valid bounds
        interpolated = torch.clamp(interpolated, min=min_bound, max=max_bound)

        # Check if still adversarial
        with torch.no_grad():
            outputs = model(interpolated)
            if targeted:
                # For targeted attacks, success means predicting the target class
                still_successful = outputs.argmax(dim=1) == target_labels
            else:
                # For untargeted attacks, success means not predicting the true class
                still_successful = outputs.argmax(dim=1) != labels

        # Update alphas with binary search
        for i in successful_indices:
            if still_successful[i]:
                # If still successful, try moving closer to original
                alphas[i] = alphas[i] + (1 - alphas[i]) / 2
                # Update the refined image with this better version
                refined_images[i] = interpolated[i]
            else:
                # If not successful, move back toward adversarial
                alphas[i] = alphas[i] / 2

    # Calculate improvement in perturbation metrics
    orig_l2 = (
        torch.norm((adv_images - original_images).view(batch_size, -1), dim=1)
        .mean()
        .item()
    )
    refined_l2 = (
        torch.norm((refined_images - original_images).view(batch_size, -1), dim=1)
        .mean()
        .item()
    )

    # Calculate SSIM improvement if skimage is available
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        # Calculate SSIM for original adversarial examples
        orig_adv_np = adv_images.detach().cpu().permute(0, 2, 3, 1).numpy()
        orig_np = original_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Calculate SSIM for refined adversarial examples
        refined_np = refined_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        # Calculate SSIM for each image
        orig_ssim_values = []
        refined_ssim_values = []

        for i in range(batch_size):
            try:
                # Try newer scikit-image API (>=0.19.0) with channel_axis
                orig_ssim = ssim(
                    orig_np[i], orig_adv_np[i], channel_axis=2, data_range=1.0
                )
                refined_ssim = ssim(
                    orig_np[i], refined_np[i], channel_axis=2, data_range=1.0
                )
            except TypeError:
                # Fall back to older API (<0.19.0) with multichannel
                orig_ssim = ssim(
                    orig_np[i], orig_adv_np[i], multichannel=True, data_range=1.0
                )
                refined_ssim = ssim(
                    orig_np[i], refined_np[i], multichannel=True, data_range=1.0
                )

            orig_ssim_values.append(orig_ssim)
            refined_ssim_values.append(refined_ssim)

        orig_ssim = np.mean(orig_ssim_values)
        refined_ssim = np.mean(refined_ssim_values)

        if verbose:
            l2_reduction = (orig_l2 - refined_l2) / orig_l2 * 100
            ssim_improvement = (refined_ssim - orig_ssim) / (1 - orig_ssim) * 100
            print(f"Refinement reduced L2 perturbation by {l2_reduction:.2f}%")
            print(
                f"Refinement improved SSIM from {orig_ssim:.4f} to {refined_ssim:.4f} ({ssim_improvement:.2f}% improvement)"
            )
    except ImportError:
        if verbose:
            l2_reduction = (orig_l2 - refined_l2) / orig_l2 * 100
            print(f"Refinement reduced L2 perturbation by {l2_reduction:.2f}%")
            print("SSIM calculation skipped (scikit-image not available)")

    return refined_images
