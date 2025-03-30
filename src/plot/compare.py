"""
Visualization utility.
Original Image -> Perturbation -> Adversarial Image
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union, List


def visualize_attack(
    original: torch.Tensor,
    perturbation: torch.Tensor,
    adversarial: torch.Tensor,
    results: Dict[str, Union[str, float]],
    epsilon: float,
    method_name: str = "FGSM",
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize an adversarial attack showing original, perturbation, and adversarial images.

    Args:
        original: Original input image tensor [C, H, W]
        perturbation: Perturbation tensor [C, H, W]
        adversarial: Adversarial example tensor [C, H, W]
        results: Dictionary with classification results including:
            - original_class: Name of the original class
            - original_confidence: Confidence score for original class
            - adversarial_class: Name of the adversarial class
            - adversarial_confidence: Confidence score for adversarial class
        epsilon: Perturbation magnitude used in the attack
        method_name: Name of the attack method
        mean: Normalization mean tensor [C]
        std: Normalization standard deviation tensor [C]
        save_path: Path to save the visualization figure
        show: Whether to display the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    # Denormalize if mean and std are provided
    if mean is not None and std is not None:
        original_img = denormalize(original, mean, std)
        adversarial_img = denormalize(adversarial, mean, std)
    else:
        original_img = original
        adversarial_img = adversarial

    # Convert to numpy for visualization
    original_np = to_numpy(original_img)
    adversarial_np = to_numpy(adversarial_img)

    # For perturbation, we want to show the sign (direction) rather than magnitude
    perturbation_np = to_numpy(perturbation)

    # Convert from [-1, 0, 1] to a colorful visualization by shifting to [0, 1] range
    sign_pert = np.sign(perturbation_np)
    sign_pert = (sign_pert + 1) / 2  # Now in range [0, 1]

    # Create figure for visualization
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Plot original image
    axs[0].imshow(original_np)
    axs[0].set_title(
        f"{results['original_class']}\n{results['original_confidence']*100:.1f}% confidence"
    )
    axs[0].axis("off")

    # Plot sign of perturbation (like in the panda example)
    axs[1].imshow(sign_pert)
    axs[1].set_title(f"+ {epsilon} ×\nsign(∇J(θ, x, y))")
    axs[1].axis("off")

    # Plot adversarial image
    axs[2].imshow(adversarial_np)
    axs[2].set_title(
        f"{results['adversarial_class']}\n{results['adversarial_confidence']*100:.1f}% confidence"
    )
    axs[2].axis("off")

    # Add main title
    fig.suptitle(f"{method_name} Attack (ε = {epsilon})", fontsize=16)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # Display figure if show is True
    if show:
        plt.show()

    return fig


def denormalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Denormalize a tensor using the given mean and standard deviation.

    Args:
        tensor: Input tensor to denormalize [C, H, W]
        mean: Mean tensor [C]
        std: Standard deviation tensor [C]

    Returns:
        Denormalized tensor
    """
    # Clone tensor to avoid modifying the original
    img = tensor.clone().detach()

    # Ensure mean and std have the right shape for broadcasting
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1)

    # Denormalize
    img = img * std + mean

    # Clamp to ensure valid image range [0, 1]
    img = torch.clamp(img, 0, 1)

    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Input tensor [C, H, W]

    Returns:
        Numpy array [H, W, C]
    """
    # Make sure tensor is on CPU and detached from computation graph
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Detach if tensor requires gradients
    if tensor.requires_grad:
        tensor = tensor.detach()

    # Convert to numpy and transpose from [C, H, W] to [H, W, C] for plotting
    if tensor.dim() == 4:  # [B, C, H, W]
        return tensor[0].permute(1, 2, 0).numpy()
    else:  # [C, H, W]
        return tensor.permute(1, 2, 0).numpy()


def visualize_attack_grid(
    originals: List[torch.Tensor],
    perturbations: List[torch.Tensor],
    adversarials: List[torch.Tensor],
    results_list: List[Dict[str, Union[str, float]]],
    epsilon: float,
    method_name: str = "FGSM",
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a grid visualization of multiple adversarial examples.

    Args:
        originals: List of original input images
        perturbations: List of perturbations
        adversarials: List of adversarial examples
        results_list: List of dictionaries with classification results
        epsilon: Perturbation magnitude used in the attack
        method_name: Name of the attack method
        mean: Normalization mean tensor
        std: Normalization standard deviation tensor
        save_path: Path to save the visualization figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure object
    """
    num_examples = len(originals)

    # Create a figure with rows for each example
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))

    # If only one example, make axes indexable
    if num_examples == 1:
        axs = axs.reshape(1, -1)

    for i in range(num_examples):
        # Get the data for this example
        original = originals[i]
        perturbation = perturbations[i]
        adversarial = adversarials[i]
        results = results_list[i]

        # Denormalize if mean and std are provided
        if mean is not None and std is not None:
            original_img = denormalize(original, mean, std)
            adversarial_img = denormalize(adversarial, mean, std)
        else:
            original_img = original
            adversarial_img = adversarial

        # Convert to numpy for visualization
        original_np = to_numpy(original_img)
        adversarial_np = to_numpy(adversarial_img)

        # For perturbation, we want to show the sign (direction) rather than magnitude
        perturbation_np = to_numpy(perturbation)
        sign_pert = np.sign(perturbation_np)
        sign_pert = (sign_pert + 1) / 2  # Convert to [0, 1] range

        # Plot original image
        axs[i, 0].imshow(original_np)
        axs[i, 0].set_title(
            f"{results['original_class']}\n{results['original_confidence']*100:.1f}% confidence"
        )
        axs[i, 0].axis("off")

        # Plot sign of perturbation
        axs[i, 1].imshow(sign_pert)
        axs[i, 1].set_title(f"+ {epsilon} ×\nsign(∇J(θ, x, y))")
        axs[i, 1].axis("off")

        # Plot adversarial image
        axs[i, 2].imshow(adversarial_np)
        axs[i, 2].set_title(
            f"{results['adversarial_class']}\n{results['adversarial_confidence']*100:.1f}% confidence"
        )
        axs[i, 2].axis("off")

    # Add main title
    fig.suptitle(f"{method_name} Attack (ε = {epsilon})", fontsize=16)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # Display figure if show is True
    if show:
        plt.show()

    return fig


def visualize_cw_attack_grid(
    originals: List[torch.Tensor],
    perturbations: List[torch.Tensor],
    adversarials: List[torch.Tensor],
    results_list: List[Dict[str, Union[str, float]]],
    l2_norms: List[float],
    config_names: List[str],
    method_name: str = "C&W L2",
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a grid visualization of C&W adversarial examples with L2 perturbation visualization.

    Args:
        originals: List of original input images
        perturbations: List of perturbations
        adversarials: List of adversarial examples
        results_list: List of dictionaries with classification results
        l2_norms: List of L2 norms for each perturbation
        config_names: List of configuration names for each example
        method_name: Name of the attack method
        mean: Normalization mean tensor
        std: Normalization standard deviation tensor
        save_path: Path to save the visualization figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure object
    """
    num_examples = len(originals)

    # Create a figure with rows for each example
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))

    # If only one example, make axes indexable
    if num_examples == 1:
        axs = axs.reshape(1, -1)

    for i in range(num_examples):
        # Get the data for this example
        original = originals[i]
        perturbation = perturbations[i]
        adversarial = adversarials[i]
        results = results_list[i]
        l2_norm = l2_norms[i]
        config_name = config_names[i]

        # Denormalize if mean and std are provided
        if mean is not None and std is not None:
            original_img = denormalize(original, mean, std)
            adversarial_img = denormalize(adversarial, mean, std)
        else:
            original_img = original
            adversarial_img = adversarial

        # Convert to numpy for visualization
        original_np = to_numpy(original_img)
        adversarial_np = to_numpy(adversarial_img)

        # For perturbation, visualize the magnitude (L2) rather than sign
        perturbation_np = to_numpy(perturbation)

        # Convert perturbation to magnitude and normalize for visualization
        # We use grayscale colormaps to show intensity better
        perturbation_mag = np.abs(perturbation_np)
        # Scale to enhance visibility - you might need to adjust these values
        perturbation_mag = np.clip(perturbation_mag * 5, 0, 1)

        # Plot original image
        axs[i, 0].imshow(original_np)
        axs[i, 0].set_title(
            f"{results['original_class']}\n{results['original_confidence']*100:.1f}% confidence"
        )
        axs[i, 0].axis("off")

        # Plot perturbation intensity (L2)
        axs[i, 1].imshow(np.mean(perturbation_mag, axis=2), cmap="hot")
        axs[i, 1].set_title(f"L2 Perturbation\nNorm: {l2_norm:.2f}")
        axs[i, 1].axis("off")

        # Plot adversarial image
        axs[i, 2].imshow(adversarial_np)
        axs[i, 2].set_title(
            f"{results['adversarial_class']}\n{results['adversarial_confidence']*100:.1f}% confidence"
        )
        axs[i, 2].axis("off")

        # Add the configuration name to the left of each row
        axs[i, 0].text(
            -0.1,
            0.5,
            config_name,
            transform=axs[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=12,
        )

    # Add main title
    fig.suptitle(f"{method_name} Attack", fontsize=16)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # Display figure if show is True
    if show:
        plt.show()

    return fig
