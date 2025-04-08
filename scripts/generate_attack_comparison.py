#!/usr/bin/env python
"""
Generate comparison images for untargeted and targeted attacks.
This script creates a visualization comparing original image with
untargeted and targeted CW attacks on a specific ImageNet image.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model
from src.attacks.baseline.attack_cw import CW


def denormalize(x):
    """Convert from normalized to [0,1] range for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)

    img = x.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def visualize_perturbation(perturbation):
    """Create a diverging colormap visualization of the perturbation"""
    perturbation_sign = perturbation.cpu()
    max_pert = (
        perturbation_sign.abs().max().item()
        if perturbation_sign.abs().max().item() > 0
        else 1.0
    )
    perturbation_sign = perturbation_sign / (max_pert * 0.1)  # Normalize and enhance
    perturbation_sign = torch.clamp(perturbation_sign, -1, 1)

    # Convert to RGB (red=increase, blue=decrease)
    pert_rgb = torch.zeros((3, perturbation_sign.shape[1], perturbation_sign.shape[2]))
    pert_rgb[0] = torch.clamp(perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1)  # Red
    pert_rgb[2] = torch.clamp(-perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1)  # Blue
    pert_rgb[1] = torch.clamp(
        1.0 - perturbation_sign.mean(dim=0).abs() * 0.5, 0, 1
    )  # Green

    return pert_rgb.permute(1, 2, 0).numpy()


def compute_ssim(original, perturbed):
    """Compute SSIM between original and perturbed images"""
    # Convert to numpy arrays in [0,1] range for SSIM computation
    orig_img = denormalize(original).cpu().permute(1, 2, 0).numpy()
    pert_img = denormalize(perturbed).cpu().permute(1, 2, 0).numpy()

    # Compute SSIM (average over channels)
    ssim_value = np.mean(
        [ssim(orig_img[..., i], pert_img[..., i], data_range=1.0) for i in range(3)]
    )

    return ssim_value


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model("resnet18").to(device)
    model.eval()

    # Load dataset
    dataset = get_dataset("imagenet", data_dir="data", max_samples=1000)

    # Find saluki image (class 176)
    saluki_idx = None
    for idx, (_, label) in enumerate(dataset):
        if label == 176:  # Saluki class
            saluki_idx = idx
            break

    if saluki_idx is None:
        raise ValueError("Saluki image not found in dataset")

    # Get the image
    image, label = dataset[saluki_idx]
    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    # Create CW attacks
    cw_untargeted = CW(model, c=1.0, kappa=0, steps=100, lr=0.01)
    cw_targeted = CW(model, c=10.0, kappa=20, steps=200, lr=0.01)

    # Set modes for attacks
    cw_untargeted.set_mode_default()  # Untargeted mode
    cw_targeted.set_mode_targeted_least_likely()  # Targeted mode using least likely class

    # Generate adversarial examples
    print("Generating untargeted attack...")
    adv_untargeted = cw_untargeted(image, label)

    print("Generating targeted attack...")
    adv_targeted = cw_targeted(image, label)

    # Get predictions and compute SSIM
    with torch.no_grad():
        orig_pred = model(image).argmax(dim=1).item()
        untarg_pred = model(adv_untargeted).argmax(dim=1).item()
        targ_pred = model(adv_targeted).argmax(dim=1).item()

        # Compute SSIM for both attacks
        untarg_ssim = compute_ssim(image[0], adv_untargeted[0])
        targ_ssim = compute_ssim(image[0], adv_targeted[0])

    # Set up matplotlib for LaTeX rendering
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 18,
            "axes.titlesize": 22,
        }
    )

    # Create visualization with spacing between groups
    plt.rcParams["figure.facecolor"] = "white"
    fig = plt.figure(figsize=(25, 5))

    # Create a grid with specific width ratios for the three main sections
    gs = plt.GridSpec(1, 5, width_ratios=[1, 0.1, 2, 0.1, 2])

    # Original image
    ax0 = fig.add_subplot(gs[0])
    orig_img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    ax0.imshow(orig_img)
    ax0.set_title("Original\n" + r"Predicted: \texttt{saluki}", pad=20)
    ax0.axis("off")

    # First separator (empty subplot)
    sep1 = fig.add_subplot(gs[1])
    sep1.axis("off")

    # Untargeted attack group
    untargeted_group = plt.subplot(gs[2])
    untargeted_group.axis("off")

    # Create nested grid for untargeted attack
    gs_untarg = plt.GridSpec(1, 2)
    gs_untarg.update(
        left=untargeted_group.get_position().x0,
        right=untargeted_group.get_position().x1,
        bottom=untargeted_group.get_position().y0,
        top=untargeted_group.get_position().y1,
    )

    # Untargeted perturbation and result
    ax_pert_untarg = fig.add_subplot(gs_untarg[0])
    pert_untarg = visualize_perturbation(adv_untargeted[0] - image[0])
    ax_pert_untarg.imshow(pert_untarg)
    ax_pert_untarg.set_title("Untargeted\nPerturbation", pad=20)
    ax_pert_untarg.axis("off")

    ax_res_untarg = fig.add_subplot(gs_untarg[1])
    untarg_img = denormalize(adv_untargeted[0]).cpu().permute(1, 2, 0).numpy()
    ax_res_untarg.imshow(untarg_img)
    ax_res_untarg.set_title(
        f"Result ({untarg_ssim*100:.1f}\% similar)\n" + r"Predicted: \texttt{beagle}",
        pad=20,
    )
    ax_res_untarg.axis("off")

    # Second separator (empty subplot)
    sep2 = fig.add_subplot(gs[3])
    sep2.axis("off")

    # Targeted attack group
    targeted_group = plt.subplot(gs[4])
    targeted_group.axis("off")

    # Create nested grid for targeted attack
    gs_targ = plt.GridSpec(1, 2)
    gs_targ.update(
        left=targeted_group.get_position().x0,
        right=targeted_group.get_position().x1,
        bottom=targeted_group.get_position().y0,
        top=targeted_group.get_position().y1,
    )

    # Targeted perturbation and result
    ax_pert_targ = fig.add_subplot(gs_targ[0])
    pert_targ = visualize_perturbation(adv_targeted[0] - image[0])
    ax_pert_targ.imshow(pert_targ)
    ax_pert_targ.set_title("Targeted\nPerturbation", pad=20)
    ax_pert_targ.axis("off")

    ax_res_targ = fig.add_subplot(gs_targ[1])
    targ_img = denormalize(adv_targeted[0]).cpu().permute(1, 2, 0).numpy()
    ax_res_targ.imshow(targ_img)
    ax_res_targ.set_title(
        f"Result ({targ_ssim*100:.1f}\% similar)\n" + r"Predicted: \texttt{gorilla}",
        pad=20,
    )
    ax_res_targ.axis("off")

    # Add vertical lines for separation
    fig.patches.extend(
        [
            plt.Rectangle(
                (gs[1].get_position(fig).x0, 0.1),
                0.002,
                0.8,
                facecolor="black",
                transform=fig.transFigure,
            ),
            plt.Rectangle(
                (gs[3].get_position(fig).x0, 0.1),
                0.002,
                0.8,
                facecolor="black",
                transform=fig.transFigure,
            ),
        ]
    )

    plt.tight_layout()

    # Save the figure
    os.makedirs("paper/images", exist_ok=True)
    plt.savefig(
        "paper/images/attack_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved visualization to paper/images/attack_comparison.png")


if __name__ == "__main__":
    main()
