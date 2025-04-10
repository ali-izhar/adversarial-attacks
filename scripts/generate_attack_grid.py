#!/usr/bin/env python
"""
Generate a comprehensive grid comparison of adversarial attacks across different models.
Creates a visualization showing original image, perturbations, and adversarial examples
for each attack method (baseline and optimization methods) on each model architecture.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model
from src.attacks.baseline import FGSM, FFGSM, DeepFool, CW
from src.attacks import PGD, CG, LBFGS


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
    orig_img = denormalize(original).cpu().permute(1, 2, 0).numpy()
    pert_img = denormalize(perturbed).cpu().permute(1, 2, 0).numpy()
    return np.mean(
        [ssim(orig_img[..., i], pert_img[..., i], data_range=1.0) for i in range(3)]
    )


def get_attack_methods() -> Dict:
    """Initialize all attack methods with their configurations"""
    eps = 8 / 255  # Define epsilon once for consistency
    return {
        # Baseline methods
        "FGSM": lambda model: FGSM(model, eps=eps),
        "FFGSM": lambda model: FFGSM(
            model, eps=eps, alpha=0.4 * eps
        ),  # alpha as fraction of eps
        "DeepFool": lambda model: DeepFool(model, steps=50, overshoot=0.02),
        "C\\&W": lambda model: CW(
            model, c=1.0, kappa=0, steps=100, lr=0.01
        ),  # Escaped ampersand for LaTeX
        # Optimization methods
        "PGD": lambda model: PGD(
            model,
            norm="Linf",
            eps=eps,
            n_iterations=40,
            step_size=eps / 10,
            rand_init=True,
            early_stopping=True,
        ),
        "CG": lambda model: CG(model, eps=eps),  # CG only needs eps parameter
        "L-BFGS": lambda model: LBFGS(
            model,
            norm="Linf",
            eps=eps,
            n_iterations=100,  # n_iterations instead of steps
            history_size=10,  # history_size instead of history
            initial_const=1e-2,
            binary_search_steps=5,
            const_factor=10.0,
            repeat_search=True,
            rand_init=True,
            init_std=0.01,
        ),
    }


def get_model_architectures() -> List[Tuple[str, str]]:
    """Get list of model names and their display names"""
    return [
        ("resnet18", "RN18"),
        ("resnet50", "RN50"),
        ("vgg16", "VGG"),
        ("efficientnet_b0", "ENet"),
        ("mobilenet_v3_large", "MNet"),
    ]


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading ImageNet dataset...")
    dataset = get_dataset("imagenet", data_dir="data", max_samples=1000)

    # Find saluki image (class 176)
    try:
        saluki_idx = next(idx for idx, (_, label) in enumerate(dataset) if label == 176)
        image, label = dataset[saluki_idx]
        image = image.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)
    except StopIteration:
        print("Error: Could not find saluki image in dataset")
        return

    # Initialize models with proper error handling
    models = {}
    architectures = get_model_architectures()

    print("Loading models...")
    for arch_name, display_name in architectures:
        try:
            print(f"  Loading {arch_name}...")
            model = get_model(arch_name).to(device)
            model.eval()
            models[display_name] = model
        except Exception as e:
            print(f"Error loading {arch_name}: {e}")
            return

    # Initialize attacks
    print("Initializing attacks...")
    attack_methods = get_attack_methods()

    # Set up matplotlib
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 7,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.labelsize": 7,
            "figure.titlesize": 8,
            "axes.linewidth": 0.5,  # Thinner plot borders
        }
    )

    # Create figure - wider than tall to fit 6 columns better
    print("Creating visualization grid...")
    fig = plt.figure(figsize=(7.0, 6.5))  # Wider than tall for better column fit

    # Custom height ratios - make headers and section labels smaller
    height_ratios = [0.6] + [0.8] + [0.4, 1, 1, 1, 1, 0.4, 1, 1, 1]
    gs = plt.GridSpec(
        11, 6, figure=fig, height_ratios=height_ratios, hspace=0.0, wspace=0.01
    )  # Minimal spacing

    # Add column headers
    headers = ["Original"] + [name for _, name in architectures]
    for col, header in enumerate(headers):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, f"\\textbf{{{header}}}", ha="center", va="center", fontsize=7)
        ax.axis("off")

    # Original image row
    orig_img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(orig_img)
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove box around plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Show original predictions
    for col, (_, display_name) in enumerate(architectures, 1):
        model = models[display_name]
        with torch.no_grad():
            pred = model(image).argmax(dim=1).item()
        ax = fig.add_subplot(gs[1, col])
        ax.text(
            0.5,
            0.5,
            f"{dataset.class_names[pred]}",
            ha="center",
            va="center",
            fontsize=7,
        )
        ax.axis("off")

    # Add section headers - more compact
    row = 2
    for section in ["Baseline Methods", "Optimization Methods"]:
        ax = fig.add_subplot(gs[row, :])
        ax.text(0.01, 0.5, f"\\textbf{{{section}}}", ha="left", va="center", fontsize=7)
        ax.axis("off")
        row += 1

        # Process attacks for this section
        if section == "Baseline Methods":
            method_list = ["FGSM", "FFGSM", "DeepFool", "C\\&W"]
        else:
            method_list = ["PGD", "CG", "L-BFGS"]

        attacks = {k: v for k, v in attack_methods.items() if k in method_list}

        for attack_name, attack_fn in attacks.items():
            print(f"  Processing {attack_name}...")
            # Show attack name
            ax = fig.add_subplot(gs[row, 0])
            ax.text(
                0.5,
                0.5,
                f"\\textbf{{{attack_name}}}",
                ha="center",
                va="center",
                fontsize=7,
            )
            ax.axis("off")

            # Generate and show results for each model
            for col, (_, display_name) in enumerate(architectures, 1):
                model = models[display_name]
                attack = attack_fn(model)

                try:
                    adv_image = attack(image.clone(), label)
                    perturbation = adv_image - image

                    ax = fig.add_subplot(gs[row, col])
                    with torch.no_grad():
                        pred = model(adv_image).argmax(dim=1).item()
                        ssim_val = compute_ssim(image[0], adv_image[0])

                    adv_img = denormalize(adv_image[0]).cpu().permute(1, 2, 0).numpy()
                    ax.imshow(adv_img)

                    # Ultra-compact title - just SSIM % and class name
                    class_name = (
                        dataset.class_names[pred].split(",")[0].split(" ")[0]
                    )  # First word only
                    ax.set_title(
                        f"SSIM: {ssim_val*100:.1f}\\%\n{class_name}", fontsize=6, pad=1
                    )

                    # Remove all padding
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                except Exception as e:
                    print(f"    Error processing {attack_name} on {display_name}: {e}")
                    # Create empty subplot for failed attack
                    ax = fig.add_subplot(gs[row, col])
                    ax.text(
                        0.5,
                        0.5,
                        "Failed",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=6,
                    )
                    ax.axis("off")

            row += 1

    # Ultra-tight layout with zero margins
    plt.subplots_adjust(
        left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.0, hspace=0.0
    )

    # Save the figure
    print("Saving visualization...")
    os.makedirs("paper/images", exist_ok=True)
    plt.savefig(
        "paper/images/attack_grid.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,  # Almost no padding
        facecolor="white",
    )
    print("Saved visualization to paper/images/attack_grid.png")


if __name__ == "__main__":
    main()
