#!/usr/bin/env python
"""
Generate grid visualization of adversarial examples from multiple baseline attack methods.
Shows original images in the first column and corresponding adversarial examples in subsequent columns.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import logging
import traceback
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("adversarial_grid.log")],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM
from src.attacks.baseline.attack_deepfool import DeepFool
from src.attacks.baseline.attack_cw import CW


def denormalize(x):
    """Convert from normalized to [0,1] range for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)

    img = x.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img


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


def generate_adversarial_grid(
    model,
    dataset,
    num_images=10,
    attack_methods=None,
    save_path="paper/images/adversarial_grid.png",
):
    """
    Generate a grid visualization with original images and their adversarial versions

    Args:
        model: Target model to attack
        dataset: Dataset with images to attack
        num_images: Number of images to include in the grid
        attack_methods: Dictionary of attack methods to use
        save_path: Path to save the visualization
    """
    device = next(model.parameters()).device

    # Default attack methods if none provided
    if attack_methods is None:
        attack_methods = {
            "FGSM": FGSM(model, eps=8 / 255),
            "FFGSM": FFGSM(model, eps=8 / 255, alpha=0.4 * 8 / 255),
            "DeepFool": DeepFool(model, steps=50, overshoot=0.02, top_k_classes=10),
            "C&W": CW(model, c=1.0, kappa=0, steps=100, lr=0.01),
        }

    # Number of columns: 1 for original + 1 for each attack method
    num_cols = 1 + len(attack_methods)

    # Select images to attack
    image_indices = np.random.choice(len(dataset), num_images, replace=False)
    logger.info(f"Selected {num_images} random images for the grid")

    # Create figure for grid
    fig = plt.figure(figsize=(num_cols * 2.2, num_images * 2 + 0.5))

    # Create a GridSpec to make room for the header row
    gs = fig.add_gridspec(
        num_images + 1, num_cols, height_ratios=[0.2] + [1] * num_images
    )

    # Create header row for method names
    header_axes = []
    for col, name in enumerate(["Original"] + list(attack_methods.keys())):
        header_ax = fig.add_subplot(gs[0, col])
        header_ax.text(
            0.5, 0.5, name, ha="center", va="center", fontsize=14, fontweight="bold"
        )
        header_ax.axis("off")
        header_axes.append(header_ax)

    # Create axes for images
    axes = np.empty((num_images, num_cols), dtype=object)
    for row in range(num_images):
        for col in range(num_cols):
            axes[row, col] = fig.add_subplot(
                gs[row + 1, col]
            )  # +1 to account for header row

    # If only one row, ensure axes is 2D
    if num_images == 1:
        axes = axes.reshape(1, -1)

    # Create a list to store attack results metrics
    results_data = []

    # Process each image
    for row, img_idx in enumerate(
        tqdm(image_indices, desc="Generating adversarial examples")
    ):
        # Get original image and label
        img, label = dataset[img_idx]
        img = img.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        # Get original prediction
        with torch.no_grad():
            orig_pred = model(img).argmax(dim=1).item()

        # Display original image
        orig_np = denormalize(img[0]).cpu().permute(1, 2, 0).numpy()
        axes[row, 0].imshow(orig_np)
        axes[row, 0].axis("off")

        # Set row label with class name
        class_name = dataset.class_names[label.item()]
        short_class = class_name.split(",")[
            0
        ]  # Take only the first part of the class name
        axes[row, 0].set_ylabel(f"{short_class}", fontsize=10)

        # Show predicted class on original image
        pred_class = dataset.class_names[orig_pred].split(",")[0]
        if len(pred_class) > 10:  # Truncate long class names
            pred_class = pred_class[:10] + "..."

        # Add prediction text to original image
        axes[row, 0].text(
            0.5,
            0.95,
            f"{pred_class}",
            transform=axes[row, 0].transAxes,
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

        # Generate and display adversarial examples for each attack
        for col, (attack_name, attack) in enumerate(attack_methods.items(), start=1):
            # Set attack to untargeted mode
            attack.set_mode_default()

            # Generate adversarial example
            try:
                adv_img = attack(img, label)

                # Get adversarial prediction
                with torch.no_grad():
                    adv_pred = model(adv_img).argmax(dim=1).item()

                # Success check
                success = adv_pred != orig_pred

                # Calculate metrics
                ssim_value = compute_ssim(img[0], adv_img[0])
                perturbation = adv_img[0] - img[0]
                l2_norm = torch.norm(perturbation.flatten(), p=2).item()
                linf_norm = torch.norm(perturbation.flatten(), p=float("inf")).item()

                # Choose the primary norm to display based on the attack
                if attack_name in ["FGSM", "FFGSM"]:
                    primary_norm = f"Linf={linf_norm:.4f}"
                else:
                    primary_norm = f"L2={l2_norm:.4f}"

                # Display adversarial image
                adv_np = denormalize(adv_img[0]).cpu().permute(1, 2, 0).numpy()
                axes[row, col].imshow(adv_np)

                # Add success indicator
                border_color = "lime" if success else "red"
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)

                # Add subtle text annotation with primary norm and new class
                adv_class = dataset.class_names[adv_pred].split(",")[0]
                if len(adv_class) > 10:  # Truncate long class names
                    adv_class = adv_class[:10] + "..."

                # Add prediction text at bottom of image (small, subtle)
                if success:
                    axes[row, col].text(
                        0.5,
                        0.95,
                        f"{adv_class}",
                        transform=axes[row, col].transAxes,
                        fontsize=8,
                        ha="center",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                    )

                # Store results
                results_data.append(
                    {
                        "image_idx": img_idx,
                        "attack": attack_name,
                        "success": success,
                        "ssim": ssim_value,
                        "l2_norm": l2_norm,
                        "linf_norm": linf_norm,
                        "orig_class": dataset.class_names[orig_pred],
                        "adv_class": dataset.class_names[adv_pred],
                    }
                )

            except Exception as e:
                logger.error(
                    f"Error generating {attack_name} adversarial example for image {img_idx}: {str(e)}"
                )
                axes[row, col].imshow(np.zeros_like(orig_np))

            axes[row, col].axis("off")

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved adversarial grid to {save_path}")

    # Print summary statistics
    logger.info("\n===== ATTACK SUMMARY STATISTICS =====")
    attack_results = {}
    for attack_name in attack_methods.keys():
        # Calculate statistics for each attack method
        attack_data = [r for r in results_data if r["attack"] == attack_name]
        if attack_data:
            success_rate = sum(r["success"] for r in attack_data) / len(attack_data)
            avg_l2 = np.mean([r["l2_norm"] for r in attack_data])
            avg_linf = np.mean([r["linf_norm"] for r in attack_data])
            avg_ssim = np.mean([r["ssim"] for r in attack_data])
        else:
            success_rate, avg_l2, avg_linf, avg_ssim = 0, 0, 0, 0

        attack_results[attack_name] = {
            "success_rate": success_rate * 100,
            "avg_l2": avg_l2,
            "avg_linf": avg_linf,
            "avg_ssim": avg_ssim * 100,
            "avg_time": 0,  # Assuming avg_time is not available in the results_data
        }

        # Log statistics
        logger.info(f"Attack: {attack_name}")
        logger.info(f"  Success Rate: {success_rate:.2%}")
        logger.info(f"  Avg L2 Norm: {avg_l2:.4f}")
        logger.info(f"  Avg Linf Norm: {avg_linf:.4f}")
        logger.info(f"  Avg SSIM: {avg_ssim:.4f}")
        logger.info(f"  Avg Time: {attack_results[attack_name]['avg_time']:.4f} ms")

    # Add success rate information to the header labels
    for col, attack_name in enumerate(list(attack_methods.keys()), start=1):
        success_rate = attack_results[attack_name]["success_rate"]
        header_axes[col].text(
            0.5,
            0.2,
            f"({success_rate:.0f}% success)",
            ha="center",
            va="center",
            fontsize=10,
            fontstyle="italic",
        )

    # Adjust layout
    plt.tight_layout()

    # Remove ticks from all subplots for cleaner look
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, results_data


def create_multi_model_grid(
    dataset, models, num_images=5, save_path="paper/images/multi_model_grid.png"
):
    """
    Create a grid comparing adversarial examples across different models

    Args:
        dataset: Dataset with images
        models: Dictionary of models {name: model}
        num_images: Number of images to include
        save_path: Path to save the visualization
    """
    # This is a placeholder for potential future extension
    pass


def main():
    """Main function to generate adversarial attack grid visualizations"""
    parser = argparse.ArgumentParser(
        description="Generate adversarial attack grid visualizations"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to include in the grid",
    )
    parser.add_argument("--model", type=str, default="resnet18", help="Model to attack")
    parser.add_argument(
        "--output",
        type=str,
        default="paper/images/adversarial_grid.png",
        help="Output file path",
    )
    args = parser.parse_args()

    logger.info("Starting adversarial attack grid visualization generation")

    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model = get_model(args.model).to(device)
        model.eval()
        logger.info(f"Loaded {args.model} model")

        # Load dataset
        dataset = get_dataset("imagenet", data_dir="data", max_samples=1000)
        logger.info(f"Loaded dataset with {len(dataset)} images")

        # Generate the grid visualization
        generate_adversarial_grid(
            model=model,
            dataset=dataset,
            num_images=args.num_images,
            save_path=args.output,
        )

        logger.info("Adversarial grid visualization complete!")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
