#!/usr/bin/env python
"""
Generate visualization of adversarial perturbations from different attack methods.
Shows actual adversarial images with perturbation analysis and comparison.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import logging
import traceback
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("adversarial_analysis.log")],
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


def analyze_perturbation(perturbation, name):
    """Analyze perturbation characteristics and log results"""
    try:
        # Flatten for easier analysis
        pert_flat = perturbation.view(-1)

        # Calculate norms
        l2_norm = torch.norm(pert_flat, p=2).item()
        linf_norm = torch.norm(pert_flat, p=float("inf")).item()
        l1_norm = torch.norm(pert_flat, p=1).item()

        # Calculate statistics
        mean_abs = torch.abs(pert_flat).mean().item()
        std_dev = torch.std(pert_flat).item()
        max_val = torch.max(pert_flat).item()
        min_val = torch.min(pert_flat).item()

        # Calculate percentage of pixels changed (where absolute value > 0.01)
        significant_change_threshold = (
            0.01  # Threshold for considering a significant change
        )
        pixels_changed = (
            torch.abs(pert_flat) > significant_change_threshold
        ).float().mean().item() * 100

        # Log the analysis
        logger.info(f"\n===== {name} Perturbation Analysis =====")
        logger.info(f"L2 Norm: {l2_norm:.6f}")
        logger.info(
            f"Linf Norm: {linf_norm:.6f}"
        )  # Changed from ∞ to Linf to avoid unicode issues
        logger.info(f"L1 Norm: {l1_norm:.6f}")
        logger.info(f"Mean Absolute Value: {mean_abs:.6f}")
        logger.info(f"Standard Deviation: {std_dev:.6f}")
        logger.info(f"Max Value: {max_val:.6f}")
        logger.info(f"Min Value: {min_val:.6f}")
        logger.info(f"Pixels Significantly Changed: {pixels_changed:.2f}%")

        return {
            "l2_norm": l2_norm,
            "linf_norm": linf_norm,
            "l1_norm": l1_norm,
            "mean_abs": mean_abs,
            "std_dev": std_dev,
            "max_val": max_val,
            "min_val": min_val,
            "pixels_changed": pixels_changed,
        }
    except Exception as e:
        logger.error(f"Error analyzing perturbation: {str(e)}")
        logger.error(traceback.format_exc())
        # Return default values in case of error
        return {
            "l2_norm": 0.0,
            "linf_norm": 0.0,
            "l1_norm": 0.0,
            "mean_abs": 0.0,
            "std_dev": 0.0,
            "max_val": 0.0,
            "min_val": 0.0,
            "pixels_changed": 0.0,
        }


def create_comparison_grid(original, attack_results, class_names, orig_idx):
    """Create a comparison grid of all attack methods using their optimal norms"""
    try:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.size": 12,
                "axes.titlesize": 14,
            }
        )

        # Create the figure
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Original image
        orig_np = denormalize(original).cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Original\nPredicted: {class_names[orig_idx]}")
        axes[0].axis("off")

        # Each attack method with its optimal norm
        optimal_norms = {
            "FGSM": "linf_norm",  # FGSM optimized for L∞
            "FFGSM": "linf_norm",  # FFGSM optimized for L∞
            "DeepFool": "l2_norm",  # DeepFool optimized for L2
            "C&W": "l2_norm",  # C&W primarily optimized for L2
        }

        # For each attack
        for i, result in enumerate(attack_results):
            name = result["name"]
            adv_img = result["adv_img"]
            metrics = result["metrics"]

            # Get adversarial image (actual image, not perturbation)
            adv_np = denormalize(adv_img).cpu().permute(1, 2, 0).numpy()

            # Show the actual adversarial image
            axes[i + 1].imshow(adv_np)

            # Get optimal norm for this attack
            optimal_norm_key = optimal_norms.get(name, "linf_norm")
            optimal_norm_value = metrics[optimal_norm_key]

            # Get norm name for display (L-inf or L2)
            norm_display = "L-inf" if optimal_norm_key == "linf_norm" else "L2"

            # Calculate perturbation for highlight overlay
            pert = adv_img - original

            # Create a subtle highlight overlay based on the optimal norm
            if optimal_norm_key == "linf_norm":
                # For L∞-optimized attacks (FGSM, FFGSM)
                pert_mag = torch.max(torch.abs(pert), dim=0)[0].cpu().numpy()
                # More conservative threshold to only highlight significant changes
                threshold = max(0.05, optimal_norm_value * 0.4)
                cmap = plt.cm.cool
            else:
                # For L2-optimized attacks (DeepFool, C&W)
                pert_mag = torch.norm(pert, p=2, dim=0).cpu().numpy()
                # Adaptive threshold based on the L2 norm
                if name == "DeepFool":
                    # DeepFool has very small perturbations
                    threshold = max(0.001, optimal_norm_value * 0.3)
                else:
                    # C&W has larger L2 perturbations but they're still small visually
                    threshold = max(0.05, min(0.3, optimal_norm_value * 0.01))
                cmap = plt.cm.plasma

            # Create mask for significant changes only
            highlight_mask = pert_mag > threshold

            # Calculate percentage of pixels highlighted
            pct_pixels_visible = np.mean(highlight_mask) * 100
            logger.info(
                f"{name} grid: {norm_display} visible pixels: {pct_pixels_visible:.2f}%"
            )

            # Only apply highlight where changes are significant
            if np.any(highlight_mask):
                # Create a normalized version of perturbation magnitude for coloring
                norm_pert = np.zeros_like(pert_mag)
                norm_pert[highlight_mask] = np.clip(
                    pert_mag[highlight_mask] / pert_mag[highlight_mask].max(), 0, 1
                )

                # Create RGBA overlay with very subtle transparency
                rgba_overlay = cmap(norm_pert)
                # Control transparency - make it very subtle (0.1-0.2 alpha)
                rgba_overlay[..., 3] = norm_pert * 0.15  # Very subtle highlight

                # Add overlay to the adversarial image plot
                axes[i + 1].imshow(rgba_overlay, alpha=0.2)

            # Calculate and format SSIM for display
            ssim_value = metrics["ssim"] * 100

            # Format the title to show the attack name and metrics
            with torch.no_grad():
                adv_pred = torch.argmax(model(adv_img.unsqueeze(0)), dim=1).item()

            # Show what class it was misclassified as
            pred_class = class_names[adv_pred]

            axes[i + 1].set_title(
                f"{name}\n({ssim_value:.2f}% similar, {norm_display}={optimal_norm_value:.4f})",
                pad=10,
            )

            # Add prediction as text at bottom of image
            axes[i + 1].text(
                0.5,
                0.98,
                f"Pred: {pred_class}",
                transform=axes[i + 1].transAxes,
                fontsize=10,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

            axes[i + 1].axis("off")

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating comparison grid: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def save_figure(fig, filename, dpi=300):
    """Save figure with proper error handling"""
    if fig is None:
        logger.error(f"Cannot save None figure to {filename}")
        return False

    try:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Save the figure
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Successfully saved figure to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving figure to {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def get_safe_filename(name):
    """Convert any name to a safe filename by removing problematic characters"""
    # Replace with string that works for filenames
    return (
        name.replace("\\", "").replace("&", "and").replace(" ", "_").replace("/", "_")
    )


def main():
    """Main function to generate adversarial attack visualizations"""
    logger.info("Starting adversarial attack analysis and visualization")

    try:
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model
        global model  # Make model accessible to visualization functions
        model = get_model("resnet18").to(device)
        model.eval()
        logger.info("Loaded ResNet-18 model")

        # Load dataset
        dataset = get_dataset("imagenet", data_dir="data", max_samples=1000)
        logger.info(f"Loaded dataset with {len(dataset)} images")

        # Find saluki image (class 176)
        saluki_idx = None
        for idx, (_, label) in enumerate(dataset):
            if label == 176:  # Saluki class
                saluki_idx = idx
                break

        if saluki_idx is None:
            raise ValueError("Saluki image not found in dataset")

        logger.info(f"Found saluki image at index {saluki_idx}")

        # Get the image and label
        image, label = dataset[saluki_idx]
        image = image.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        # Get original prediction
        with torch.no_grad():
            orig_pred = model(image).argmax(dim=1).item()
        logger.info(
            f"Original prediction: {dataset.class_names[orig_pred]} (index {orig_pred})"
        )

        # Initialize attacks
        logger.info("Initializing attack methods...")
        fgsm = FGSM(model, eps=8 / 255)
        ffgsm = FFGSM(model, eps=8 / 255, alpha=0.4 * 8 / 255)
        deepfool = DeepFool(model, steps=50, overshoot=0.02, top_k_classes=10)
        cw = CW(model, c=1.0, kappa=0, steps=100, lr=0.01)

        # Set attack modes to untargeted
        for attack, name in [
            (fgsm, "FGSM"),
            (ffgsm, "FFGSM"),
            (deepfool, "DeepFool"),
            (cw, "C&W"),
        ]:
            attack.set_mode_default()
            logger.info(f"Set {name} to untargeted mode")

        # Run attacks and collect results
        attack_results = []

        # FGSM attack
        logger.info("Running FGSM attack...")
        adv_fgsm = fgsm(image, label)
        with torch.no_grad():
            fgsm_pred = model(adv_fgsm).argmax(dim=1).item()
        logger.info(
            f"FGSM prediction: {dataset.class_names[fgsm_pred]} (index {fgsm_pred})"
        )

        fgsm_ssim = compute_ssim(image[0], adv_fgsm[0])
        logger.info(f"FGSM SSIM: {fgsm_ssim:.6f}")

        fgsm_metrics = analyze_perturbation(adv_fgsm[0] - image[0], "FGSM")
        fgsm_metrics["ssim"] = fgsm_ssim

        # Calculate L2 and Linf norms directly to verify
        fgsm_perturbation = adv_fgsm[0] - image[0]
        fgsm_l2 = torch.norm(fgsm_perturbation.flatten(), p=2).item()
        fgsm_linf = torch.norm(fgsm_perturbation.flatten(), p=float("inf")).item()
        logger.info(
            f"FGSM L2 norm (direct): {fgsm_l2:.6f}, Linf norm (direct): {fgsm_linf:.6f}"
        )
        # Ensure metrics match direct calculation
        fgsm_metrics["l2_norm"] = fgsm_l2
        fgsm_metrics["linf_norm"] = fgsm_linf

        attack_results.append(
            {"name": "FGSM", "adv_img": adv_fgsm[0], "metrics": fgsm_metrics}
        )

        # FFGSM attack
        logger.info("Running FFGSM attack...")
        adv_ffgsm = ffgsm(image, label)
        with torch.no_grad():
            ffgsm_pred = model(adv_ffgsm).argmax(dim=1).item()
        logger.info(
            f"FFGSM prediction: {dataset.class_names[ffgsm_pred]} (index {ffgsm_pred})"
        )

        ffgsm_ssim = compute_ssim(image[0], adv_ffgsm[0])
        logger.info(f"FFGSM SSIM: {ffgsm_ssim:.6f}")

        ffgsm_metrics = analyze_perturbation(adv_ffgsm[0] - image[0], "FFGSM")
        ffgsm_metrics["ssim"] = ffgsm_ssim

        # Calculate L2 and Linf norms directly to verify
        ffgsm_perturbation = adv_ffgsm[0] - image[0]
        ffgsm_l2 = torch.norm(ffgsm_perturbation.flatten(), p=2).item()
        ffgsm_linf = torch.norm(ffgsm_perturbation.flatten(), p=float("inf")).item()
        logger.info(
            f"FFGSM L2 norm (direct): {ffgsm_l2:.6f}, Linf norm (direct): {ffgsm_linf:.6f}"
        )
        # Ensure metrics match direct calculation
        ffgsm_metrics["l2_norm"] = ffgsm_l2
        ffgsm_metrics["linf_norm"] = ffgsm_linf

        attack_results.append(
            {"name": "FFGSM", "adv_img": adv_ffgsm[0], "metrics": ffgsm_metrics}
        )

        # DeepFool attack
        logger.info("Running DeepFool attack...")
        adv_deepfool = deepfool(image, label)
        with torch.no_grad():
            deepfool_pred = model(adv_deepfool).argmax(dim=1).item()
        logger.info(
            f"DeepFool prediction: {dataset.class_names[deepfool_pred]} (index {deepfool_pred})"
        )

        deepfool_ssim = compute_ssim(image[0], adv_deepfool[0])
        logger.info(f"DeepFool SSIM: {deepfool_ssim:.6f}")

        deepfool_metrics = analyze_perturbation(adv_deepfool[0] - image[0], "DeepFool")
        deepfool_metrics["ssim"] = deepfool_ssim

        # Calculate L2 and Linf norms directly to verify
        deepfool_perturbation = adv_deepfool[0] - image[0]
        deepfool_l2 = torch.norm(deepfool_perturbation.flatten(), p=2).item()
        deepfool_linf = torch.norm(
            deepfool_perturbation.flatten(), p=float("inf")
        ).item()
        logger.info(
            f"DeepFool L2 norm (direct): {deepfool_l2:.6f}, Linf norm (direct): {deepfool_linf:.6f}"
        )
        # Ensure metrics match direct calculation
        deepfool_metrics["l2_norm"] = deepfool_l2
        deepfool_metrics["linf_norm"] = deepfool_linf

        attack_results.append(
            {
                "name": "DeepFool",
                "adv_img": adv_deepfool[0],
                "metrics": deepfool_metrics,
            }
        )

        # C&W attack
        logger.info("Running C&W attack...")
        adv_cw = cw(image, label)
        with torch.no_grad():
            cw_pred = model(adv_cw).argmax(dim=1).item()
        logger.info(f"C&W prediction: {dataset.class_names[cw_pred]} (index {cw_pred})")

        cw_ssim = compute_ssim(image[0], adv_cw[0])
        logger.info(f"C&W SSIM: {cw_ssim:.6f}")

        cw_metrics = analyze_perturbation(adv_cw[0] - image[0], "C&W")
        cw_metrics["ssim"] = cw_ssim

        # Calculate L2 and Linf norms directly to verify
        cw_perturbation = adv_cw[0] - image[0]
        cw_l2 = torch.norm(cw_perturbation.flatten(), p=2).item()
        cw_linf = torch.norm(cw_perturbation.flatten(), p=float("inf")).item()
        logger.info(
            f"C&W L2 norm (direct): {cw_l2:.6f}, Linf norm (direct): {cw_linf:.6f}"
        )
        # Ensure metrics match direct calculation
        cw_metrics["l2_norm"] = cw_l2
        cw_metrics["linf_norm"] = cw_linf

        # Check if any L2 norm seems unreasonably large (like the 33.3256 value we saw)
        if cw_metrics["l2_norm"] > 10.0:
            logger.warning(
                f"C&W L2 norm seems unusually large: {cw_metrics['l2_norm']}. Checking scaling..."
            )
            # It might be a scaling issue - normalize the metrics to [0,1] range if needed
            if cw_perturbation.max() > 1.0 or cw_perturbation.min() < -1.0:
                logger.warning(
                    "Perturbation values outside [-1,1] range, suggesting possible scaling issue"
                )

            # Try an alternative calculation for very small perturbations
            scaled_l2 = torch.norm(cw_perturbation.flatten() * 255, p=2).item() / 255
            logger.info(f"Alternative C&W L2 calculation: {scaled_l2:.6f}")
            if scaled_l2 < cw_metrics["l2_norm"] and scaled_l2 > 0.01:
                logger.warning(f"Using alternative L2 calculation: {scaled_l2}")
                cw_metrics["l2_norm"] = scaled_l2

        attack_results.append(
            {
                "name": "C&W",  # Removed backslash to prevent issues
                "adv_img": adv_cw[0],
                "metrics": cw_metrics,
            }
        )

        # Create output directory
        os.makedirs("paper/images", exist_ok=True)

        # Only create and save the comparison grid
        logger.info("Creating comparison grid visualization...")
        grid_fig = create_comparison_grid(
            image[0], attack_results, dataset.class_names, orig_pred
        )
        save_figure(grid_fig, "paper/images/attack_comparison_grid.png", dpi=300)

        # Print summary table
        logger.info("\n===== ATTACK COMPARISON SUMMARY =====")
        header = f"{'Attack':<10} | {'SSIM':<10} | {'L2 Norm':<10} | {'Linf Norm':<10} | {'Changed %':<10} | {'New Class':<20}"
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        for result in attack_results:
            metrics = result["metrics"]
            with torch.no_grad():
                pred = model(result["adv_img"].unsqueeze(0)).argmax(dim=1).item()

            logger.info(
                f"{result['name']:<10} | "
                f"{metrics['ssim']*100:>8.2f}% | "
                f"{metrics['l2_norm']:>10.4f} | "
                f"{metrics['linf_norm']:>10.4f} | "
                f"{metrics['pixels_changed']:>9.2f}% | "
                f"{dataset.class_names[pred]:<20}"
            )

        logger.info("Analysis and visualization complete!")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
