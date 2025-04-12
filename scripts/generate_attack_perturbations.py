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


def create_visualizations(
    original, perturbed, name, metrics, class_names, idx=0, thresholds=None
):
    """
    Create detailed visualization for one attack method.
    Shows original, adversarial, and difference highlighting using L-infinity norm.
    """
    try:
        if thresholds is None:
            # Adaptive thresholds based on the perturbation magnitude
            thresholds = {
                "highlight": max(
                    0.003, metrics["linf_norm"] * 0.1
                ),  # 10% of L-infinity norm with minimum
                "significant": max(
                    0.01, metrics["linf_norm"] * 0.5
                ),  # 50% of L-infinity norm with minimum
            }

        # Get image data
        orig_img = denormalize(original).cpu()
        adv_img = denormalize(perturbed).cpu()

        # Calculate perturbation
        perturbation = perturbed - original

        # Convert to numpy for matplotlib
        orig_np = orig_img.permute(1, 2, 0).numpy()
        adv_np = adv_img.permute(1, 2, 0).numpy()

        # Calculate pixel-wise absolute difference using L-infinity norm (max change across RGB channels)
        pixel_diff = torch.max(torch.abs(perturbation), dim=0)[0].cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(orig_np)
        axes[0].set_title(f"Original\nPredicted: {class_names[idx]}")
        axes[0].axis("off")

        # Adversarial image
        axes[1].imshow(adv_np)
        with torch.no_grad():
            adv_pred = torch.argmax(model(perturbed.unsqueeze(0)), dim=1).item()
        axes[1].set_title(f"Adversarial ({name})\nPredicted: {class_names[adv_pred]}")
        axes[1].axis("off")

        # Difference visualization (overlay on adversarial)
        axes[2].imshow(adv_np)

        # Create custom colormap for differences
        # Blue for minimal changes, yellow for moderate, red for significant
        colors = [
            (0, 0, 1, 0),  # Transparent blue (minimal)
            (0, 0, 1, 0.3),  # Semi-transparent blue (very small)
            (1, 1, 0, 0.5),  # Semi-transparent yellow (moderate)
            (1, 0, 0, 0.7),
        ]  # Semi-transparent red (significant)

        # Create mask where differences are above threshold
        highlight_mask = pixel_diff > thresholds["highlight"]

        # Calculate percentage of pixels changed for this visualization
        pct_pixels_visible = np.mean(highlight_mask) * 100
        logger.info(
            f"{name}: Percentage of pixels highlighted with L-inf norm: {pct_pixels_visible:.2f}%"
        )

        # Create heatmap of differences
        cmap = LinearSegmentedColormap.from_list("custom_diff", colors, N=100)

        # Normalize difference values for colormap
        # Small differences will be blue, large differences will be red
        norm_diff = np.zeros_like(pixel_diff)
        if pixel_diff.max() > 0:
            norm_diff = np.clip(pixel_diff / thresholds["significant"], 0, 3) / 3

        # Plot only where the mask is True
        masked_diff = np.zeros_like(norm_diff)
        masked_diff[highlight_mask] = norm_diff[highlight_mask]

        # Plot heatmap on top of adversarial image
        diff_plot = axes[2].imshow(masked_diff, cmap=cmap, alpha=0.7)
        axes[2].set_title(
            f"Perturbation Highlights (L-inf)\nL-inf={metrics['linf_norm']:.4f}, SSIM={metrics['ssim']:.4f}"
        )
        axes[2].axis("off")

        # Add colorbar
        cbar = plt.colorbar(diff_plot, ax=axes[2], orientation="vertical", shrink=0.7)
        cbar.set_label("Perturbation Intensity")

        # Add metrics as text in the plot
        metrics_text = (
            f"L-inf Norm: {metrics['linf_norm']:.4f}\n"
            f"L2 Norm: {metrics['l2_norm']:.4f}\n"
            f"SSIM: {metrics['ssim']:.4f}\n"
            f"Visible Pixels: {pct_pixels_visible:.2f}%\n"
            f"Changed Pixels: {metrics['pixels_changed']:.2f}%"
        )

        axes[2].text(
            0.02,
            0.02,
            metrics_text,
            transform=axes[2].transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating visualization for {name}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def create_comparison_grid(original, attack_results, class_names, orig_idx):
    """Create a comparison grid of all attack methods using L-infinity norm for visualization"""
    try:
        # Disable LaTeX rendering to avoid font and special character issues
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

        # Get max L-infinity norm for relative color intensity
        max_linf = max([result["metrics"]["linf_norm"] for result in attack_results])

        # For each attack
        for i, result in enumerate(attack_results):
            name = result["name"]
            adv_img = result["adv_img"]
            metrics = result["metrics"]

            # Get adversarial image
            adv_np = denormalize(adv_img).cpu().permute(1, 2, 0).numpy()

            # Show adversarial image
            axes[i + 1].imshow(adv_np)

            # Calculate normalized intensity for this attack (for color intensity)
            intensity = min(1.0, metrics["linf_norm"] / max_linf)

            # Get perturbation heat map using L-infinity norm
            pert = adv_img - original
            # Max absolute change across RGB channels for each pixel (L-infinity)
            pert_mag = torch.max(torch.abs(pert), dim=0)[0].cpu().numpy()

            # Create mask for significant changes
            threshold = max(
                0.003, metrics["linf_norm"] * 0.1
            )  # 10% of linf with minimum
            highlight_mask = pert_mag > threshold

            # Calculate percentage of pixels changed for this visualization
            pct_pixels_visible = np.mean(highlight_mask) * 100
            logger.info(f"{name} grid: L-inf visible pixels: {pct_pixels_visible:.2f}%")

            # Create a heatmap color overlay
            cmap = plt.colormaps.get_cmap("viridis")

            # Normalize perturbation magnitude
            if pert_mag.max() > 0:
                norm_mag = pert_mag / pert_mag.max()
            else:
                norm_mag = pert_mag

            # Create RGBA heatmap with proper alpha
            heatmap = cmap(norm_mag)
            # Make it transparent where changes are below threshold
            heatmap[..., 3] = np.zeros_like(norm_mag)  # Start with all transparent
            heatmap[highlight_mask, 3] = 0.5 * intensity  # Add alpha where significant

            # Overlay heatmap
            axes[i + 1].imshow(heatmap)

            # Add title with metrics
            if metrics["ssim"] > 0.9999:
                ssim_display = 99.99  # Cap at 99.99%
            else:
                ssim_display = metrics["ssim"] * 100

            axes[i + 1].set_title(
                f"{name}\n({ssim_display:.2f}% similar, L-inf={metrics['linf_norm']:.4f})",
                pad=10,
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

        attack_results.append(
            {
                "name": "C&W",  # Removed backslash to prevent issues
                "adv_img": adv_cw[0],
                "metrics": cw_metrics,
            }
        )

        # Create output directory
        os.makedirs("paper/images", exist_ok=True)

        # Create detailed visualizations for each attack
        logger.info("Creating detailed visualizations...")
        for result in attack_results:
            safe_name = get_safe_filename(result["name"])
            fig = create_visualizations(
                image[0],
                result["adv_img"],
                result["name"],
                result["metrics"],
                dataset.class_names,
                orig_pred,
            )
            save_figure(fig, f"paper/images/attack_details_{safe_name}.png", dpi=300)

        # Create comparison grid visualization
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
