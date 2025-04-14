#!/usr/bin/env python
"""Generate grid visualization of adversarial examples from multiple attack methods.

USAGE::
    >>> python attack_grid.py
    >>> python attack_grid.py --model <model_name>
    >>> python attack_grid.py --output <output_path>

model_name: name of the model to attack (default: resnet18)
output_path: path to save the output image (default: images/attack_comparison.png)
"""

import os
import sys
import logging
import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch

# Configure matplotlib to use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    }
)

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

# Baseline attacks
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM
from src.attacks.baseline.attack_deepfool import DeepFool
from src.attacks.baseline.attack_cw import CW

# Optimization-based attacks
from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import CG
from src.attacks.attack_lbfgs import LBFGS


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


def safe_set_attr(obj, attr_name, value):
    """Safely set an attribute if it exists"""
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)
        return True
    return False


def generate_adversarial_grid(
    model,
    dataset,
    num_images=4,
    attack_methods=None,
    save_path="images/attack_comparison.png",
    max_attempts_per_image=5,  # Maximum attempts per image to find successful attacks
    max_total_candidates=10,  # Maximum number of total images to try
):
    """
    Generate a grid visualization with original images and their adversarial versions.
    Keeps trying different images until finding ones where all attacks succeed.

    Args:
        model: Target model to attack
        dataset: Dataset with images to attack
        num_images: Number of images to include in the grid
        attack_methods: Dictionary of attack methods to use
        save_path: Path to save the visualization
        max_attempts_per_image: Maximum number of attempts to succeed with all attacks on one image
        max_total_candidates: Maximum number of total images to try
    """
    device = next(model.parameters()).device

    # Default attack methods if none provided
    if attack_methods is None:
        attack_methods = {
            # Baseline methods
            "FGSM": FGSM(model, eps=8 / 255),
            "FFGSM": FFGSM(model, eps=8 / 255, alpha=0.4 * 8 / 255),
            "DeepFool": DeepFool(model, steps=50, overshoot=0.02, top_k_classes=10),
            "C&W": CW(model, c=1.0, kappa=0, steps=100, lr=0.01),
            # Optimization methods
            "PGD": PGD(
                model,
                norm="Linf",
                eps=8 / 255,
                n_iterations=40,
                step_size=2 / 255,
                rand_init=True,
                early_stopping=True,
            ),
            "CG": CG(
                model,
                norm="Linf",
                eps=0.0314,
                n_iter=50,
                beta_method="HS",
                restart_interval=10,
                tv_lambda=0.05,
                color_lambda=0.05,
                perceptual_lambda=0.05,
                rand_init=True,
                fgsm_init=True,
                adaptive_restart=True,
                early_stopping=True,
                strict_epsilon_constraint=True,
            ),
            "L-BFGS": LBFGS(
                model,
                norm="L2",
                eps=1.0,
                n_iterations=100,
                history_size=10,
                initial_const=1e-1,
                binary_search_steps=10,
                const_factor=10.0,
                repeat_search=True,
            ),
        }

    # Number of columns: 1 for original + 1 for each attack method
    num_cols = 1 + len(attack_methods)

    # Generate a pool of candidate images
    candidate_indices = np.random.choice(
        len(dataset), min(max_total_candidates, len(dataset)), replace=False
    )
    logger.info(f"Generated a pool of {len(candidate_indices)} candidate images")

    # Store selected images and their attack results
    successful_images = []  # List of tuples: (img_idx, attack_results)
    candidate_results = {}  # Store all attack results keyed by image_idx

    # Process each candidate image until we have enough successful ones
    pbar = tqdm(candidate_indices, desc="Finding successful adversarial examples")
    for img_idx in pbar:
        # Skip if we already have enough successful images
        if len(successful_images) >= num_images:
            break

        # Get original image and label
        img, label = dataset[img_idx]
        img = img.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        # Get original prediction
        with torch.no_grad():
            outputs = model(img)
            orig_pred = outputs.argmax(dim=1).item()

            # Skip if the model already misclassifies this image
            if orig_pred != label.item():
                pbar.set_postfix({"status": f"Skipping misclassified image {img_idx}"})
                continue

        # Track success for each attack method
        all_success = True
        attack_results = {}

        # Try each attack method
        for attack_name, attack in attack_methods.items():
            attack_success = False
            attack_metrics = {}

            # Multiple attempts for this attack on this image
            for attempt in range(max_attempts_per_image):
                try:
                    # Set attack to untargeted mode
                    attack.set_mode_default()

                    # Tune attack parameters based on attempt number
                    if attempt > 0:
                        # Escalate parameters for each attempt
                        if attack_name == "L-BFGS":
                            # L-BFGS: binary search steps and initial constant
                            safe_set_attr(
                                attack, "binary_search_steps", min(14, 8 + attempt)
                            )

                            # More aggressive adjustments for L-BFGS
                            if hasattr(attack, "initial_const"):
                                # Exponentially increase the initial constant with each attempt
                                attack.initial_const *= 3.0 ** min(attempt, 4)

                            # Increase epsilon for more effective attack
                            if hasattr(attack, "eps"):
                                attack.eps = min(
                                    4.0, attack.eps * (1.5 ** min(attempt, 3))
                                )

                            # Enable repeat search after the first few attempts
                            if attempt >= 2 and hasattr(attack, "repeat_search"):
                                attack.repeat_search = True

                            # Increase iterations for more thorough optimization
                            safe_set_attr(
                                attack, "n_iterations", min(200, 100 + attempt * 20)
                            )

                        elif attack_name == "PGD":
                            # PGD: epsilon, iterations, step size
                            if hasattr(attack, "eps"):
                                attack.eps = min(
                                    1.0, attack.eps * (1.2 ** min(attempt, 3))
                                )
                            safe_set_attr(
                                attack, "n_iterations", min(200, 40 + attempt * 20)
                            )
                            safe_set_attr(
                                attack,
                                "step_size",
                                min(
                                    0.1,
                                    (
                                        attack.step_size
                                        if hasattr(attack, "step_size")
                                        else 0.01
                                    )
                                    * 1.2,
                                ),
                            )

                        elif attack_name == "CG":
                            # CG: epsilon, iterations (using various possible attribute names)
                            if hasattr(attack, "eps"):
                                attack.eps = min(
                                    1.0, attack.eps * (1.2 ** min(attempt, 3))
                                )
                            # Try different names for the iterations parameter
                            success = False
                            for iter_attr in ["n_iter", "n_iterations", "iterations"]:
                                if safe_set_attr(
                                    attack,
                                    iter_attr,
                                    min(200, getattr(attack, iter_attr, 50) + 20),
                                ):
                                    success = True
                                    break

                        elif attack_name == "DeepFool":
                            # DeepFool: steps, overshoot
                            safe_set_attr(attack, "steps", min(100, 30 + attempt * 10))
                            safe_set_attr(
                                attack,
                                "overshoot",
                                min(0.1, 0.02 * (1 + attempt * 0.5)),
                            )

                        elif attack_name == "C&W":
                            # C&W: steps, learning rate, confidence
                            safe_set_attr(attack, "steps", min(200, 100 + attempt * 20))
                            if hasattr(attack, "c"):
                                attack.c *= 1.5 ** min(attempt, 3)
                            safe_set_attr(
                                attack, "lr", min(0.05, 0.01 * (1 + attempt * 0.2))
                            )

                        elif attack_name in ["FGSM", "FFGSM"]:
                            # FGSM/FFGSM: epsilon
                            if hasattr(attack, "eps"):
                                attack.eps = min(
                                    32 / 255, attack.eps * (1.2 ** min(attempt, 3))
                                )

                    # Generate adversarial example
                    adv_img = attack(img, label)

                    # Check if attack succeeded
                    with torch.no_grad():
                        adv_outputs = model(adv_img)
                        adv_pred = adv_outputs.argmax(dim=1).item()

                    attack_success = adv_pred != orig_pred

                    # Calculate metrics
                    perturbation = adv_img[0] - img[0]
                    l2_norm = torch.norm(perturbation.flatten(), p=2).item()
                    linf_norm = torch.norm(
                        perturbation.flatten(), p=float("inf")
                    ).item()
                    ssim_value = compute_ssim(img[0], adv_img[0])

                    # Store metrics
                    attack_metrics = {
                        "adv_img": adv_img,
                        "success": attack_success,
                        "l2_norm": l2_norm,
                        "linf_norm": linf_norm,
                        "ssim": ssim_value,
                        "adv_pred": adv_pred,
                        "attempt": attempt + 1,
                    }

                    # Break if successful
                    if attack_success:
                        break

                except Exception as e:
                    logger.warning(f"Error in {attack_name} attack: {str(e)}")
                    # Continue to next attempt

            # Store result for this attack
            attack_results[attack_name] = attack_metrics

            # If any attack fails, mark the whole image as unsuccessful
            if not attack_success:
                all_success = False
                pbar.set_postfix(
                    {"status": f"Failed on {attack_name} after {attempt+1} attempts"}
                )

        # Store results for this image
        candidate_results[img_idx] = {
            "img": img,
            "label": label.item(),
            "orig_pred": orig_pred,
            "attack_results": attack_results,
            "all_success": all_success,
            # Count successful attacks for this image
            "success_count": sum(
                1 for a in attack_results.values() if a.get("success", False)
            ),
            # Specific flag for L-BFGS success (but don't require it for overall success)
            "lbfgs_success": attack_results.get("L-BFGS", {}).get("success", False),
        }

        # Consider success threshold - allow images where at least N-1 attacks succeeded
        # This makes it easier to find good candidate images when L-BFGS struggles
        min_success_threshold = len(attack_methods) - 1  # Allow one attack to fail
        if candidate_results[img_idx]["success_count"] >= min_success_threshold:
            successful_images.append(img_idx)
            pbar.set_postfix(
                {
                    "status": f"Success! {len(successful_images)}/{num_images} (with {candidate_results[img_idx]['success_count']} attacks)"
                }
            )

        # If we already have found some successes but need more, lower the threshold
        elif (
            len(successful_images) > 0
            and len(successful_images) < num_images // 2
            and candidate_results[img_idx]["success_count"] >= len(attack_methods) - 2
        ):
            # Allow two failed attacks if we're struggling to find perfect matches
            successful_images.append(img_idx)
            pbar.set_postfix(
                {
                    "status": f"Partial success! {len(successful_images)}/{num_images} (with {candidate_results[img_idx]['success_count']} attacks)"
                }
            )

    # If we don't have enough completely successful images, find the best partial successes
    if len(successful_images) < num_images:
        logger.warning(
            f"Could only find {len(successful_images)} images with all attacks successful. "
            f"Adding {num_images - len(successful_images)} images with partial success."
        )

        # Score remaining candidates by number of successful attacks
        remaining_candidates = [
            (idx, data)
            for idx, data in candidate_results.items()
            if idx not in successful_images
        ]

        # Sort by:
        # 1. Total success count (higher is better)
        # 2. Average SSIM of successful attacks (higher is better)
        # 3. L-BFGS success specifically (gives some weight to this attack)
        sorted_candidates = sorted(
            remaining_candidates,
            key=lambda x: (
                x[1]["success_count"],  # Count of successful attacks
                sum(
                    a.get("ssim", 0)
                    for a in x[1]["attack_results"].values()
                    if a.get("success", False)
                )
                / max(1, x[1]["success_count"]),  # Average SSIM of successful attacks
                (
                    1 if x[1].get("lbfgs_success", False) else 0
                ),  # Give some priority to L-BFGS success
            ),
            reverse=True,
        )

        # Add best partial successes until we have enough images
        for idx, _ in sorted_candidates:
            if len(successful_images) >= num_images:
                break
            successful_images.append(idx)

    # If we still don't have enough images, something is wrong
    if len(successful_images) < num_images:
        raise ValueError(
            f"Could not find {num_images} images to attack. Found only {len(successful_images)}."
        )

    # Select the final images to display
    final_image_indices = successful_images[:num_images]
    logger.info(f"Selected {len(final_image_indices)} images for the grid")

    # Create figure for grid
    fig = plt.figure(figsize=(num_cols * 2.2, num_images * 2 + 0.5))

    # Create a GridSpec to make room for the header row
    gs = fig.add_gridspec(
        num_images + 1, num_cols, height_ratios=[0.2] + [1] * num_images
    )

    # Create header row for method names
    header_axes = []
    for col, name in enumerate(["Original"] + list(attack_methods.keys())):
        # Escape special LaTeX characters for rendering
        display_name = name.replace("&", "\\&")
        header_ax = fig.add_subplot(gs[0, col])
        header_ax.text(
            0.5,
            0.5,
            display_name,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
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

    # Create a list to store attack results metrics for reporting
    results_data = []

    # Process each selected image
    for row, img_idx in enumerate(final_image_indices):
        # Get data for this image
        image_data = candidate_results[img_idx]
        img = image_data["img"]
        label = image_data["label"]
        orig_pred = image_data["orig_pred"]

        # Display original image
        orig_np = denormalize(img[0]).cpu().permute(1, 2, 0).numpy()
        axes[row, 0].imshow(orig_np)
        axes[row, 0].axis("off")

        # Set row label with class name
        class_name = dataset.class_names[label]
        short_class = class_name.split(",")[
            0
        ]  # Take only the first part of the class name
        axes[row, 0].set_ylabel(f"{short_class}", fontsize=14)

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
            fontsize=12,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

        # Display adversarial examples for each attack
        for col, attack_name in enumerate(attack_methods.keys(), start=1):
            attack_result = image_data["attack_results"].get(attack_name, {})

            if attack_result and "adv_img" in attack_result:
                adv_img = attack_result["adv_img"]
                success = attack_result["success"]
                l2_norm = attack_result["l2_norm"]
                linf_norm = attack_result["linf_norm"]
                ssim_value = attack_result["ssim"]
                adv_pred = attack_result["adv_pred"]

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
                        fontsize=12,
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
            else:
                # No successful attack
                logger.warning(f"No result for {attack_name} on image {img_idx}")
                axes[row, col].imshow(np.zeros_like(orig_np))
                axes[row, col].text(
                    0.5,
                    0.5,
                    "Failed",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )

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
            "avg_time": 0,  # Time not tracked in this function
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
        avg_ssim = attack_results[attack_name]["avg_ssim"]

        # Format metrics text
        metrics_text = f"({success_rate:.0f}\\% success)"

        # Add SSIM if available and some attacks succeeded
        if success_rate > 0:
            metrics_text = f"({success_rate:.0f}\\% success, SSIM: {avg_ssim/100:.2f})"

        header_axes[col].text(
            0.5,
            0.2,
            metrics_text,
            ha="center",
            va="center",
            fontsize=12,
            fontstyle="italic",
        )

    # Adjust layout
    plt.tight_layout()

    # Remove ticks from all subplots for cleaner look
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, results_data


def main():
    """Main function to generate adversarial attack grid visualizations"""
    parser = argparse.ArgumentParser(
        description="Generate adversarial attack grid visualizations"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to include in the grid",
    )
    parser.add_argument("--model", type=str, default="resnet18", help="Model to attack")
    parser.add_argument(
        "--output",
        type=str,
        default="images/attack_comparison.png",
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
