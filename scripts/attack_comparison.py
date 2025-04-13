#!/usr/bin/env python
"""This script creates a visualization comparing original image with
untargeted and targeted attacks on a specific ImageNet image.

USAGE::
    >>> python attack_comparison.py
    >>> python attack_comparison.py --image_idx <image_index>
    >>> python attack_comparison.py --show_all

image_index: index of the image to use from the ImageNet dataset
show_all: show all attacks [FGSM, FFGSM, DeepFool, CW, PGD, CG, LBFGS] (default: only untargeted CW)
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
from skimage.metrics import structural_similarity as ssim

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model

# Baseline attacks
from src.attacks.baseline import CW
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM
from src.attacks.baseline.attack_deepfool import DeepFool

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


def get_class_name_and_confidence(model, image, class_idx):
    """Get class name and confidence score for a given prediction"""
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, class_idx].item() * 100
    return confidence


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize different attack methods")
    parser.add_argument(
        "--image_idx",
        type=int,
        default=None,
        help="Index of ImageNet image to use (default: find first saluki image)",
    )
    parser.add_argument(
        "--show_all",
        action="store_true",
        help="Show all attacks instead of just untargeted/targeted CW",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model("resnet18").to(device)
    model.eval()

    # Load dataset
    dataset = get_dataset("imagenet", data_dir="data", max_samples=1000)

    # Select image
    image_idx = args.image_idx
    if image_idx is None:
        # Find saluki image (class 176)
        for idx, (_, label) in enumerate(dataset):
            if label == 176:  # Saluki class
                image_idx = idx
                break

        if image_idx is None:
            raise ValueError("Saluki image not found in dataset")

    # Get the image
    image, label = dataset[image_idx]
    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    # Generate adversarial examples
    print("Generating adversarial examples...")
    adv_examples = {}
    predictions = {}
    metrics = {}

    # Track original image predictions
    with torch.no_grad():
        orig_logits = model(image)
        orig_probs = torch.softmax(orig_logits, dim=1)
        orig_pred = orig_logits.argmax(dim=1).item()
        orig_conf = orig_probs[0, orig_pred].item() * 100

        predictions["Original"] = {"pred": orig_pred, "conf": orig_conf}

    if args.show_all:
        # Define all attacks
        attacks = {}

        # Baseline attacks
        attacks["FGSM"] = FGSM(model, eps=8 / 255)
        attacks["FFGSM"] = FFGSM(model, eps=8 / 255, alpha=0.4 * 8 / 255)
        attacks["DeepFool"] = DeepFool(
            model, steps=50, overshoot=0.02, top_k_classes=10
        )
        attacks["CW"] = CW(model, c=1.0, kappa=0, steps=100, lr=0.01)

        # Optimization-based attacks
        attacks["PGD"] = PGD(
            model,
            norm="Linf",
            eps=8 / 255,
            n_iterations=40,
            step_size=2 / 255,
            rand_init=True,
            early_stopping=True,
        )
        attacks["CG"] = CG(
            model,
            norm="L2",
            eps=3.0 / 255.0,
            n_iter=50,
            tv_lambda=0.1,
            color_lambda=0.1,
            perceptual_lambda=0.1,
            early_stopping=True,
        )
        attacks["L-BFGS"] = LBFGS(
            model,
            norm="L2",
            eps=1.0,
            n_iterations=100,
            history_size=10,
            initial_const=1e-1,
            binary_search_steps=10,
            const_factor=10.0,
            repeat_search=True,
        )

        # Set modes for all attacks to untargeted
        for attack_name, attack in attacks.items():
            attack.set_mode_default()  # Untargeted mode

        # Generate adversarial examples for each attack
        for attack_name, attack in attacks.items():
            print(f"Running {attack_name}...")
            adv_examples[attack_name] = attack(image, label)
    else:
        # Only run CW attacks (untargeted)
        print("Running untargeted CW attack...")
        cw_untargeted = CW(model, c=1.0, kappa=0, steps=100, lr=0.01)
        cw_untargeted.set_mode_default()  # Untargeted mode
        adv_examples["CW"] = cw_untargeted(image, label)

    # Always run targeted CW attack
    print("Running targeted CW attack...")
    cw_targeted = CW(model, c=10.0, kappa=20, steps=200, lr=0.01)
    cw_targeted.set_mode_targeted_least_likely()  # Targeted mode using least likely class
    adv_examples["CW (targeted)"] = cw_targeted(image, label)

    # Calculate predictions and metrics for all generated examples
    with torch.no_grad():
        # Process each attack
        for attack_name, adv_img in adv_examples.items():
            adv_logits = model(adv_img)
            adv_probs = torch.softmax(adv_logits, dim=1)
            adv_pred = adv_logits.argmax(dim=1).item()
            adv_conf = adv_probs[0, adv_pred].item() * 100

            # Compute metrics
            perturbation = adv_img[0] - image[0]
            l2_norm = torch.norm(perturbation.flatten(), p=2).item()
            linf_norm = torch.norm(perturbation.flatten(), p=float("inf")).item()
            ssim_value = compute_ssim(image[0], adv_img[0])

            predictions[attack_name] = {"pred": adv_pred, "conf": adv_conf}

            metrics[attack_name] = {
                "l2_norm": l2_norm,
                "linf_norm": linf_norm,
                "ssim": ssim_value,
            }

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

    # Create visualization
    plt.rcParams["figure.facecolor"] = "white"

    if args.show_all:
        # Configure full layout with all attacks
        num_rows = 3  # Original, Perturbation, Adversarial
        num_cols = len(adv_examples) + 1  # All attacks + original

        fig = plt.figure(figsize=(num_cols * 3, num_rows * 3))

        # Create grid
        gs = plt.GridSpec(num_rows, num_cols, figure=fig)

        # Get original class name
        orig_class_name = dataset.class_names[orig_pred].split(",")[0]

        # Display original image in first column
        ax_orig = fig.add_subplot(gs[0, 0])
        orig_img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
        ax_orig.imshow(orig_img)
        ax_orig.set_title(
            f"Original\nPredicted: \\texttt{{{orig_class_name}}}\n({orig_conf:.1f}\\%)",
            pad=20,
        )
        ax_orig.axis("off")

        # Empty perturbation cell for original
        ax_orig_pert = fig.add_subplot(gs[1, 0])
        ax_orig_pert.text(
            0.5,
            0.5,
            "No perturbation",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax_orig_pert.axis("off")

        # Empty bottom cell for original
        ax_orig_bot = fig.add_subplot(gs[2, 0])
        ax_orig_bot.axis("off")

        # Add each attack method
        col_idx = 1
        for attack_name, adv_img in adv_examples.items():
            # Skip targeted CW attack - we'll put it last
            if attack_name == "CW (targeted)" and col_idx < num_cols - 1:
                continue

            # Get prediction info
            adv_pred = predictions[attack_name]["pred"]
            adv_conf = predictions[attack_name]["conf"]
            ssim_value = metrics[attack_name]["ssim"]

            # Top row: adversarial example
            ax_adv = fig.add_subplot(gs[0, col_idx])
            adv_img_np = denormalize(adv_img[0]).cpu().permute(1, 2, 0).numpy()
            ax_adv.imshow(adv_img_np)
            pred_name = dataset.class_names[adv_pred].split(",")[0]
            ax_adv.set_title(
                f"{attack_name}\nPredicted: \\texttt{{{pred_name}}}\n({adv_conf:.1f}\\%)",
                pad=20,
            )
            ax_adv.axis("off")

            # Middle row: perturbation visualization
            ax_pert = fig.add_subplot(gs[1, col_idx])
            pert = visualize_perturbation(adv_img[0] - image[0])
            ax_pert.imshow(pert)
            ax_pert.set_title("Perturbation", pad=10)
            ax_pert.axis("off")

            # Bottom row: SSIM and norms
            ax_metrics = fig.add_subplot(gs[2, col_idx])
            l2_norm = metrics[attack_name]["l2_norm"]
            linf_norm = metrics[attack_name]["linf_norm"]
            ax_metrics.text(
                0.5,
                0.5,
                f"SSIM: {ssim_value*100:.1f}\\%\n$L_2$: {l2_norm:.3f}\n$L_\\infty$: {linf_norm:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax_metrics.axis("off")

            col_idx += 1

        # Add targeted CW attack as last column if not already added
        if "CW (targeted)" in adv_examples and col_idx < num_cols:
            attack_name = "CW (targeted)"
            adv_img = adv_examples[attack_name]

            # Get prediction info
            adv_pred = predictions[attack_name]["pred"]
            adv_conf = predictions[attack_name]["conf"]
            ssim_value = metrics[attack_name]["ssim"]

            # Top row: adversarial example
            ax_adv = fig.add_subplot(gs[0, -1])
            adv_img_np = denormalize(adv_img[0]).cpu().permute(1, 2, 0).numpy()
            ax_adv.imshow(adv_img_np)
            pred_name = dataset.class_names[adv_pred].split(",")[0]
            ax_adv.set_title(
                f"{attack_name}\nPredicted: \\texttt{{{pred_name}}}\n({adv_conf:.1f}\\%)",
                pad=20,
            )
            ax_adv.axis("off")

            # Middle row: perturbation visualization
            ax_pert = fig.add_subplot(gs[1, -1])
            pert = visualize_perturbation(adv_img[0] - image[0])
            ax_pert.imshow(pert)
            ax_pert.set_title("Perturbation", pad=10)
            ax_pert.axis("off")

            # Bottom row: SSIM and norms
            ax_metrics = fig.add_subplot(gs[2, -1])
            l2_norm = metrics[attack_name]["l2_norm"]
            linf_norm = metrics[attack_name]["linf_norm"]
            ax_metrics.text(
                0.5,
                0.5,
                f"SSIM: {ssim_value*100:.1f}\\%\n$L_2$: {l2_norm:.3f}\n$L_\\infty$: {linf_norm:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax_metrics.axis("off")

    else:
        # Traditional CW comparison with only untargeted and targeted
        fig = plt.figure(figsize=(25, 5))

        # Create a grid with specific width ratios for the three main sections
        gs = plt.GridSpec(1, 5, width_ratios=[1, 0.1, 2, 0.1, 2])

        # Get original class name
        orig_class_name = dataset.class_names[orig_pred].split(",")[0]

        # Original image
        ax0 = fig.add_subplot(gs[0])
        orig_img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
        ax0.imshow(orig_img)
        ax0.set_title(
            f"Original\nPredicted: \\texttt{{{orig_class_name}}}\n({orig_conf:.1f}\\%)",
            pad=20,
        )
        ax0.axis("off")

        # First separator (empty subplot)
        sep1 = fig.add_subplot(gs[1])
        sep1.axis("off")

        # Untargeted attack group
        untargeted_group = plt.subplot(gs[2])
        untargeted_group.axis("off")

        # Create nested grid for untargeted attack
        gs_untarg = plt.GridSpec(2, 2, height_ratios=[1, 0.2])
        gs_untarg.update(
            left=untargeted_group.get_position().x0,
            right=untargeted_group.get_position().x1,
            bottom=untargeted_group.get_position().y0,
            top=untargeted_group.get_position().y1,
        )

        # Untargeted perturbation and result
        ax_pert_untarg = fig.add_subplot(gs_untarg[0, 0])
        pert_untarg = visualize_perturbation(adv_examples["CW"][0] - image[0])
        ax_pert_untarg.imshow(pert_untarg)
        ax_pert_untarg.set_title("Untargeted\nPerturbation", pad=20)
        ax_pert_untarg.axis("off")

        ax_res_untarg = fig.add_subplot(gs_untarg[0, 1])
        untarg_img = denormalize(adv_examples["CW"][0]).cpu().permute(1, 2, 0).numpy()
        ax_res_untarg.imshow(untarg_img)
        untarg_pred = predictions["CW"]["pred"]
        untarg_conf = predictions["CW"]["conf"]
        pred_name = dataset.class_names[untarg_pred].split(",")[0]
        ax_res_untarg.set_title(
            f"Predicted: \\texttt{{{pred_name}}}\n({untarg_conf:.1f}\\%)", pad=20
        )
        ax_res_untarg.axis("off")

        # Add SSIM text below the result image
        ax_ssim_untarg = fig.add_subplot(gs_untarg[1, 1])
        untarg_ssim = metrics["CW"]["ssim"]
        ax_ssim_untarg.text(
            0.5,
            0.5,
            f"SSIM: {untarg_ssim*100:.1f}\\%",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax_ssim_untarg.axis("off")

        # Second separator (empty subplot)
        sep2 = fig.add_subplot(gs[3])
        sep2.axis("off")

        # Targeted attack group
        targeted_group = plt.subplot(gs[4])
        targeted_group.axis("off")

        # Create nested grid for targeted attack
        gs_targ = plt.GridSpec(2, 2, height_ratios=[1, 0.2])
        gs_targ.update(
            left=targeted_group.get_position().x0,
            right=targeted_group.get_position().x1,
            bottom=targeted_group.get_position().y0,
            top=targeted_group.get_position().y1,
        )

        # Targeted perturbation and result
        ax_pert_targ = fig.add_subplot(gs_targ[0, 0])
        pert_targ = visualize_perturbation(adv_examples["CW (targeted)"][0] - image[0])
        ax_pert_targ.imshow(pert_targ)
        ax_pert_targ.set_title("Targeted\nPerturbation", pad=20)
        ax_pert_targ.axis("off")

        ax_res_targ = fig.add_subplot(gs_targ[0, 1])
        targ_img = (
            denormalize(adv_examples["CW (targeted)"][0]).cpu().permute(1, 2, 0).numpy()
        )
        ax_res_targ.imshow(targ_img)
        targ_pred = predictions["CW (targeted)"]["pred"]
        targ_conf = predictions["CW (targeted)"]["conf"]
        pred_name = dataset.class_names[targ_pred].split(",")[0]
        ax_res_targ.set_title(
            f"Predicted: \\texttt{{{pred_name}}}\n({targ_conf:.1f}\\%)", pad=20
        )
        ax_res_targ.axis("off")

        # Add SSIM text below the result image
        ax_ssim_targ = fig.add_subplot(gs_targ[1, 1])
        targ_ssim = metrics["CW (targeted)"]["ssim"]
        ax_ssim_targ.text(
            0.5,
            0.5,
            f"SSIM: {targ_ssim*100:.1f}\\%",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax_ssim_targ.axis("off")

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
    os.makedirs("images", exist_ok=True)
    output_filename = (
        "attack_comparison_full.png" if args.show_all else "attack_comparison.png"
    )
    plt.savefig(
        f"images/{output_filename}",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved visualization to images/{output_filename}")


if __name__ == "__main__":
    main()
