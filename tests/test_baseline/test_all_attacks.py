#!/usr/bin/env python
"""
Tests for all baseline attacks.

This script tests all baseline attacks (FGSM, FFGSM, DeepFool, CW, MIFGSM)
on a small sample of ImageNet data, using our model wrappers
that handle normalized inputs correctly.

Usage:
    python -m tests.test_baseline.test_all_attacks
"""

import os
import sys
import time
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM
from src.attacks.baseline.attack_deepfool import DeepFool
from src.attacks.baseline.attack_cw import CW
from src.attacks.baseline.attack_mifgsm import MIFGSM


def plot_images(
    original, adversarial, attack_name, original_label, adv_prediction, save_path=None
):
    """
    Create a comparison plot between original and adversarial images.

    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor
        attack_name: Name of the attack
        original_label: Original label
        adv_prediction: Prediction on adversarial example
        save_path: Path to save the figure
    """
    # Denormalize images for visualization
    mean = (
        torch.tensor([0.485, 0.456, 0.406], dtype=original.dtype)
        .view(3, 1, 1)
        .to(original.device)
    )
    std = (
        torch.tensor([0.229, 0.224, 0.225], dtype=original.dtype)
        .view(3, 1, 1)
        .to(original.device)
    )

    def denormalize(x):
        """Convert from normalized to [0,1] range for visualization"""
        img = x.cpu().clone()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        return img

    # Get numpy versions for plotting
    original_np = denormalize(original).permute(1, 2, 0).numpy()
    adversarial_np = denormalize(adversarial).permute(1, 2, 0).numpy()

    # Calculate perturbation for visualization
    perturbation = adversarial - original

    # Sign/direction view (shows how values change: increase=red, decrease=blue)
    # Create a diverging colormap visualization
    perturbation_sign = perturbation.cpu()
    # Scale the perturbation for better visualization, with diverging colors
    max_pert = (
        perturbation_sign.abs().max().item()
        if perturbation_sign.abs().max().item() > 0
        else 1.0
    )
    perturbation_sign = perturbation_sign / (max_pert * 0.1)  # Normalize and enhance
    perturbation_sign = torch.clamp(perturbation_sign, -1, 1)

    # Convert to a diverging colormap (red-white-blue)
    # Red for positive changes, blue for negative
    pert_rgb = torch.zeros(
        (3, perturbation_sign.shape[1], perturbation_sign.shape[2]),
        device=perturbation_sign.device,
    )
    pert_rgb[0] = torch.clamp(
        perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1
    )  # Red channel
    pert_rgb[2] = torch.clamp(
        -perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1
    )  # Blue channel
    pert_rgb[1] = torch.clamp(
        1.0 - perturbation_sign.mean(dim=0).abs() * 0.5, 0, 1
    )  # Green channel

    # Convert to numpy
    diverging_pert = pert_rgb.permute(1, 2, 0).numpy()

    # Create the figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax[0].imshow(original_np)
    ax[0].set_title(f"Original\nLabel: {original_label}")
    ax[0].axis("off")

    # Plot perturbation (sign)
    ax[1].imshow(diverging_pert)
    ax[1].set_title(f"Perturbation (Direction)\nRed=Increase, Blue=Decrease")
    ax[1].axis("off")

    # Plot adversarial image
    ax[2].imshow(adversarial_np)
    ax[2].set_title(f"Adversarial\nPredicted: {adv_prediction}")
    ax[2].axis("off")

    # Add attack information
    plt.suptitle(f"Adversarial Example - {attack_name}", fontsize=16)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def test_attack(attack, model, dataset, attack_name, args):
    """
    Test an attack on a dataset and report metrics.

    Args:
        attack: Attack instance
        model: Target model
        dataset: Dataset to attack
        attack_name: Name of the attack for reporting
        args: Command-line arguments

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*50}")
    print(f"Testing {attack_name} attack")
    print(f"{'='*50}")

    # Create a dataloader with the test batch size
    dataloader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    class_names = dataset.class_names

    # Store metrics
    metrics = {
        "name": attack_name,
        "success_rate": 0.0,
        "l2_norm": 0.0,
        "linf_norm": 0.0,
        "ssim": 0.0,
        "time_per_sample": 0.0,
        "iterations": 0.0,
        "gradient_calls": 0.0,
    }

    # Reset attack metrics before starting
    attack.reset_metrics()

    # Track successful adversarial examples for visualization
    successful_examples = []

    # Run attack on a limited number of batches
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(
        tqdm(dataloader, desc=f"{attack_name}")
    ):
        if batch_idx >= args.max_batches:
            break

        # Ensure inputs and labels are the right type and on the right device
        inputs = inputs.to(args.device).float()  # Explicitly convert to float32
        labels = labels.to(args.device)

        total_samples += inputs.size(0)

        # Get original predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, original_preds = torch.max(outputs, 1)

            # Skip samples that are already misclassified
            correct_mask = original_preds == labels
            if not correct_mask.any():
                print(
                    f"  All samples in this batch are already misclassified, skipping"
                )
                continue

            # Only attack correctly classified samples
            correct_inputs = inputs[correct_mask]
            correct_labels = labels[correct_mask]

            if len(correct_inputs) == 0:
                continue

        # Attack only the correctly classified samples
        attack_start = time.time()
        adversarial = attack(correct_inputs, correct_labels)
        attack_time = time.time() - attack_start

        # Explicitly evaluate attack success - this updates the attack's internal metrics
        batch_success_rate, success_mask, (orig_preds, adv_preds) = (
            attack.evaluate_attack_success(correct_inputs, adversarial, correct_labels)
        )

        # Explicitly compute perturbation metrics - this updates the attack's internal metrics
        perturbation_metrics = attack.compute_perturbation_metrics(
            correct_inputs, adversarial
        )

        # Display batch results
        print(
            f"  Batch {batch_idx+1}: Success Rate = {batch_success_rate:.2f}%, "
            f"L2: {perturbation_metrics['l2_norm']:.4f}, "
            f"L∞: {perturbation_metrics['linf_norm']:.4f}, "
            f"SSIM: {perturbation_metrics['ssim']:.4f}"
        )

        # Save a successful example for visualization
        if success_mask.any() and len(successful_examples) < args.num_vis:
            success_idx = torch.where(success_mask)[0][0].item()

            original_img = correct_inputs[success_idx].cpu()
            adversarial_img = adversarial[success_idx].cpu()
            original_label_idx = correct_labels[success_idx].item()
            adv_pred_idx = adv_preds[success_idx].item()

            successful_examples.append(
                {
                    "original": original_img,
                    "adversarial": adversarial_img,
                    "original_label": class_names[original_label_idx],
                    "original_idx": original_label_idx,
                    "adv_prediction": class_names[adv_pred_idx],
                    "adv_idx": adv_pred_idx,
                }
            )

    # Get metrics from the attack
    attack_metrics = attack.get_metrics()

    # Combine metrics
    metrics["success_rate"] = attack_metrics["success_rate"]
    metrics["l2_norm"] = attack_metrics["l2_norm"]
    metrics["linf_norm"] = attack_metrics["linf_norm"]
    metrics["ssim"] = attack_metrics["ssim"]
    metrics["time_per_sample"] = attack_metrics["time_per_sample"]
    metrics["iterations"] = attack_metrics["iterations"]
    metrics["gradient_calls"] = attack_metrics["gradient_calls"]

    # Print metrics summary
    print(f"\nMetrics for {attack_name}:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}%")
    print(f"  L2 Norm: {metrics['l2_norm']:.4f}")
    print(f"  L∞ Norm: {metrics['linf_norm']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Time per sample: {metrics['time_per_sample']*1000:.2f} ms")
    print(f"  Average iterations: {metrics['iterations']:.2f}")
    print(f"  Average gradient calls: {metrics['gradient_calls']:.2f}")

    # Create visualizations
    os.makedirs(args.output_dir, exist_ok=True)

    for i, example in enumerate(successful_examples):
        plot_images(
            original=example["original"],
            adversarial=example["adversarial"],
            attack_name=attack_name,
            original_label=f"{example['original_label']} ({example['original_idx']})",
            adv_prediction=f"{example['adv_prediction']} ({example['adv_idx']})",
            save_path=f"{args.output_dir}/{attack_name.lower().replace(' ', '_')}_example_{i+1}.png",
        )

    return metrics


def main(args):
    """Main function."""
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Load dataset
    print("Loading ImageNet dataset...")
    dataset = get_dataset(
        dataset_name="imagenet", data_dir=args.data_dir, max_samples=args.num_samples
    )

    # Load model
    print(f"Loading {args.model_name} model...")
    model = get_model(args.model_name).to(args.device)
    model.eval()

    # Initialize all attacks
    attacks = []

    # FGSM attacks with different epsilon values
    if "fgsm" in args.attacks or "all" in args.attacks:
        attacks.append((FGSM(model, eps=4 / 255), "FGSM (ε=4/255)"))
        attacks.append((FGSM(model, eps=8 / 255), "FGSM (ε=8/255)"))

    # FFGSM attack
    if "ffgsm" in args.attacks or "all" in args.attacks:
        attacks.append(
            (FFGSM(model, eps=8 / 255, alpha=6 / 255), "FFGSM (ε=8/255, α=6/255)")
        )

    # DeepFool attack
    if "deepfool" in args.attacks or "all" in args.attacks:
        attacks.append(
            (DeepFool(model, steps=50, overshoot=0.02), "DeepFool (steps=50)")
        )

    # CW attack - use fewer steps for testing
    if "cw" in args.attacks or "all" in args.attacks:
        attacks.append(
            (CW(model, c=1.0, kappa=0, steps=100, lr=0.01), "CW (c=1.0, steps=100)")
        )

    # MIFGSM attack
    if "mifgsm" in args.attacks or "all" in args.attacks:
        attacks.append((MIFGSM(model, steps=10, alpha=0.01), "MIFGSM (steps=10)"))

    # Test each attack and collect results
    all_metrics = []

    for attack, attack_name in attacks:
        metrics = test_attack(
            attack=attack,
            model=model,
            dataset=dataset,
            attack_name=attack_name,
            args=args,
        )
        all_metrics.append(metrics)

    # Print comparison table
    print("\n\n" + "=" * 80)
    print("ATTACK COMPARISON")
    print("=" * 80)

    headers = ["Attack", "Success %", "L2 Norm", "L∞ Norm", "SSIM", "Time (ms)"]
    rows = []

    for metrics in all_metrics:
        rows.append(
            [
                metrics["name"],
                f"{metrics['success_rate']:.2f}%",
                f"{metrics['l2_norm']:.4f}",
                f"{metrics['linf_norm']:.4f}",
                f"{metrics['ssim']:.4f}",
                f"{metrics['time_per_sample']*1000:.2f}",
            ]
        )

    # Print as a formatted table
    col_widths = [
        max(len(row[i]) for row in rows + [headers]) for i in range(len(headers))
    ]

    # Print header
    header_str = " | ".join(
        f"{headers[i]:{col_widths[i]}}" for i in range(len(headers))
    )
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in rows:
        row_str = " | ".join(f"{row[i]:{col_widths[i]}}" for i in range(len(headers)))
        print(row_str)

    print(f"\nResults and visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test baseline adversarial attacks")

    # Dataset and model parameters
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Base directory for datasets"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet50",
            "vgg16",
            "efficientnet_b0",
            "mobilenet_v3_large",
        ],
        help="Model architecture to attack",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to load from dataset",
    )

    # Attack selection
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "fgsm", "ffgsm", "deepfool", "cw", "mifgsm"],
        help="Which attacks to test",
    )

    # Test parameters
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for testing"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum number of batches to test per attack",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=3,
        help="Number of adversarial examples to visualize per attack",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/test_baseline",
        help="Directory to save test results and visualizations",
    )

    args = parser.parse_args()

    main(args)
