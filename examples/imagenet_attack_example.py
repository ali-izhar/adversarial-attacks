"""
Example script for visualizing adversarial attacks on ImageNet images.

This script demonstrates how to generate adversarial examples using various
attack methods on pretrained models with ImageNet images.

Usage:
    python imagenet_attack_example.py --method [cg|pgd|lbfgs] [--targeted] [--eps 0.5]

Arguments:
    --method: Attack method to use (pgd, cg, lbfgs)
    --targeted: Use targeted attack (default: untargeted)
    --eps: Perturbation budget (epsilon) for the attack
    --norm: Norm to use for constraining perturbations (L2, Linf)
    --iterations: Number of optimization iterations
    --output: Output directory for saving visualizations
    --show-norms: Display detailed norm analysis visualizations
"""

import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import argparse

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_lbfgs import LBFGS
from src.datasets.loader import get_dataset, get_dataloader
from examples.plot import (
    visualize_results,
    visualize_perturbations,
    visualize_convergence,
    visualize_norm_comparison,
    compare_norms,
)


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_IMAGES = 5  # Number of images to attack
IMAGENET_DATA_DIR = "data/imagenet"  # Path to ImageNet dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adversarial Attack Visualization on ImageNet"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pgd", "cg", "lbfgs"],
        default="pgd",
        help="Attack method to use (pgd, cg, lbfgs)",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Use targeted attack (default: untargeted)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.03,
        help="Perturbation budget (epsilon) for the attack",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["L2", "Linf"],
        default="L2",
        help="Norm to use for constraining perturbations",
    )
    parser.add_argument(
        "--iterations", type=int, default=40, help="Number of optimization iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for saving visualizations",
    )
    parser.add_argument(
        "--show-norms",
        action="store_true",
        help="Display detailed norm analysis visualizations",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load from dataset",
    )

    return parser.parse_args()


def load_model():
    """Load a pretrained model with ImageNet weights."""
    # Use a pretrained ResNet50 model
    model = models.resnet50(pretrained=True)
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    return model


def load_imagenet_data(max_samples=None):
    """Load ImageNet sample images."""
    # Define normalization for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Create transform pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Create dataset and dataloader
    try:
        dataset = get_dataset(
            dataset_name="imagenet",
            data_dir="data",
            transform=transform,
            max_samples=max_samples,
        )

        # Load just a few samples for visualization
        dataloader = get_dataloader(
            dataset=dataset, batch_size=NUM_IMAGES, shuffle=True
        )

        # Get a single batch
        images, labels = next(iter(dataloader))

        # Load class names for display
        with open(os.path.join(IMAGENET_DATA_DIR, "imagenet_classes.txt"), "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        return images.to(DEVICE), labels.to(DEVICE), class_names
    except Exception as e:
        print(f"Error loading ImageNet data: {e}")
        raise


def create_attack(model, method, args):
    """Create an attack based on the specified method."""
    common_params = {
        "model": model,
        "norm": args.norm,
        "eps": args.eps,
        "targeted": args.targeted,
        "n_iterations": args.iterations,
        "verbose": True,
    }

    if method == "pgd":
        # PGD-specific parameters
        attack_params = {
            **common_params,
            "alpha_init": 0.01,  # Smaller step size for ImageNet
            "alpha_type": "diminishing",
            "rand_init": True,
            "init_std": 0.01,
            "early_stopping": True,
        }
        attack = PGD(**attack_params)

    elif method == "cg":
        # CG-specific parameters
        attack_params = {
            **common_params,
            "fletcher_reeves": True,
            "restart_interval": 10,
            "backtracking_factor": 0.7,
            "sufficient_decrease": 1e-4,
            "early_stopping": True,
        }
        attack = ConjugateGradient(**attack_params)

    elif method == "lbfgs":
        # L-BFGS-specific parameters
        attack_params = {
            **common_params,
            "history_size": 10,
            "line_search_fn": "strong_wolfe",
            "initial_step": 0.1,
            "rand_init": False,
            "early_stopping": True,
        }
        attack = LBFGS(**attack_params)

    else:
        raise ValueError(f"Unknown attack method: {method}")

    return attack, attack_params


def prepare_targets(labels, class_names, targeted=False):
    """Prepare target labels for the attack."""
    num_classes = len(class_names)

    if targeted:
        # For targeted attacks, choose a random but consistent target different from original
        targets = (
            labels + torch.randint(1, num_classes - 1, labels.shape).to(DEVICE)
        ) % num_classes
        return targets
    else:
        # For untargeted attacks, use the true labels
        return labels


def generate_adversarial_examples(
    model, images, labels, attack, class_names, targeted=False
):
    """Generate adversarial examples using the specified attack."""
    # Prepare target labels
    targets = prepare_targets(labels, class_names, targeted)

    # Generate adversarial examples
    adv_images, metrics = attack.generate(images, targets)

    return adv_images, metrics, targets


def reverse_normalization(
    images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    Reverse the normalization for visualization.

    Args:
        images: Normalized images
        mean: Normalization mean
        std: Normalization standard deviation

    Returns:
        Denormalized images suitable for visualization
    """
    # Create clones to avoid modifying originals
    denorm_images = images.clone().detach()

    # Reverse normalization
    for i in range(3):  # For each channel
        denorm_images[:, i, :, :] = denorm_images[:, i, :, :] * std[i] + mean[i]

    # Ensure values are in valid range [0, 1]
    denorm_images = torch.clamp(denorm_images, 0, 1)

    return denorm_images


def main():
    # Parse command line arguments
    args = parse_args()

    # Set output directory
    output_dir = args.output

    # Load model
    print(f"Loading pretrained model...")
    model = load_model()

    # Load data
    print(f"Loading ImageNet data...")
    images, labels, class_names = load_imagenet_data(args.max_samples)

    # Get original predictions
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # Create the attack
    print(f"Creating {args.method.upper()} attack...")
    attack, attack_params = create_attack(model, args.method, args)

    # Generate adversarial examples
    print(f"Generating adversarial examples...")
    adv_images, metrics, targets = generate_adversarial_examples(
        model, images, labels, attack, class_names, args.targeted
    )

    # Get adversarial predictions
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)

    # Unnormalize images for visualization
    orig_images_viz = reverse_normalization(images)
    adv_images_viz = reverse_normalization(adv_images)

    # Calculate perturbation norms (on normalized images)
    norms = compare_norms(images, adv_images)

    # Print results
    print("\nResults:")
    print(f"Attack method: {args.method.upper()}")
    print(f"Attack type: {'Targeted' if args.targeted else 'Untargeted'}")
    print(f"Reported Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Average L2 Norm: {norms['L2'].mean().item():.4f}")
    print(f"Average Linf Norm: {norms['Linf'].mean().item():.4f}")

    # Calculate actual success rate from individual results
    successful_attacks = 0
    total_attacks = len(images)

    # Show which images were successfully attacked
    for i in range(total_attacks):
        if args.targeted:
            is_success = adv_predictions[i] == targets[i]
            success_marker = "✓" if is_success else "✗"
            print(
                f"Image {i+1}: {class_names[labels[i]]} → {class_names[adv_predictions[i]]} "
                f"(Target: {class_names[targets[i]]}) "
                f"(L2: {norms['L2'][i]:.4f}, Linf: {norms['Linf'][i]:.4f}) {success_marker}"
            )
            if is_success:
                successful_attacks += 1
        else:
            is_success = predictions[i] != adv_predictions[i]
            success_marker = "✓" if is_success else "✗"
            print(
                f"Image {i+1}: {class_names[labels[i]]} → {class_names[adv_predictions[i]]} "
                f"(L2: {norms['L2'][i]:.4f}, Linf: {norms['Linf'][i]:.4f}) {success_marker}"
            )
            if is_success:
                successful_attacks += 1

    # Calculate and display actual success rate
    actual_success_rate = (successful_attacks / total_attacks) * 100
    print(
        f"\nActual Success Rate: {actual_success_rate:.1f}% ({successful_attacks}/{total_attacks})"
    )

    # If there's a discrepancy, warn the user
    if abs(actual_success_rate - metrics["success_rate"]) > 1.0:
        print(f"\nWARNING: Discrepancy between reported and actual success rates!")
        print(
            f"This may indicate that the attack is counting initially successful examples incorrectly."
        )

    # Visualize the results
    print("\nVisualizing results...")
    visualize_results(
        orig_images_viz,
        adv_images_viz,
        labels,
        targets,
        predictions,
        adv_predictions,
        attack_params,
        metrics,
        args.method,
        args.targeted,
        output_dir,
        class_names,
    )

    # Visualize perturbations
    visualize_perturbations(
        orig_images_viz,
        adv_images_viz,
        labels,
        targets,
        predictions,
        adv_predictions,
        args.method,
        args.targeted,
        5,  # Enhancement factor
        output_dir,
        NUM_IMAGES,
        class_names,
    )

    # Show detailed norm comparison if requested
    if args.show_norms:
        visualize_norm_comparison(norms, args.method, args.targeted, output_dir)

    # Visualize convergence if available (mainly for L-BFGS)
    if "loss_trajectory" in metrics and metrics["loss_trajectory"]:
        visualize_convergence(metrics, args.method, args.targeted, output_dir)


if __name__ == "__main__":
    main()
