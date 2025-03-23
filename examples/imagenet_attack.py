"""
Example script for visualizing adversarial attacks on ImageNet images.

This script demonstrates how to generate adversarial examples using various
attack methods on pretrained models with ImageNet images.

Usage:
    python imagenet_attack.py --method [cg|pgd|lbfgs] [--targeted] [--eps 1e-8] [--model resnet50]

Arguments:
    --method: Attack method to use (pgd, cg, lbfgs)
    --targeted: Use targeted attack (default: untargeted)
    --eps: Perturbation budget (epsilon) for the attack
    --norm: Norm to use for constraining perturbations (L2, Linf)
    --iterations: Number of optimization iterations
    --output: Output directory for saving visualizations
    --show-norms: Display detailed norm analysis visualizations
    --model: Model architecture to use (e.g., resnet50, vgg16, efficientnet_b0)
    --num-test-images: Number of images to search through for correctly classified ones
    --num-attack-images: Number of correctly classified images to attack
"""

import os
import sys
import torch
import torchvision.transforms as transforms
import argparse
import random

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_lbfgs import LBFGS
from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model
from examples.plot import (
    visualize_results,
    visualize_perturbations,
    visualize_convergence,
    visualize_norm_comparison,
    compare_norms,
)


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        default=1e-8,  # Extremely small epsilon for truly imperceptible perturbations
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
        "--iterations", type=int, default=100, help="Number of optimization iterations"
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
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_large",
        help="Model architecture to use (e.g., resnet50, vgg16, efficientnet_b0)",
    )
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=50,
        help="Number of images to search through for correctly classified ones",
    )
    parser.add_argument(
        "--num-attack-images",
        type=int,
        default=5,
        help="Number of correctly classified images to attack",
    )

    return parser.parse_args()


def load_model(model_name):
    """Load a model using the wrappers."""
    try:
        # Use the get_model factory function to create the specified model
        model = get_model(model_name)
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        return model
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Available models:")
        print("  - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152")
        print("  - VGG: vgg11, vgg13, vgg16, vgg19")
        print("  - EfficientNet: efficientnet_b0 through efficientnet_b7")
        print("  - MobileNet: mobilenet_v3_large, mobilenet_v3_small")
        sys.exit(1)


def load_imagenet_data(model, batch_size=1, shuffle=True, max_samples=None):
    """Load ImageNet sample images."""
    # Use the model's normalization parameters
    mean = model.mean
    std = model.std

    # Create transform pipeline with model-specific normalization
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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

        # Load samples
        dataloader = get_dataloader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle
        )

        # Load class names for display
        with open(os.path.join(IMAGENET_DATA_DIR, "imagenet_classes.txt"), "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        return dataloader, class_names
    except Exception as e:
        print(f"Error loading ImageNet data: {e}")
        raise


def find_correctly_classified_images(model, dataloader, num_images, class_names):
    """Find images that are correctly classified by the model."""
    correctly_classified = []
    tested_images = 0

    print("Searching for correctly classified images...")

    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Check which images are correctly classified
            correct_mask = predictions == labels

            # For each correct image in this batch
            for i in range(len(images)):
                if correct_mask[i]:
                    correctly_classified.append(
                        (images[i].unsqueeze(0), labels[i].unsqueeze(0))
                    )
                    print(
                        f"Found correctly classified image {len(correctly_classified)}: {class_names[labels[i]]}"
                    )

                    if len(correctly_classified) >= num_images:
                        # Return the required number of images
                        return correctly_classified

        tested_images += len(images)
        print(
            f"Tested {tested_images} images, found {len(correctly_classified)} correctly classified"
        )

    print(
        f"Warning: Only found {len(correctly_classified)} correctly classified images after testing {tested_images} images"
    )
    return correctly_classified


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
            "alpha_init": args.eps / 10,  # Step size based on epsilon
            "alpha_type": "diminishing",
            "rand_init": True,
            "init_std": args.eps / 100,  # Even smaller init noise
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


def reverse_normalization(images, mean, std):
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
    print(f"Loading pretrained model: {args.model}...")
    model = load_model(args.model)

    # Load data with model-specific normalization
    print(f"Loading ImageNet data...")
    dataloader, class_names = load_imagenet_data(
        model,
        batch_size=args.num_test_images,
        shuffle=True,
        max_samples=args.max_samples,
    )

    # Find correctly classified images
    correctly_classified = find_correctly_classified_images(
        model, dataloader, args.num_attack_images, class_names
    )

    if not correctly_classified:
        print("No correctly classified images found. Exiting.")
        return

    # Select the requested number of images
    if len(correctly_classified) > args.num_attack_images:
        correctly_classified = correctly_classified[: args.num_attack_images]

    # Combine into batches
    images = torch.cat([img for img, _ in correctly_classified], dim=0)
    labels = torch.cat([lbl for _, lbl in correctly_classified], dim=0)

    # Get original predictions (to verify they're correctly classified)
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # Check that all selected images are correctly classified
    assert torch.all(
        predictions == labels
    ), "Not all selected images are correctly classified!"

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
    orig_images_viz = reverse_normalization(images, model.mean, model.std)
    adv_images_viz = reverse_normalization(adv_images, model.mean, model.std)

    # Calculate perturbation norms (on normalized images)
    norms = compare_norms(images, adv_images)

    # Print results
    print("\nResults:")
    print(f"Model: {args.model}")
    print(f"Attack method: {args.method.upper()}")
    print(f"Attack type: {'Targeted' if args.targeted else 'Untargeted'}")
    print(f"Epsilon (perturbation budget): {args.eps}")

    # Print individual results for diagnosis
    print("\nOriginal predictions vs true labels:")
    for i in range(len(images)):
        print(
            f"Image {i+1}: {class_names[labels[i]]} → predicted as {class_names[predictions[i]]} "
            + ("(misclassified)" if predictions[i] != labels[i] else "(correct)")
        )

    # Print information about the norms (in a more human-readable way)
    print(f"\nPerturbation statistics:")
    print(f"Average L2 Norm: {norms['L2'].mean().item():.6f}")
    print(f"Average L2 per dimension (RMS): {norms['L2_avg'].mean().item():.8f}")
    print(f"Average Linf Norm: {norms['Linf'].mean().item():.8f}")
    print(
        f"Average percentage of pixels changed: {norms['L0_percent'].mean().item():.4f}%"
    )

    # Calculate actual success rate
    successful_attacks = 0
    total_attacks = len(images)

    # Show which images were successfully attacked
    print("\nAttack results:")
    for i in range(total_attacks):
        if args.targeted:
            is_success = adv_predictions[i] == targets[i]
            success_marker = "✓" if is_success else "✗"
            success_type = " (SUCCESSFUL ATTACK)" if is_success else " (failed)"

            print(
                f"Image {i+1}: {class_names[labels[i]]} → {class_names[adv_predictions[i]]} "
                f"(Target: {class_names[targets[i]]}) "
                f"(L2: {norms['L2'][i]:.6f}, RMS: {norms['L2_avg'][i]:.8f}, Linf: {norms['Linf'][i]:.6f}) {success_marker}{success_type}"
            )
            if is_success:
                successful_attacks += 1
        else:
            # For untargeted attacks on correctly classified images
            is_success = (
                adv_predictions[i] != labels[i]
            )  # Success = misclassified after attack
            success_marker = "✓" if is_success else "✗"
            success_type = (
                " (SUCCESSFUL ATTACK)" if is_success else " (failed to misclassify)"
            )

            print(
                f"Image {i+1}: {class_names[labels[i]]} → {class_names[adv_predictions[i]]} "
                f"(L2: {norms['L2'][i]:.6f}, RMS: {norms['L2_avg'][i]:.8f}, Linf: {norms['Linf'][i]:.6f}) {success_marker}{success_type}"
            )

            if is_success:
                successful_attacks += 1

    # Calculate and display success rate
    success_rate = (successful_attacks / total_attacks) * 100
    print(
        f"\nAdversarial Attack Success Rate: {success_rate:.1f}% ({successful_attacks}/{total_attacks})"
    )

    # Create a metrics dictionary that includes both metrics for visualization
    vis_metrics = {
        **metrics,
        "total_success_rate": success_rate,
        "model": args.model,
    }

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
        vis_metrics,
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
        10,  # Higher enhancement factor for very small perturbations
        output_dir,
        len(images),
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
