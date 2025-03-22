"""
Example script for visualizing different adversarial attacks.

This script demonstrates how to generate adversarial examples using various
attack methods and visualizes the original and perturbed images side by side.

Usage:
    python attack_example.py --method [cg|pgd|lbfgs] [--targeted] [--eps 0.5]

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
import torchvision
import torchvision.transforms as transforms
import argparse

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_lbfgs import LBFGS
from examples.plot import (
    visualize_results,
    visualize_perturbations,
    visualize_convergence,
    visualize_norm_comparison,
    compare_norms,
    CLASSES,
)


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/resnet18_cifar10.pth"  # Path to a pre-trained model
DATA_PATH = "data"  # Path to dataset
NUM_IMAGES = 5  # Number of images to attack


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adversarial Attack Visualization")
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
        default=0.5,
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

    return parser.parse_args()


def load_model():
    """Load a pre-trained model."""
    # Use a pre-trained ResNet18 model
    model = torchvision.models.resnet18(pretrained=False)
    # Modify for CIFAR-10 (10 classes)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # Load trained weights if available
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(
            f"Warning: Pre-trained model not found at {MODEL_PATH}. Using untrained model."
        )

    model = model.to(DEVICE)
    model.eval()
    return model


def load_data():
    """Load a few sample images from CIFAR-10 for testing."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Don't normalize as it complicates visualization
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform
    )

    # Take a few samples
    loader = torch.utils.data.DataLoader(dataset, batch_size=NUM_IMAGES, shuffle=True)

    # Get a single batch
    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


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
            "alpha_init": 0.1,
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


def prepare_targets(labels, targeted=False):
    """Prepare target labels for the attack."""
    if targeted:
        # For targeted attacks, choose targets different from original labels
        targets = (labels + 1) % 10
        return targets
    else:
        # For untargeted attacks, use the true labels
        return labels


def generate_adversarial_examples(model, images, labels, attack, targeted=False):
    """Generate adversarial examples using the specified attack."""
    # Prepare target labels
    targets = prepare_targets(labels, targeted)

    # Generate adversarial examples
    adv_images, metrics = attack.generate(images, targets)

    return adv_images, metrics, targets


def main():
    # Parse command line arguments
    args = parse_args()

    # Set output directory
    output_dir = args.output

    # Load model
    print(f"Loading model...")
    model = load_model()

    # Load data
    print(f"Loading data...")
    images, labels = load_data()

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
        model, images, labels, attack, args.targeted
    )

    # Get adversarial predictions
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)

    # Calculate perturbation norms
    norms = compare_norms(images, adv_images)

    # Print results
    print("\nResults:")
    print(f"Attack method: {args.method.upper()}")
    print(f"Attack type: {'Targeted' if args.targeted else 'Untargeted'}")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Average L2 Norm: {norms['L2'].mean().item():.4f}")
    print(f"Average Linf Norm: {norms['Linf'].mean().item():.4f}")

    # Show which images were successfully attacked
    for i in range(len(images)):
        if args.targeted:
            success = "✓" if adv_predictions[i] == targets[i] else "✗"
            print(
                f"Image {i+1}: {CLASSES[labels[i]]} → {CLASSES[adv_predictions[i]]} (Target: {CLASSES[targets[i]]}) (L2: {norms['L2'][i]:.4f}, Linf: {norms['Linf'][i]:.4f}) {success}"
            )
        else:
            success = "✓" if predictions[i] != adv_predictions[i] else "✗"
            print(
                f"Image {i+1}: {CLASSES[labels[i]]} → {CLASSES[adv_predictions[i]]} (L2: {norms['L2'][i]:.4f}, Linf: {norms['Linf'][i]:.4f}) {success}"
            )

    # Visualize the results
    print("\nVisualizing results...")
    visualize_results(
        images,
        adv_images,
        labels,
        targets,
        predictions,
        adv_predictions,
        attack_params,
        metrics,
        args.method,
        args.targeted,
        output_dir,
    )

    # Visualize perturbations
    visualize_perturbations(
        images,
        adv_images,
        labels,
        targets,
        predictions,
        adv_predictions,
        args.method,
        args.targeted,
        5,
        output_dir,
        NUM_IMAGES,
    )

    # Show detailed norm comparison if requested
    if args.show_norms:
        visualize_norm_comparison(norms, args.method, args.targeted, output_dir)

    # Visualize convergence if available (mainly for L-BFGS)
    if "loss_trajectory" in metrics and metrics["loss_trajectory"]:
        visualize_convergence(metrics, args.method, args.targeted, output_dir)


if __name__ == "__main__":
    main()
