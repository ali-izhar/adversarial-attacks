"""
Example script for visualizing different adversarial attacks.

This script demonstrates how to generate adversarial examples using various
attack methods and visualizes the original and perturbed images side by side.

Usage:
    python attack_example.py --method [cg|pgd|lbfgs] [--targeted] [--eps 0.5]
"""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_lbfgs import LBFGS


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/resnet18_cifar10.pth"  # Path to a pre-trained model
DATA_PATH = "data"  # Path to dataset
NUM_IMAGES = 5  # Number of images to attack
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)  # CIFAR-10 classes


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


def compute_perturbation_visualization(images, adv_images, factor=5):
    """
    Compute and visualize the perturbation with enhanced contrast.

    Args:
        images: Original images
        adv_images: Adversarial images
        factor: Enhancement factor to make perturbations more visible

    Returns:
        The enhanced perturbation visualization
    """
    perturbation = adv_images - images

    # Enhance the perturbation to make it more visible
    enhanced_perturbation = perturbation * factor

    # Ensure the values are in a valid range
    enhanced_perturbation = torch.clamp(enhanced_perturbation + 0.5, 0, 1)

    return enhanced_perturbation


def compare_norms(images, adv_images):
    """
    Compare different norms of the perturbation.

    Args:
        images: Original images
        adv_images: Adversarial images

    Returns:
        Dictionary with L0, L1, L2, and Linf norms
    """
    perturbation = adv_images - images
    batch_size = perturbation.shape[0]

    # Reshape to (batch_size, -1) for norm calculations
    perturbation_flat = perturbation.reshape(batch_size, -1)

    # Calculate L0 norm (number of changed pixels)
    l0_norm = (perturbation_flat.abs() > 1e-5).float().sum(dim=1)

    # Calculate L1 norm (sum of absolute values)
    l1_norm = perturbation_flat.abs().sum(dim=1)

    # Calculate L2 norm (Euclidean distance)
    l2_norm = torch.norm(perturbation_flat, dim=1, p=2)

    # Calculate Linf norm (maximum absolute value)
    linf_norm = perturbation_flat.abs().max(dim=1)[0]

    return {"L0": l0_norm, "L1": l1_norm, "L2": l2_norm, "Linf": linf_norm}


def visualize_results(
    images,
    adv_images,
    labels,
    targets,
    predictions,
    adv_predictions,
    attack_params,
    metrics,
    method,
    targeted,
    output_dir=None,
):
    """Visualize original and adversarial images side by side with a legend."""
    n = len(images)

    # Create a figure with n rows and 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5 * n))

    # Format the attack name for the title
    method_name = method.upper()
    if method == "cg":
        method_name = "Conjugate Gradient"
    elif method == "lbfgs":
        method_name = "L-BFGS"

    # Add a title for the entire figure
    attack_type = "Targeted" if targeted else "Untargeted"
    fig.suptitle(
        f"{method_name} {attack_type} Attack: Original vs Adversarial Images",
        fontsize=16,
    )

    # Add a legend with attack parameters in the bottom
    param_text = "Attack Parameters:\n"
    for key, value in attack_params.items():
        if key not in ["model", "verbose"]:
            param_text += f"- {key}: {value}\n"

    # Add metrics
    param_text += f"\nResults:\n- Success Rate: {metrics['success_rate']:.1f}%\n"
    param_text += f"- Avg. Iterations: {metrics['iterations']:.1f}\n"
    param_text += f"- Time: {metrics['time']:.2f}s"

    # Add the text in a separate axes
    fig.text(0.1, 0.01, param_text, fontsize=10, verticalalignment="bottom")

    # Function to denormalize images if needed
    def denormalize(img):
        return img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    # Plot each image
    for i in range(n):
        # Original image
        if n > 1:
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
        else:
            ax1 = axes[0]
            ax2 = axes[1]

        ax1.imshow(denormalize(images[i]))
        ax1.set_title(
            f"Original: {CLASSES[labels[i]]}\nPredicted: {CLASSES[predictions[i]]}"
        )
        ax1.axis("off")

        # Adversarial image - different title based on targeted/untargeted
        ax2.imshow(denormalize(adv_images[i]))
        if targeted:
            ax2.set_title(
                f"Adversarial (Target: {CLASSES[targets[i]]})\nPredicted: {CLASSES[adv_predictions[i]]}"
            )
        else:
            ax2.set_title(f"Adversarial\nPredicted: {CLASSES[adv_predictions[i]]}")
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the text

    # Save the figure if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{method}_{attack_type.lower()}_visualization.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")

    plt.show()


def visualize_perturbations(
    images,
    adv_images,
    labels,
    targets,
    adv_predictions,
    method,
    targeted,
    enhancement_factor=5,
    output_dir=None,
):
    """Create a visualization showing original, perturbation, and adversarial images."""
    # Create an enhanced visualization with perturbations
    perturbation_vis = compute_perturbation_visualization(
        images, adv_images, enhancement_factor
    )

    # Create a figure showing original, perturbation, and adversarial
    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(15, 3 * NUM_IMAGES))

    # Format the attack name for the title
    method_name = method.upper()
    if method == "cg":
        method_name = "Conjugate Gradient"
    elif method == "lbfgs":
        method_name = "L-BFGS"

    attack_type = "Targeted" if targeted else "Untargeted"
    fig.suptitle(
        f"{method_name} {attack_type} Attack Analysis: Original, Perturbation (Enhanced), and Adversarial",
        fontsize=16,
    )

    for i in range(NUM_IMAGES):
        # Plot original
        axes[i, 0].imshow(images[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[i, 0].set_title(f"Original: {CLASSES[labels[i]]}")
        axes[i, 0].axis("off")

        # Plot enhanced perturbation
        axes[i, 1].imshow(perturbation_vis[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[i, 1].set_title(f"Perturbation (Enhanced {enhancement_factor}x)")
        axes[i, 1].axis("off")

        # Plot adversarial
        axes[i, 2].imshow(adv_images[i].permute(1, 2, 0).detach().cpu().numpy())
        if targeted:
            axes[i, 2].set_title(f"Adversarial (Target: {CLASSES[targets[i]]})")
        else:
            axes[i, 2].set_title(f"Adversarial: {CLASSES[adv_predictions[i]]}")
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{method}_{attack_type.lower()}_analysis.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")

    plt.show()


def visualize_convergence(metrics, method, targeted, output_dir=None):
    """
    Visualize the optimization convergence based on loss trajectory.

    Args:
        metrics: Dictionary containing optimization metrics
    """
    if "loss_trajectory" not in metrics or not metrics["loss_trajectory"]:
        print("Loss trajectory not available in metrics")
        return

    loss_values = metrics["loss_trajectory"]
    iterations = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, loss_values, "b-", linewidth=2)

    # Format the attack name for the title
    method_name = method.upper()
    if method == "cg":
        method_name = "Conjugate Gradient"
    elif method == "lbfgs":
        method_name = "L-BFGS"

    attack_type = "Targeted" if targeted else "Untargeted"
    plt.title(
        f"{method_name} {attack_type} Attack Optimization Convergence", fontsize=14
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.yscale("log")  # Use log scale to better visualize the decay
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the figure if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{method}_{attack_type.lower()}_convergence.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)

    plt.show()


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
        adv_predictions,
        args.method,
        args.targeted,
        5,
        output_dir,
    )

    # Visualize convergence if available (mainly for L-BFGS)
    if "loss_trajectory" in metrics and metrics["loss_trajectory"]:
        visualize_convergence(metrics, args.method, args.targeted, output_dir)


if __name__ == "__main__":
    main()
