"""Example script for visualizing Projected Gradient Descent (PGD) adversarial attacks."""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.attack_pgd import PGD


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


def generate_adversarial_examples(model, images, labels):
    """Generate adversarial examples using PGD attack."""
    # Define attack parameters
    attack_params = {
        "norm": "L2",
        "eps": 0.5,
        "targeted": False,
        "n_iterations": 40,
        "alpha_init": 0.1,
        "alpha_type": "diminishing",
        "rand_init": True,
        "init_std": 0.01,
        "early_stopping": True,
        "verbose": True,
    }

    # Initialize the attack
    attack = PGD(model=model, **attack_params)

    # Generate adversarial examples
    adv_images, metrics = attack.generate(images, labels)

    return adv_images, metrics, attack_params


def visualize_results(
    images, adv_images, labels, predictions, adv_predictions, attack_params, metrics
):
    """Visualize original and adversarial images side by side with a legend."""
    n = len(images)

    # Create a figure with n rows and 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5 * n))

    # Add a title for the entire figure
    fig.suptitle("PGD Attack: Original vs Adversarial Images", fontsize=16)

    # Add a legend with attack parameters in the bottom
    param_text = "Attack Parameters:\n"
    for key, value in attack_params.items():
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

        # Adversarial image
        ax2.imshow(denormalize(adv_images[i]))
        ax2.set_title(f"Adversarial\nPredicted: {CLASSES[adv_predictions[i]]}")
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the text
    plt.savefig("pgd_attack_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()


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


def main():
    # Load model
    print("Loading model...")
    model = load_model()

    # Load data
    print("Loading data...")
    images, labels = load_data()

    # Get original predictions
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # Generate adversarial examples
    print("Generating adversarial examples...")
    adv_images, metrics, attack_params = generate_adversarial_examples(
        model, images, labels
    )

    # Get adversarial predictions
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)

    # Calculate perturbation norms
    norms = compare_norms(images, adv_images)

    # Print results
    print("\nResults:")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Average L2 Norm: {norms['L2'].mean().item():.4f}")
    print(f"Average Linf Norm: {norms['Linf'].mean().item():.4f}")

    # Show which images were successfully attacked
    for i in range(len(images)):
        success = "✓" if predictions[i] != adv_predictions[i] else "✗"
        print(
            f"Image {i+1}: {CLASSES[labels[i]]} → {CLASSES[adv_predictions[i]]} (L2: {norms['L2'][i]:.4f}, Linf: {norms['Linf'][i]:.4f}) {success}"
        )

    # Visualize the results
    print("\nVisualizing results...")
    visualize_results(
        images, adv_images, labels, predictions, adv_predictions, attack_params, metrics
    )

    # Create an enhanced visualization with perturbations
    enhancement_factor = 5  # Factor to enhance perturbation visibility
    perturbation_vis = compute_perturbation_visualization(
        images, adv_images, enhancement_factor
    )

    # Create a figure showing original, perturbation, and adversarial
    fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(15, 3 * NUM_IMAGES))
    fig.suptitle(
        "PGD Attack Analysis: Original, Perturbation (Enhanced), and Adversarial",
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
        axes[i, 2].set_title(f"Adversarial: {CLASSES[adv_predictions[i]]}")
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("pgd_attack_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
