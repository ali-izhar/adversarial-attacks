"""Tests for analyzing patterns in adversarial perturbations."""

import os
import sys
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision.models as models
import torchvision.transforms as transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks import PGD, ConjugateGradient, LBFGS
from src.datasets import get_dataset, get_dataloader


# Default test parameters
DEFAULT_PARAMS = {
    "data_dir": "data",
    "dataset": "imagenet",
    "batch_size": 10,
    "num_samples": 50,  # Test with more samples for better pattern analysis
    "eps": 0.1,  # Fixed epsilon for pattern analysis
    "norm": "L2",  # Fixed norm for pattern analysis
    "output_dir": "test_results/perturbation_analysis",
}

# Read parameters from environment variables
TEST_PARAMS = DEFAULT_PARAMS.copy()

if "TEST_OUTPUT_DIR" in os.environ:
    TEST_PARAMS["output_dir"] = os.environ["TEST_OUTPUT_DIR"]

if "TEST_EPS" in os.environ:
    TEST_PARAMS["eps"] = float(os.environ["TEST_EPS"])

if "TEST_NUM_SAMPLES" in os.environ:
    TEST_PARAMS["num_samples"] = int(os.environ["TEST_NUM_SAMPLES"])

if "TEST_DEVICE" in os.environ:
    # This will be used when creating the model
    device_override = os.environ["TEST_DEVICE"]
else:
    device_override = None


@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dataset():
    """Load the dataset once for all tests."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    dataset = get_dataset(
        TEST_PARAMS["dataset"],
        data_dir=TEST_PARAMS["data_dir"],
        transform=preprocess,
        max_samples=TEST_PARAMS["num_samples"],
    )
    return dataset


@pytest.fixture(scope="module")
def dataloader(dataset):
    """Create dataloader for testing."""
    return get_dataloader(dataset, batch_size=TEST_PARAMS["batch_size"], shuffle=False)


def generate_perturbations(model, dataloader, attack_method):
    """
    Generate perturbations using the specified attack method.

    Args:
        model: The model to attack
        dataloader: DataLoader for input samples
        attack_method: String indicating which attack to use ('pgd', 'cg', or 'lbfgs')

    Returns:
        List of perturbations, original images, predictions, and success indicators
    """
    device = next(model.parameters()).device
    eps = TEST_PARAMS["eps"]
    norm = TEST_PARAMS["norm"]

    # Initialize the appropriate attack
    if attack_method == "pgd":
        attack = PGD(
            model=model,
            norm=norm,
            eps=eps,
            targeted=False,
            n_iterations=100,
            alpha_init=eps / 10,
            verbose=False,
            device=device,
        )
    elif attack_method == "cg":
        attack = ConjugateGradient(
            model=model,
            norm=norm,
            eps=eps,
            targeted=False,
            n_iterations=50,
            verbose=False,
            device=device,
        )
    elif attack_method == "lbfgs":
        attack = LBFGS(
            model=model,
            norm=norm,
            eps=eps,
            targeted=False,
            n_iterations=30,
            verbose=False,
            device=device,
        )
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")

    # Collect perturbations
    perturbations = []
    original_images = []
    original_preds = []
    adv_preds = []
    successes = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Get original predictions
        with torch.no_grad():
            outputs = model(images)
            _, predictions = outputs.max(1)

        # Skip images that are already misclassified
        valid_indices = predictions == labels
        if not valid_indices.any():
            continue

        # Only keep correctly classified images
        valid_images = images[valid_indices]
        valid_labels = labels[valid_indices]
        valid_preds = predictions[valid_indices]

        if valid_images.size(0) == 0:
            continue

        # Generate adversarial examples
        adv_images, _ = attack.generate(valid_images, valid_labels)

        # Calculate perturbations
        pert = adv_images - valid_images

        # Check success
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predictions = adv_outputs.max(1)

        success = adv_predictions != valid_labels

        # Store data
        for i in range(valid_images.size(0)):
            perturbations.append(pert[i].cpu().numpy())
            original_images.append(valid_images[i].cpu().numpy())
            original_preds.append(valid_preds[i].item())
            adv_preds.append(adv_predictions[i].item())
            successes.append(success[i].item())

    return perturbations, original_images, original_preds, adv_preds, successes


def visualize_perturbation_heatmap(perturbations, successes, attack_method, output_dir):
    """
    Create heatmap of average perturbation magnitude across image channels.

    Args:
        perturbations: List of perturbation tensors
        successes: List of success indicators
        attack_method: Name of the attack method
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only analyze successful perturbations
    successful_perts = [p for p, s in zip(perturbations, successes) if s]

    if not successful_perts:
        print(f"No successful perturbations for {attack_method}. Skipping heatmap.")
        return

    # Convert to numpy arrays
    successful_perts = np.array(successful_perts)

    # Calculate average perturbation
    avg_pert = np.mean(successful_perts, axis=0)

    # Calculate magnitude across channels
    pert_magnitude = np.sqrt(np.sum(avg_pert**2, axis=0))

    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(pert_magnitude, cmap="hot")
    plt.colorbar(label="Perturbation Magnitude")
    plt.title(f"Average Perturbation Heatmap - {attack_method.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attack_method}_heatmap.png"))
    plt.close()

    # Also create channel-wise heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_names = ["Red", "Green", "Blue"]
    for i in range(3):
        im = axes[i].imshow(np.abs(avg_pert[i]), cmap="viridis")
        axes[i].set_title(f"{channel_names[i]} Channel")
        plt.colorbar(im, ax=axes[i])

    plt.suptitle(f"Channel-wise Perturbation Analysis - {attack_method.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attack_method}_channel_heatmaps.png"))
    plt.close()


def analyze_perturbation_distribution(
    perturbations, successes, attack_method, output_dir
):
    """
    Analyze the statistical distribution of perturbation values.

    Args:
        perturbations: List of perturbation tensors
        successes: List of success indicators
        attack_method: Name of the attack method
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only analyze successful perturbations
    successful_perts = [p for p, s in zip(perturbations, successes) if s]

    if not successful_perts:
        print(
            f"No successful perturbations for {attack_method}. Skipping distribution analysis."
        )
        return

    # Flatten perturbations to 1D arrays
    flat_perts = [p.flatten() for p in successful_perts]

    # Create histograms
    plt.figure(figsize=(12, 6))
    for i, pert in enumerate(
        flat_perts[: min(10, len(flat_perts))]
    ):  # Limit to 10 examples
        plt.hist(pert, alpha=0.3, bins=50, label=f"Example {i+1}")

    plt.xlabel("Perturbation Value")
    plt.ylabel("Frequency")
    plt.title(f"Perturbation Distribution - {attack_method.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attack_method}_distribution.png"))
    plt.close()

    # Calculate and plot statistics
    all_values = np.concatenate(flat_perts)
    mean_val = np.mean(all_values)
    median_val = np.median(all_values)
    std_val = np.std(all_values)

    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=100, alpha=0.7)
    plt.axvline(mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.4f}")
    plt.axvline(
        median_val, color="g", linestyle="--", label=f"Median: {median_val:.4f}"
    )
    plt.xlabel("Perturbation Value")
    plt.ylabel("Frequency")
    plt.title(
        f"Combined Perturbation Distribution - {attack_method.upper()} (σ={std_val:.4f})"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attack_method}_combined_distribution.png"))
    plt.close()


def visualize_perturbation_similarity(
    perturbations, original_preds, attack_methods, output_dir
):
    """
    Visualize similarity between perturbations using dimensionality reduction.

    Args:
        perturbations: Dictionary mapping attack methods to lists of perturbations
        original_preds: Dictionary mapping attack methods to lists of original predictions
        attack_methods: List of attack method names
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all perturbations and their metadata
    all_perts = []
    all_labels = []  # Original class
    all_methods = []  # Attack method

    for method in attack_methods:
        method_perts = perturbations[method]
        method_preds = original_preds[method]

        for pert, pred in zip(method_perts, method_preds):
            all_perts.append(pert.flatten())
            all_labels.append(pred)
            all_methods.append(method)

    if not all_perts:
        print("No perturbations to analyze. Skipping similarity analysis.")
        return

    # Convert to numpy arrays
    all_perts = np.array(all_perts)

    # PCA for dimensionality reduction
    try:
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(all_perts)

        # t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(pca_result)

        # Plot by attack method
        plt.figure(figsize=(10, 8))
        method_colors = {"pgd": "blue", "cg": "red", "lbfgs": "green"}

        for method in attack_methods:
            indices = [i for i, m in enumerate(all_methods) if m == method]
            if indices:
                plt.scatter(
                    tsne_result[indices, 0],
                    tsne_result[indices, 1],
                    alpha=0.7,
                    color=method_colors[method],
                    label=method.upper(),
                )

        plt.title("t-SNE Projection of Perturbations by Attack Method")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perturbation_tsne_by_method.png"))
        plt.close()

        # Plot by original class (use only first 10 classes for clarity)
        plt.figure(figsize=(12, 10))
        class_set = sorted(list(set(all_labels)))[:10]

        for cls in class_set:
            indices = [i for i, c in enumerate(all_labels) if c == cls]
            if indices:
                plt.scatter(
                    tsne_result[indices, 0],
                    tsne_result[indices, 1],
                    alpha=0.7,
                    label=f"Class {cls}",
                )

        plt.title("t-SNE Projection of Perturbations by Original Class")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "perturbation_tsne_by_class.png"))
        plt.close()

    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")


def test_perturbation_analysis(model, dataloader):
    """
    Test to analyze patterns in adversarial perturbations.
    """
    output_dir = TEST_PARAMS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Attack methods to analyze
    attack_methods = ["pgd", "cg", "lbfgs"]

    # Generate perturbations for each method
    all_perturbations = {}
    all_images = {}
    all_orig_preds = {}
    all_adv_preds = {}
    all_successes = {}

    for method in attack_methods:
        print(f"Generating perturbations using {method.upper()}...")
        pert, orig, orig_preds, adv_preds, success = generate_perturbations(
            model, dataloader, method
        )

        all_perturbations[method] = pert
        all_images[method] = orig
        all_orig_preds[method] = orig_preds
        all_adv_preds[method] = adv_preds
        all_successes[method] = success

        # Basic statistics
        success_rate = sum(success) / len(success) * 100 if success else 0
        print(
            f"{method.upper()} success rate: {success_rate:.2f}% ({sum(success)}/{len(success)})"
        )

    # Analyze each method's perturbations
    for method in attack_methods:
        print(f"Analyzing {method.upper()} perturbations...")

        # Generate heatmaps
        visualize_perturbation_heatmap(
            all_perturbations[method], all_successes[method], method, output_dir
        )

        # Analyze distribution
        analyze_perturbation_distribution(
            all_perturbations[method], all_successes[method], method, output_dir
        )

    # Compare perturbations across methods
    print("Comparing perturbations across methods...")
    visualize_perturbation_similarity(
        all_perturbations, all_orig_preds, attack_methods, output_dir
    )

    # Save examples with their perturbations
    for method in attack_methods:
        # Only save up to 5 successful examples for each method
        successful_indices = [i for i, s in enumerate(all_successes[method]) if s][:5]

        if not successful_indices:
            continue

        fig, axes = plt.subplots(
            len(successful_indices), 3, figsize=(15, 5 * len(successful_indices))
        )
        if len(successful_indices) == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(successful_indices):
            # Original image
            orig_img = np.transpose(all_images[method][idx], (1, 2, 0))
            orig_img = np.clip(orig_img, 0, 1)
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original (Class {all_orig_preds[method][idx]})")
            axes[i, 0].axis("off")

            # Perturbation (enhanced for visibility)
            pert = np.transpose(all_perturbations[method][idx], (1, 2, 0))
            pert_vis = pert * 10 + 0.5
            pert_vis = np.clip(pert_vis, 0, 1)
            axes[i, 1].imshow(pert_vis)
            axes[i, 1].set_title("Perturbation (10x)")
            axes[i, 1].axis("off")

            # Adversarial image
            adv_img = orig_img + np.transpose(all_perturbations[method][idx], (1, 2, 0))
            adv_img = np.clip(adv_img, 0, 1)
            axes[i, 2].imshow(adv_img)
            axes[i, 2].set_title(f"Adversarial (Class {all_adv_preds[method][idx]})")
            axes[i, 2].axis("off")

        plt.suptitle(
            f"{method.upper()} Attack Examples (ε={TEST_PARAMS['eps']}, {TEST_PARAMS['norm']} norm)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_examples.png"))
        plt.close()

    # Assertions to confirm analysis worked
    for method in attack_methods:
        assert (
            len(all_perturbations[method]) > 0
        ), f"No perturbations generated for {method}"
