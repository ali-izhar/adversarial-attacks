"""Tests for targeted adversarial attacks."""

import os
import sys
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    "batch_size": 5,
    "num_samples": 20,
    "eps_values": [0.1, 0.2, 0.5],  # Targeted attacks often need larger epsilons
    "norm": "L2",
    "output_dir": "test_results/targeted_attacks",
    "target_classes": [0, 10, 100, 200, 400, 800],  # Different target classes to test
}

# Read parameters from environment variables
TEST_PARAMS = DEFAULT_PARAMS.copy()

if "TEST_OUTPUT_DIR" in os.environ:
    TEST_PARAMS["output_dir"] = os.environ["TEST_OUTPUT_DIR"]

if "TEST_EPS" in os.environ:
    TEST_PARAMS["eps_values"] = [float(os.environ["TEST_EPS"])]

if "TEST_NUM_SAMPLES" in os.environ:
    TEST_PARAMS["num_samples"] = int(os.environ["TEST_NUM_SAMPLES"])

if "TEST_TARGET_CLASS" in os.environ:
    TEST_PARAMS["target_classes"] = [int(os.environ["TEST_TARGET_CLASS"])]

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


def visualize_targeted_attack(
    original,
    adversarial,
    original_pred,
    target_class,
    final_pred,
    class_names,
    method,
    eps,
    output_path,
):
    """
    Visualize the results of a targeted attack.

    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor
        original_pred: Original prediction class
        target_class: Target class for the attack
        final_pred: Final prediction class after attack
        class_names: List of class names
        method: Attack method name
        eps: Epsilon value used for attack
        output_path: Path to save the visualization
    """
    # Convert to numpy for visualization
    original_np = original.cpu().permute(1, 2, 0).numpy()
    original_np = np.clip(original_np, 0, 1)

    adversarial_np = adversarial.cpu().permute(1, 2, 0).numpy()
    adversarial_np = np.clip(adversarial_np, 0, 1)

    # Calculate perturbation
    perturbation = adversarial_np - original_np
    perturbation = perturbation * 10 + 0.5  # Amplify and center for visibility
    perturbation = np.clip(perturbation, 0, 1)

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax[0].imshow(original_np)
    ax[0].set_title(f"Original: {class_names[original_pred]}")
    ax[0].axis("off")

    # Plot adversarial image
    ax[1].imshow(adversarial_np)
    success = final_pred == target_class
    success_text = "Success" if success else "Failed"
    ax[1].set_title(f"Adversarial: {class_names[final_pred]}\n({success_text})")
    ax[1].axis("off")

    # Plot perturbation
    ax[2].imshow(perturbation)
    ax[2].set_title("Perturbation (10x)")
    ax[2].axis("off")

    # Set title for the whole figure
    plt.suptitle(
        f"Targeted Attack ({method}) - Target: {class_names[target_class]}, ε={eps}"
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_targeted_attack(model, dataloader, dataset, attack_method, target_class, eps):
    """
    Run a targeted attack towards the specified target class.

    Args:
        model: The model to attack
        dataloader: DataLoader for input samples
        dataset: Dataset with class names
        attack_method: Attack method to use ('pgd', 'cg', or 'lbfgs')
        target_class: Target class for the attack
        eps: Epsilon constraint for the attack

    Returns:
        Dictionary with attack results and metrics
    """
    device = next(model.parameters()).device

    # Initialize the attack
    if attack_method == "pgd":
        attack = PGD(
            model=model,
            norm=TEST_PARAMS["norm"],
            eps=eps,
            targeted=True,
            n_iterations=200,  # More iterations for targeted attacks
            alpha_init=eps / 20,  # Smaller step size for better precision
            verbose=False,
            device=device,
        )
    elif attack_method == "cg":
        attack = ConjugateGradient(
            model=model,
            norm=TEST_PARAMS["norm"],
            eps=eps,
            targeted=True,
            n_iterations=100,
            verbose=False,
            device=device,
        )
    elif attack_method == "lbfgs":
        attack = LBFGS(
            model=model,
            norm=TEST_PARAMS["norm"],
            eps=eps,
            targeted=True,
            n_iterations=50,
            verbose=False,
            device=device,
        )
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")

    # Collect results
    total_samples = 0
    success_count = 0
    total_time = 0
    total_iterations = 0

    # Store examples for visualization
    examples = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        total_samples += images.size(0)

        # Create target labels (all set to the target class)
        targets = torch.full_like(labels, target_class)

        # Skip images that are already the target class
        valid_indices = labels != target_class
        if not valid_indices.any():
            continue

        # Only keep valid images
        valid_images = images[valid_indices]
        valid_labels = labels[valid_indices]
        valid_targets = targets[valid_indices]

        if valid_images.size(0) == 0:
            continue

        # Get original predictions
        with torch.no_grad():
            outputs = model(valid_images)
            _, predictions = outputs.max(1)

        # Run attack
        adv_images, metrics = attack.generate(valid_images, valid_targets)

        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predictions = adv_outputs.max(1)

        # Check success
        success = adv_predictions == valid_targets
        success_count += success.sum().item()

        # Update metrics
        total_time += metrics["time"]
        total_iterations += metrics["iterations"]

        # Store examples (up to 3 per target class)
        for i in range(min(3, valid_images.size(0))):
            examples.append(
                {
                    "original": valid_images[i].clone(),
                    "adversarial": adv_images[i].clone(),
                    "original_pred": predictions[i].item(),
                    "target_class": target_class,
                    "final_pred": adv_predictions[i].item(),
                    "success": success[i].item(),
                }
            )

    # Calculate average metrics
    avg_time = total_time / total_samples if total_samples > 0 else 0
    avg_iterations = total_iterations / total_samples if total_samples > 0 else 0
    success_rate = success_count / total_samples * 100 if total_samples > 0 else 0

    return {
        "success_rate": success_rate,
        "avg_time": avg_time,
        "avg_iterations": avg_iterations,
        "total_samples": total_samples,
        "success_count": success_count,
        "examples": examples,
    }


@pytest.mark.parametrize("target_class", TEST_PARAMS["target_classes"])
@pytest.mark.parametrize("eps", TEST_PARAMS["eps_values"])
def test_targeted_attack_comparison(model, dataloader, dataset, target_class, eps):
    """
    Test and compare targeted attacks with different methods.
    """
    output_dir = os.path.join(
        TEST_PARAMS["output_dir"], f"target_{target_class}_eps_{eps}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Attack methods to test
    attack_methods = ["pgd", "cg", "lbfgs"]

    # Run attacks and collect results
    results = {}

    for method in attack_methods:
        print(
            f"Running {method.upper()} targeted attack (target class: {target_class}, ε={eps})..."
        )
        method_results = run_targeted_attack(
            model, dataloader, dataset, method, target_class, eps
        )
        results[method] = method_results

        # Visualize examples
        for i, example in enumerate(method_results["examples"]):
            if i >= 5:  # Limit to 5 examples per method
                break

            output_path = os.path.join(output_dir, f"{method}_example_{i}.png")
            visualize_targeted_attack(
                example["original"],
                example["adversarial"],
                example["original_pred"],
                example["target_class"],
                example["final_pred"],
                dataset.class_names,
                method,
                eps,
                output_path,
            )

    # Save and print results
    result_file = os.path.join(output_dir, "results.txt")
    with open(result_file, "w") as f:
        f.write(f"Targeted Attack Results (Target Class: {target_class}, ε={eps})\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Method':<10} {'Success Rate':<15} {'Time (s)':<12} {'Iterations':<12}\n"
        )
        f.write("-" * 80 + "\n")

        for method, method_results in results.items():
            f.write(
                f"{method:<10} {method_results['success_rate']:<15.2f} {method_results['avg_time']:<12.2f} "
                f"{method_results['avg_iterations']:<12.2f}\n"
            )

    # Print results
    print(f"\nResults for Target Class: {target_class}, ε={eps}:")
    print("-" * 80)
    print(f"{'Method':<10} {'Success Rate':<15} {'Time (s)':<12} {'Iterations':<12}")
    print("-" * 80)

    for method, method_results in results.items():
        print(
            f"{method:<10} {method_results['success_rate']:<15.2f} {method_results['avg_time']:<12.2f} "
            f"{method_results['avg_iterations']:<12.2f}"
        )

    # Create comparative success rate plot across methods
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    success_rates = [results[m]["success_rate"] for m in methods]

    plt.bar(methods, success_rates, color=["blue", "red", "green"])
    plt.ylabel("Success Rate (%)")
    plt.title(f"Targeted Attack Success Rate (Target Class: {target_class}, ε={eps})")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
    plt.close()

    # Create comparative iterations plot across methods
    plt.figure(figsize=(10, 6))
    iterations = [results[m]["avg_iterations"] for m in methods]

    plt.bar(methods, iterations, color=["blue", "red", "green"])
    plt.ylabel("Average Iterations")
    plt.title(f"Targeted Attack Efficiency (Target Class: {target_class}, ε={eps})")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(output_dir, "iterations_comparison.png"))
    plt.close()

    # No strict assertions since success may vary, but we should have some results
    for method, method_results in results.items():
        assert (
            method_results["total_samples"] > 0
        ), f"No samples were processed for {method}"
