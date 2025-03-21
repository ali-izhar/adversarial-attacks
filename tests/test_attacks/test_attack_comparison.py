"""Tests comparing different attack methods."""

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
    "num_samples": 20,  # Test with more samples for statistical significance
    "eps_values": [0.01, 0.05, 0.1, 0.2],  # Test different epsilon values
    "norm_types": ["L2", "Linf"],
    "targeted": [False, True],  # Test both targeted and untargeted attacks
    "target_class": 0,  # Target class for targeted attacks
    "output_dir": "test_results/comparison",
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
    TEST_PARAMS["target_class"] = int(os.environ["TEST_TARGET_CLASS"])

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


def visualize_comparison(
    original,
    pgd_adv,
    cg_adv,
    lbfgs_adv,
    original_pred,
    pgd_pred,
    cg_pred,
    lbfgs_pred,
    class_names,
    eps,
    norm,
    targeted,
    output_path,
):
    """
    Visualize and compare results from all three attack methods.
    """
    # Convert to numpy for visualization
    original_np = original.cpu().permute(1, 2, 0).numpy()
    original_np = np.clip(original_np, 0, 1)

    pgd_np = pgd_adv.cpu().permute(1, 2, 0).numpy()
    pgd_np = np.clip(pgd_np, 0, 1)

    cg_np = cg_adv.cpu().permute(1, 2, 0).numpy()
    cg_np = np.clip(cg_np, 0, 1)

    lbfgs_np = lbfgs_adv.cpu().permute(1, 2, 0).numpy()
    lbfgs_np = np.clip(lbfgs_np, 0, 1)

    # Calculate perturbations
    pgd_pert = pgd_np - original_np
    pgd_pert = pgd_pert * 10 + 0.5  # Amplify and center for visibility
    pgd_pert = np.clip(pgd_pert, 0, 1)

    cg_pert = cg_np - original_np
    cg_pert = cg_pert * 10 + 0.5
    cg_pert = np.clip(cg_pert, 0, 1)

    lbfgs_pert = lbfgs_np - original_np
    lbfgs_pert = lbfgs_pert * 10 + 0.5
    lbfgs_pert = np.clip(lbfgs_pert, 0, 1)

    # Create subplots
    fig, ax = plt.subplots(4, 2, figsize=(12, 16))

    # Original image
    ax[0, 0].imshow(original_np)
    ax[0, 0].set_title(f"Original: {class_names[original_pred]}")
    ax[0, 0].axis("off")

    # Empty spot
    ax[0, 1].axis("off")

    # PGD results
    ax[1, 0].imshow(pgd_np)
    ax[1, 0].set_title(f"PGD: {class_names[pgd_pred]}")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(pgd_pert)
    ax[1, 1].set_title("PGD Perturbation (10x)")
    ax[1, 1].axis("off")

    # CG results
    ax[2, 0].imshow(cg_np)
    ax[2, 0].set_title(f"CG: {class_names[cg_pred]}")
    ax[2, 0].axis("off")

    ax[2, 1].imshow(cg_pert)
    ax[2, 1].set_title("CG Perturbation (10x)")
    ax[2, 1].axis("off")

    # L-BFGS results
    ax[3, 0].imshow(lbfgs_np)
    ax[3, 0].set_title(f"L-BFGS: {class_names[lbfgs_pred]}")
    ax[3, 0].axis("off")

    ax[3, 1].imshow(lbfgs_pert)
    ax[3, 1].set_title("L-BFGS Perturbation (10x)")
    ax[3, 1].axis("off")

    # Figure title
    attack_type = "Targeted" if targeted else "Untargeted"
    plt.suptitle(f"{attack_type} Attack Comparison (ε={eps}, norm={norm})")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_attack_comparison(model, dataloader, dataset, eps, norm, targeted, output_dir):
    """
    Run all three attack methods and compare their performance.
    """
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)

    # Initialize attack methods
    pgd = PGD(
        model=model,
        norm=norm,
        eps=eps,
        targeted=targeted,
        n_iterations=100,
        alpha_init=eps / 10,
        verbose=False,
        device=device,
    )

    cg = ConjugateGradient(
        model=model,
        norm=norm,
        eps=eps,
        targeted=targeted,
        n_iterations=50,
        verbose=False,
        device=device,
    )

    lbfgs = LBFGS(
        model=model,
        norm=norm,
        eps=eps,
        targeted=targeted,
        n_iterations=30,
        verbose=False,
        device=device,
    )

    # Track metrics
    results = {
        "PGD": {"success": 0, "time": 0, "iterations": 0, "l2_norm": 0, "linf_norm": 0},
        "CG": {"success": 0, "time": 0, "iterations": 0, "l2_norm": 0, "linf_norm": 0},
        "LBFGS": {
            "success": 0,
            "time": 0,
            "iterations": 0,
            "l2_norm": 0,
            "linf_norm": 0,
        },
    }

    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        total_samples += images.size(0)

        # Create targets based on whether this is targeted or untargeted
        if targeted:
            # For targeted attacks, all targets are set to the target class
            targets = torch.full(
                (images.size(0),),
                TEST_PARAMS["target_class"],
                dtype=torch.long,
                device=device,
            )
        else:
            # For untargeted attacks, targets are the true labels
            targets = labels

        # Run PGD attack
        pgd_adv, pgd_metrics = pgd.generate(images, targets)

        # Run CG attack
        cg_adv, cg_metrics = cg.generate(images, targets)

        # Run L-BFGS attack
        lbfgs_adv, lbfgs_metrics = lbfgs.generate(images, targets)

        # Get original and adversarial predictions
        with torch.no_grad():
            orig_outputs = model(images)
            _, orig_predictions = orig_outputs.max(1)

            pgd_outputs = model(pgd_adv)
            _, pgd_predictions = pgd_outputs.max(1)

            cg_outputs = model(cg_adv)
            _, cg_predictions = cg_outputs.max(1)

            lbfgs_outputs = model(lbfgs_adv)
            _, lbfgs_predictions = lbfgs_outputs.max(1)

        # Check success
        if targeted:
            pgd_success = pgd_predictions == targets
            cg_success = cg_predictions == targets
            lbfgs_success = lbfgs_predictions == targets
        else:
            pgd_success = pgd_predictions != targets
            cg_success = cg_predictions != targets
            lbfgs_success = lbfgs_predictions != targets

        # Calculate perturbation norms
        pgd_pert = pgd_adv - images
        pgd_l2 = torch.norm(pgd_pert.view(images.size(0), -1), p=2, dim=1).mean().item()
        pgd_linf = (
            torch.norm(pgd_pert.view(images.size(0), -1), p=float("inf"), dim=1)
            .mean()
            .item()
        )

        cg_pert = cg_adv - images
        cg_l2 = torch.norm(cg_pert.view(images.size(0), -1), p=2, dim=1).mean().item()
        cg_linf = (
            torch.norm(cg_pert.view(images.size(0), -1), p=float("inf"), dim=1)
            .mean()
            .item()
        )

        lbfgs_pert = lbfgs_adv - images
        lbfgs_l2 = (
            torch.norm(lbfgs_pert.view(images.size(0), -1), p=2, dim=1).mean().item()
        )
        lbfgs_linf = (
            torch.norm(lbfgs_pert.view(images.size(0), -1), p=float("inf"), dim=1)
            .mean()
            .item()
        )

        # Update metrics
        results["PGD"]["success"] += pgd_success.sum().item()
        results["PGD"]["time"] += pgd_metrics["time"]
        results["PGD"]["iterations"] += pgd_metrics["iterations"]
        results["PGD"]["l2_norm"] += pgd_l2 * images.size(0)
        results["PGD"]["linf_norm"] += pgd_linf * images.size(0)

        results["CG"]["success"] += cg_success.sum().item()
        results["CG"]["time"] += cg_metrics["time"]
        results["CG"]["iterations"] += cg_metrics["iterations"]
        results["CG"]["l2_norm"] += cg_l2 * images.size(0)
        results["CG"]["linf_norm"] += cg_linf * images.size(0)

        results["LBFGS"]["success"] += lbfgs_success.sum().item()
        results["LBFGS"]["time"] += lbfgs_metrics["time"]
        results["LBFGS"]["iterations"] += lbfgs_metrics["iterations"]
        results["LBFGS"]["l2_norm"] += lbfgs_l2 * images.size(0)
        results["LBFGS"]["linf_norm"] += lbfgs_linf * images.size(0)

        # Visualize examples
        for i in range(images.size(0)):
            if pgd_success[i] or cg_success[i] or lbfgs_success[i]:
                img_name = f"sample_{batch_idx}_{i}_eps_{eps}_norm_{norm}_{'targeted' if targeted else 'untargeted'}.png"
                output_path = os.path.join(output_dir, img_name)

                visualize_comparison(
                    images[i],
                    pgd_adv[i],
                    cg_adv[i],
                    lbfgs_adv[i],
                    orig_predictions[i].item(),
                    pgd_predictions[i].item(),
                    cg_predictions[i].item(),
                    lbfgs_predictions[i].item(),
                    dataset.class_names,
                    eps,
                    norm,
                    targeted,
                    output_path,
                )

    # Calculate average metrics
    for method in results:
        if total_samples > 0:
            results[method]["success_rate"] = (
                results[method]["success"] / total_samples * 100
            )
            results[method]["avg_time"] = results[method]["time"] / total_samples
            results[method]["avg_iterations"] = (
                results[method]["iterations"] / total_samples
            )
            results[method]["avg_l2_norm"] = results[method]["l2_norm"] / total_samples
            results[method]["avg_linf_norm"] = (
                results[method]["linf_norm"] / total_samples
            )

    # Save results to file
    result_file = os.path.join(
        output_dir,
        f"results_eps_{eps}_norm_{norm}_{'targeted' if targeted else 'untargeted'}.txt",
    )
    with open(result_file, "w") as f:
        f.write(
            f"Attack Comparison Results (ε={eps}, norm={norm}, {'targeted' if targeted else 'untargeted'})\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Method':<10} {'Success Rate':<15} {'Time (s)':<12} {'Iterations':<12} {'L2 Norm':<10} {'Linf Norm':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for method, metrics in results.items():
            f.write(
                f"{method:<10} {metrics['success_rate']:<15.2f} {metrics['avg_time']:<12.2f} "
                f"{metrics['avg_iterations']:<12.2f} {metrics['avg_l2_norm']:<10.4f} {metrics['avg_linf_norm']:<10.4f}\n"
            )

    return results


@pytest.mark.parametrize("eps", TEST_PARAMS["eps_values"])
@pytest.mark.parametrize("norm", TEST_PARAMS["norm_types"])
@pytest.mark.parametrize("targeted", TEST_PARAMS["targeted"])
def test_attack_comparison(model, dataloader, dataset, eps, norm, targeted):
    """
    Test and compare all three attack methods.
    """
    output_dir = os.path.join(
        TEST_PARAMS["output_dir"],
        f"eps_{eps}_norm_{norm}_{'targeted' if targeted else 'untargeted'}",
    )

    results = run_attack_comparison(
        model, dataloader, dataset, eps, norm, targeted, output_dir
    )

    # Print results
    print(
        f"\nResults for ε={eps}, norm={norm}, {'targeted' if targeted else 'untargeted'}:"
    )
    print("-" * 80)
    print(
        f"{'Method':<10} {'Success Rate':<15} {'Time (s)':<12} {'Iterations':<12} {'L2 Norm':<10} {'Linf Norm':<10}"
    )
    print("-" * 80)

    for method, metrics in results.items():
        print(
            f"{method:<10} {metrics['success_rate']:<15.2f} {metrics['avg_time']:<12.2f} "
            f"{metrics['avg_iterations']:<12.2f} {metrics['avg_l2_norm']:<10.4f} {metrics['avg_linf_norm']:<10.4f}"
        )

    # Assertions to ensure attacks are working properly
    # These are minimal checks - we mainly want to collect and visualize data
    for method, metrics in results.items():
        # Check that the attack produces perturbations within epsilon bound
        if norm == "L2":
            assert (
                metrics["avg_l2_norm"] <= eps + 1e-5
            ), f"{method} exceeded L2 constraint"
        else:  # Linf
            assert (
                metrics["avg_linf_norm"] <= eps + 1e-5
            ), f"{method} exceeded Linf constraint"
