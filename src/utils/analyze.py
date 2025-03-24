"""
Adversarial attack analysis utility.

This script provides functionality to analyze adversarial attacks on image classification
models. It works with the ImageNet dataset and supports various attack methods.

Usage:
    python -m src.utils.analyze --model resnet50 --attack cw --confidence 0.5

The script performs the following steps:
1. Loads a specified pretrained model
2. Finds a set of correctly classified images from the ImageNet dataset
3. Applies the specified adversarial attack method
4. Analyzes and reports the attack success rate and perturbation metrics
"""

import os
import sys
import torch
import time
import argparse
import numpy as np
from typing import Tuple, Dict, Any, List

from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model

from src.attacks.attack_pgd import PGD
from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_lbfgs import LBFGS
from src.attacks.attack_fgsm import FGSM
from src.attacks.attack_deepfool import DeepFool
from src.attacks.attack_cw import CW


# Configure device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str) -> torch.nn.Module:
    """
    Load a pretrained model.

    Args:
        model_name: Name of the model to load (e.g., 'resnet50', 'vgg16')

    Returns:
        The loaded model
    """
    try:
        model = get_model(model_name)
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded {model_name} model")
        return model
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Available models:")
        print("  - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152")
        print("  - VGG: vgg11, vgg13, vgg16, vgg19")
        print("  - EfficientNet: efficientnet_b0 through efficientnet_b7")
        print("  - MobileNet: mobilenet_v3_large, mobilenet_v3_small")
        sys.exit(1)


def load_data(
    model: torch.nn.Module, max_samples: int = None
) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    """
    Load the ImageNet dataset with proper normalization for the given model.

    Args:
        model: The model for which to normalize the data
        max_samples: Maximum number of samples to load from the dataset

    Returns:
        A tuple of (dataloader, class_names)
    """
    import torchvision.transforms as transforms

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

    # Load dataset
    try:
        dataset = get_dataset(
            dataset_name="imagenet",
            data_dir="data",
            transform=transform,
            max_samples=max_samples,
        )

        # Create dataloader
        dataloader = get_dataloader(
            dataset=dataset, batch_size=1, shuffle=True  # Process one image at a time
        )

        # Load class names
        with open(os.path.join("data/imagenet", "imagenet_classes.txt"), "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        return dataloader, class_names

    except Exception as e:
        print(f"Error loading ImageNet data: {e}")
        sys.exit(1)


def find_correctly_classified_images(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_images: int,
    class_names: List[str],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Find images that are correctly classified by the model.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the images
        num_images: Number of correctly classified images to find
        class_names: List of class names for display

    Returns:
        List of tuples (image, label) for correctly classified images
    """
    correctly_classified = []
    tested_images = 0

    print("Searching for correctly classified images...")

    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Check if image is correctly classified
            if predictions[0] == labels[0]:
                correctly_classified.append((images, labels))
                print(
                    f"Found correctly classified image {len(correctly_classified)}: {class_names[labels[0]]}"
                )

                if len(correctly_classified) >= num_images:
                    return correctly_classified

        tested_images += 1
        if tested_images % 10 == 0:
            print(
                f"Tested {tested_images} images, found {len(correctly_classified)} correctly classified"
            )

    print(
        f"Warning: Only found {len(correctly_classified)} correctly classified images after testing {tested_images} images"
    )
    return correctly_classified


def create_attack(
    model: torch.nn.Module, attack_name: str, args: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create an attack based on the specified method.

    Args:
        model: The model to attack
        attack_name: Name of the attack method (e.g., 'pgd', 'cw')
        args: Dictionary of attack parameters

    Returns:
        Tuple of (attack_instance, attack_params)
    """
    # Common parameters for all attacks
    common_params = {
        "model": model,
        "norm": args.get("norm", "L2"),
        "eps": args.get("eps", 1.0),
        "targeted": args.get("targeted", False),
        "loss_fn": args.get("loss_fn", "cross_entropy"),
        "verbose": True,
        "device": DEVICE,
    }

    # Create attack based on name
    if attack_name == "pgd":
        attack_params = {
            **common_params,
            "alpha_init": args.get("alpha_init", common_params["eps"] / 10),
            "alpha_type": args.get("alpha_type", "constant"),
            "rand_init": True,
            "init_std": args.get("eps", 1.0) / 20,
            "early_stopping": True,
            "n_iterations": args.get("iterations", 100),
        }
        attack = PGD(**attack_params)

    elif attack_name == "cg":
        attack_params = {
            **common_params,
            "fletcher_reeves": True,
            "restart_interval": 10,
            "backtracking_factor": 0.7,
            "sufficient_decrease": 1e-4,
            "early_stopping": True,
            "n_iterations": args.get("iterations", 100),
        }
        attack = ConjugateGradient(**attack_params)

    elif attack_name == "lbfgs":
        attack_params = {
            **common_params,
            "history_size": 10,
            "line_search_fn": "strong_wolfe",
            "initial_step": 0.1,
            "rand_init": False,
            "early_stopping": True,
            "n_iterations": args.get("iterations", 100),
        }
        attack = LBFGS(**attack_params)

    elif attack_name == "fgsm":
        attack_params = {
            **common_params,
            "clip_min": 0.0,
            "clip_max": 1.0,
        }
        attack = FGSM(**attack_params)

    elif attack_name == "deepfool":
        attack_params = {
            "model": model,
            "norm": args.get("norm", "L2"),
            "num_classes": 1000,  # ImageNet has 1000 classes
            "overshoot": 0.02,
            "max_iter": 50,
            "verbose": True,
            "device": DEVICE,
        }
        attack = DeepFool(**attack_params)

    elif attack_name == "cw":
        attack_params = {
            "model": model,
            "confidence": args.get("confidence", 0.0),
            "c_init": args.get("c_init", 0.1),
            "max_iter": args.get("iterations", 500),
            "binary_search_steps": args.get("binary_search_steps", 5),
            "learning_rate": args.get("learning_rate", 0.01),
            "targeted": args.get("targeted", False),
            "abort_early": True,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "verbose": True,
            "device": DEVICE,
        }
        attack = CW(**attack_params)

    else:
        raise ValueError(f"Unknown attack method: {attack_name}")

    return attack, attack_params


def prepare_targets(labels: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    """
    Prepare target labels for the attack.

    Args:
        labels: Original labels
        targeted: Whether the attack is targeted

    Returns:
        Target labels for the attack
    """
    if targeted:
        # For targeted attacks, choose a random target different from original
        num_classes = 1000  # ImageNet has 1000 classes
        targets = (
            labels + torch.randint(1, num_classes - 1, labels.shape).to(DEVICE)
        ) % num_classes
        return targets
    else:
        # For untargeted attacks, use the true labels
        return labels


def generate_adversarial_examples(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack: Any,
    targeted: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """
    Generate adversarial examples using the specified attack.

    Args:
        model: The model to attack
        images: Input images
        labels: True labels
        attack: Attack instance
        targeted: Whether the attack is targeted

    Returns:
        Tuple of (adversarial_examples, metrics, targets)
    """
    # Prepare target labels
    targets = prepare_targets(labels, targeted)

    # Generate adversarial examples
    start_time = time.time()
    adv_images, metrics = attack.generate(images, targets)
    attack_time = time.time() - start_time

    # Add timing information to metrics
    metrics["attack_time"] = attack_time
    metrics["time_per_image"] = attack_time / len(images)

    return adv_images, metrics, targets


def calculate_norms(
    original_images: torch.Tensor, adversarial_images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate various perturbation norms.

    Args:
        original_images: Clean images
        adversarial_images: Adversarial images

    Returns:
        Dictionary of different norm metrics
    """
    # Calculate perturbation
    perturbation = adversarial_images - original_images

    # Reshape for per-image calculations
    batch_size = original_images.shape[0]
    flat_perturbation = perturbation.view(batch_size, -1)

    # Calculate different norms
    l0_norm = (flat_perturbation != 0).float().sum(dim=1)
    l0_percent = 100 * l0_norm / flat_perturbation.shape[1]
    l1_norm = flat_perturbation.abs().sum(dim=1)
    l2_norm = torch.norm(flat_perturbation, p=2, dim=1)
    l2_avg = l2_norm / np.sqrt(flat_perturbation.shape[1])  # Root mean square
    linf_norm = flat_perturbation.abs().max(dim=1)[0]

    return {
        "L0": l0_norm,
        "L0_percent": l0_percent,
        "L1": l1_norm,
        "L2": l2_norm,
        "L2_avg": l2_avg,
        "Linf": linf_norm,
    }


def analyze_attack_results(
    model: torch.nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    metrics: Dict[str, Any],
    attack_params: Dict[str, Any],
    class_names: List[str],
    targeted: bool = False,
) -> None:
    """
    Analyze and print attack results.

    Args:
        model: The model that was attacked
        original_images: Clean images
        adversarial_images: Adversarial images
        labels: True labels
        targets: Target labels
        metrics: Attack metrics
        attack_params: Attack parameters
        class_names: Class names for human-readable output
        targeted: Whether the attack was targeted
    """
    # Get predictions for original and adversarial images
    with torch.no_grad():
        original_outputs = model(original_images)
        original_predictions = original_outputs.argmax(dim=1)

        adversarial_outputs = model(adversarial_images)
        adversarial_predictions = adversarial_outputs.argmax(dim=1)

    # Calculate perturbation norms
    norms = calculate_norms(original_images, adversarial_images)

    # Calculate attack success rate
    if targeted:
        # For targeted attacks, success means prediction matches target
        success_mask = adversarial_predictions == targets
    else:
        # For untargeted attacks, success means prediction differs from original
        success_mask = adversarial_predictions != original_predictions

    success_rate = success_mask.float().mean().item() * 100

    # Print attack configuration
    print("\n" + "=" * 50)
    print("ADVERSARIAL ATTACK ANALYSIS")
    print("=" * 50)

    print(f"\nAttack configuration:")
    for key, value in attack_params.items():
        if key not in ["model", "device", "verbose"]:
            print(f"  {key}: {value}")

    # Print metrics
    print(f"\nAttack metrics:")
    print(f"  Success rate: {success_rate:.2f}%")
    print(f"  Average L2 norm: {norms['L2'].mean().item():.6f}")
    print(f"  Average L2 per dimension: {norms['L2_avg'].mean().item():.8f}")
    print(f"  Average Linf norm: {norms['Linf'].mean().item():.8f}")
    print(
        f"  Average percentage of pixels changed: {norms['L0_percent'].mean().item():.4f}%"
    )
    print(
        f"  Average attack time per image: {metrics.get('time_per_image', 'N/A'):.4f} seconds"
    )

    # Print detailed results for each image
    print("\nDetailed results:")
    for i in range(len(original_images)):
        is_success = success_mask[i].item()
        success_marker = "✓" if is_success else "✗"

        if targeted:
            print(f"\nImage {i+1}: [{success_marker}]")
            print(
                f"  Original class: {class_names[labels[i]]} (predicted: {class_names[original_predictions[i]]})"
            )
            print(f"  Target class: {class_names[targets[i]]}")
            print(
                f"  Adversarial prediction: {class_names[adversarial_predictions[i]]}"
            )
        else:
            print(f"\nImage {i+1}: [{success_marker}]")
            print(
                f"  Original class: {class_names[labels[i]]} (predicted: {class_names[original_predictions[i]]})"
            )
            print(
                f"  Adversarial prediction: {class_names[adversarial_predictions[i]]}"
            )

        print(f"  Perturbation norms:")
        print(f"    L2: {norms['L2'][i]:.6f}")
        print(f"    L2 (per dimension): {norms['L2_avg'][i]:.8f}")
        print(f"    Linf: {norms['Linf'][i]:.8f}")
        print(f"    Percentage of pixels changed: {norms['L0_percent'][i]:.4f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adversarial Attack Analysis")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture to use (e.g., resnet50, vgg16)",
    )

    # Attack method and parameters
    parser.add_argument(
        "--attack",
        type=str,
        default="pgd",
        choices=["pgd", "cg", "lbfgs", "fgsm", "deepfool", "cw"],
        help="Attack method to use",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Use targeted attack (default: untargeted)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["L2", "Linf"],
        default="L2",
        help="Norm to use for constraining perturbations",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Perturbation budget (epsilon) for the attack",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of optimization iterations"
    )

    # Attack-specific parameters
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Confidence parameter for C&W attack",
    )
    parser.add_argument(
        "--c-init", type=float, default=0.1, help="Initial c value for C&W attack"
    )
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=None,
        help="Initial step size for PGD attack",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "margin"],
        help="Loss function to use for the attack",
    )

    # Analysis parameters
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of correctly classified images to attack",
    )

    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    # Parse arguments
    args = parse_args()

    # Convert arguments to dictionary for easier passing
    attack_args = {
        "norm": args.norm,
        "eps": args.eps,
        "targeted": args.targeted,
        "iterations": args.iterations,
        "confidence": args.confidence,
        "c_init": args.c_init,
        "alpha_init": args.alpha_init,
        "loss_fn": args.loss_fn,
    }

    # Load model
    print(f"Loading {args.model} model...")
    model = load_model(args.model)

    # Load data
    print("Loading ImageNet data...")
    dataloader, class_names = load_data(model)

    # Find correctly classified images
    correctly_classified = find_correctly_classified_images(
        model, dataloader, args.num_images, class_names
    )

    if not correctly_classified:
        print("No correctly classified images found. Exiting.")
        return

    # Combine found images into batches
    print(
        f"Selected {len(correctly_classified)} correctly classified images for attack"
    )
    images = torch.cat([img for img, _ in correctly_classified], dim=0)
    labels = torch.cat([lbl for _, lbl in correctly_classified], dim=0)

    # Create attack
    print(f"Creating {args.attack.upper()} attack...")
    attack, attack_params = create_attack(model, args.attack, attack_args)

    # Generate adversarial examples
    print("Generating adversarial examples...")
    adv_images, metrics, targets = generate_adversarial_examples(
        model, images, labels, attack, args.targeted
    )

    # Analyze results
    analyze_attack_results(
        model,
        images,
        adv_images,
        labels,
        targets,
        metrics,
        attack_params,
        class_names,
        args.targeted,
    )


if __name__ == "__main__":
    main()
