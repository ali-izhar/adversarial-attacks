#!/usr/bin/env python
"""Tests for all optimization attacks (PGD, CG, LBFGS) on a small sample of ImageNet data.

USAGE::
    >>> python -m tests.test_attacks.test_optimize_attacks
"""

import os
import sys
import time
import torch
import argparse
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

# Add the project root to the path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model
from src.attacks import CG, PGD, LBFGS


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # Calculate optimal batch size based on number of samples
    num_samples = len(dataset)
    batch_size = (
        min(args.batch_size, num_samples) if args.batch_size > 0 else num_samples
    )

    # Create a dataloader with the calculated batch size
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    class_names = dataset.class_names

    # Store metrics
    metrics = {
        "name": attack_name,
        "attack_success_rate": 0.0,  # How often attack succeeded in fooling the model
        "model_accuracy": 0.0,  # Model's accuracy on adversarial examples
        "l2_norm": 0.0,
        "linf_norm": 0.0,
        "ssim": 0.0,
        "time_per_sample": 0.0,
        "iterations": 0.0,
        "gradient_calls": 0.0,
    }

    # Reset attack metrics before starting
    attack.reset_metrics()

    # Set targeted mode if requested
    if args.targeted:
        if args.target_method == "random":
            attack.set_mode_targeted_random()
            attack_name = f"{attack_name} (Targeted-Random)"
        elif args.target_method == "least-likely":
            attack.set_mode_targeted_least_likely()
            attack_name = f"{attack_name} (Targeted-Least-Likely)"
        print(f"Running in targeted mode: {args.target_method}")
    else:
        attack.set_mode_default()

    # Track successful adversarial examples for visualization
    successful_examples = []

    # Run attack on all batches
    total_samples = 0
    total_batches = len(dataloader)

    print(
        f"Processing {num_samples} samples in {total_batches} batches (batch size: {batch_size})"
    )

    for batch_idx, (inputs, labels) in enumerate(
        tqdm(dataloader, desc=f"{attack_name}", total=total_batches)
    ):
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
                    f"  All samples in batch {batch_idx+1}/{total_batches} are already misclassified, skipping"
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

        # Get model predictions on adversarial examples
        with torch.no_grad():
            adv_outputs = model(adversarial)
            _, adv_preds = torch.max(adv_outputs, 1)

            # Calculate model accuracy based on attack type
            if args.targeted:
                # For targeted attacks, accuracy is low when predictions match target labels
                target_labels = attack.get_target_label(correct_inputs, correct_labels)
                model_accuracy = (
                    100 - (adv_preds == target_labels).float().mean().item() * 100
                )
            else:
                # For untargeted attacks, accuracy is how often predictions match true labels
                model_accuracy = (
                    adv_preds == correct_labels
                ).float().mean().item() * 100

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
            f"  Batch {batch_idx+1}/{total_batches}: "
            f"Attack Success = {batch_success_rate:.2f}%, "
            f"Model Accuracy = {model_accuracy:.2f}%, "
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
    metrics["attack_success_rate"] = attack_metrics["success_rate"]
    metrics["model_accuracy"] = (
        100 - attack_metrics["success_rate"]
    )  # Since attack success = 100 - model accuracy
    metrics["l2_norm"] = attack_metrics["l2_norm"]
    metrics["linf_norm"] = attack_metrics["linf_norm"]
    metrics["ssim"] = attack_metrics["ssim"]
    metrics["time_per_sample"] = attack_metrics["time_per_sample"]
    metrics["iterations"] = attack_metrics["iterations"]
    metrics["gradient_calls"] = attack_metrics.get(
        "gradient_calls", 0
    )  # May not exist in simplified version

    # Print metrics summary
    print(f"\nMetrics for {attack_name}:")
    print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    print(f"  Model Accuracy on Adversarial Examples: {metrics['model_accuracy']:.2f}%")
    print(f"  L2 Norm: {metrics['l2_norm']:.4f}")
    print(f"  L∞ Norm: {metrics['linf_norm']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Time per sample: {metrics['time_per_sample']*1000:.2f} ms")
    print(f"  Average iterations: {metrics['iterations']:.2f}")
    if "gradient_calls" in attack_metrics:
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


def parse_fraction(value):
    """Parse string fractions like '4/255' or convert floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and "/" in value:
        num, denom = value.split("/")
        return float(num) / float(denom)
    return float(value)


def create_attack(model, attack_type, config, targeted=False, pgd_variant=None):
    """
    Create an attack instance based on the attack type and configuration.

    Args:
        model: The model to attack
        attack_type: Type of attack (e.g., "CG", "LBFGS", "PGD")
        config: Configuration dictionary
        targeted: Whether the attack is targeted
        pgd_variant: Specific PGD variant to use (basic, margin, l2)

    Returns:
        tuple: (attack_instance, attack_name)
    """
    # Get attack parameters from config
    attack_params = config["attack"]["params"]
    attack_mode = "targeted" if targeted else "untargeted"

    # Default norm type (first in the list)
    norm_type = config["attack"]["norm_types"][0]

    if attack_type == "CG":
        # Get CG parameters from config
        params = attack_params["CG"][attack_mode]
        eps_value = params["eps_values"][norm_type][0]  # Default to first epsilon value
        eps = parse_fraction(eps_value)

        # Get parameters from config with sensible defaults
        n_iter = params.get("n_iter", 50)
        beta_method = params.get("beta_method", "HS")
        restart_interval = params.get("restart_interval", 10)
        tv_lambda = params.get("tv_lambda", 0.05)
        color_lambda = params.get("color_lambda", 0.05)
        perceptual_lambda = params.get("perceptual_lambda", 0.05)
        rand_init = params.get("rand_init", True)
        fgsm_init = params.get("fgsm_init", True)
        adaptive_restart = params.get("adaptive_restart", True)
        early_stopping = params.get("early_stopping", True)
        strict_epsilon_constraint = params.get("strict_epsilon_constraint", True)

        # Create CG with parameters from config
        attack = CG(
            model,
            norm=norm_type,
            eps=eps,
            n_iter=n_iter,
            beta_method=beta_method,
            restart_interval=restart_interval,
            tv_lambda=tv_lambda,
            color_lambda=color_lambda,
            perceptual_lambda=perceptual_lambda,
            rand_init=rand_init,
            fgsm_init=fgsm_init,
            adaptive_restart=adaptive_restart,
            early_stopping=early_stopping,
            strict_epsilon_constraint=strict_epsilon_constraint,
            verbose=True,  # Enable verbose output for debugging
        )
        attack_name = f"CG (ε={eps:.4f})"

    elif attack_type == "LBFGS":
        # Get LBFGS parameters from config
        params = attack_params["LBFGS"][attack_mode]
        eps_value = params["eps_values"]["Linf"][0]  # Default to first epsilon value
        eps = parse_fraction(eps_value)

        # Get enhanced LBFGS parameters with defaults if not present
        n_iterations = params.get("n_iterations", 50)
        history_size = params.get("history_size", 10)
        initial_const = params.get("initial_const", 1e-2)
        binary_search_steps = params.get("binary_search_steps", 5)
        const_factor = params.get("const_factor", 10.0)
        repeat_search = params.get("repeat_search", True)
        rand_init = params.get("rand_init", True)
        init_std = params.get("init_std", 0.01)

        attack = LBFGS(
            model,
            norm=norm_type,
            eps=eps,
            n_iterations=n_iterations,
            history_size=history_size,
            initial_const=initial_const,
            binary_search_steps=binary_search_steps,
            const_factor=const_factor,
            repeat_search=repeat_search,
            rand_init=rand_init,
            init_std=init_std,
        )

        attack_name = f"LBFGS (ε={eps:.4f}, bs_steps={binary_search_steps})"

    elif attack_type == "PGD":
        # Get the base PGD parameters from config
        params = attack_params["PGD"][attack_mode]
        norm = "Linf"  # Default norm
        eps_value = params["eps_values"][norm][0]  # Default to first epsilon value
        eps = parse_fraction(eps_value)

        # If a specific PGD variant is specified, use that configuration
        if pgd_variant:
            # Get variant configuration from config file
            if pgd_variant in config["attack"]["params"]["PGD"]["variants"]:
                variant_config = config["attack"]["params"]["PGD"]["variants"][
                    pgd_variant
                ]
                variant_params = variant_config["params"].copy()

                # Set epsilon based on config file
                variant_params["eps"] = eps

                # Parse any fraction values
                for param, value in variant_params.items():
                    if isinstance(value, str) and "/" in value:
                        variant_params[param] = parse_fraction(value)

                # Create attack with the specific variant parameters
                attack = PGD(model, **variant_params)
                attack_name = f"{variant_config['name']} (ε={eps:.4f})"

                # Add information about special features to the name
                if variant_params.get("loss_fn") != "cross_entropy":
                    attack_name += f", {variant_params.get('loss_fn')} loss"
            else:
                print(
                    f"Warning: PGD variant '{pgd_variant}' not found in config, using default"
                )
                # Fall back to standard PGD parameters
                n_iterations = params["n_iterations"]
                step_size_key = f"step_size_{norm.lower()}"
                step_size = parse_fraction(params[step_size_key])
                rand_init = params["rand_init"]
                early_stopping = params["early_stopping"]
                loss_fn = params.get("loss_fn", "cross_entropy")

                attack = PGD(
                    model,
                    norm=norm,
                    eps=eps,
                    n_iterations=n_iterations,
                    step_size=step_size,
                    loss_fn=loss_fn,
                    rand_init=rand_init,
                    early_stopping=early_stopping,
                )
                attack_name = (
                    f"PGD ({norm}, ε={eps:.4f}, steps={n_iterations}, {loss_fn})"
                )
        else:
            # Use standard PGD parameters from config
            n_iterations = params["n_iterations"]
            step_size_key = f"step_size_{norm.lower()}"
            step_size = parse_fraction(params[step_size_key])
            rand_init = params["rand_init"]
            early_stopping = params["early_stopping"]
            loss_fn = params.get("loss_fn", "cross_entropy")

            attack = PGD(
                model,
                norm=norm,
                eps=eps,
                n_iterations=n_iterations,
                step_size=step_size,
                loss_fn=loss_fn,
                rand_init=rand_init,
                early_stopping=early_stopping,
            )
            attack_name = f"PGD ({norm}, ε={eps:.4f}, steps={n_iterations}, {loss_fn})"

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return attack, attack_name


def main(args):
    """Main function."""
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Load configuration file
    config_path = (
        args.config_file
        if args.config_file
        else os.path.join(project_root, "config", "config.yaml")
    )
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Validate targeted attack parameters
    if args.targeted and not args.target_method:
        print("Error: --target-method must be specified when --targeted is set")
        print("Choose from: random, least-likely")
        return

    # Adjust output directory if needed for targeted attacks
    if args.targeted:
        args.output_dir = os.path.join(
            args.output_dir, f"targeted_{args.target_method}"
        )

    # Load dataset
    print("Loading ImageNet dataset...")
    num_samples = args.num_samples or config["dataset"]["num_images"]
    data_dir = args.data_dir or config["dataset"]["image_dir"]

    # Make sure we're using a valid path for the dataset
    if not os.path.exists(data_dir) and "imagenet" not in data_dir:
        # Try with imagenet subdirectory
        alt_path = os.path.join(data_dir, "imagenet")
        if os.path.exists(alt_path):
            data_dir = alt_path
        else:
            # Try with project root
            project_data_dir = os.path.join(project_root, "data", "imagenet")
            if os.path.exists(project_data_dir):
                data_dir = project_data_dir
            else:
                # Just use as is and let the dataset loader handle it
                pass

    print(f"Using dataset path: {data_dir}")
    dataset = get_dataset(
        dataset_name="imagenet", data_dir=data_dir, max_samples=num_samples
    )

    # Load model
    print(f"Loading {args.model_name} model...")
    model = get_model(args.model_name).to(args.device)
    model.eval()

    # Initialize attacks based on config
    attacks = []

    # Map attack request to creation function
    for attack_type in args.attacks:
        if attack_type.lower() == "all":
            # Load all configured attacks
            for attack_name in ["CG", "LBFGS", "PGD"]:
                try:
                    attack, attack_name = create_attack(
                        model, attack_name, config, args.targeted
                    )
                    attacks.append((attack, attack_name))
                except Exception as e:
                    print(f"Error creating attack {attack_name}: {e}")
        elif attack_type.lower() == "pgd":
            # If pgd_variant is specified, create that specific variant
            if args.pgd_variant:
                try:
                    attack, attack_name = create_attack(
                        model,
                        "PGD",
                        config,
                        args.targeted,
                        pgd_variant=args.pgd_variant,
                    )
                    attacks.append((attack, attack_name))
                except Exception as e:
                    print(f"Error creating PGD variant {args.pgd_variant}: {e}")
            else:
                # Create default PGD
                try:
                    attack, attack_name = create_attack(
                        model, "PGD", config, args.targeted
                    )
                    attacks.append((attack, attack_name))
                except Exception as e:
                    print(f"Error creating PGD attack: {e}")
        else:
            # Load specific attack type (normalize to uppercase for consistent comparison)
            attack_type_upper = attack_type.upper()
            # Map to correct case based on known attack types
            attack_type_map = {
                "CG": "CG",
                "LBFGS": "LBFGS",
                "PGD": "PGD",
            }
            try:
                if attack_type_upper in attack_type_map:
                    attack, attack_name = create_attack(
                        model, attack_type_map[attack_type_upper], config, args.targeted
                    )
                    attacks.append((attack, attack_name))
                else:
                    print(f"Unknown attack type: {attack_type}")
            except Exception as e:
                print(f"Error creating attack {attack_type}: {e}")

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

    headers = [
        "Attack",
        "Attack Success %",
        "Model Accuracy %",
        "L2 Norm",
        "L∞ Norm",
        "SSIM",
        "Time (ms)",
    ]
    rows = []

    for metrics in all_metrics:
        rows.append(
            [
                metrics["name"],
                f"{metrics['attack_success_rate']:.2f}%",
                f"{metrics['model_accuracy']:.2f}%",
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

    attack_mode = "targeted" if args.targeted else "untargeted"
    print(f"\nResults and visualizations saved to {args.output_dir}")
    print(f"Attack mode: {attack_mode.upper()}")
    if args.targeted:
        print(f"Target method: {args.target_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test baseline adversarial attacks")

    # Dataset and model parameters
    parser.add_argument(
        "--data-dir", "-d", type=str, default=None, help="Base directory for datasets"
    )
    parser.add_argument(
        "--model-name",
        "-m",
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
        "-n",
        type=int,
        default=None,
        help="Number of samples to load from dataset (defaults to config value)",
    )

    # Config file
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default=None,
        help="Path to configuration file (defaults to config/config.yaml)",
    )

    # Attack selection
    parser.add_argument(
        "--attacks",
        "-a",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "cg", "lbfgs", "pgd"],
        help="Which attacks to test",
    )

    # PGD variant selection
    parser.add_argument(
        "--pgd-variant",
        "-pv",
        type=str,
        choices=["basic", "momentum", "l2"],
        default="momentum",
        help="Specific PGD variant to test (for --attacks pgd)",
    )

    # Test parameters
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4, help="Batch size for testing"
    )
    parser.add_argument(
        "--max-batches",
        "-mb",
        type=int,
        default=1,
        help="Maximum number of batches to test per attack",
    )
    parser.add_argument(
        "--num-vis",
        "-v",
        type=int,
        default=3,
        help="Number of adversarial examples to visualize per attack",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/test_attacks",
        help="Directory to save test results and visualizations",
    )

    # Targeted attack parameters
    parser.add_argument(
        "--targeted",
        "-t",
        action="store_true",
        help="Run targeted attacks",
    )
    parser.add_argument(
        "--target-method",
        "-tm",
        type=str,
        choices=["random", "least-likely"],
        help="Target method for targeted attacks",
    )

    args = parser.parse_args()

    main(args)
