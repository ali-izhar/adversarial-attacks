#!/usr/bin/env python
import argparse
import torch
import torchvision.models as models
from tqdm import tqdm
import os
import time
import numpy as np

from src.attacks import PGDAttack, CGAttack, LBFGSAttack
from src.utils.data import load_dataset
from src.utils.metrics import calculate_metrics
from src.utils.visualization import save_comparison
from src.experiments.configs import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run adversarial attack experiments")

    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["resnet18"],
        choices=["resnet18", "vgg16", "mobilenetv2"],
        help="Models to attack",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["pgd", "cg", "lbfgs"],
        choices=["pgd", "cg", "lbfgs"],
        help="Attack methods to use",
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images to attack"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.03, help="Maximum perturbation magnitude"
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="L2",
        choices=["L2", "Linf"],
        help="Norm for perturbation constraint",
    )
    parser.add_argument(
        "--targeted", action="store_true", help="Perform targeted attacks"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    return parser.parse_args()


def get_model(model_name):
    """Load a pretrained model."""
    if model_name == "resnet18":
        return models.resnet18(pretrained=True).eval()
    elif model_name == "vgg16":
        return models.vgg16(pretrained=True).eval()
    elif model_name == "mobilenetv2":
        return models.mobilenet_v2(pretrained=True).eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_attack(attack_name, model, args):
    """Initialize an attack method."""
    if attack_name == "pgd":
        return PGDAttack(
            model, epsilon=args.epsilon, norm=args.norm, targeted=args.targeted
        )
    elif attack_name == "cg":
        return CGAttack(
            model, epsilon=args.epsilon, norm=args.norm, targeted=args.targeted
        )
    elif attack_name == "lbfgs":
        return LBFGSAttack(
            model, epsilon=args.epsilon, norm=args.norm, targeted=args.targeted
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def main():
    """Main function to run experiments."""
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    images, labels = load_dataset(args.dataset, args.num_images)

    # Get experiment configuration
    config = get_config(args.dataset)

    # Results dictionary
    results = {}

    # For each model
    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        model = get_model(model_name).to(device)

        model_results = {}

        # For each attack method
        for attack_name in args.attacks:
            print(f"\nRunning {attack_name} attack...")
            attack = get_attack(attack_name, model, args)

            # Track metrics
            success_rates = []
            l2_norms = []
            linf_norms = []
            ssim_values = []
            runtimes = []

            # Run attack on each image
            for i, (image, label) in enumerate(
                tqdm(zip(images, labels), total=len(images))
            ):
                image = image.to(device)
                label = label.to(device)

                # Time the attack
                start_time = time.time()
                adv_image = attack.generate(image.unsqueeze(0), label.unsqueeze(0))
                end_time = time.time()

                # Calculate metrics
                metrics = calculate_metrics(
                    image.unsqueeze(0), adv_image, label.unsqueeze(0), model
                )

                # Save examples
                if i < 10:  # Save first 10 examples
                    save_comparison(
                        image.cpu(),
                        adv_image[0].cpu(),
                        os.path.join(
                            args.output_dir, f"{model_name}_{attack_name}_{i}.png"
                        ),
                        metrics,
                    )

                # Record metrics
                success_rates.append(metrics["success"])
                l2_norms.append(metrics["l2_norm"])
                linf_norms.append(metrics["linf_norm"])
                ssim_values.append(metrics["ssim"])
                runtimes.append(end_time - start_time)

            # Aggregate results
            attack_results = {
                "success_rate": np.mean(success_rates) * 100,
                "l2_norm": np.mean(l2_norms),
                "linf_norm": np.mean(linf_norms),
                "ssim": np.mean(ssim_values),
                "runtime_per_image": np.mean(runtimes),
            }

            model_results[attack_name] = attack_results

            # Print summary
            print(f"\n{attack_name.upper()} Attack Results:")
            print(f"Success Rate: {attack_results['success_rate']:.2f}%")
            print(f"Avg L2 Norm: {attack_results['l2_norm']:.4f}")
            print(f"Avg L∞ Norm: {attack_results['linf_norm']:.4f}")
            print(f"Avg SSIM: {attack_results['ssim']:.4f}")
            print(f"Avg Runtime: {attack_results['runtime_per_image']:.4f} seconds")

        results[model_name] = model_results

    # Save all results
    torch.save(results, os.path.join(args.output_dir, "results.pt"))

    print(
        "\nExperiment complete. Results saved to:",
        os.path.join(args.output_dir, "results.pt"),
    )


if __name__ == "__main__":
    main()
