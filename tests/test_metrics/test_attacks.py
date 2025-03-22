"""Test script to evaluate adversarial attacks."""

import os
import sys
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets import get_dataset, get_dataloader
from src.attacks import PGD, ConjugateGradient, LBFGS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test adversarial attacks")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to test"
    )
    parser.add_argument("--eps", type=float, default=0.05, help="Perturbation size")
    parser.add_argument(
        "--norm",
        type=str,
        default="L2",
        choices=["L2", "Linf"],
        help="Perturbation norm",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pgd",
        choices=["pgd", "cg", "lbfgs", "all"],
        help="Attack method",
    )
    parser.add_argument(
        "--targeted", action="store_true", help="Perform targeted attack"
    )
    parser.add_argument(
        "--target-class", type=int, default=0, help="Target class for targeted attack"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )

    return parser.parse_args()


def visualize_results(
    original,
    adversarial,
    original_pred,
    adversarial_pred,
    class_names,
    method,
    output_path,
):
    """Visualize the original and adversarial images."""
    # Convert to numpy for visualization
    original_np = original.cpu().permute(1, 2, 0).numpy()
    original_np = np.clip(original_np, 0, 1)

    adversarial_np = adversarial.cpu().permute(1, 2, 0).numpy()
    adversarial_np = np.clip(adversarial_np, 0, 1)

    # Calculate perturbation
    perturbation = adversarial_np - original_np
    perturbation = (
        perturbation * 10 + 0.5
    )  # Amplify and center perturbation for visibility
    perturbation = np.clip(perturbation, 0, 1)

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax[0].imshow(original_np)
    ax[0].set_title(f"Original: {class_names[original_pred]}")
    ax[0].axis("off")

    # Plot adversarial image
    ax[1].imshow(adversarial_np)
    ax[1].set_title(f"Adversarial: {class_names[adversarial_pred]}")
    ax[1].axis("off")

    # Plot perturbation
    ax[2].imshow(perturbation)
    ax[2].set_title("Perturbation (10x)")
    ax[2].axis("off")

    # Set title for the whole figure
    plt.suptitle(f"Attack Method: {method}")

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # Image preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        transform=preprocess,
        max_samples=args.num_samples,
    )
    dataloader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create attack methods
    attacks = {}
    if args.method == "pgd" or args.method == "all":
        attacks["PGD"] = PGD(
            model=model,
            norm=args.norm,
            eps=args.eps,
            targeted=args.targeted,
            n_iterations=100,
            alpha_init=args.eps / 10,
            verbose=True,
            device=device,
        )

    if args.method == "cg" or args.method == "all":
        attacks["CG"] = ConjugateGradient(
            model=model,
            norm=args.norm,
            eps=args.eps,
            targeted=args.targeted,
            n_iterations=50,
            verbose=True,
            device=device,
        )

    if args.method == "lbfgs" or args.method == "all":
        attacks["L-BFGS"] = LBFGS(
            model=model,
            norm=args.norm,
            eps=args.eps,
            targeted=args.targeted,
            n_iterations=30,
            verbose=True,
            device=device,
        )

    # Run attacks
    results = {}
    all_metrics = {}

    for attack_name, attack in attacks.items():
        print(f"\nRunning {attack_name} attack...")
        success_count = 0

        # Track metrics across all samples
        all_metrics[attack_name] = {
            "success_rate": 0.0,
            "avg_time": 0.0,
            "avg_iterations": 0.0,
            "l2_norm": 0.0,
            "linf_norm": 0.0,
        }

        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            # For targeted attacks, set the target label
            if args.targeted:
                targets = torch.tensor([args.target_class] * images.size(0)).to(device)
            else:
                targets = labels.to(device)

            # Get original prediction
            with torch.no_grad():
                outputs = model(images)
                _, predictions = outputs.max(1)

            # Run attack
            adv_images, metrics = attack.generate(images, targets)

            # Check if attack succeeded
            with torch.no_grad():
                adv_outputs = model(adv_images)
                _, adv_predictions = adv_outputs.max(1)

            # Count successes
            if args.targeted:
                success = adv_predictions == targets
            else:
                success = adv_predictions != targets

            success_count += success.sum().item()

            # Calculate perturbation norms
            perturbation = adv_images - images
            l2_norm = (
                torch.norm(perturbation.view(images.size(0), -1), p=2, dim=1)
                .mean()
                .item()
            )
            linf_norm = (
                torch.norm(perturbation.view(images.size(0), -1), p=float("inf"), dim=1)
                .mean()
                .item()
            )

            # Update metrics
            all_metrics[attack_name]["avg_time"] += metrics["time"]
            all_metrics[attack_name]["avg_iterations"] += metrics["iterations"]
            all_metrics[attack_name]["l2_norm"] += l2_norm
            all_metrics[attack_name]["linf_norm"] += linf_norm

            # Visualize results
            for j in range(images.size(0)):
                if success[j]:
                    output_path = os.path.join(
                        args.output_dir,
                        f"{attack_name}_sample_{i * args.batch_size + j}.png",
                    )
                    visualize_results(
                        images[j],
                        adv_images[j],
                        predictions[j].item(),
                        adv_predictions[j].item(),
                        dataset.class_names,
                        attack_name,
                        output_path,
                    )

        # Compute average metrics
        num_samples = len(dataloader.dataset)
        all_metrics[attack_name]["success_rate"] = success_count / num_samples * 100
        all_metrics[attack_name]["avg_time"] /= num_samples
        all_metrics[attack_name]["avg_iterations"] /= num_samples
        all_metrics[attack_name]["l2_norm"] /= num_samples
        all_metrics[attack_name]["linf_norm"] /= num_samples

        # Print results
        print(f"\nResults for {attack_name}:")
        print(f"  Success rate: {all_metrics[attack_name]['success_rate']:.2f}%")
        print(f"  Average time: {all_metrics[attack_name]['avg_time']:.2f} seconds")
        print(f"  Average iterations: {all_metrics[attack_name]['avg_iterations']:.2f}")
        print(f"  Average L2 norm: {all_metrics[attack_name]['l2_norm']:.4f}")
        print(f"  Average Linf norm: {all_metrics[attack_name]['linf_norm']:.4f}")

    # Compare attack methods if multiple were used
    if len(attacks) > 1:
        print("\nAttack Comparison:")
        print("-" * 80)
        print(
            f"{'Method':<10} {'Success Rate':<15} {'Time (s)':<12} {'Iterations':<12} {'L2 Norm':<10} {'Linf Norm':<10}"
        )
        print("-" * 80)

        for attack_name, metrics in all_metrics.items():
            print(
                f"{attack_name:<10} {metrics['success_rate']:<15.2f} {metrics['avg_time']:<12.2f} "
                f"{metrics['avg_iterations']:<12.2f} {metrics['l2_norm']:<10.4f} {metrics['linf_norm']:<10.4f}"
            )


if __name__ == "__main__":
    main()
