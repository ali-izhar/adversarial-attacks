#!/usr/bin/env python
"""
Evaluate the Carlini & Wagner (C&W) attack on ImageNet.

This script runs the C&W attack on a set of correctly classified images
and generates comprehensive evaluation metrics and visualizations.
"""

import os
import torch
import argparse
from src.attacks.attack_cw import CW
from src.models.wrappers import get_model
from src.datasets.imagenet import get_dataset, get_dataloader
from src.utils.eval import AdversarialAttackEvaluator


def cw_attack_wrapper(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    norm: str,
    targeted: bool,
) -> torch.Tensor:
    """
    Custom wrapper for C&W attack that properly handles parameters.

    Args:
        model: Model to attack
        images: Input images
        labels: Target/true labels
        epsilon: Perturbation budget (used for c_init in C&W)
        norm: Norm type (ignored for C&W as it's L2 only)
        targeted: Whether attack is targeted

    Returns:
        Adversarial examples
    """
    # Create CW attack with proper parameters
    attack = CW(
        model=model,
        confidence=args.confidence,
        c_init=epsilon * 0.1,  # Scale c_init based on epsilon
        max_iter=500,
        binary_search_steps=5,
        learning_rate=0.01,
        targeted=targeted,
        abort_early=True,
        clip_min=0.0,
        clip_max=1.0,
        verbose=True,
        device=images.device,
    )

    # Generate adversarial examples
    adv_images, _ = attack.generate(images, labels)

    return adv_images


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate C&W attack")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model to attack (e.g., resnet50, vgg16)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of correctly classified images to attack",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Confidence parameter for C&W attack",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Perform targeted attack instead of untargeted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cw_evaluation",
        help="Directory to save results",
    )
    global args
    args = parser.parse_args()

    # Configure L2 perturbation budgets to evaluate
    # For C&W attack, epsilon acts as a constraint in the binary search process
    epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"Evaluating C&W attack against {args.model}")
    print(
        f"Attack configuration: confidence={args.confidence}, "
        + f"targeted={args.targeted}, binary_search_steps=5"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model)
    model.to(device)
    model.eval()

    # Load data
    print("Loading ImageNet data...")
    # Get the dataset first
    dataset = get_dataset("imagenet", "data")
    # Get class names from the dataset
    class_names = dataset.class_names
    # Get dataloader (separately)
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=True)

    # Find correctly classified images
    print(f"Finding {args.num_images} correctly classified images...")
    correctly_classified = []
    tested_images = 0

    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            if predictions[0] == labels[0]:
                correctly_classified.append((images, labels))
                print(
                    f"Found correctly classified image {len(correctly_classified)}: {class_names[labels[0]]}"
                )

                if len(correctly_classified) >= args.num_images:
                    break

        tested_images += 1
        if tested_images % 10 == 0:
            print(
                f"Tested {tested_images} images, found {len(correctly_classified)} correctly classified"
            )

    # Initialize evaluator
    evaluator = AdversarialAttackEvaluator(
        model,
        correctly_classified,
        class_names,
        output_dir=os.path.join(args.output_dir, args.model),
        device=device,
    )

    # Add attack with custom wrapper
    evaluator.add_attack(
        "C&W",
        cw_attack_wrapper,
        description=f"C&W attack with L2 norm (confidence={args.confidence})",
    )

    # Run evaluation
    evaluator.evaluate_all(
        epsilon_values=epsilon_values,
        norm="L2",
        targeted=args.targeted,
        save_adv_examples=True,
    )

    # Generate report
    report_path = os.path.join(args.output_dir, args.model, "evaluation_report.md")
    evaluator.generate_report(report_path)

    # Get the results for display
    results = evaluator.results["C&W"]
    mid_idx = len(epsilon_values) // 2
    mid_eps = epsilon_values[mid_idx]

    # Get summary metrics from the middle epsilon evaluation
    detailed_metrics = results["detailed_metrics"]

    print("\n" + "=" * 50)
    print(f"C&W Attack Results Summary (ε={mid_eps}):")
    print("=" * 50)
    print(f"Success Rate: {detailed_metrics['success_rate']:.2f}%")
    print(f"Average L2 Perturbation: {detailed_metrics['L2']:.4f}")
    print(f"Average L-infinity Perturbation: {detailed_metrics['Linf']:.4f}")
    print(f"PSNR: {detailed_metrics['PSNR_dB']:.2f} dB")
    print(f"Average % of Pixels Changed: {detailed_metrics['L0_percent']:.2f}%")
    print("\nFull evaluation report saved to:")
    print(f"  {report_path}")
    print(f"Visualizations saved to:")
    print(f"  {os.path.join(args.output_dir, args.model, 'C&W')}")


if __name__ == "__main__":
    main()
