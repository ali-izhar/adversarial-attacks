#!/usr/bin/env python
"""
Script to evaluate and print accuracy metrics for different models on ImageNet data.

This script runs inference on all available models and reports:
- Top-1 and Top-5 accuracy
- Inference time
- Model parameter count
- Confusion patterns

# Basic usage (all samples)
python tests/test_models/print_accuracy.py

# Limit samples (faster for testing)
python tests/test_models/print_accuracy.py --max-samples 100

# Save results to file
python tests/test_models/print_accuracy.py --output results.json --visualize
"""

import os
import sys
import time
import argparse
from collections import defaultdict
import json
from tabulate import tabulate
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import get_model
from src.datasets.imagenet import get_dataset, get_dataloader


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_with_direct_forward(model, dataloader, class_names, device):
    """
    Evaluate a model by directly accessing the underlying torchvision model,
    bypassing our wrapper's normalization.

    This is used to avoid double normalization when the dataset is already normalized.
    """
    # Access the underlying model directly
    if hasattr(model, "_model"):
        base_model = model._model
    else:
        base_model = model

    base_model.eval()
    base_model.to(device)

    # Track metrics
    total = 0
    top1_correct = 0
    top5_correct = 0
    inference_time = 0
    misclassifications = []

    first_batch = True

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)

            # For the first batch, print statistics to debug
            if first_batch:
                print(f"Input shape: {images.shape}")
                print(f"Input range: [{images.min():.4f}, {images.max():.4f}]")
                print(f"Input mean: {images.mean():.4f}, std: {images.std():.4f}")
                first_batch = False

            # Measure inference time
            start_time = time.time()
            # Use the base model directly (skip our normalization)
            logits = base_model(images)
            inference_time += time.time() - start_time

            # Calculate top-1 predictions
            _, top1_preds = torch.max(logits, 1)

            # Top-5 predictions
            _, top5_indices = torch.topk(logits, 5, dim=1)

            # Update top-1 accuracy
            top1_correct += (top1_preds == labels).sum().item()

            # Record misclassifications for top-1
            for i, (label, pred) in enumerate(zip(labels, top1_preds)):
                if label.item() != pred.item():
                    misclassifications.append(
                        {
                            "true_class": class_names[label.item()],
                            "pred_class": class_names[pred.item()],
                            "true_idx": label.item(),
                            "pred_idx": pred.item(),
                        }
                    )

            # Update top-5 accuracy
            labels_expanded = labels.view(-1, 1).expand_as(top5_indices)
            top5_correct += torch.sum(torch.eq(top5_indices, labels_expanded)).item()

            total += batch_size

    # Calculate final metrics
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    avg_inference_time = inference_time / total

    # Print summary
    print(f"  Evaluated {total} images")
    print(f"  Top-1 correct predictions: {top1_correct}/{total}")
    print(f"  Top-5 correct predictions: {top5_correct}/{total}")

    # Analyze misclassifications
    confusion_pairs = defaultdict(int)
    for m in misclassifications:
        confusion_pairs[(m["true_class"], m["pred_class"])] += 1

    # Get most common confusions
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "inference_time": avg_inference_time,
        "parameters": count_parameters(
            model if not hasattr(model, "_model") else model._model
        ),
        "top_confusions": top_confusions,
    }


def main():
    """Main function to evaluate all models."""
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on ImageNet")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing ImageNet data",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to specified JSON file"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of results"
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load the ImageNet dataset
    print("Loading ImageNet dataset...")

    # First, check the normalization of the dataset
    custom_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = get_dataset(
        "imagenet",
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        transform=custom_transform,  # Use explicit transform
    )
    dataloader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )  # Set num_workers=0 for easier debugging

    # Load class names
    imagenet_dir = os.path.join(args.data_dir, "imagenet")
    with open(os.path.join(imagenet_dir, "imagenet_classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Define models to evaluate
    models_to_evaluate = [
        "resnet18",
        "resnet50",
        "vgg16",
        "efficientnet_b0",
        "mobilenet_v3_large",
    ]

    # Store results
    results = {}

    # Evaluate each model
    for model_name in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")

        # Explicitly clear GPU cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create a fresh model instance with pretrained weights
        model = get_model(model_name, pretrained=True)

        # Force model to eval mode
        model.eval()
        model = model.to(device)

        # Time the evaluation - use direct evaluation to avoid double normalization
        with torch.no_grad():
            start_time = time.time()
            metrics = evaluate_with_direct_forward(
                model, dataloader, class_names, device
            )
            eval_time = time.time() - start_time

        # Add to results
        results[model_name] = {"metrics": metrics, "eval_time": eval_time}

        # Print basic results
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"  Inference time: {metrics['inference_time']*1000:.2f} ms per image")
        print(f"  Parameters: {metrics['parameters']:,}")

        # Print top confusions
        print("  Top confusions:")
        for (true_class, pred_class), count in metrics["top_confusions"]:
            print(f"    {true_class} → {pred_class}: {count} times")

        # Free up memory
        del model

    # Print final comparison table
    headers = ["Model", "Top-1 Acc", "Top-5 Acc", "Inf. Time (ms)", "Params (M)"]
    table_data = []

    for model_name, data in results.items():
        metrics = data["metrics"]
        row = [
            model_name,
            f"{metrics['top1_accuracy']:.4f}",
            f"{metrics['top5_accuracy']:.4f}",
            f"{metrics['inference_time']*1000:.2f}",
            f"{metrics['parameters']/1e6:.2f}",
        ]
        table_data.append(row)

    print("\n=== Model Comparison ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save results if requested
    if args.output:
        # Convert to serializable format
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                "top1_accuracy": data["metrics"]["top1_accuracy"],
                "top5_accuracy": data["metrics"]["top5_accuracy"],
                "inference_time_ms": data["metrics"]["inference_time"] * 1000,
                "parameters": data["metrics"]["parameters"],
                "eval_time": data["eval_time"],
            }

        with open(args.output, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Create visualization if requested
    if args.visualize:
        create_comparison_plot(results)


def create_comparison_plot(results):
    """Create a visualization of model performance."""
    # Extract data for plotting
    models = list(results.keys())
    top1_accuracies = [results[m]["metrics"]["top1_accuracy"] for m in models]
    inf_times = [results[m]["metrics"]["inference_time"] * 1000 for m in models]  # ms
    parameter_counts = [results[m]["metrics"]["parameters"] / 1e6 for m in models]  # M

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy vs Inference Time scatter plot
    scatter = ax1.scatter(
        inf_times,
        top1_accuracies,
        s=[p * 5 for p in parameter_counts],  # Size proportional to parameters
        alpha=0.7,
    )

    ax1.set_xlabel("Inference Time (ms)")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_title("Accuracy vs. Inference Time")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Add model names as annotations
    for i, model in enumerate(models):
        ax1.annotate(
            model,
            (inf_times[i], top1_accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Bar chart comparing accuracy
    x = np.arange(len(models))
    width = 0.35

    ax2.bar(x, top1_accuracies, width, label="Top-1 Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracies")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Add a second y-axis for inference time
    ax3 = ax2.twinx()
    ax3.plot(x, inf_times, "ro-", label="Inference Time")
    ax3.set_ylabel("Inference Time (ms)", color="r")
    ax3.tick_params(axis="y", labelcolor="r")

    # Add legends
    ax2.legend(loc="upper left")
    ax3.legend(loc="upper right")

    plt.tight_layout()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/model_comparison.png", dpi=300, bbox_inches="tight")
    print("Visualization saved to results/model_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
