#!/usr/bin/env python
"""
Test script to verify model accuracy on clean images.
This helps diagnose issues with the adversarial attack evaluation pipeline.
"""

import torch
import os
import sys
import random

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model

# Sample of ImageNet class indices for common objects
# These are real classes the models were trained to recognize
IMAGENET_SAMPLE_CLASSES = {
    0: 491,  # strawberry
    1: 285,  # Egyptian cat
    2: 970,  # alp
    3: 385,  # badger
    4: 388,  # giant panda
    5: 13,  # junco
    6: 153,  # Maltese dog
    7: 417,  # balloon
    8: 497,  # church
    9: 920,  # book jacket
}


def main():
    # Configuration
    data_dir = "data"  # Update this to your ImageNet data directory
    num_samples = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load models
    models = {
        "ResNet-18": get_model("resnet18").to(device),
        # "ResNet-50": get_model("resnet50").to(device),
        # "VGG-16": get_model("vgg16").to(device),
        # "EfficientNet-B0": get_model("efficientnet_b0").to(device),
        # "MobileNet-V3": get_model("mobilenet_v3_large").to(device),
    }

    # Set models to evaluation mode
    for model in models.values():
        model.eval()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = get_dataset(
        dataset_name="imagenet", data_dir=data_dir, max_samples=num_samples
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=num_samples, shuffle=False
    )

    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)

    print(f"Loaded {len(images)} images with original labels: {labels.tolist()}")

    # Convert our sequential labels to actual ImageNet class indices
    # In a real attack, we would use these mapped labels
    mapped_labels = torch.tensor(
        [IMAGENET_SAMPLE_CLASSES[l.item()] for l in labels], device=device
    )
    print(f"Mapped to ImageNet classes: {mapped_labels.tolist()}")

    # Check model accuracy on clean images
    for model_name, model in models.items():
        print(f"\nTesting {model_name} with original labels:")
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy with original labels (should be poor)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(images)

            print(f"Accuracy with original labels: {accuracy:.2f}%")
            print(f"Predictions: {predicted.tolist()}")
            print(f"Original labels: {labels.tolist()}")

        print(f"\nTesting {model_name} with mapped ImageNet labels:")
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy with mapped labels (should be better)
            correct = (predicted == mapped_labels).sum().item()
            accuracy = 100 * correct / len(images)

            print(f"Accuracy with mapped labels: {accuracy:.2f}%")
            print(f"Predictions: {predicted.tolist()}")
            print(f"Mapped labels: {mapped_labels.tolist()}")

            # Print some sample class names to verify mapping
            class_names = list(dataset.class_names)
            print("\nSample class mappings:")
            for i in range(min(5, len(labels))):
                orig_idx = labels[i].item()
                mapped_idx = mapped_labels[i].item()
                pred_idx = predicted[i].item()

                print(f"Image {i}:")
                print(f"  Original label {orig_idx}")
                print(
                    f"  Mapped to ImageNet class {mapped_idx} ({class_names[mapped_idx]})"
                )
                print(f"  Predicted as class {pred_idx} ({class_names[pred_idx]})")


if __name__ == "__main__":
    main()
