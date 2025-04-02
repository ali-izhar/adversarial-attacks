#!/usr/bin/env python
"""
Test script to verify that the ImageNetDataset implementation correctly loads
and maps images to their class indices based on synset IDs in filenames.
"""
import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model


def main():
    # Configuration
    data_dir = "data"  # Update this to your ImageNet data directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load a model for testing
    model = get_model("resnet18").to(device)
    model.eval()

    # Load dataset with a small number of samples
    print(f"Loading dataset from {data_dir}...")

    for num_samples in [10, 20]:
        print(f"\nTesting with {num_samples} samples:")
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

        print(f"Loaded {len(images)} images with labels: {labels.tolist()}")

        # Check if the labels are indeed 0 to num_samples-1 as expected
        expected = list(range(min(num_samples, len(labels))))
        if labels.tolist()[: len(expected)] == expected:
            print("✓ Labels match expected sequential order")
        else:
            print("✗ Labels do not match expected sequential order")

        # Get model predictions
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(images)

            print(f"Model accuracy: {accuracy:.2f}%")
            print(f"Predictions: {predicted.tolist()}")

            # Print first 5 sample details
            print("\nSample images:")
            for i in range(min(5, len(labels))):
                img_path = dataset.image_paths[i][0]
                filename = os.path.basename(img_path)
                label = labels[i].item()
                pred = predicted[i].item()

                print(f"  Image {i}: {filename}")
                print(f"    Label: {label} ({dataset.class_names[label]})")
                print(f"    Predicted: {pred} ({dataset.class_names[pred]})")


if __name__ == "__main__":
    main()
