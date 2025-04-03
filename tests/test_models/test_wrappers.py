"""Tests for evaluating model accuracy on ImageNet data."""

import os
import sys
import torch

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.datasets.imagenet import get_dataset
from src.models.wrappers import get_model


def main():
    # Configuration
    data_dir = "data"  # Update this to your ImageNet data directory
    num_samples = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load models
    models = {
        "ResNet-18": get_model("resnet18").to(device),
        "ResNet-50": get_model("resnet50").to(device),
        "VGG-16": get_model("vgg16").to(device),
        "EfficientNet-B0": get_model("efficientnet_b0").to(device),
        "MobileNet-V3": get_model("mobilenet_v3_large").to(device),
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

    print(f"Loaded {len(images)} images with labels: {labels.tolist()}")

    # Check for duplicate class names in the dataset
    if hasattr(dataset, "duplicate_classes") and dataset.duplicate_classes:
        print("\nDetected duplicate class names:")
        for name, indices in dataset.duplicate_classes.items():
            print(f"  '{name}' appears at indices: {indices}")

    # Check model accuracy on clean images
    for model_name, model in models.items():
        print(f"\nTesting {model_name}:")
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy with labels from our dataset
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(images)

            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Predictions: {predicted.tolist()}")
            print(f"Labels: {labels.tolist()}")

            # Print some sample class names
            class_names = list(dataset.class_names)
            print("\nSample predictions:")
            for i in range(min(5, len(labels))):
                img_path = dataset.image_paths[i][0]
                filename = os.path.basename(img_path)
                label_idx = labels[i].item()
                pred_idx = predicted[i].item()

                # Extract synset ID from filename if available
                synset_id = filename.split("_")[0] if "_" in filename else "unknown"

                print(f"Image {i}: {filename}")
                print(f"  Synset ID: {synset_id}")
                print(f"  True label: {label_idx} ({class_names[label_idx]})")
                print(f"  Predicted: {pred_idx} ({class_names[pred_idx]})")
                print(f"  {'✓ Correct' if label_idx == pred_idx else '✗ Incorrect'}")


if __name__ == "__main__":
    main()
