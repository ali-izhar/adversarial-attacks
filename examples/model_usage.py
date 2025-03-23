"""
Example script demonstrating how to use the models with ImageNet data.

This script shows how to:
1. Load different pretrained models
2. Process images from the ImageNet dataset
3. Run inference and get predictions
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.datasets.imagenet import get_dataset, get_dataloader


def main():
    """Main function demonstrating model usage."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a model (supported models: resnet50, vgg16, efficientnet_b0, mobilenet_v3_large, etc.)
    model_name = "resnet50"
    print(f"Loading {model_name}...")
    model = get_model(model_name, pretrained=True).to(device)
    model.eval()

    # Load the ImageNet class names
    imagenet_dir = os.path.join("data", "imagenet")
    with open(os.path.join(imagenet_dir, "imagenet_classes.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load a sample dataset with 10 images
    print("Loading dataset...")
    dataset = get_dataset(
        "imagenet",
        data_dir="data",
        max_samples=10,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    )

    # Create a dataloader
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=False)

    # Run inference on the dataset
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            predictions, probabilities = model.predict(images)

            # Print results for each image in the batch
            for i in range(images.size(0)):
                true_label = classes[labels[i].item()]
                pred_label = classes[predictions[i].item()]
                confidence = probabilities[i, predictions[i]].item() * 100

                print(f"Image {batch_idx * 4 + i + 1}:")
                print(f"  True label: {true_label}")
                print(f"  Predicted: {pred_label} ({confidence:.2f}% confidence)")
                print()

    print("Trying a different model...")
    # Load a different model
    model = get_model("efficientnet_b0", pretrained=True).to(device)

    # Process a single image
    sample_idx = 0
    image, label = dataset[sample_idx]
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get prediction
    with torch.no_grad():
        prediction, prob = model.predict(image)

    # Print result
    true_label = classes[label]
    pred_label = classes[prediction.item()]
    confidence = prob[0, prediction].item() * 100

    print(f"EfficientNet-B0 prediction:")
    print(f"  True label: {true_label}")
    print(f"  Predicted: {pred_label} ({confidence:.2f}% confidence)")

    print("\nDone!")


if __name__ == "__main__":
    main()
