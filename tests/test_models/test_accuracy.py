"""Tests for evaluating model accuracy on ImageNet data."""

import os
import sys
import pytest
import torch
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import get_model
from src.datasets.imagenet import get_dataset, get_dataloader


@pytest.fixture
def imagenet_data():
    """Fixture to load a small subset of ImageNet data for testing."""
    # Load a small subset of images (50 for faster testing)
    dataset = get_dataset("imagenet", data_dir="data", max_samples=50)
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False)

    # Load the class names
    imagenet_dir = os.path.join("data", "imagenet")
    with open(os.path.join(imagenet_dir, "imagenet_classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return dataloader, class_names


@pytest.mark.slow  # Mark these tests as slow since they involve model inference
class TestModelAccuracy:
    """Tests for model accuracy on the ImageNet dataset."""

    def test_single_model_accuracy(self, imagenet_data):
        """Test the accuracy of a single model on ImageNet data."""
        dataloader, class_names = imagenet_data

        # Choose a model known for good performance
        model_name = "resnet50"
        model = get_model(model_name, pretrained=True)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Compute accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                predictions, _ = model.predict(images)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"\n{model_name} accuracy: {accuracy:.4f} ({correct}/{total})")

        # Accuracy should be better than random guessing by a significant margin
        assert accuracy > 0.1, f"Accuracy of {accuracy} is too low!"

    @pytest.mark.parametrize(
        "model_name",
        ["resnet18", "resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"],
    )
    def test_model_accuracy_comparison(self, imagenet_data, model_name):
        """Test and compare the accuracy of different models."""
        dataloader, class_names = imagenet_data

        model = get_model(model_name, pretrained=True)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Track metrics
        correct = 0
        total = 0
        inference_time = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                # Measure inference time
                start_time = time.time()
                predictions, _ = model.predict(images)
                inference_time += time.time() - start_time

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_inference_time = inference_time / total
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Avg Inference Time: {avg_inference_time*1000:.2f} ms per image")

        # Store results for this model (could save to file for comparison)
        result = {
            "model": model_name,
            "accuracy": accuracy,
            "inference_time": avg_inference_time,
        }

        # All models should be better than random guessing
        assert accuracy > 0.05, f"Accuracy for {model_name} is too low: {accuracy}"

    def test_top5_accuracy(self, imagenet_data):
        """Test the top-5 accuracy of a model."""
        dataloader, class_names = imagenet_data

        model_name = "resnet50"
        model = get_model(model_name, pretrained=True)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Compute top-5 accuracy
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                # Get logits directly from forward pass
                logits = model(images)

                # Calculate top-1 and top-5 accuracy
                _, top1_indices = logits.topk(1, dim=1)
                _, top5_indices = logits.topk(5, dim=1)

                top1_indices = top1_indices.squeeze()

                # Check if true label is in top-1 predictions
                top1_correct += (top1_indices == labels).sum().item()

                # Check if true label is in top-5 predictions
                for i, label in enumerate(labels):
                    if label in top5_indices[i]:
                        top5_correct += 1

                total += labels.size(0)

        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total

        print(f"\n{model_name} results:")
        print(f"  Top-1 accuracy: {top1_accuracy:.4f}")
        print(f"  Top-5 accuracy: {top5_accuracy:.4f}")

        # Top-5 should be better than top-1
        assert top5_accuracy >= top1_accuracy
        # Top-5 should be significantly better than random guessing
        assert top5_accuracy > 0.2

    def test_model_confusion_analysis(self, imagenet_data):
        """Test model performance by analyzing misclassifications."""
        dataloader, class_names = imagenet_data

        model_name = "resnet50"
        model = get_model(model_name, pretrained=True)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Track misclassifications
        misclassifications = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                predictions, probabilities = model.predict(images)

                # Find misclassified samples
                for i in range(images.size(0)):
                    if predictions[i].item() != labels[i].item():
                        true_class = class_names[labels[i].item()]
                        pred_class = class_names[predictions[i].item()]
                        confidence = probabilities[i, predictions[i]].item()

                        misclassifications.append(
                            {
                                "true_class": true_class,
                                "pred_class": pred_class,
                                "confidence": confidence,
                            }
                        )

        # Analyze misclassifications
        if misclassifications:
            # Count how often each class is confused
            confused_pairs = defaultdict(int)
            for m in misclassifications:
                confused_pairs[(m["true_class"], m["pred_class"])] += 1

            # Print the most common confusions
            print("\nMost common confusion pairs:")
            for (true_class, pred_class), count in sorted(
                confused_pairs.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"  {true_class} -> {pred_class}: {count} times")

            # Check confidence of misclassifications
            confidences = [m["confidence"] for m in misclassifications]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average confidence of misclassifications: {avg_confidence:.4f}")

            # Confidences for wrong predictions should generally be lower
            assert avg_confidence < 0.9, "Model is too confident in wrong predictions"


class TestVisualization:
    """Tests for visualizing model performance and comparisons."""

    def test_accuracy_comparison_plot(self):
        """Create a visualization comparing model accuracies."""
        # Example results for demonstration
        # In a real test, these would come from actual evaluations
        results = [
            {"model": "ResNet-18", "accuracy": 0.69, "inference_time": 0.0035},
            {"model": "ResNet-50", "accuracy": 0.76, "inference_time": 0.0047},
            {"model": "VGG-16", "accuracy": 0.71, "inference_time": 0.0055},
            {"model": "EfficientNet-B0", "accuracy": 0.77, "inference_time": 0.0040},
            {"model": "MobileNet-V3", "accuracy": 0.72, "inference_time": 0.0022},
        ]

        # Skip actual plotting for CI environments
        if "CI" in os.environ:
            pytest.skip("Skipping visualization test in CI environment")

        # Create a simple comparison chart
        models = [r["model"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        inf_times = [r["inference_time"] * 1000 for r in results]  # Convert to ms

        x = np.arange(len(models))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        rects1 = ax1.bar(
            x - width / 2, accuracies, width, label="Accuracy", color="blue"
        )
        rects2 = ax2.bar(
            x + width / 2, inf_times, width, label="Inference Time (ms)", color="orange"
        )

        ax1.set_ylabel("Top-1 Accuracy")
        ax2.set_ylabel("Inference Time (ms)")
        ax1.set_title("Model Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        fig.tight_layout()

        # Save the figure for review (if needed for CI)
        os.makedirs("test_outputs", exist_ok=True)
        plt.savefig("test_outputs/model_comparison.png")
        plt.close()

        # The test passes if the visualization code runs without error
        assert True
