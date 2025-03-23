#!/usr/bin/env python
"""
Script to analyze the ImageNet dataset and verify its integrity.

This script performs several checks on the ImageNet dataset:
1. Dataset statistics (total samples, class distribution)
2. Image format validation
3. Label extraction validation
4. Sample image visualization with assigned labels
5. Normalization check
"""

import os
import sys
import argparse
from collections import Counter, defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset


def analyze_class_distribution(dataset):
    """Analyze the distribution of classes in the dataset."""
    labels = [label for _, label in dataset.image_paths]
    class_counts = Counter(labels)

    # Get class names for each index
    class_names = {idx: name for idx, name in enumerate(dataset.class_names)}

    # Calculate statistics
    total_samples = len(labels)
    num_classes_with_samples = len(class_counts)
    min_class_count = min(class_counts.values()) if class_counts else 0
    max_class_count = max(class_counts.values()) if class_counts else 0

    print(f"Dataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(
        f"  Number of classes with samples: {num_classes_with_samples}/{len(dataset.class_names)}"
    )
    print(f"  Min samples per class: {min_class_count}")
    print(f"  Max samples per class: {max_class_count}")

    # Print most common classes
    print("\nMost Common Classes:")
    for class_idx, count in class_counts.most_common(10):
        class_name = class_names.get(class_idx, f"Unknown (ID: {class_idx})")
        print(f"  {class_name}: {count} samples")

    # Print classes with no samples
    missing_classes = set(range(len(dataset.class_names))) - set(class_counts.keys())
    if missing_classes:
        print(f"\n{len(missing_classes)} classes have no samples. First 10:")
        for idx in sorted(list(missing_classes))[:10]:
            print(f"  {class_names.get(idx, f'Unknown (ID: {idx})')}")

    # Create a histogram of class frequencies
    plt.figure(figsize=(12, 6))
    plt.hist(list(class_counts.values()), bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Samples per Class")
    plt.xlabel("Number of Samples")
    plt.ylabel("Number of Classes")
    plt.grid(axis="y", alpha=0.75)

    # Save the plot
    os.makedirs("analysis_results", exist_ok=True)
    plt.savefig("analysis_results/class_distribution.png")
    plt.close()

    return class_counts


def check_image_formats(dataset):
    """Check the formats and dimensions of images in the dataset."""
    format_counts = Counter()
    dimensions = []
    corrupt_count = 0

    print("\nChecking image formats...")

    # Sample up to 100 random images
    sample_size = min(100, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)

    for idx in sample_indices:
        img_path, _ = dataset.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                format_counts[img.format] += 1
                dimensions.append(img.size)
        except Exception as e:
            print(f"  Error opening image {img_path}: {e}")
            corrupt_count += 1

    print(f"Image Formats (from {sample_size} samples):")
    for format_name, count in format_counts.most_common():
        print(f"  {format_name}: {count} ({count/sample_size*100:.1f}%)")

    if dimensions:
        widths, heights = zip(*dimensions)
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        print(f"\nAverage dimensions: {avg_width:.1f} x {avg_height:.1f}")
        print(f"Min dimensions: {min(widths)} x {min(heights)}")
        print(f"Max dimensions: {max(widths)} x {max(heights)}")

    if corrupt_count > 0:
        print(f"\nWARNING: Found {corrupt_count} corrupt images!")

    return format_counts, dimensions, corrupt_count


def validate_label_extraction(dataset):
    """Validate the label extraction logic by examining filenames and assigned labels."""
    print("\nValidating label extraction...")

    # Group files by assigned label
    files_by_label = defaultdict(list)
    for path, label in dataset.image_paths:
        files_by_label[label].append(os.path.basename(path))

    # Check raw filenames to see what's happening
    print("\nExamining 10 random filenames and their extracted labels:")
    sample_paths = random.sample(dataset.image_paths, min(10, len(dataset.image_paths)))

    for path, label in sample_paths:
        filename = os.path.basename(path)

        # Extract what should be the class name
        if "_" in filename:
            expected_class = filename.split("_", 1)[1]
            if "." in expected_class:
                expected_class = expected_class.split(".")[0]

            # Get the synset ID
            synset_id = filename.split("_")[0]

            # Get the actual class name from the assigned label
            actual_class = (
                dataset.class_names[label]
                if label < len(dataset.class_names)
                else "Unknown"
            )

            print(f"  Filename: {filename}")
            print(f"    Synset: {synset_id}")
            print(f"    Extracted class name: '{expected_class}'")
            print(f"    Assigned label: {label} ('{actual_class}')")

            # Check if there's a mismatch
            if expected_class != actual_class:
                print(
                    f"    ⚠️ MISMATCH! Expected '{expected_class}', got '{actual_class}'"
                )

            # Check if expected_class exists in class names
            if expected_class in dataset.class_names:
                expected_idx = dataset.class_names.index(expected_class)
                if expected_idx != label:
                    print(
                        f"    ⚠️ INCORRECT INDEX! '{expected_class}' should be index {expected_idx}"
                    )
            else:
                print(
                    f"    ⚠️ WARNING: '{expected_class}' not found in class names list"
                )

            print()

    # Analyze filename patterns for each class as before
    pattern_analysis = {}

    for label, filenames in files_by_label.items():
        if len(filenames) == 0:
            continue

        # Check for numeric prefixes
        numeric_prefixes = [
            fn.split("_")[0]
            for fn in filenames
            if "_" in fn and fn.split("_")[0].isdigit()
        ]

        # Check for synset IDs
        synset_prefixes = [
            fn.split("_")[0]
            for fn in filenames
            if "_" in fn
            and fn.split("_")[0].startswith("n")
            and fn.split("_")[0][1:].isdigit()
        ]

        pattern_analysis[label] = {
            "total_files": len(filenames),
            "numeric_prefix_count": len(numeric_prefixes),
            "synset_prefix_count": len(synset_prefixes),
            "numeric_prefixes": Counter(numeric_prefixes).most_common(3),
            "synset_prefixes": Counter(synset_prefixes).most_common(3),
        }

    # Print analysis of a few classes
    print("Label extraction analysis for first 5 classes with samples:")
    for i, (label, analysis) in enumerate(pattern_analysis.items()):
        if i >= 5:
            break

        class_name = (
            dataset.class_names[label]
            if label < len(dataset.class_names)
            else "Unknown"
        )
        print(f"\nClass {label} ({class_name}):")
        print(f"  Total files: {analysis['total_files']}")
        print(f"  Files with numeric prefixes: {analysis['numeric_prefix_count']}")
        print(f"  Files with synset prefixes: {analysis['synset_prefix_count']}")

        if analysis["numeric_prefixes"]:
            print(f"  Most common numeric prefixes: {analysis['numeric_prefixes']}")
        if analysis["synset_prefixes"]:
            print(f"  Most common synset prefixes: {analysis['synset_prefixes']}")

        # Show some sample filenames
        if len(files_by_label[label]) > 0:
            sample_names = random.sample(
                files_by_label[label], min(3, len(files_by_label[label]))
            )
            print(f"  Sample filenames: {sample_names}")

    return pattern_analysis


def visualize_samples(dataset, num_samples=5):
    """Visualize random samples from the dataset with their labels."""
    print("\nVisualizing random samples...")

    # Use the inverse normalization to display images correctly
    inv_normalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            )
        ]
    )

    # Sample random indices, with preference for classes that have few samples
    label_counts = Counter([label for _, label in dataset.image_paths])
    rare_classes = [label for label, count in label_counts.items() if count < 5]

    # If we have rare classes, include them in the visualization
    indices = []
    if rare_classes:
        rare_indices = []
        for idx, (_, label) in enumerate(dataset.image_paths):
            if label in rare_classes:
                rare_indices.append(idx)
                if len(rare_indices) >= num_samples // 2:
                    break
        indices.extend(rare_indices)

    # Add random indices to reach num_samples
    remaining = num_samples - len(indices)
    if remaining > 0:
        available = set(range(len(dataset))) - set(indices)
        indices.extend(random.sample(list(available), min(remaining, len(available))))

    # Create figure for visualization
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        # Get image and label
        image, label = dataset[idx]

        # Get original filename
        orig_path, _ = dataset.image_paths[idx]
        filename = os.path.basename(orig_path)

        # If the image is from the default transformation, denormalize it
        if hasattr(dataset, "transform") and dataset.transform is not None:
            try:
                image = inv_normalize(image)
            except:
                pass  # Skip if error (might not be normalized)

        # Convert to numpy for display
        img_np = image.permute(1, 2, 0).cpu().numpy()

        # Clip values to [0, 1] range
        img_np = np.clip(img_np, 0, 1)

        # Display image
        axes[i].imshow(img_np)

        # Get class name
        class_name = (
            dataset.class_names[label]
            if label < len(dataset.class_names)
            else "Unknown"
        )

        # Add title with filename and class
        axes[i].set_title(f"{filename}\nClass: {class_name} (ID: {label})", fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("analysis_results/sample_images.png", dpi=150)
    plt.close()


def check_normalization(dataset):
    """Check if images are properly normalized."""
    print("\nChecking image normalization...")

    # Sample a few images
    samples = min(10, len(dataset))
    means = []
    stds = []

    for i in range(samples):
        img, _ = dataset[i]
        means.append(img.mean(dim=[1, 2]).numpy())
        stds.append(img.std(dim=[1, 2]).numpy())

    avg_mean = np.mean(means, axis=0)
    avg_std = np.mean(stds, axis=0)

    print(
        f"Average RGB mean: [{avg_mean[0]:.4f}, {avg_mean[1]:.4f}, {avg_mean[2]:.4f}]"
    )
    print(f"Average RGB std: [{avg_std[0]:.4f}, {avg_std[1]:.4f}, {avg_std[2]:.4f}]")

    # Check if images are likely to be normalized
    expected_mean = [0, 0, 0]  # Approximately 0 if normalized
    expected_std = [1, 1, 1]  # Approximately 1 if normalized

    # Calculate mean absolute difference
    mean_diff = np.mean(np.abs(avg_mean - expected_mean))
    std_diff = np.mean(np.abs(avg_std - expected_std))

    if mean_diff < 0.5 and std_diff < 0.5:
        print("✓ Images appear to be normalized (mean ≈ 0, std ≈ 1)")
    else:
        # Check if they match ImageNet normalization values
        raw_mean = [0.485, 0.456, 0.406]
        raw_std = [0.229, 0.224, 0.225]

        if (
            np.mean(np.abs(avg_mean - raw_mean)) < 0.1
            and np.mean(np.abs(avg_std - raw_std)) < 0.1
        ):
            print("⚠️ Images appear to be in raw format without normalization")
        else:
            print(
                "⚠️ Image normalization is unclear - values don't match expected patterns"
            )

    return avg_mean, avg_std


def check_class_file(data_dir):
    """Check the imagenet_classes.txt file for issues."""
    print("\nAnalyzing class file...")

    class_file = os.path.join(data_dir, "imagenet", "imagenet_classes.txt")
    if not os.path.exists(class_file):
        print(f"Error: Class file not found at {class_file}")
        return

    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    # Check for duplicate class names
    name_to_indices = {}
    for i, name in enumerate(class_names):
        if name in name_to_indices:
            name_to_indices[name].append(i)
        else:
            name_to_indices[name] = [i]

    duplicates = {
        name: indices for name, indices in name_to_indices.items() if len(indices) > 1
    }

    if duplicates:
        print(f"Found {len(duplicates)} duplicate class names:")
        for name, indices in duplicates.items():
            print(f"  '{name}' appears at indices: {indices} (0-indexed)")
            # Check if we have filename mapping for this
            for i, idx in enumerate(indices):
                print(f"    Need to map image for '{name}' #{i+1} to index {idx}")
    else:
        print("No duplicate class names found - class file looks good!")

    # Check for unusual class names
    unusual_names = []
    for name in class_names:
        # Check for trailing spaces
        if name != name.strip():
            unusual_names.append((name, "has trailing/leading spaces"))
        # Check for unusual characters
        elif any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ "
            for c in name
        ):
            unusual_names.append((name, "has unusual characters"))

    if unusual_names:
        print("\nFound unusual class names:")
        for name, reason in unusual_names:
            print(f"  '{name}': {reason}")

    return class_names


def main():
    """Main function to analyze the dataset."""
    parser = argparse.ArgumentParser(description="Analyze ImageNet dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing ImageNet data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (None for all)",
    )
    args = parser.parse_args()

    print(f"Analyzing ImageNet dataset in {args.data_dir}...")

    # First check the class file
    check_class_file(args.data_dir)

    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)

    # Load dataset without normalization for analysis
    raw_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    try:
        dataset = get_dataset(
            "imagenet",
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            transform=raw_transform,
        )

        # Perform the analysis
        class_counts = analyze_class_distribution(dataset)
        format_counts, dimensions, corrupt_count = check_image_formats(dataset)
        pattern_analysis = validate_label_extraction(dataset)
        visualize_samples(dataset)

        # Now check normalization with the default transform
        print("\nChecking with default transform (including normalization)...")
        default_dataset = get_dataset(
            "imagenet", data_dir=args.data_dir, max_samples=args.max_samples
        )
        avg_mean, avg_std = check_normalization(default_dataset)

        # Print summary
        print("\n===== Analysis Summary =====")
        print(f"Total images: {len(dataset)}")
        print(f"Classes with samples: {len(class_counts)}/{len(dataset.class_names)}")
        print(f"Potentially problematic images: {corrupt_count}")

        # Save some metadata
        with open("analysis_results/dataset_info.txt", "w") as f:
            f.write(f"Dataset path: {args.data_dir}\n")
            f.write(f"Total images: {len(dataset)}\n")
            f.write(f"Number of classes: {len(dataset.class_names)}\n")
            f.write(f"Classes with samples: {len(class_counts)}\n")
            f.write(
                f"Average RGB mean: [{avg_mean[0]:.4f}, {avg_mean[1]:.4f}, {avg_mean[2]:.4f}]\n"
            )
            f.write(
                f"Average RGB std: [{avg_std[0]:.4f}, {avg_std[1]:.4f}, {avg_std[2]:.4f}]\n"
            )

        print(f"\nAnalysis results saved to the 'analysis_results' directory")

    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
