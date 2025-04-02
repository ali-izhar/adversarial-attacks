"""
Dataset loading utilities for adversarial attack evaluation.

This module provides classes and functions for loading and processing image datasets
to be used in adversarial attack experiments. It includes:

- ImageNetDataset: Custom dataset class for ImageNet-style data.
- get_dataset: Factory function to create datasets by name.
- get_dataloader: Utility to create PyTorch DataLoaders.

The implementation focuses on flexibility and proper handling of image data
with appropriate transformations for model input.
"""

import os
from typing import Tuple, Optional, List, Callable

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError


class ImageNetDataset(Dataset):
    """
    Dataset for loading ImageNet-style data.

    This dataset expects:
      1. A directory of images (assumed to be under 'sample_images' subfolder).
      2. A text file 'imagenet_classes.txt' that maps class indices to class names.

    It applies preprocessing transformations and uses heuristics to extract labels from filenames.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the ImageNet dataset.

        Args:
            data_dir: Base directory for the dataset (e.g., 'data/imagenet/').
            transform: Transformation to apply to images (if None, a default transform is used).
            max_samples: Maximum number of samples to load (useful for quick tests).

        Raises:
            FileNotFoundError: If required files/directories are not found.
            IndexError: If the class names file is empty.
            ValueError: If other validation issues occur.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load class names and mapping from class names to indices.
        self.load_class_names()

        # Get list of image file paths and associated labels.
        self.image_paths = self.get_image_paths(max_samples)

        # Use a default set of transformations if none is provided.
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # ImageNet normalization
                ]
            )

    def load_class_names(self):
        """
        Load class names from a file 'imagenet_classes.txt' located in the data directory.

        Raises:
            FileNotFoundError: If the class file doesn't exist.
            IndexError: If the class file is empty.
        """
        class_file = os.path.join(self.data_dir, "imagenet_classes.txt")

        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                # Each line is expected to contain one class name.
                self.class_names = [
                    line.strip() for line in f.readlines() if line.strip()
                ]

            # Check if the class file is empty
            if not self.class_names:
                raise IndexError(f"Class names file '{class_file}' is empty.")

            # Check for duplicate class names and warn
            name_counts = {}
            for i, name in enumerate(self.class_names):
                if name in name_counts:
                    name_counts[name].append(i)
                else:
                    name_counts[name] = [i]

            duplicates = {
                name: indices
                for name, indices in name_counts.items()
                if len(indices) > 1
            }
            if duplicates:
                print(f"Warning: Found {len(duplicates)} duplicate class names:")
                for name, indices in duplicates.items():
                    print(f"  '{name}' appears at indices: {indices}")

            # Create a mapping from class name to index for later use.
            # For duplicates, use the first occurrence
            self.class_to_idx = {}
            for i, name in enumerate(self.class_names):
                if name not in self.class_to_idx:
                    self.class_to_idx[name] = i
        else:
            raise FileNotFoundError(f"Class names file not found at {class_file}")

    def get_image_paths(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Get a list of (image_path, label) tuples.

        This method looks into the 'sample_images' subdirectory for files named with
        ImageNet synset IDs like 'n01440764_tench.JPEG' and maps them to their correct
        ImageNet class index.

        Args:
            max_samples: If provided, limits the number of samples returned.

        Returns:
            A list of tuples where each tuple contains:
              - image_path (str): The path to the image file.
              - label (int): The associated class label (0-999 for ImageNet-1k).
        """
        # Look in the 'sample_images' subdirectory
        img_dir = os.path.join(self.data_dir, "sample_images")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        # ImageNet synset ID to class index mapping
        # These are the standard synset IDs used in the ILSVRC2012 dataset
        synset_to_class = {
            "n01440764": 0,  # tench
            "n01443537": 1,  # goldfish
            "n01484850": 2,  # great white shark
            "n01491361": 3,  # tiger shark
            "n01494475": 4,  # hammerhead shark
            "n01496331": 5,  # electric ray
            # ... and so on for all 1000 classes
        }

        # If a synset is not found in the mapping, we'll try to infer it from class_names
        if not synset_to_class:
            print("Warning: No pre-defined synset-to-class mapping available.")
            print(
                "Will attempt to match filenames to class names in imagenet_classes.txt"
            )

        # Collect image files with valid extensions
        image_files = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
        for filename in os.listdir(img_dir):
            if any(filename.lower().endswith(ext.lower()) for ext in valid_extensions):
                image_files.append(os.path.join(img_dir, filename))

        # Sort for reproducibility
        image_files.sort()

        # Parse filenames and match to class indices
        image_paths = []
        assigned_classes = set()

        # Keep track of images by class for balanced sampling
        images_by_class = {}

        for img_path in image_files:
            filename = os.path.basename(img_path)
            label = None

            # Try to extract synset ID from filename (format: n01440764_tench.JPEG)
            # First look for the standard n########_ pattern
            synset_match = filename.split("_")[0] if "_" in filename else None

            if synset_match and synset_match in synset_to_class:
                # Directly map using known synset-to-class mapping
                label = synset_to_class[synset_match]
            else:
                # Try to extract class name from filename after the underscore
                if "_" in filename:
                    class_name_part = filename.split("_", 1)[1]
                    # Remove file extension if present
                    if "." in class_name_part:
                        class_name_part = class_name_part.split(".")[0]

                    # Try to match to a class name
                    for idx, name in enumerate(self.class_names):
                        # Try different matching approaches
                        if (
                            name.lower() == class_name_part.lower()
                            or name.lower() == class_name_part.lower().replace("_", " ")
                        ):
                            label = idx
                            break

                # If still no match, try to use numeric part of filename if present
                if label is None:
                    # For files like: "0_tench.JPEG" - extract the numeric prefix
                    if filename[0].isdigit() and "_" in filename:
                        try:
                            num_prefix = int(filename.split("_")[0])
                            if 0 <= num_prefix < len(self.class_names):
                                label = num_prefix
                        except ValueError:
                            pass

            # If we found a valid label, add the image to our dataset
            if label is not None:
                if label not in images_by_class:
                    images_by_class[label] = []
                images_by_class[label].append((img_path, label))
                assigned_classes.add(label)

        # Verify we have images
        total_images = sum(len(images) for images in images_by_class.values())
        if total_images == 0:
            print(f"Warning: No valid images found in directory {img_dir}")
            return []

        print(
            f"Found {total_images} valid images across {len(assigned_classes)} classes"
        )

        # If max_samples is specified, select a balanced subset
        if max_samples is not None and max_samples < total_images:
            # Try to get samples from the first max_samples classes
            # for better compatibility with our IMAGENET_SAMPLE_CLASSES mapping
            selected_images = []

            # First, try to select one image from each class in order
            for class_idx in range(min(max_samples, len(self.class_names))):
                if class_idx in images_by_class and images_by_class[class_idx]:
                    selected_images.append(images_by_class[class_idx][0])

                    if len(selected_images) >= max_samples:
                        break

            # If we still need more samples, take from available classes
            if len(selected_images) < max_samples:
                # Get list of all classes that have images
                available_classes = list(images_by_class.keys())
                available_classes.sort()  # Sort for reproducibility

                for class_idx in available_classes:
                    class_images = images_by_class[class_idx]

                    # Skip if we already took the first image from this class
                    start_idx = (
                        1 if class_idx < min(max_samples, len(self.class_names)) else 0
                    )

                    # Add remaining images from this class
                    for img_idx in range(start_idx, len(class_images)):
                        selected_images.append(class_images[img_idx])
                        if len(selected_images) >= max_samples:
                            break

                    if len(selected_images) >= max_samples:
                        break

            image_paths = selected_images[:max_samples]
        else:
            # If no max_samples or max_samples >= total_images, include all images
            image_paths = []
            for images in images_by_class.values():
                image_paths.extend(images)

        # Print distribution of classes in the final dataset
        class_counts = {}
        for _, label in image_paths:
            class_counts[label] = class_counts.get(label, 0) + 1

        print(f"Final dataset has {len(image_paths)} images with class distribution:")
        # Print top 5 most frequent classes
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for class_idx, count in top_classes:
            print(
                f"  Class {class_idx} ({self.class_names[class_idx]}): {count} images"
            )

        return image_paths

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve the image and its label at a given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (image, label) where:
              - image is a transformed tensor.
              - label is an integer representing the class.

        Note:
            For corrupt images, we return a black tensor with the top-left
            pixel set to a special value (-1,-1,-1) to mark it as corrupt.
        """
        img_path, label = self.image_paths[idx]
        try:
            # Open the image and convert to RGB.
            img = Image.open(img_path).convert("RGB")
            # Apply the specified transformation.
            if self.transform:
                img = self.transform(img)
            return img, label
        except (UnidentifiedImageError, OSError, IOError) as e:
            # More specific handling of image loading failures
            print(f"Error loading image {img_path}: {e}")
            # Create a black image with the correct dimensions based on the transform
            if self.transform:
                # Use a simple black PIL image that will work with our transforms
                temp_img = Image.new("RGB", (224, 224), color=0)
                img_tensor = self.transform(temp_img)
                # Mark this as corrupt by setting a special pixel pattern
                # Set the top-left pixel to a special value (-1,-1,-1)
                if img_tensor.shape[1] > 0 and img_tensor.shape[2] > 0:
                    img_tensor[:, 0, 0] = -1.0
                return img_tensor, label
            else:
                # Default size if no transform is specified
                img_tensor = torch.zeros((3, 224, 224))
                # Mark as corrupt with special pixel
                img_tensor[:, 0, 0] = -1.0
                return img_tensor, label
        except Exception as e:
            # Catch any other exceptions
            print(f"Unexpected error loading image {img_path}: {e}")
            # Return a placeholder image (black image) if loading fails.
            img_tensor = torch.zeros((3, 224, 224))
            # Mark as corrupt with special pixel
            img_tensor[:, 0, 0] = -1.0
            return img_tensor, label


def get_dataset(
    dataset_name: str,
    data_dir: str = "data",
    split: str = "val",
    transform: Optional[Callable] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Create and return a dataset by name.

    This factory function supports various dataset names. Currently, it supports 'imagenet'.
    The function checks that the dataset directory exists and applies any provided transformations.

    Args:
        dataset_name: Name of the dataset (e.g., 'imagenet', 'cifar10', etc.).
        data_dir: Base directory where datasets are stored.
        split: Data split to use ('train', 'val', 'test'); currently unused for ImageNet.
        transform: Optional transformation to apply to the images.
        max_samples: Optional limit on the number of samples to load.

    Returns:
        A PyTorch Dataset object corresponding to the requested dataset.

    Raises:
        ValueError: If the requested dataset is not supported.
        FileNotFoundError: If the dataset directory cannot be found.
    """
    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if dataset_name.lower() == "imagenet":
        return ImageNetDataset(dataset_dir, transform, max_samples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the provided dataset.

    This function wraps the dataset in a DataLoader, which provides convenient batch iteration,
    shuffling, and multi-process data loading.

    Args:
        dataset: The dataset from which to load data.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        A PyTorch DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Use pinned memory if CUDA is available.
    )
