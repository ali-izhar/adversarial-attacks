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
import glob
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

            # Create a mapping from class name to index for later use.
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        else:
            raise FileNotFoundError(f"Class names file not found at {class_file}")

    def get_image_paths(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Get a list of (image_path, label) tuples.

        This method looks into a common subdirectory ('sample_images') for images.
        It uses a heuristic to extract labels from filenames, such as filenames that start
        with a digit or a synset ID (e.g., 'n01440764_tench.jpg').

        Args:
            max_samples: If provided, limits the number of samples returned.

        Returns:
            A list of tuples where each tuple contains:
              - image_path (str): The path to the image file.
              - label (int): The associated class label.

        Raises:
            FileNotFoundError: If the image directory doesn't exist.
        """
        # Assume images are stored under a 'sample_images' subdirectory.
        img_dir = os.path.join(self.data_dir, "sample_images")

        if os.path.exists(img_dir):
            images = []
            # Collect images with common file extensions.
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                images.extend(glob.glob(os.path.join(img_dir, ext)))
            # Sort to ensure reproducibility.
            images.sort()

            image_paths = []
            for img_path in images:
                filename = os.path.basename(img_path)

                # Try to extract a label from the filename.
                if "_" in filename:
                    prefix = filename.split("_")[0]

                    if prefix.isdigit():
                        # If the prefix is a number, use it as the label.
                        label = int(prefix)
                    elif prefix.startswith("n") and prefix[1:].isdigit():
                        # For synset IDs like 'n01440764', attempt to match with class names.
                        # This is a heuristic: we loop over known class names to see if one is present.
                        class_name = None
                        for name in self.class_names:
                            if name in filename:
                                class_name = name
                                break
                        # Fallback to label 0 if no match is found.
                        label = (
                            self.class_to_idx.get(class_name, 0)
                            if class_name is not None
                            else 0
                        )
                    else:
                        # Default to label 0 if the prefix cannot be parsed.
                        label = 0
                else:
                    # No separator found; assign default label 0.
                    label = 0

                image_paths.append((img_path, label))

            # Validate there's at least one image
            if not image_paths:
                print(f"Warning: No images found in directory {img_dir}")
        else:
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        # If max_samples is specified, truncate the list.
        if max_samples is not None and max_samples < len(image_paths):
            image_paths = image_paths[:max_samples]

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
