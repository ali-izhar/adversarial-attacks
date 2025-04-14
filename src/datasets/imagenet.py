#!/usr/bin/env python
"""Dataset loading utilities for adversarial attack evaluation.

USAGE::
    >>> from src.datasets.imagenet import get_dataset
    >>> dataset = get_dataset("imagenet")
    >>> dataloader = get_dataloader(dataset)

This module provides classes and functions for loading ImageNet-style data with proper normalization.

INPUT/OUTPUT SPECIFICATIONS:
- Input: Raw image files (jpg, jpeg, png) from disk (data/imagenet/sample_images/)
- Output: Normalized tensors in the range [-2.64, 2.64] after ImageNet normalization
         with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

IMPORTANT: Images from this dataloader are ALREADY NORMALIZED for ImageNet models.
"""

import os
from typing import Tuple, Optional, List, Callable

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError


class ImageNetDataset(Dataset):
    """Dataset for loading ImageNet-style data.

    Expected directory structure:
    - data_dir/sample_images/: Contains image files named as "nXXXXXXXX_classname.JPEG"
    - data_dir/imagenet_classes.txt: Text file mapping class indices to names

    INPUT/OUTPUT SPECIFICATIONS:
    - Input: Image files on disk (data/imagenet/sample_images/)
    - Output: Normalized tensors of shape [3, 224, 224] in range [-2.64, 2.64]
              and class indices [0-999] for ImageNet
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize the ImageNet dataset.

        Args:
            data_dir: Base directory for the dataset (e.g., 'data/imagenet/').
            transform: Transformation to apply to images (if None, a default transform is used).
            max_samples: Optional limit on the number of samples to load.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load class names from text file
        self.load_class_names()

        # Get list of image paths and associated labels
        self.image_paths = self.get_image_paths(max_samples)

        # Use a default set of transformations if none is provided
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
        """Load class names from imagenet_classes.txt."""
        class_file = os.path.join(self.data_dir, "imagenet_classes.txt")

        if not os.path.exists(class_file):
            raise FileNotFoundError(f"Class names file not found at {class_file}")

        with open(class_file, "r") as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        if not self.class_names:
            raise ValueError(f"No class names found in {class_file}")

        # Known synset mappings for special cases
        self.synset_to_idx = {
            # Animals and birds
            "n01531178": 11,  # goldfinch
            "n02113186": 264,  # Cardigan Welsh Corgi
            # Objects and clothing
            "n02963159": 474,  # cardigan (sweater)
            "n03126707": 517,  # crane (construction)
            "n03710637": 638,  # maillot (swimsuit)
            "n03710721": 639,  # maillot (tank suit)
        }

    def normalize_class_name(self, name: str) -> str:
        """Normalize class name by removing spaces, hyphens, and special characters."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("'", "").strip()

    def get_image_paths(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """Get list of (image_path, label) tuples from the sample_images directory.

        Args:
            max_samples: Optional limit on number of samples to return.

        Returns:
            List of tuples containing (image_path, class_index).
        """
        img_dir = os.path.join(self.data_dir, "sample_images")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        # Create mapping of normalized class names to indices
        class_to_idx = {
            self.normalize_class_name(name): idx
            for idx, name in enumerate(self.class_names)
        }

        image_paths = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")

        for filename in sorted(os.listdir(img_dir)):
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                continue

            filepath = os.path.join(img_dir, filename)

            # Extract synset ID and class name
            if "_" not in filename:
                continue

            synset_id = filename.split("_")[0]
            class_name = filename.split("_", 1)[1].split(".")[0]

            # First try synset mapping for special cases
            if synset_id in self.synset_to_idx:
                label = self.synset_to_idx[synset_id]
                image_paths.append((filepath, label))
                continue

            # Try exact match with normalized class name
            normalized_class = self.normalize_class_name(class_name)
            if normalized_class in class_to_idx:
                label = class_to_idx[normalized_class]
                image_paths.append((filepath, label))

        print(f"\nFound {len(image_paths)} valid images")

        # Limit to max_samples if specified
        if max_samples is not None and max_samples < len(image_paths):
            image_paths = image_paths[:max_samples]
            print(f"Using {max_samples} samples as requested")

        return image_paths

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retrieve the image and its label at a given index.

        Returns:
            Tuple of (image, label) where:
              - image is a normalized tensor of shape [3, 224, 224]
              - label is an integer class index [0-999]
        """
        img_path, label = self.image_paths[idx]

        try:
            # Open the image and apply transformations
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

        except (UnidentifiedImageError, OSError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image with correct dimensions
            img_tensor = torch.zeros((3, 224, 224))
            return img_tensor, label


def get_dataset(
    dataset_name: str,
    data_dir: str = "data",
    split: str = "val",
    transform: Optional[Callable] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Create and return an ImageNet dataset.

    Args:
        dataset_name: Name of the dataset (must be 'imagenet').
        data_dir: Base directory where datasets are stored.
        split: Data split to use (unused, kept for API compatibility).
        transform: Optional transformation to apply to the images.
        max_samples: Optional limit on the number of samples to load.

    Returns:
        A PyTorch Dataset that outputs normalized tensors.
    """
    if dataset_name.lower() != "imagenet":
        raise ValueError(f"Only 'imagenet' dataset is supported, got: {dataset_name}")

    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return ImageNetDataset(dataset_dir, transform, max_samples)


def get_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the provided dataset.

    Args:
        dataset: The dataset from which to load data.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses for data loading.

    Returns:
        A PyTorch DataLoader providing normalized image batches.
        Each batch contains:
        - images: Tensor [batch_size, 3, 224, 224] with values in range [-2.64, 2.64]
        - labels: Tensor [batch_size] with class indices [0-999]
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
