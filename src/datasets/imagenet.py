"""
Dataset loading utilities for adversarial attack evaluation.

This module provides classes and functions for loading ImageNet-style data with proper normalization.

INPUT/OUTPUT SPECIFICATIONS:
- Input: Raw image files (jpg, jpeg, png) from disk (data/imagenet/sample_images/)
- Output: Normalized tensors in the range [-2.64, 2.64] after ImageNet normalization
         with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

IMPORTANT: Images from this dataloader are ALREADY NORMALIZED for ImageNet models.
"""

import os
from typing import Tuple, Optional, List, Callable, Dict

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError


class ImageNetDataset(Dataset):
    """
    Dataset for loading ImageNet-style data.

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
        """
        Initialize the ImageNet dataset.

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
        """
        Load class names from imagenet_classes.txt and handle duplicates.

        ImageNet-1k has some duplicate class names (like 'crane' and 'maillot'),
        where multiple synset IDs map to the same class name.
        """
        class_file = os.path.join(self.data_dir, "imagenet_classes.txt")

        if not os.path.exists(class_file):
            raise FileNotFoundError(f"Class names file not found at {class_file}")

        with open(class_file, "r") as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        if not self.class_names:
            raise ValueError(f"No class names found in {class_file}")

        # Find duplicate class names and their indices
        name_to_indices: Dict[str, List[int]] = {}
        for idx, name in enumerate(self.class_names):
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(idx)

        # Report duplicates
        self.duplicate_classes = {
            name: indices
            for name, indices in name_to_indices.items()
            if len(indices) > 1
        }
        if self.duplicate_classes:
            print("Warning: Found duplicate class names:")
            for name, indices in self.duplicate_classes.items():
                print(f"  '{name}' appears at indices: {indices}")

        # Create mapping for class name to indices (handling duplicates)
        self.class_to_indices = name_to_indices

        # Standard mapping for synset ID to class index
        # This is particularly important for duplicated class names
        self.synset_to_idx = {}

        # Known mappings for some common synset IDs
        # Add specific mappings for duplicated classes
        self.synset_to_idx = {
            # Specifically handle the "crane" duplicates
            "n01531178": 134,  # Crane (bird)
            "n03126707": 517,  # Crane (construction)
            # Specifically handle the "maillot" duplicates
            "n03710637": 638,  # Maillot (swimsuit)
            "n03710721": 639,  # Maillot (tank suit)
        }

    def get_image_paths(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Get list of (image_path, label) tuples from the sample_images directory.

        Handles the case where different synset IDs map to the same class name,
        by using synset ID to determine the correct class index.
        """
        img_dir = os.path.join(self.data_dir, "sample_images")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        image_paths = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")

        for filename in sorted(os.listdir(img_dir)):
            if not any(
                filename.lower().endswith(ext.lower()) for ext in valid_extensions
            ):
                continue

            filepath = os.path.join(img_dir, filename)
            label = None

            # First, try to extract synset ID from filename (format: nXXXXXXXX_classname.JPEG)
            if "_" in filename:
                synset_id = filename.split("_")[0]

                # If synset ID is in our known mapping, use it directly
                if synset_id in self.synset_to_idx:
                    label = self.synset_to_idx[synset_id]
                    image_paths.append((filepath, label))
                    continue

                # Extract the class name part to match with our class list
                class_name_part = filename.split("_", 1)[1]
                if "." in class_name_part:
                    class_name_part = class_name_part.split(".")[0]

                # Match class name, properly handling duplicates
                matched_name = None
                for name in self.class_to_indices:
                    if name.lower() == class_name_part.lower():
                        matched_name = name
                        break

                if matched_name:
                    indices = self.class_to_indices[matched_name]

                    # If duplicate class names exist, handle specially
                    if len(indices) > 1:
                        # For special cases, try to disambiguate using synset ID
                        if synset_id and matched_name in self.duplicate_classes:
                            if matched_name == "crane":
                                # Bird crane has synset starting with "n015..."
                                if synset_id.startswith("n015"):
                                    label = 134  # Bird crane
                                else:
                                    label = 517  # Construction crane
                            elif matched_name == "maillot":
                                # Different types of maillot
                                if synset_id == "n03710637":
                                    label = 638
                                else:
                                    label = 639
                    else:
                        # Non-duplicate case, just use the single index
                        label = indices[0]

                    if label is not None:
                        image_paths.append((filepath, label))

        print(f"Found {len(image_paths)} valid images")

        # Limit to max_samples if specified
        if max_samples is not None and max_samples < len(image_paths):
            image_paths = image_paths[:max_samples]
            print(f"Using {max_samples} samples as requested")

        return image_paths

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve the image and its label at a given index.

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
    """
    Create and return an ImageNet dataset.

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
    """
    Create a DataLoader for the provided dataset.

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
