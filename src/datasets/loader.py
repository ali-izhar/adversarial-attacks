"""Dataset loaders for adversarial attack evaluation."""

import os
import glob
from typing import Tuple, Optional, List, Callable

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageNetDataset(Dataset):
    """
    Dataset for loading ImageNet-style data.

    This dataset is designed to work with:
    1. A directory of images
    2. A text file mapping class indices to class names
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
            data_dir: Base directory for the dataset (e.g., 'data/imagenet/')
            transform: Transformation to apply to images
            max_samples: Maximum number of samples to load (useful for testing)
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load class names
        self.load_class_names()

        # Load image paths
        self.image_paths = self.get_image_paths(max_samples)

        # Default preprocessing if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )

    def load_class_names(self):
        """Load class names from file."""
        # Try to find the class names file
        class_file = os.path.join(self.data_dir, "imagenet_classes.txt")

        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        else:
            raise FileNotFoundError(f"Class names file not found at {class_file}")

    def get_image_paths(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Get list of (image_path, label) tuples.

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            List of (image_path, label) tuples
        """
        # Try common directory structures
        img_dir = os.path.join(self.data_dir, "sample_images")

        if os.path.exists(img_dir):
            # Sample images directory exists - assume all images are labeled with filename format
            images = []
            # Search for common image formats
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                images.extend(glob.glob(os.path.join(img_dir, ext)))

            # Sort for reproducibility
            images.sort()

            # Try to extract labels from filenames if they include class information
            # Fallback to assigning class 0 if unable to determine
            image_paths = []

            # Check if the image files have class indices in their names
            for img_path in images:
                filename = os.path.basename(img_path)

                # Look for class index in the filename (e.g., "n01440764_tench.jpg" or "0_tench.jpg")
                # This is a heuristic and might need to be adjusted based on your actual naming convention
                if "_" in filename:
                    prefix = filename.split("_")[0]

                    # Try to match with class index (number) or class ID (starting with 'n')
                    if prefix.isdigit():
                        label = int(prefix)
                    elif prefix.startswith("n") and prefix[1:].isdigit():
                        # Extract index for synset IDs like 'n01440764'
                        class_name = self.class_names[0]  # Fallback
                        for i, name in enumerate(self.class_names):
                            if name in filename:
                                class_name = name
                                break
                        label = self.class_to_idx.get(class_name, 0)
                    else:
                        # Use first class as default
                        label = 0
                else:
                    # Use first class as default
                    label = 0

                image_paths.append((img_path, label))
        else:
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        # Limit number of samples if specified
        if max_samples is not None and max_samples < len(image_paths):
            image_paths = image_paths[:max_samples]

        return image_paths

    def __len__(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get the image and label at the given index.

        Args:
            idx: Index

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.image_paths[idx]

        # Load and transform image
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder tensor if image cannot be loaded
            return torch.zeros((3, 224, 224)), label


def get_dataset(
    dataset_name: str,
    data_dir: str = "data",
    split: str = "val",
    transform: Optional[Callable] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Get dataset by name.

    Args:
        dataset_name: Name of the dataset ('imagenet', 'cifar10', etc.)
        data_dir: Base directory for all datasets
        split: Data split ('train', 'val', 'test')
        transform: Transformation to apply to images
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object

    Raises:
        ValueError: If dataset is not supported
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
    Create a DataLoader from a Dataset.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes

    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
