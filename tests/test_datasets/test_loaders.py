"""Unit tests for the dataset loader module."""

import os
import sys
import pytest
import torch
import tempfile
import shutil
from PIL import Image
import torchvision.transforms as transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.loader import ImageNetDataset, get_dataset, get_dataloader


@pytest.fixture
def mock_imagenet_data():
    """Create mock ImageNet data for testing."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create sample_images directory
    sample_dir = os.path.join(temp_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)

    # Create class file
    classes = ["tench", "goldfish", "great_white_shark"]
    with open(os.path.join(temp_dir, "imagenet_classes.txt"), "w") as f:
        f.write("\n".join(classes))

    # Create sample images
    for i, class_name in enumerate(classes):
        # Create with class index prefix
        img_path = os.path.join(sample_dir, f"{i}_{class_name}.jpg")
        Image.new("RGB", (32, 32), color=(i * 50, 100, 150)).save(img_path)

        # Create with synset format
        synset_path = os.path.join(sample_dir, f"n{1000000+i}_{class_name}.jpg")
        Image.new("RGB", (32, 32), color=(i * 50, 150, 100)).save(synset_path)

    yield temp_dir

    # Cleanup after tests
    shutil.rmtree(temp_dir)


class TestImageNetDataset:
    """Tests for the ImageNetDataset class."""

    def test_init(self, mock_imagenet_data):
        """Test dataset initialization."""
        dataset = ImageNetDataset(mock_imagenet_data)
        assert len(dataset) == 6  # 3 classes × 2 images per class

        # Test with custom transform
        custom_transform = transforms.Compose(
            [transforms.Resize(100), transforms.ToTensor()]
        )
        dataset = ImageNetDataset(mock_imagenet_data, transform=custom_transform)
        assert dataset.transform == custom_transform

    def test_load_class_names(self, mock_imagenet_data):
        """Test loading class names."""
        dataset = ImageNetDataset(mock_imagenet_data)
        assert len(dataset.class_names) == 3
        assert dataset.class_names == ["tench", "goldfish", "great_white_shark"]
        assert dataset.class_to_idx["tench"] == 0
        assert dataset.class_to_idx["goldfish"] == 1

    def test_get_image_paths(self, mock_imagenet_data):
        """Test retrieving image paths and labels."""
        dataset = ImageNetDataset(mock_imagenet_data)

        # Should have 6 images (2 per class × 3 classes)
        assert len(dataset.image_paths) == 6

        # Check label extraction from filenames
        for path, label in dataset.image_paths:
            if "0_tench" in path or "n1000000_tench" in path:
                assert label == 0
            elif "1_goldfish" in path or "n1000001_goldfish" in path:
                assert label == 1
            elif "2_great_white_shark" in path or "n1000002_great_white_shark" in path:
                assert label == 2

    def test_max_samples(self, mock_imagenet_data):
        """Test limiting the number of samples."""
        dataset = ImageNetDataset(mock_imagenet_data, max_samples=3)
        assert len(dataset) == 3

    def test_getitem(self, mock_imagenet_data):
        """Test retrieving individual items."""
        dataset = ImageNetDataset(mock_imagenet_data)
        img, label = dataset[0]

        # Check image shape and type
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)  # Default transform output shape
        assert isinstance(label, int)


class TestGetDataset:
    """Tests for the get_dataset function."""

    @pytest.fixture
    def mock_data_dir(self, mock_imagenet_data):
        """Create a mock data directory structure."""
        temp_dir = tempfile.mkdtemp()
        # Create ImageNet directory with the mock data
        imagenet_dir = os.path.join(temp_dir, "imagenet")
        os.makedirs(imagenet_dir, exist_ok=True)

        # Create a CIFAR10 directory as well for testing unsupported datasets
        cifar10_dir = os.path.join(temp_dir, "cifar10")
        os.makedirs(cifar10_dir, exist_ok=True)

        # Copy mock data into the imagenet directory
        sample_dir = os.path.join(imagenet_dir, "sample_images")
        os.makedirs(sample_dir, exist_ok=True)

        # Copy class file
        shutil.copy(
            os.path.join(mock_imagenet_data, "imagenet_classes.txt"),
            os.path.join(imagenet_dir, "imagenet_classes.txt"),
        )

        # Copy sample images
        for file in os.listdir(os.path.join(mock_imagenet_data, "sample_images")):
            src = os.path.join(mock_imagenet_data, "sample_images", file)
            dst = os.path.join(sample_dir, file)
            shutil.copy(src, dst)

        yield temp_dir

        # Cleanup after tests
        shutil.rmtree(temp_dir)

    def test_get_imagenet_dataset(self, mock_data_dir):
        """Test getting an ImageNet dataset."""
        dataset = get_dataset("imagenet", data_dir=mock_data_dir)
        assert isinstance(dataset, ImageNetDataset)
        assert len(dataset) == 6

    def test_unsupported_dataset(self, mock_data_dir):
        """Test error when requesting unsupported dataset."""
        with pytest.raises(ValueError):
            get_dataset("cifar10", data_dir=mock_data_dir)

    def test_nonexistent_dir(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_dataset("imagenet", data_dir="/nonexistent_dir")


class TestGetDataloader:
    """Tests for the get_dataloader function."""

    def test_dataloader_creation(self, mock_imagenet_data):
        """Test creating a dataloader."""
        dataset = ImageNetDataset(mock_imagenet_data)
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=True)

        assert dataloader.batch_size == 2
        # DataLoader object doesn't expose shuffle as an attribute

        # Test iteration
        batch = next(iter(dataloader))
        imgs, labels = batch

        assert imgs.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)

        # Test with different settings
        dataloader = get_dataloader(dataset, batch_size=3, shuffle=False, num_workers=2)
        assert dataloader.batch_size == 3
        assert dataloader.num_workers == 2
