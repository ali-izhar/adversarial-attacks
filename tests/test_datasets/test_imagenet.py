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

from src.datasets.imagenet import ImageNetDataset, get_dataset, get_dataloader


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

    # Create sample images with distinctive pixel values for each class
    for i, class_name in enumerate(classes):
        # Create with class index prefix
        img_path = os.path.join(sample_dir, f"{i}_{class_name}.jpg")
        Image.new("RGB", (32, 32), color=(i * 50, 100, 150)).save(img_path)

        # Create with synset format
        synset_path = os.path.join(sample_dir, f"n{1000000+i}_{class_name}.jpg")
        Image.new("RGB", (32, 32), color=(i * 50, 150, 100)).save(synset_path)

    # Add edge cases
    # 1. Image with no pattern match in filename (should default to label 0)
    no_pattern_path = os.path.join(sample_dir, "no_pattern_match.jpg")
    Image.new("RGB", (32, 32), color=(200, 200, 200)).save(no_pattern_path)

    # 2. Image with synset ID but no matching class name (should default to label 0)
    unknown_synset_path = os.path.join(sample_dir, "n9999999_unknown_class.jpg")
    Image.new("RGB", (32, 32), color=(210, 210, 210)).save(unknown_synset_path)

    # 3. Corrupt image file
    corrupt_img_path = os.path.join(sample_dir, "0_corrupt.jpg")
    with open(corrupt_img_path, "wb") as f:
        f.write(b"Not a valid image file")

    # 4. Duplicate image with same label
    duplicate_path = os.path.join(sample_dir, "0_tench_duplicate.jpg")
    Image.new("RGB", (32, 32), color=(0, 100, 150)).save(duplicate_path)

    yield temp_dir

    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def empty_class_file():
    """Create mock data with empty class file."""
    temp_dir = tempfile.mkdtemp()
    sample_dir = os.path.join(temp_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)

    # Create empty class file
    with open(os.path.join(temp_dir, "imagenet_classes.txt"), "w") as f:
        f.write("")

    # Add a test image
    img_path = os.path.join(sample_dir, "0_test.jpg")
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img_path)

    yield temp_dir
    shutil.rmtree(temp_dir)


class TestImageNetDataset:
    """Tests for the ImageNetDataset class."""

    def test_init(self, mock_imagenet_data):
        """Test dataset initialization."""
        dataset = ImageNetDataset(mock_imagenet_data)
        # 3 classes × 2 images per class + 4 edge case images
        assert len(dataset) == 10

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

    def test_empty_class_file(self, empty_class_file):
        """Test behavior with empty class file."""
        with pytest.raises(IndexError):
            ImageNetDataset(empty_class_file)

    def test_get_image_paths(self, mock_imagenet_data):
        """Test retrieving image paths and labels."""
        dataset = ImageNetDataset(mock_imagenet_data)

        # Should have 10 images (6 regular + 4 edge cases)
        assert len(dataset.image_paths) == 10

        # Check for duplicate file paths
        paths = [path for path, _ in dataset.image_paths]
        assert len(paths) == len(set(paths)), "Duplicate paths detected"

        # Verify all expected images are found
        filenames = [os.path.basename(path) for path, _ in dataset.image_paths]
        assert "0_tench.jpg" in filenames
        assert "n1000000_tench.jpg" in filenames
        assert "no_pattern_match.jpg" in filenames
        assert "0_corrupt.jpg" in filenames

        # Check label extraction from filenames comprehensively
        label_map = {
            os.path.basename(path): label for path, label in dataset.image_paths
        }

        # Regular pattern matches
        assert label_map["0_tench.jpg"] == 0
        assert label_map["1_goldfish.jpg"] == 1
        assert label_map["2_great_white_shark.jpg"] == 2
        assert label_map["n1000000_tench.jpg"] == 0
        assert label_map["n1000001_goldfish.jpg"] == 1
        assert label_map["n1000002_great_white_shark.jpg"] == 2

        # Edge cases
        assert (
            label_map["no_pattern_match.jpg"] == 0
        ), "No pattern should default to label 0"
        assert (
            label_map["n9999999_unknown_class.jpg"] == 0
        ), "Unknown synset should default to label 0"
        assert (
            label_map["0_corrupt.jpg"] == 0
        ), "Corrupt image should have correct label"
        assert (
            label_map["0_tench_duplicate.jpg"] == 0
        ), "Duplicate should have same label"

    def test_max_samples(self, mock_imagenet_data):
        """Test limiting the number of samples."""
        dataset = ImageNetDataset(mock_imagenet_data, max_samples=5)
        assert len(dataset) == 5

        # Ensure the first 5 samples are returned in order
        original_dataset = ImageNetDataset(mock_imagenet_data)
        limited_dataset = ImageNetDataset(mock_imagenet_data, max_samples=5)

        for i in range(5):
            assert limited_dataset.image_paths[i] == original_dataset.image_paths[i]

    def test_getitem(self, mock_imagenet_data):
        """Test retrieving individual items."""
        dataset = ImageNetDataset(mock_imagenet_data)

        # Test regular image
        img, label = dataset[0]  # First image should be valid
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)  # Default transform output shape
        assert isinstance(label, int)

        # Find index of corrupt image
        corrupt_idx = None
        for i, (path, _) in enumerate(dataset.image_paths):
            if "corrupt" in path:
                corrupt_idx = i
                break

        # Test corrupt image returns black tensor with special marking
        corrupt_img, corrupt_label = dataset[corrupt_idx]
        assert (
            corrupt_img[:, 1:, 1:].sum().item() == 0
        ), "Corrupt image should be mostly black"
        assert (
            corrupt_img[:, 0, 0] == -1.0
        ).all(), "Corrupt image should have a special marker at (0,0)"
        assert corrupt_img.shape == (3, 224, 224)

    def test_image_content_matches_label(self, mock_imagenet_data):
        """Test that image content matches the expected label."""
        # Use a simple transform that preserves colors
        transform = transforms.Compose(
            [
                transforms.Resize(32),  # Keep original size
                transforms.ToTensor(),
            ]
        )

        dataset = ImageNetDataset(mock_imagenet_data, transform=transform)

        # Check standard images with predictable colors
        for i, (path, label) in enumerate(dataset.image_paths):
            basename = os.path.basename(path)
            if basename.startswith("0_tench") and "corrupt" not in basename:
                # Should have color (0, 100, 150)
                img, img_label = dataset[i]
                # Check the center pixel
                center_r = img[0, 16, 16].item()
                center_g = img[1, 16, 16].item()
                center_b = img[2, 16, 16].item()

                # Account for compression artifacts with tolerance
                assert (
                    0 <= center_r < 0.1
                ), f"Expected red channel near 0, got {center_r}"
                assert (
                    0.35 <= center_g < 0.45
                ), f"Expected green channel near 0.4, got {center_g}"
                assert (
                    0.55 <= center_b < 0.65
                ), f"Expected blue channel near 0.6, got {center_b}"
                assert img_label == 0, f"Expected label 0, got {img_label}"

    def test_shuffled_data_consistency(self, mock_imagenet_data):
        """Test that shuffling data maintains image-label consistency."""
        dataset = ImageNetDataset(mock_imagenet_data)

        # Get original image paths and labels
        original_paths = [path for path, _ in dataset.image_paths]
        original_labels = [label for _, label in dataset.image_paths]

        # Create a map from path to expected label
        path_to_label = {path: label for path, label in dataset.image_paths}

        # Create a dataloader with shuffling
        dataloader = get_dataloader(dataset, batch_size=1, shuffle=True)

        # Get all items from one epoch
        retrieved_images = []
        retrieved_labels = []
        for img, label in dataloader:
            retrieved_images.append(img)
            retrieved_labels.append(label.item())

        # Check if counts match
        assert len(retrieved_labels) == len(original_labels)

        # Count occurrences of each label in original and retrieved
        original_label_counts = {}
        for label in original_labels:
            original_label_counts[label] = original_label_counts.get(label, 0) + 1

        retrieved_label_counts = {}
        for label in retrieved_labels:
            retrieved_label_counts[label] = retrieved_label_counts.get(label, 0) + 1

        # Check if label counts match after shuffling
        assert original_label_counts == retrieved_label_counts


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
        assert len(dataset) == 10  # Now 10 with edge cases

    def test_unsupported_dataset(self, mock_data_dir):
        """Test error when requesting unsupported dataset."""
        with pytest.raises(ValueError):
            get_dataset("cifar10", data_dir=mock_data_dir)

    def test_nonexistent_dir(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_dataset("imagenet", data_dir="/nonexistent_dir")

    def test_missing_class_file(self, mock_data_dir):
        """Test error when class file is missing."""
        # Remove the class file
        os.remove(os.path.join(mock_data_dir, "imagenet", "imagenet_classes.txt"))

        with pytest.raises(FileNotFoundError):
            get_dataset("imagenet", data_dir=mock_data_dir)

    def test_missing_images_dir(self, mock_data_dir):
        """Test error when sample_images directory is missing."""
        # Remove the images directory
        shutil.rmtree(os.path.join(mock_data_dir, "imagenet", "sample_images"))

        with pytest.raises(FileNotFoundError):
            get_dataset("imagenet", data_dir=mock_data_dir)


class TestGetDataloader:
    """Tests for the get_dataloader function."""

    def test_dataloader_creation(self, mock_imagenet_data):
        """Test creating a dataloader."""
        dataset = ImageNetDataset(mock_imagenet_data)
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=True)

        assert dataloader.batch_size == 2

        # Test iteration
        batch = next(iter(dataloader))
        imgs, labels = batch

        assert imgs.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)

        # Test with different settings
        dataloader = get_dataloader(dataset, batch_size=3, shuffle=False, num_workers=2)
        assert dataloader.batch_size == 3
        assert dataloader.num_workers == 2

    def test_dataloader_complete_iteration(self, mock_imagenet_data):
        """Test that dataloader returns all samples once per epoch."""
        dataset = ImageNetDataset(mock_imagenet_data)
        total_samples = len(dataset)

        # Test with batch size that doesn't divide evenly into sample count
        batch_size = 3
        dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

        seen_samples = 0
        for batch in dataloader:
            imgs, labels = batch
            seen_samples += len(labels)

        assert seen_samples == total_samples, "Dataloader should return all samples"

    def test_batch_consistency(self, mock_imagenet_data):
        """Test that images and labels in a batch stay consistent."""
        # Create a dataset with a simple identity transform to preserve colors
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageNetDataset(mock_imagenet_data, transform=transform)

        # Create a dataloader with a batch size > 1
        dataloader = get_dataloader(dataset, batch_size=4, shuffle=False)

        # Get first batch
        imgs, labels = next(iter(dataloader))

        # Check label-color consistency for the first class (tench)
        # Images should have distinctive colors per class
        for i in range(len(labels)):
            if labels[i] == 0:  # tench class
                # Skip corrupt images (identified by special pixel pattern)
                if (imgs[i, :, 0, 0] == -1.0).all():
                    continue

                # Should have color near (0, 100, 150) normalized to [0,1]
                # Check center pixel
                center_r = imgs[i, 0, 16, 16].item()
                center_g = imgs[i, 1, 16, 16].item()
                center_b = imgs[i, 2, 16, 16].item()

                # Allow for some tolerance due to image processing
                assert (
                    0 <= center_r < 0.1
                ), f"Tench should have low red value, got {center_r}"
                assert (
                    0.35 <= center_g < 0.45
                ), f"Tench should have medium green value, got {center_g}"
                assert (
                    0.55 <= center_b < 0.65
                ), f"Tench should have high blue value, got {center_b}"
