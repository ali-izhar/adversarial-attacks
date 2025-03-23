"""Common pytest fixtures for model testing."""

import os
import sys
import pytest
import torch
import torchvision.transforms as transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.imagenet import get_dataset


def pytest_addoption(parser):
    """Add command-line options for model tests."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow model tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def device():
    """Fixture to provide the device for model testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def standard_transform():
    """Standard image transformation for model testing."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


@pytest.fixture(scope="session")
def sample_images(standard_transform):
    """Fixture to provide a few sample images for quick tests."""
    # Create some dummy tensors (for very fast tests)
    samples = torch.rand(5, 3, 224, 224)
    labels = torch.randint(0, 1000, (5,))
    return samples, labels


@pytest.fixture(scope="session")
def mini_imagenet_dataset(standard_transform):
    """Fixture to provide a very small ImageNet dataset for basic tests."""
    try:
        return get_dataset(
            "imagenet", data_dir="data", max_samples=10, transform=standard_transform
        )
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"ImageNet data not available: {e}")
