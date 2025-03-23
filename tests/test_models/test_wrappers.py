"""Unit tests for model wrappers and factory functions."""

import os
import sys
import pytest
import torch

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import (
    ResNetModel,
    VGGModel,
    EfficientNetModel,
    MobileNetModel,
    get_model,
)


class TestModelFactory:
    """Tests for the model factory function."""

    def test_get_model_basic(self):
        """Test that get_model returns the correct type for each model."""
        # ResNet family
        assert isinstance(get_model("resnet18", pretrained=False), ResNetModel)
        assert isinstance(get_model("resnet50", pretrained=False), ResNetModel)

        # VGG family
        assert isinstance(get_model("vgg16", pretrained=False), VGGModel)

        # EfficientNet family
        assert isinstance(
            get_model("efficientnet_b0", pretrained=False), EfficientNetModel
        )

        # MobileNet family
        assert isinstance(
            get_model("mobilenet_v3_large", pretrained=False), MobileNetModel
        )

    def test_get_model_invalid(self):
        """Test that get_model raises ValueError for invalid model names."""
        with pytest.raises(ValueError):
            get_model("not_a_real_model")

    def test_get_model_kwargs(self):
        """Test that get_model passes kwargs correctly."""
        # Test pretrained flag
        model = get_model("resnet18", pretrained=False)
        assert model._model is not None


class TestModelWrappers:
    """Tests for the model wrapper classes."""

    def test_model_initialization(self):
        """Test that models initialize correctly."""
        # Test with pretrained=False to speed up tests
        models_to_test = [
            ResNetModel(variant="resnet18", pretrained=False),
            VGGModel(variant="vgg11", pretrained=False),
            EfficientNetModel(variant="efficientnet_b0", pretrained=False),
            MobileNetModel(variant="mobilenet_v3_small", pretrained=False),
        ]

        for model in models_to_test:
            assert model._model is not None
            assert hasattr(model, "forward")
            assert hasattr(model, "predict")
            assert hasattr(model, "get_gradients")

    def test_model_device(self):
        """Test that models can be moved between devices."""
        model = ResNetModel(variant="resnet18", pretrained=False)

        # Check default device
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert model.device == default_device

        # Move to CPU explicitly
        model.to("cpu")
        assert model.device == torch.device("cpu")
        assert next(model.parameters()).device == torch.device("cpu")

    def test_input_processing(self):
        """Test that model handles input correctly."""
        # Create a small model for faster testing
        model = MobileNetModel(variant="mobilenet_v3_small", pretrained=False)
        model.to("cpu")  # Ensure on CPU for testing

        # Create a dummy input
        batch_size = 2
        dummy_input = torch.rand(batch_size, 3, 224, 224)

        # Check that forward pass works
        with torch.no_grad():
            output = model(dummy_input)

        # Output should be logits for 1000 ImageNet classes
        assert output.shape == (batch_size, 1000)

        # Check predict method
        with torch.no_grad():
            classes, probs = model.predict(dummy_input)

        assert classes.shape == (batch_size,)
        assert probs.shape == (batch_size, 1000)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_computation(self):
        """Test that get_gradients computes input gradients correctly."""
        model = ResNetModel(variant="resnet18", pretrained=False)
        model.to("cpu")  # Ensure on CPU for testing

        # Create a dummy input and target
        batch_size = 2
        dummy_input = torch.rand(batch_size, 3, 224, 224, requires_grad=True)
        target_class = torch.tensor([0, 10])  # Arbitrary target classes

        # Compute gradients
        gradients = model.get_gradients(dummy_input, target_class)

        # Gradients should have the same shape as input
        assert gradients.shape == dummy_input.shape
        # Gradients should not be all zeros
        assert not torch.allclose(gradients, torch.zeros_like(gradients))
        # Gradients should be detached
        assert not gradients.requires_grad


class TestModelConsistency:
    """Tests for model behavior consistency."""

    def test_normalization_consistency(self):
        """Test that all models apply the same normalization."""
        # Create instances of each model type
        models_to_test = [
            ResNetModel(variant="resnet18", pretrained=False),
            VGGModel(variant="vgg11", pretrained=False),
            EfficientNetModel(variant="efficientnet_b0", pretrained=False),
            MobileNetModel(variant="mobilenet_v3_small", pretrained=False),
        ]

        # All models should use ImageNet normalization
        for model in models_to_test:
            assert model.mean == [0.485, 0.456, 0.406]
            assert model.std == [0.229, 0.224, 0.225]

            # Create test input that's easily distinguishable before and after normalization
            test_input = torch.ones(1, 3, 10, 10)  # All ones
            normalized = model._normalize(test_input)

            # Manually compute what the normalized values should be
            expected = (
                test_input - torch.tensor(model.mean).view(1, 3, 1, 1)
            ) / torch.tensor(model.std).view(1, 3, 1, 1)

            # Check that normalization is applied correctly
            assert torch.allclose(normalized, expected)
