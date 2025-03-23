"""
Model wrappers for adversarial attack evaluation.

This module provides wrapper classes for various pretrained models to be used
in adversarial attack experiments. It includes:

- ImageNetModel: Base abstract class defining the common interface.
- Implementation classes for various model architectures.
- get_model: Factory function to create model instances by name.

The implementation focuses on making it easy to use pretrained models in a
standardized way for adversarial attack evaluation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Union
from abc import ABC, abstractmethod


class ImageNetModel(nn.Module, ABC):
    """
    Base class for all ImageNet model wrappers.

    This abstract class defines the common interface that all model implementations
    must follow for consistency in the adversarial attack experiments.
    """

    def __init__(self):
        """Initialize the base class."""
        super().__init__()
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_size = (224, 224)  # Default input size (height, width)
        self._mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self._std = [0.229, 0.224, 0.225]  # ImageNet std

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get the expected input size (height, width)."""
        return self._input_size

    @property
    def mean(self) -> List[float]:
        """Get the mean values for normalization."""
        return self._mean

    @property
    def std(self) -> List[float]:
        """Get the standard deviation values for normalization."""
        return self._std

    def to(self, device: Union[str, torch.device]) -> "ImageNetModel":
        """Move the model to the specified device."""
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if self._model is not None:
            self._model.to(self._device)
        return self

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
               Expected to be in range [0, 1] and unnormalized.

        Returns:
            Tensor of logits with shape (batch_size, num_classes).
        """
        pass

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get class predictions and probabilities for input images.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
               Expected to be in range [0, 1] and unnormalized.

        Returns:
            Tuple of (predicted_classes, probabilities) where:
            - predicted_classes: Tensor of shape (batch_size) with predicted class indices.
            - probabilities: Tensor of shape (batch_size, num_classes) with class probabilities.
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities

    def get_gradients(
        self, x: torch.Tensor, target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute input gradients with respect to the target class.

        This is useful for gradient-based adversarial attacks like FGSM.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            target_class: Target class indices of shape (batch_size).

        Returns:
            Gradients tensor of same shape as input.
        """
        x = x.clone().detach().requires_grad_(True).to(self.device)

        # Forward pass
        logits = self.forward(x)

        # Create one-hot encoding of target classes
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        # Compute gradients
        loss = torch.sum(one_hot * logits)
        loss.backward()

        return x.grad.detach()


class ResNetModel(ImageNetModel):
    """ResNet model wrapper for ImageNet classification."""

    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        """
        Initialize a ResNet model.

        Args:
            variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()

        if variant not in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Load the appropriate model
        if variant == "resnet18":
            self._model = models.resnet18(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "resnet34":
            self._model = models.resnet34(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "resnet50":
            self._model = models.resnet50(
                weights="IMAGENET1K_V2" if pretrained else None
            )
        elif variant == "resnet101":
            self._model = models.resnet101(
                weights="IMAGENET1K_V2" if pretrained else None
            )
        elif variant == "resnet152":
            self._model = models.resnet152(
                weights="IMAGENET1K_V2" if pretrained else None
            )

        self._model.eval()
        self._model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        x = x.to(self.device)
        # Apply ImageNet normalization
        x = self._normalize(x)
        return self._model(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input according to ImageNet standards."""
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class VGGModel(ImageNetModel):
    """VGG model wrapper for ImageNet classification."""

    def __init__(self, variant: str = "vgg16", pretrained: bool = True):
        """
        Initialize a VGG model.

        Args:
            variant: VGG variant ('vgg11', 'vgg13', 'vgg16', 'vgg19').
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()

        if variant not in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            raise ValueError(f"Unsupported VGG variant: {variant}")

        # Load the appropriate model with batch normalization
        if variant == "vgg11":
            self._model = models.vgg11_bn(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "vgg13":
            self._model = models.vgg13_bn(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "vgg16":
            self._model = models.vgg16_bn(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "vgg19":
            self._model = models.vgg19_bn(
                weights="IMAGENET1K_V1" if pretrained else None
            )

        self._model.eval()
        self._model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        x = x.to(self.device)
        # Apply ImageNet normalization
        x = self._normalize(x)
        return self._model(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input according to ImageNet standards."""
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class EfficientNetModel(ImageNetModel):
    """EfficientNet model wrapper for ImageNet classification."""

    def __init__(self, variant: str = "efficientnet_b0", pretrained: bool = True):
        """
        Initialize an EfficientNet model.

        Args:
            variant: EfficientNet variant ('efficientnet_b0' through 'efficientnet_b7').
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()

        valid_variants = [f"efficientnet_b{i}" for i in range(8)]
        if variant not in valid_variants:
            raise ValueError(
                f"Unsupported EfficientNet variant: {variant}. "
                f"Valid options are: {', '.join(valid_variants)}"
            )

        # Load the appropriate model
        if variant == "efficientnet_b0":
            self._model = models.efficientnet_b0(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b1":
            self._model = models.efficientnet_b1(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b2":
            self._model = models.efficientnet_b2(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b3":
            self._model = models.efficientnet_b3(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b4":
            self._model = models.efficientnet_b4(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b5":
            self._model = models.efficientnet_b5(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b6":
            self._model = models.efficientnet_b6(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "efficientnet_b7":
            self._model = models.efficientnet_b7(
                weights="IMAGENET1K_V1" if pretrained else None
            )

        self._model.eval()
        self._model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        x = x.to(self.device)
        # Apply ImageNet normalization
        x = self._normalize(x)
        return self._model(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input according to ImageNet standards."""
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class MobileNetModel(ImageNetModel):
    """MobileNet model wrapper for ImageNet classification."""

    def __init__(self, variant: str = "mobilenet_v3_large", pretrained: bool = True):
        """
        Initialize a MobileNet model.

        Args:
            variant: MobileNet variant ('mobilenet_v3_large', 'mobilenet_v3_small').
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()

        if variant not in ["mobilenet_v3_large", "mobilenet_v3_small"]:
            raise ValueError(f"Unsupported MobileNet variant: {variant}")

        # Load the appropriate model
        if variant == "mobilenet_v3_large":
            self._model = models.mobilenet_v3_large(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif variant == "mobilenet_v3_small":
            self._model = models.mobilenet_v3_small(
                weights="IMAGENET1K_V1" if pretrained else None
            )

        self._model.eval()
        self._model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        x = x.to(self.device)
        # Apply ImageNet normalization
        x = self._normalize(x)
        return self._model(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input according to ImageNet standards."""
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


def get_model(model_name: str, **kwargs) -> ImageNetModel:
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model architecture and variant, e.g.,
                  'resnet50', 'vgg16', 'efficientnet_b0', 'mobilenet_v3_large'.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        An initialized model instance.

    Raises:
        ValueError: If the specified model is not supported.
    """
    # Parse architecture and variant from model_name
    model_map = {
        # ResNet family
        "resnet18": (ResNetModel, {"variant": "resnet18"}),
        "resnet34": (ResNetModel, {"variant": "resnet34"}),
        "resnet50": (ResNetModel, {"variant": "resnet50"}),
        "resnet101": (ResNetModel, {"variant": "resnet101"}),
        "resnet152": (ResNetModel, {"variant": "resnet152"}),
        # VGG family
        "vgg11": (VGGModel, {"variant": "vgg11"}),
        "vgg13": (VGGModel, {"variant": "vgg13"}),
        "vgg16": (VGGModel, {"variant": "vgg16"}),
        "vgg19": (VGGModel, {"variant": "vgg19"}),
        # EfficientNet family
        "efficientnet_b0": (EfficientNetModel, {"variant": "efficientnet_b0"}),
        "efficientnet_b1": (EfficientNetModel, {"variant": "efficientnet_b1"}),
        "efficientnet_b2": (EfficientNetModel, {"variant": "efficientnet_b2"}),
        "efficientnet_b3": (EfficientNetModel, {"variant": "efficientnet_b3"}),
        "efficientnet_b4": (EfficientNetModel, {"variant": "efficientnet_b4"}),
        "efficientnet_b5": (EfficientNetModel, {"variant": "efficientnet_b5"}),
        "efficientnet_b6": (EfficientNetModel, {"variant": "efficientnet_b6"}),
        "efficientnet_b7": (EfficientNetModel, {"variant": "efficientnet_b7"}),
        # MobileNet family
        "mobilenet_v3_large": (MobileNetModel, {"variant": "mobilenet_v3_large"}),
        "mobilenet_v3_small": (MobileNetModel, {"variant": "mobilenet_v3_small"}),
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models are: {', '.join(model_map.keys())}"
        )

    # Get model class and default arguments
    model_class, default_args = model_map[model_name]

    # Merge default arguments with provided arguments
    merged_args = {**default_args, **kwargs}

    # Create and return model instance
    return model_class(**merged_args)
