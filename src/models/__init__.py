"""Model interfaces for adversarial attack evaluation."""

from .wrappers import (
    ImageNetModel,
    ResNetModel,
    VGGModel,
    EfficientNetModel,
    MobileNetModel,
    get_model,
)

__all__ = [
    "ImageNetModel",
    "ResNetModel",
    "VGGModel",
    "EfficientNetModel",
    "MobileNetModel",
    "get_model",
]
