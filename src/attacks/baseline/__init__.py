"""Baseline attacks implemented from research papers"""

from .attack_cw import CW
from .attack_deepfool import DeepFool
from .attack_ffgsm import FFGSM
from .attack_fgsm import FGSM

__all__ = ["CW", "DeepFool", "FFGSM", "FGSM"]
