"""Utility functions for adversarial attack implementations."""

from .projections import (
    project_box,
    project_l2_ball,
    project_linf_ball,
    project_perturbation,
    project_adversarial_example,
)

__all__ = [
    "project_box",
    "project_l2_ball",
    "project_linf_ball",
    "project_perturbation",
    "project_adversarial_example",
]
