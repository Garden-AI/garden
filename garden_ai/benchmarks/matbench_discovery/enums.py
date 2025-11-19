"""Enums for Matbench Discovery benchmark tasks."""

from enum import Enum


class MatbenchTask(Enum):
    """Available Matbench Discovery benchmark tasks.

    Currently only IS2RE is implemented for the MVP.
    Future tasks could include:
    - RS2RE: Relaxed Structure to Relaxed Energy
    - S2EFS: Structure to Energy, Forces, and Stress
    """

    IS2RE = "is2re"  # Initial Structure to Relaxed Energy
