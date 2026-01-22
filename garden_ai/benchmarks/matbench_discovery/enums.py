"""Enums for Matbench Discovery benchmark tasks."""

from enum import Enum


class MatbenchTask(Enum):
    """Available Matbench Discovery benchmark tasks."""

    IS2RE = "IS2RE"  # Initial Structure to Relaxed Energy
    RS2RE = "RS2RE"  # Relaxed Structure to Relaxed Energy
    S2EFS = "S2EFS"  # Structure to Energy, Forces, Stress
    S2EF = "S2EF"  # Structure to Energy, Force
    S2EFSM = "S2EFSM"  # Structure to Energy, Force, Stress, Magmoms
    IS2E = "IS2E"  # Initial Structure to Energy
    S2E = "S2E"  # Structure to Energy
    S2RE = "S2RE"  # Structure to Relaxed Energy
    RP2RE = "RP2RE"  # Relaxed Prototype to Relaxed Energy
    IP2E = "IP2E"  # Initial Prototype to Energy


class DatasetSize(str, Enum):
    """Predefined dataset sizes for Matbench Discovery benchmarks.

    These correspond to different subsets of the WBM test set that are commonly
    used for evaluating materials discovery models.
    """

    FULL = "full"
    """Full WBM test set (~257k structures)"""

    UNIQUE_PROTOS = "unique_protos"
    """Unique prototypes subset (~215k structures) - removes duplicate prototypes"""

    RANDOM_10K = "random_10k"
    """Random 10k structures from the unique prototypes subset (fixed seed)"""

    RANDOM_100 = "random_100"
    """Random 100 structures for quick testing (fixed seed)"""

    def seed(self, seed: int) -> "DatasetConfig":
        """Return a configuration with a custom random seed."""
        return DatasetConfig(self, seed)


class DatasetConfig:
    """Configuration for a dataset subset with a specific random seed."""

    def __init__(self, subset: DatasetSize, seed: int):
        self.subset = subset
        self.seed = seed

    def __repr__(self):
        return f"{self.subset.name}(seed={self.seed})"
