"""EGASD: Entropy-Guided Adaptive Speculative Decoding."""

__version__ = "1.0.0"
__author__ = "Thesis code maintainers"

from .models import AcceptancePredictionHead, PivotClassifier
from .entropy_utils import compute_entropy, normalize_entropy, compute_dynamic_threshold
from .egasd_decode import EGASDConfig, EGASDDecoder, egasd_generate

__all__ = [
    "AcceptancePredictionHead",
    "PivotClassifier",
    "compute_entropy",
    "normalize_entropy",
    "compute_dynamic_threshold",
    "EGASDConfig",
    "EGASDDecoder",
    "egasd_generate",
]
