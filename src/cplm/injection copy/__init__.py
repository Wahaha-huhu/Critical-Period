"""Injection datasets for structural uptake and factual retention experiments."""

from .wordhop import build_wordhop_dataset, transform_sentence
from .facts import build_factual_dataset

__all__ = ["build_wordhop_dataset", "transform_sentence", "build_factual_dataset"]
