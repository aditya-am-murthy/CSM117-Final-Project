"""
Mini-batch forgetting with dynamic pruning and gradient replay buffers.

This module implements adaptive unlearning strategies that isolate forget regions
through dynamic pruning techniques and gradient replay buffers.
"""

from .dynamic_pruning import DynamicPruningUnlearning
from .gradient_replay import GradientReplayBufferUnlearning

__all__ = ['DynamicPruningUnlearning', 'GradientReplayBufferUnlearning']

