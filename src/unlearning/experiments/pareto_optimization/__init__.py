"""
Multi-objective learning frameworks for unlearning.

This module implements Pareto optimization techniques that jointly maximize
forgetting accuracy and retention fidelity.
"""

from .pareto_unlearning import ParetoOptimizationUnlearning

__all__ = ['ParetoOptimizationUnlearning']

