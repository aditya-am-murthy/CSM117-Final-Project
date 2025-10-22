"""
Data Handling and Preprocessing

This module provides utilities for loading, splitting, and distributing
datasets across federated learning clients.
"""

from .dataset_manager import DatasetManager, UnlearningDataSplitter

__all__ = ['DatasetManager', 'UnlearningDataSplitter']
