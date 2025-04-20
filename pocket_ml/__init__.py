"""Pocket ML - A simplified machine learning library

This library provides easy-to-use tools for common machine learning tasks.
"""

from .model import Classifier
from .preprocessing import DataPreprocessor
from .visualization import Visualizer

__version__ = '0.1.0'
__author__ = 'Raghul'
__all__ = ['Classifier', 'DataPreprocessor', 'Visualizer']