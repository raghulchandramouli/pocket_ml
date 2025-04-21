"""Pocket ML - A simplified machine learning library

This library provides easy-to-use tools for common machine learning tasks.
"""

from .algorithms.classification.classifier import Classifier
from .preprocessing.data_preprocessor import DataPreprocessor
from .visualization.visualizer import Visualizer

__version__ = '0.1.2'
__author__ = 'Raghul'
__all__ = ['Classifier', 'DataPreprocessor', 'Visualizer']