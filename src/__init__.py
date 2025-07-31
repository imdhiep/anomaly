"""
Source Code Package
==================

Main source code package containing core and model modules.
"""

from . import core
from . import models

# Import main classes for convenience
from .core import AnomalyDetectionPipeline
from .models import (
    Simple3DCNN, ResNet3D, Autoencoder, 
    VariationalAutoencoder, train_autoencoder
)

__all__ = [
    'core', 'models',
    'AnomalyDetectionPipeline',
    'Simple3DCNN', 'ResNet3D', 'Autoencoder', 
    'VariationalAutoencoder', 'train_autoencoder'
]
