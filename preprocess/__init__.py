"""
2D to 3D Preprocessing Pipeline
==============================

This module provides comprehensive preprocessing tools for converting 2D RGB images
to 4-channel RGB-D tensors using depth estimation and image enhancement techniques.

Main Components:
- RGB image denoising (Gaussian/Median filtering)
- Histogram equalization (YUV/HSV color spaces)
- Depth estimation using pretrained models (MiDaS, DepthNet)
- RGB-D volume generation and visualization
"""

from .core.rgb_to_rgbd import RGBToRGBDProcessor
from .core.filters import ImageFilter
from .core.depth_estimation import DepthEstimator
from .core.histogram_equalization import HistogramEqualizer
from .utils.visualization import RGBDVisualizer

__version__ = "1.0.0"
__author__ = "Final Year Project"

__all__ = [
    "RGBToRGBDProcessor",
    "ImageFilter", 
    "DepthEstimator",
    "RGBDVisualizer",
    "HistogramEqualizer"
]
