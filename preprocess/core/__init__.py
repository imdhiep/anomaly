"""
Core Processing Components
=========================

Core modules for RGB to RGB-D conversion pipeline.
"""

from .rgb_to_rgbd import RGBToRGBDProcessor, create_processor_presets
from .filters import ImageFilter, FilterType, create_filter_presets
from .histogram_equalization import HistogramEqualizer, ColorSpace, create_equalization_presets
from .depth_estimation import DepthEstimator, DepthPostProcessor

__all__ = [
    "RGBToRGBDProcessor",
    "create_processor_presets",
    "ImageFilter",
    "FilterType", 
    "create_filter_presets",
    "HistogramEqualizer",
    "ColorSpace",
    "create_equalization_presets",
    "DepthEstimator",
    "DepthPostProcessor"
]
