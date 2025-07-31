"""
Core Pipeline Components
=======================

This module contains the core functionality of the 3D anomaly detection pipeline.
"""

from .pipeline import AnomalyDetectionPipeline
from .data_loader import load_and_normalize_volume, load_dataset
from .patch_extraction import extract_patches_3d, extract_patches_with_coordinates
from .inference import inference_pipeline, create_anomaly_score_map
from .evaluation import comprehensive_evaluation, compute_roc_auc
from .thresholding import compute_threshold_from_normal_scores, apply_threshold

__all__ = [
    'AnomalyDetectionPipeline',
    'load_and_normalize_volume',
    'load_dataset', 
    'extract_patches_3d',
    'extract_patches_with_coordinates',
    'inference_pipeline',
    'create_anomaly_score_map',
    'comprehensive_evaluation',
    'compute_roc_auc',
    'compute_threshold_from_normal_scores',
    'apply_threshold'
]
