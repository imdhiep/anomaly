"""
3D Anomaly Detection Pipeline
============================

A comprehensive PyTorch-based pipeline for 3D medical image anomaly detection using autoencoders.

This package provides all the components needed to build, train, and evaluate 3D anomaly detection models,
including data loading, feature extraction, patch processing, autoencoder training, and evaluation.

Example Usage:
-------------
```python
from pipeline import create_simple_pipeline
from data_loader import load_dataset

# Create pipeline
pipeline = create_simple_pipeline()

# Load training data (normal samples only)
train_volumes = load_dataset(train_paths, target_shape=(64, 64, 64))

# Fit pipeline
pipeline.fit(train_volumes)

# Load test data
test_volume = load_and_normalize_volume('test_scan.nii.gz', target_shape=(64, 64, 64))

# Get anomaly score map
score_map, metadata = pipeline.score_map(test_volume)

# Get binary prediction
binary_mask = pipeline.predict(test_volume)

# Evaluate on test set with ground truth
metrics = pipeline.evaluate(test_volumes, ground_truth_masks)
```

Components:
----------
- data_loader: Load and preprocess 3D volumes (NIfTI, NumPy)
- patch_extraction: Extract overlapping 3D patches
- feature_extractor: 3D CNN models (Simple CNN, 3D-ResNet)
- feature_embedding: Convert patches to fixed-length vectors
- autoencoder: Various autoencoder architectures
- training: Training loops and utilities
- inference: Inference pipeline and anomaly scoring
- thresholding: Threshold computation and application
- evaluation: Comprehensive evaluation metrics
- pipeline: End-to-end modular pipeline
"""

from .data_loader import (
    load_and_normalize_volume,
    load_dataset,
    augment_volume
)

from .patch_extraction import (
    extract_patches_3d,
    extract_patches_with_coordinates,
    reconstruct_from_patches,
    create_patch_grid
)

from .feature_extractor import (
    FeatureExtractor3D,
    Simple3DCNN,
    ResNet3D,
    BasicBlock3D,
    Bottleneck3D,
    resnet3d_18,
    resnet3d_34,
    resnet3d_50
)

from .feature_embedding import (
    FeatureEmbedding,
    PatchSampler,
    MultiScaleFeatureEmbedding
)

from .autoencoder import (
    Autoencoder,
    VariationalAutoencoder,
    DenoisingAutoencoder,
    ContractiveAutoencoder
)

from .training import (
    train_autoencoder,
    FeatureVectorDataset,
    plot_training_curves,
    load_trained_autoencoder
)

from .inference import (
    inference_pipeline,
    extract_test_features,
    compute_reconstruction_errors,
    create_anomaly_score_map,
    batch_inference,
    ensemble_inference
)

from .thresholding import (
    compute_threshold_from_normal_scores,
    apply_threshold,
    adaptive_threshold,
    multi_threshold_analysis,
    morphological_postprocessing
)

from .evaluation import (
    comprehensive_evaluation,
    compute_roc_auc,
    compute_precision_recall,
    compute_pixel_wise_iou,
    compute_dice_coefficient,
    plot_roc_curves,
    plot_precision_recall_curves,
    compute_optimal_threshold
)

from .pipeline import (
    AnomalyDetectionPipeline,
    create_simple_pipeline,
    create_resnet_pipeline
)

__version__ = "1.0.0"
__author__ = "3D Anomaly Detection Team"

__all__ = [
    # Data loading
    'load_and_normalize_volume',
    'load_dataset',
    'augment_volume',
    
    # Patch extraction
    'extract_patches_3d',
    'extract_patches_with_coordinates',
    'reconstruct_from_patches',
    'create_patch_grid',
    
    # Feature extraction
    'FeatureExtractor3D',
    'Simple3DCNN',
    'ResNet3D',
    'BasicBlock3D',
    'Bottleneck3D',
    'resnet3d_18',
    'resnet3d_34',
    'resnet3d_50',
    
    # Feature embedding
    'FeatureEmbedding',
    'PatchSampler',
    'MultiScaleFeatureEmbedding',
    
    # Autoencoders
    'Autoencoder',
    'VariationalAutoencoder',
    'DenoisingAutoencoder',
    'ContractiveAutoencoder',
    
    # Training
    'train_autoencoder',
    'FeatureVectorDataset',
    'plot_training_curves',
    'load_trained_autoencoder',
    
    # Inference
    'inference_pipeline',
    'extract_test_features',
    'compute_reconstruction_errors',
    'create_anomaly_score_map',
    'batch_inference',
    'ensemble_inference',
    
    # Thresholding
    'compute_threshold_from_normal_scores',
    'apply_threshold',
    'adaptive_threshold',
    'multi_threshold_analysis',
    'morphological_postprocessing',
    
    # Evaluation
    'comprehensive_evaluation',
    'compute_roc_auc',
    'compute_precision_recall',
    'compute_pixel_wise_iou',
    'compute_dice_coefficient',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'compute_optimal_threshold',
    
    # Pipeline
    'AnomalyDetectionPipeline',
    'create_simple_pipeline',
    'create_resnet_pipeline'
]
