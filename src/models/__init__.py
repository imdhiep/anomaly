"""
Deep Learning Models
===================

This module contains all the deep learning models used in the pipeline.
"""

from .feature_extractor import (
    FeatureExtractor3D, Simple3DCNN, ResNet3D,
    resnet3d_18, resnet3d_34, resnet3d_50
)
from .feature_embedding import (
    FeatureEmbedding, PatchSampler, MultiScaleFeatureEmbedding
)
from .autoencoder import (
    Autoencoder, VariationalAutoencoder, 
    DenoisingAutoencoder, ContractiveAutoencoder
)
from .training import (
    train_autoencoder, FeatureVectorDataset, 
    plot_training_curves, load_trained_autoencoder
)

__all__ = [
    # Feature extractors
    'FeatureExtractor3D', 'Simple3DCNN', 'ResNet3D',
    'resnet3d_18', 'resnet3d_34', 'resnet3d_50',
    
    # Feature embedding
    'FeatureEmbedding', 'PatchSampler', 'MultiScaleFeatureEmbedding',
    
    # Autoencoders
    'Autoencoder', 'VariationalAutoencoder', 
    'DenoisingAutoencoder', 'ContractiveAutoencoder',
    
    # Training
    'train_autoencoder', 'FeatureVectorDataset',
    'plot_training_curves', 'load_trained_autoencoder'
]
