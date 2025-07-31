import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import torchvision.models as models


class ResNet50RGBD(nn.Module):
    """ResNet50 backbone modified for 4-channel RGB-D input with multi-scale feature extraction."""
    
    def __init__(self, pretrained: bool = True):
        super(ResNet50RGBD, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace first conv layer for 4-channel input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            # Copy RGB weights and initialize depth channel
            with torch.no_grad():
                # Copy RGB weights
                self.backbone.conv1.weight[:, :3] = original_conv1.weight
                # Initialize depth channel as average of RGB channels
                self.backbone.conv1.weight[:, 3] = original_conv1.weight.mean(dim=1)
        
        # Remove final layers (we only need feature extraction)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from specified layers."""
        features = {}
        
        # Forward pass through ResNet layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 'layer1' in layer_names:
            features['layer1'] = x
        
        x = self.backbone.layer2(x)
        if 'layer2' in layer_names:
            features['layer2'] = x
            
        x = self.backbone.layer3(x)
        if 'layer3' in layer_names:
            features['layer3'] = x
            
        x = self.backbone.layer4(x)
        if 'layer4' in layer_names:
            features['layer4'] = x
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (for compatibility)."""
        return self.backbone(x)


class MultiScaleFeatureFusion(nn.Module):
    """Fuse multi-scale features with upsampling and optional smoothing."""
    
    def __init__(self, 
                 feature_channels: Dict[str, int],
                 target_size: Tuple[int, int] = (128, 128),
                 use_conv_smooth: bool = True):
        """
        Initialize multi-scale feature fusion.
        
        Args:
            feature_channels: Dict mapping layer names to channel counts
            target_size: Target spatial size for upsampling
            use_conv_smooth: Whether to use 1x1 conv for smoothing
        """
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.feature_channels = feature_channels
        self.target_size = target_size
        self.use_conv_smooth = use_conv_smooth
        
        # Optional 1x1 conv layers for smoothing
        if use_conv_smooth:
            self.smooth_convs = nn.ModuleDict()
            for layer_name, channels in feature_channels.items():
                self.smooth_convs[layer_name] = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Calculate total output channels
        self.total_channels = sum(feature_channels.values())
    
    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features.
        
        Args:
            feature_maps: Dict of feature maps from different layers
            
        Returns:
            Fused feature tensor of shape (B, total_channels, H, W)
        """
        fused_features = []
        
        for layer_name, feature_map in feature_maps.items():
            # Upsample to target size
            upsampled = F.interpolate(
                feature_map, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Optional smoothing
            if self.use_conv_smooth and layer_name in self.smooth_convs:
                upsampled = self.smooth_convs[layer_name](upsampled)
            
            fused_features.append(upsampled)
        
        # Concatenate along channel dimension
        fused = torch.cat(fused_features, dim=1)
        
        return fused


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder using only 1x1 convolutions."""
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: List[int] = [512, 256, 128],
                 latent_channels: int = 64):
        """
        Initialize Convolutional Autoencoder.
        
        Args:
            input_channels: Number of input channels from fused features
            hidden_channels: List of hidden layer channel counts
            latent_channels: Number of latent space channels
        """
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        
        # Encoder
        encoder_layers = []
        prev_channels = input_channels
        
        for hidden_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(prev_channels, hidden_ch, kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            prev_channels = hidden_ch
        
        # Final encoder layer to latent space
        encoder_layers.append(nn.Conv2d(prev_channels, latent_channels, kernel_size=1))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (symmetric to encoder)
        decoder_layers = []
        prev_channels = latent_channels
        
        for hidden_ch in reversed(hidden_channels):
            decoder_layers.extend([
                nn.Conv2d(prev_channels, hidden_ch, kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            prev_channels = hidden_ch
        
        # Final decoder layer back to input space
        decoder_layers.append(nn.Conv2d(prev_channels, input_channels, kernel_size=1))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)


class FeatureEmbedding(nn.Module):
    """Legacy wrapper for compatibility (kept for existing code)."""
    
    def __init__(self, 
                 input_channels: int,
                 patch_size: Tuple[int, int, int],
                 embedding_method: str = 'flatten',
                 pooling_type: str = 'avg',
                 mlp_hidden_dims: Optional[List[int]] = None,
                 output_dim: Optional[int] = None):
        """
        Initialize feature embedding module.
        
        Args:
            input_channels: Number of input channels from feature maps
            patch_size: Size of patches (D, H, W)
            embedding_method: 'flatten', 'pool', or 'mlp'
            pooling_type: 'avg', 'max', or 'adaptive' (if embedding_method is 'pool')
            mlp_hidden_dims: Hidden dimensions for MLP projection
            output_dim: Output dimension for MLP projection
        """
        super(FeatureEmbedding, self).__init__()
        
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embedding_method = embedding_method
        self.pooling_type = pooling_type
        
        # Calculate flattened dimension
        self.flattened_dim = input_channels * patch_size[0] * patch_size[1] * patch_size[2]
        
        if embedding_method == 'flatten':
            self.embed_dim = self.flattened_dim
        elif embedding_method == 'pool':
            if pooling_type == 'adaptive':
                self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.embed_dim = input_channels
            elif pooling_type == 'avg':
                self.pool = nn.AvgPool3d(patch_size)
                self.embed_dim = input_channels
            elif pooling_type == 'max':
                self.pool = nn.MaxPool3d(patch_size)
                self.embed_dim = input_channels
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")
        elif embedding_method == 'mlp':
            if mlp_hidden_dims is None or output_dim is None:
                raise ValueError("mlp_hidden_dims and output_dim must be specified for MLP embedding")
            
            layers = []
            prev_dim = self.flattened_dim
            
            for hidden_dim in mlp_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
            self.embed_dim = output_dim
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    def forward(self, feature_patches: torch.Tensor) -> torch.Tensor:
        """
        Convert feature patches to embeddings.
        
        Args:
            feature_patches: Tensor of shape (N, C, D, H, W) where N is number of patches
        
        Returns:
            Embedded features of shape (N, embed_dim)
        """
        if self.embedding_method == 'flatten':
            # Flatten spatial dimensions
            embeddings = feature_patches.view(feature_patches.size(0), -1)
        
        elif self.embedding_method == 'pool':
            # Apply pooling
            pooled = self.pool(feature_patches)
            embeddings = pooled.view(pooled.size(0), -1)
        
        elif self.embedding_method == 'mlp':
            # Flatten then apply MLP
            flattened = feature_patches.view(feature_patches.size(0), -1)
            embeddings = self.mlp(flattened)
        
        return embeddings


class PatchSampler(nn.Module):
    """Sample patches from CNN feature maps and convert to embeddings."""
    
    def __init__(self,
                 patch_size: Tuple[int, int, int],
                 stride: Optional[Tuple[int, int, int]] = None,
                 max_patches: Optional[int] = None,
                 sampling_strategy: str = 'uniform'):
        """
        Initialize patch sampler.
        
        Args:
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            max_patches: Maximum number of patches to sample
            sampling_strategy: 'uniform', 'random', or 'grid'
        """
        super(PatchSampler, self).__init__()
        
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches
        self.sampling_strategy = sampling_strategy
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample patches from feature map.
        
        Args:
            feature_map: Feature map tensor of shape (C, D, H, W)
        
        Returns:
            Tuple of (patches, coordinates)
            patches: Tensor of shape (N, C, patch_d, patch_h, patch_w)
            coordinates: Tensor of shape (N, 3)
        """
        if feature_map.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got {feature_map.dim()}D")
        
        # Simple patch extraction without external dependency
        C, D, H, W = feature_map.shape
        patch_d, patch_h, patch_w = self.patch_size
        
        stride_d = self.stride[0] if self.stride else patch_d
        stride_h = self.stride[1] if self.stride else patch_h
        stride_w = self.stride[2] if self.stride else patch_w
        
        patches = []
        coordinates = []
        
        for d in range(0, D - patch_d + 1, stride_d):
            for h in range(0, H - patch_h + 1, stride_h):
                for w in range(0, W - patch_w + 1, stride_w):
                    patch = feature_map[:, d:d+patch_d, h:h+patch_h, w:w+patch_w]
                    patches.append(patch)
                    coordinates.append([d, h, w])
        
        if not patches:
            # If no patches could be extracted, return empty tensors
            return torch.empty(0, C, patch_d, patch_h, patch_w), torch.empty(0, 3)
        
        patches = torch.stack(patches)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        
        # Apply sampling strategy
        if self.max_patches is not None and patches.size(0) > self.max_patches:
            if self.sampling_strategy == 'uniform':
                # Uniform sampling
                indices = torch.linspace(0, patches.size(0) - 1, self.max_patches, dtype=torch.long)
            elif self.sampling_strategy == 'random':
                # Random sampling
                indices = torch.randperm(patches.size(0))[:self.max_patches]
            elif self.sampling_strategy == 'grid':
                # Grid-based sampling
                step = patches.size(0) // self.max_patches
                indices = torch.arange(0, patches.size(0), step)[:self.max_patches]
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            patches = patches[indices]
            coordinates = coordinates[indices]
        
        return patches, coordinates


class MultiScaleFeatureEmbedding(nn.Module):
    """Extract and embed features at multiple scales (legacy compatibility)."""
    
    def __init__(self,
                 feature_extractor,
                 patch_sizes: List[Tuple[int, int, int]],
                 embedding_configs: List[Dict],
                 layer_names: List[str],
                 fusion_method: str = 'concat'):
        """
        Initialize multi-scale feature embedding.
        
        Args:
            feature_extractor: Pre-trained feature extractor
            patch_sizes: List of patch sizes for each scale
            embedding_configs: List of embedding configurations
            layer_names: List of layer names to extract features from
            fusion_method: 'concat', 'sum', or 'attention'
        """
        super(MultiScaleFeatureEmbedding, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.patch_sizes = patch_sizes
        self.layer_names = layer_names
        self.fusion_method = fusion_method
        
        # Create patch samplers and embeddings for each scale
        self.patch_samplers = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        
        for patch_size, config in zip(patch_sizes, embedding_configs):
            sampler = PatchSampler(patch_size, **config.get('sampler_args', {}))
            embedding = FeatureEmbedding(patch_size=patch_size, **config.get('embedding_args', {}))
            
            self.patch_samplers.append(sampler)
            self.embeddings.append(embedding)
        
        # Calculate total embedding dimension
        self.total_embed_dim = sum(emb.embed_dim for emb in self.embeddings)
        
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embeddings[0].embed_dim,
                num_heads=8,
                batch_first=True
            )
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from volume.
        
        Args:
            volume: Input volume tensor
        
        Returns:
            Multi-scale feature embeddings
        """
        # Extract feature maps
        feature_maps = self.feature_extractor.extract_features(volume, self.layer_names)
        
        all_embeddings = []
        
        for layer_name in self.layer_names:
            feature_map = feature_maps[layer_name]
            
            layer_embeddings = []
            for sampler, embedding in zip(self.patch_samplers, self.embeddings):
                # Sample patches
                patches, _ = sampler(feature_map.squeeze(0))  # Remove batch dim
                
                # Embed patches
                embedded = embedding(patches)
                layer_embeddings.append(embedded)
            
            # Fuse embeddings from different scales
            if self.fusion_method == 'concat':
                fused = torch.cat(layer_embeddings, dim=1)
            elif self.fusion_method == 'sum':
                # Ensure all embeddings have the same dimension
                min_dim = min(emb.size(1) for emb in layer_embeddings)
                trimmed = [emb[:, :min_dim] for emb in layer_embeddings]
                fused = torch.stack(trimmed).sum(dim=0)
            elif self.fusion_method == 'attention':
                # Use attention to fuse embeddings
                stacked = torch.stack(layer_embeddings, dim=1)  # (N, num_scales, embed_dim)
                fused, _ = self.attention(stacked, stacked, stacked)
                fused = fused.mean(dim=1)  # Average over scales
            
            all_embeddings.append(fused)
        
        # Concatenate embeddings from all layers
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        return final_embeddings
