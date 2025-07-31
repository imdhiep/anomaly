import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from src.models.feature_embedding import ResNet50RGBD, MultiScaleFeatureFusion, ConvolutionalAutoencoder


class AnomalyDetectionModel(nn.Module):
    """Complete Anomaly Detection Model with ResNet50 + CAE."""
    
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        
        # Feature extractor (frozen ResNet50)
        self.feature_extractor = ResNet50RGBD(pretrained=True)
        
        # Feature fusion module
        feature_channels = {
            'layer1': 256,
            'layer2': 512, 
            'layer3': 1024,
            'layer4': 2048
        }
        self.feature_fusion = MultiScaleFeatureFusion(
            feature_channels=feature_channels,
            target_size=(128, 128),
            use_conv_smooth=True
        )
        
        # Convolutional Autoencoder (trainable)
        total_channels = sum(feature_channels.values())  # 256+512+1024+2048 = 3840
        self.autoencoder = ConvolutionalAutoencoder(
            input_channels=total_channels,
            hidden_channels=[1024, 512, 256],
            latent_channels=128
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        with torch.no_grad():
            # Extract multi-scale features (frozen)
            feature_maps = self.feature_extractor.extract_features(
                x, ['layer1', 'layer2', 'layer3', 'layer4']
            )
            
            # Fuse features (frozen)
            fused_features = self.feature_fusion(feature_maps)
        
        # Reconstruct through autoencoder (trainable)
        reconstructed = self.autoencoder(fused_features)
        
        return fused_features, reconstructed


def load_rgbd_image(npz_path):
    """Load RGBD image from NPZ file."""
    data = np.load(npz_path)
    
    for key in data.keys():
        arr = data[key]
        if arr.ndim == 3 and arr.shape[-1] == 4:  # RGBD format (H, W, 4)
            # Convert to tensor and normalize
            rgbd = torch.from_numpy(arr).float()
            
            # Normalize channels
            for c in range(4):
                channel_data = rgbd[:, :, c]
                if channel_data.std() > 0:
                    rgbd[:, :, c] = (channel_data - channel_data.mean()) / channel_data.std()
            
            # Convert to (4, H, W) format
            rgbd = rgbd.permute(2, 0, 1)
            
            # Resize to standard size
            rgbd = F.interpolate(rgbd.unsqueeze(0), size=(256, 256), 
                               mode='bilinear', align_corners=False).squeeze(0)
            
            return rgbd, arr  # Return processed and original
    
    return None, None


def compute_anomaly_map(model, rgbd_tensor, device, original_size=None):
    """
    Compute anomaly map from RGBD tensor following Step 5:
    - Compute fused features f(x) and reconstruction f̂(x)
    - Compute per-location error: A[i,j] = ||f[i,j] – f̂[i,j]||_2
    - Upsample anomaly map to original image size with F.interpolate
    """
    model.eval()
    rgbd_tensor = rgbd_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Extract fused features f(x) and reconstruction f̂(x)
        fused_features, reconstructed_features = model(rgbd_tensor)
        
        # Compute per-location L2 error: A[i,j] = ||f[i,j] – f̂[i,j]||_2
        # fused_features and reconstructed_features shape: [1, C, H, W]
        diff = fused_features - reconstructed_features  # [1, C, H, W]
        error_map = torch.norm(diff, p=2, dim=1)  # L2 norm across channels -> [1, H, W]
        error_map = error_map.squeeze(0)  # Remove batch dimension -> [H, W]
        
        # Upsample anomaly map to original image size if provided
        if original_size is not None:
            error_map_upsampled = F.interpolate(
                error_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze()  # [orig_H, orig_W]
        else:
            # Default to 256x256 if no original size provided
            error_map_upsampled = F.interpolate(
                error_map.unsqueeze(0).unsqueeze(0), 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        
        return error_map_upsampled.cpu().numpy()


def apply_threshold(anomaly_map, method='adaptive', percentile=95, fixed_threshold=None):
    """
    Apply threshold to anomaly map to produce binary anomaly mask.
    Supports dynamic and fixed thresholding as specified in Step 5.
    """
    if method == 'fixed' and fixed_threshold is not None:
        # Fixed threshold
        threshold = fixed_threshold
    elif method == 'adaptive':
        # Dynamic threshold: mean + k*std (k=2 is common)
        threshold = np.mean(anomaly_map) + 2.0 * np.std(anomaly_map)
    elif method == 'percentile':
        # Dynamic threshold: use percentile (default 95th percentile)
        threshold = np.percentile(anomaly_map, percentile)
    elif method == 'otsu':
        # Dynamic threshold: Otsu's method
        # Convert to uint8 for cv2.threshold
        anomaly_uint8 = ((anomaly_map - anomaly_map.min()) / 
                        (anomaly_map.max() - anomaly_map.min()) * 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(anomaly_uint8, 0, 255, 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert back to original scale
        threshold = (otsu_threshold / 255.0 * 
                    (anomaly_map.max() - anomaly_map.min()) + anomaly_map.min())
    else:
        # Default fallback
        threshold = np.mean(anomaly_map) + np.std(anomaly_map)
    
    # Produce binary anomaly mask
    binary_mask = anomaly_map > threshold
    
    return binary_mask, threshold


def create_visualization(original_rgbd, anomaly_map, binary_mask, threshold, save_path):
    """
    Create comprehensive visualization.
    Now anomaly_map and binary_mask are already upsampled to original size.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB
    rgb = original_rgbd[:, :, :3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Depth map
    depth = original_rgbd[:, :, 3]
    im1 = axes[0, 1].imshow(depth, cmap='plasma')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Anomaly map (already upsampled to original size)
    im2 = axes[0, 2].imshow(anomaly_map, cmap='hot')
    axes[0, 2].set_title('Anomaly Map (Upsampled)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # RGB with anomaly overlay (sizes should match now)
    overlay = (rgb * 255).astype(np.uint8)
    overlay[binary_mask] = [255, 0, 0]  # Red for anomalies
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('RGB + Anomaly Overlay')
    axes[1, 0].axis('off')
    
    # Binary anomaly mask
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title(f'Binary Mask (Threshold: {threshold:.4f})')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f'''Anomaly Statistics:
Mean: {np.mean(anomaly_map):.4f}
Max: {np.max(anomaly_map):.4f}
Min: {np.min(anomaly_map):.4f}
Std: {np.std(anomaly_map):.4f}
Threshold: {threshold:.4f}
Anomaly Pixels: {np.sum(binary_mask)}/{binary_mask.size}
Anomaly Ratio: {np.mean(binary_mask):.4f}'''
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='center')
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def test_anomaly_detection():
    """Main function to test anomaly detection."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load trained model
    model = AnomalyDetectionModel().to(device)
    
    # Load model weights
    model_path = "anomaly_detection_model_final.pth"
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found! Please train the model first.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Test on anomaly images
    anomaly_folder = "../preprocess/output/Anomaly"
    anomaly_path = Path(anomaly_folder)
    
    if not anomaly_path.exists():
        print(f"Anomaly folder not found: {anomaly_folder}")
        return
    
    npz_files = list(anomaly_path.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {anomaly_folder}")
        return
    
    print(f"Found {len(npz_files)} anomaly files")
    
    # Create output directory
    output_dir = Path("anomaly_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    # Test on first few anomaly images
    for i, npz_file in enumerate(npz_files[:5]):  # Test first 5 files
        print(f"\nProcessing {npz_file.name}...")
        
        # Load RGBD image
        rgbd_tensor, original_rgbd = load_rgbd_image(npz_file)
        if rgbd_tensor is None:
            print(f"Failed to load {npz_file.name}")
            continue
        
        try:
            # Get original image dimensions for upsampling
            original_size = original_rgbd.shape[:2]  # (height, width)
            
            # Compute anomaly map with proper upsampling to original size
            anomaly_map = compute_anomaly_map(model, rgbd_tensor, device, original_size)
            
            # Apply threshold to get binary mask - try different methods
            binary_mask_adaptive, threshold_adaptive = apply_threshold(anomaly_map, method='adaptive')
            binary_mask_percentile, threshold_percentile = apply_threshold(anomaly_map, method='percentile', percentile=85)
            binary_mask_otsu, threshold_otsu = apply_threshold(anomaly_map, method='otsu')
            
            # Use percentile method (usually works better for anomaly detection)
            binary_mask, threshold = binary_mask_percentile, threshold_percentile
            
            # Create visualization
            save_path = output_dir / f"anomaly_result_{npz_file.stem}.png"
            create_visualization(original_rgbd, anomaly_map, binary_mask, threshold, save_path)
            
            # Print results with multiple threshold comparisons
            anomaly_ratio = np.mean(binary_mask)
            print(f"Processed {npz_file.name}:")
            print(f"  Mean anomaly score: {np.mean(anomaly_map):.6f}")
            print(f"  Max anomaly score: {np.max(anomaly_map):.6f}")
            print(f"  Min anomaly score: {np.min(anomaly_map):.6f}")
            print(f"  Std anomaly score: {np.std(anomaly_map):.6f}")
            print(f"  Adaptive threshold: {threshold_adaptive:.6f} -> ratio: {np.mean(binary_mask_adaptive):.4f}")
            print(f"  Percentile threshold: {threshold_percentile:.6f} -> ratio: {np.mean(binary_mask_percentile):.4f}")
            print(f"  Otsu threshold: {threshold_otsu:.6f} -> ratio: {np.mean(binary_mask_otsu):.4f}")
            print(f"  Selected threshold: {threshold:.6f}")
            print(f"  Selected anomaly ratio: {anomaly_ratio:.4f}")
            
        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")
            continue
    
    print(f"\nAnomaly detection results saved in: {output_dir}")


if __name__ == "__main__":
    test_anomaly_detection()
