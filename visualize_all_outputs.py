import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from train_pipeline import AnomalyDetectionModel


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


def get_reconstruction_and_anomaly(model, rgbd_tensor, device, original_size=None):
    """Get both reconstruction and anomaly map."""
    model.eval()
    rgbd_tensor = rgbd_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        fused_features, reconstructed_features = model(rgbd_tensor)
        
        # Get reconstruction error
        diff = fused_features - reconstructed_features
        l2_error = torch.norm(diff, p=2, dim=1).squeeze(0)
        
        # Cosine similarity error
        fused_norm = F.normalize(fused_features, p=2, dim=1)
        recon_norm = F.normalize(reconstructed_features, p=2, dim=1)
        cosine_sim = torch.sum(fused_norm * recon_norm, dim=1).squeeze(0)
        cosine_error = 1.0 - cosine_sim
        
        # Channel-wise variance error
        channel_diff = torch.var(diff, dim=1).squeeze(0)
        
        # Combine errors
        combined_error = (0.5 * l2_error + 0.3 * cosine_error + 0.2 * channel_diff)
        
        # Normalize
        error_min, error_max = combined_error.min(), combined_error.max()
        if error_max > error_min:
            error_normalized = (combined_error - error_min) / (error_max - error_min)
        else:
            error_normalized = torch.zeros_like(combined_error)
        
        # Gaussian smoothing
        error_smooth = torch.from_numpy(
            cv2.GaussianBlur(error_normalized.cpu().numpy(), (3, 3), 0.5)
        ).to(device)
        
        # Upsample to original size
        if original_size is not None:
            error_upsampled = F.interpolate(
                error_smooth.unsqueeze(0).unsqueeze(0),
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        else:
            error_upsampled = error_smooth
        
        # Get reconstruction visualization
        # Convert reconstructed features back to RGB-like visualization
        recon_vis = reconstructed_features.mean(dim=1).squeeze(0)  # Average across feature channels
        recon_vis = (recon_vis - recon_vis.min()) / (recon_vis.max() - recon_vis.min())
        
        # Upsample reconstruction to original size
        if original_size is not None:
            recon_upsampled = F.interpolate(
                recon_vis.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            recon_upsampled = recon_vis
        
        return error_upsampled.cpu().numpy(), recon_upsampled.cpu().numpy()


def apply_object_threshold(anomaly_map, original_rgbd):
    """Apply object-focused thresholding."""
    rgb = original_rgbd[:, :, :3]
    depth = original_rgbd[:, :, 3]
    
    # Find main object using depth
    depth_valid = depth > 0
    if np.any(depth_valid):
        depth_binary = depth_valid.astype(np.uint8)
        contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            object_mask = np.zeros_like(depth, dtype=bool)
            cv2.fillPoly(object_mask.astype(np.uint8), [largest_contour], 1)
            object_mask = object_mask.astype(bool)
            
            # Erode mask slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            if np.any(object_mask):
                object_scores = anomaly_map[object_mask]
                background_scores = anomaly_map[~object_mask]
                
                obj_mean = np.mean(object_scores)
                obj_std = np.std(object_scores)
                bg_mean = np.mean(background_scores) if len(background_scores) > 0 else 0
                
                # Conservative threshold
                if obj_mean > bg_mean:
                    threshold = obj_mean + 0.5 * obj_std
                else:
                    threshold = np.percentile(object_scores, 75)
                
                # Create focused mask
                binary_mask = (anomaly_map > threshold) & object_mask
                
                # Clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                binary_mask = binary_mask.astype(bool)
                
                return binary_mask, threshold, object_mask
    
    # Fallback
    threshold = np.percentile(anomaly_map, 80)
    binary_mask = anomaly_map > threshold
    object_mask = np.ones_like(anomaly_map, dtype=bool)
    return binary_mask, threshold, object_mask


def visualize_all_outputs(npz_path, model, device, save_path=None):
    """Visualize tất cả outputs: gốc, reconstruction, depth, anomaly, overlay."""
    # Load image
    rgbd_tensor, original_rgbd = load_rgbd_image(npz_path)
    if rgbd_tensor is None:
        print(f"Failed to load {npz_path}")
        return
    
    # Get reconstruction and anomaly map
    original_size = original_rgbd.shape[:2]
    anomaly_map, reconstruction = get_reconstruction_and_anomaly(model, rgbd_tensor, device, original_size)
    
    # Apply object-focused thresholding
    binary_mask, threshold, object_mask = apply_object_threshold(anomaly_map, original_rgbd)
    
    # Prepare images
    rgb = original_rgbd[:, :, :3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min()) if rgb.max() > rgb.min() else rgb
    
    depth = original_rgbd[:, :, 3]
    
    # Create visualization
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 1. Ảnh gốc (Original RGB)
    axes[0].imshow(rgb_norm)
    axes[0].set_title('Ảnh gốc (Original)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ảnh reconstruction
    im1 = axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Depth map
    im2 = axes[2].imshow(depth, cmap='plasma')
    axes[2].set_title('Depth Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Anomaly map
    im3 = axes[3].imshow(anomaly_map, cmap='hot')
    axes[3].set_title('Anomaly Map', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 5. Overlay (RGB + Anomaly)
    overlay = (rgb_norm * 255).astype(np.uint8)
    if np.any(binary_mask):
        overlay[binary_mask] = [255, 0, 0]  # Red for anomalies
    axes[4].imshow(overlay)
    axes[4].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[4].axis('off')
    
    # Add overall title
    filename = Path(npz_path).name
    fig.suptitle(f'Anomaly Detection Results: {filename}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nThống kê cho {filename}:")
    print(f"  Max anomaly score: {np.max(anomaly_map):.6f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Anomaly pixels: {np.sum(binary_mask):,}/{binary_mask.size:,}")
    print(f"  Anomaly ratio: {np.mean(binary_mask):.4f}")


def test_visualization():
    """Test visualization với một số file anomaly."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = AnomalyDetectionModel().to(device)
    model_path = "anomaly_detection_model_final.pth"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Get anomaly files
    anomaly_folder = "../preprocess/output/Anomaly"
    anomaly_files = list(Path(anomaly_folder).glob("*.npz"))
    
    if not anomaly_files:
        print("No anomaly files found!")
        return
    
    print(f"Found {len(anomaly_files)} anomaly files")
    
    # Create output directory
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize first few files
    for i, npz_file in enumerate(anomaly_files[:3]):  # First 3 files
        print(f"\n{'='*50}")
        print(f"Visualizing {npz_file.name}")
        print(f"{'='*50}")
        
        save_path = output_dir / f"all_outputs_{npz_file.stem}.png"
        visualize_all_outputs(npz_file, model, device, save_path)


if __name__ == "__main__":
    test_visualization()
