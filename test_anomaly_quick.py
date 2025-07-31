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
        
        # Get better fused features visualization
        fused_vis = fused_features.mean(dim=1).squeeze(0)  # Average across feature channels
        fused_vis = (fused_vis - fused_vis.min()) / (fused_vis.max() - fused_vis.min())
        
        # Upsample fused features to original size
        if original_size is not None:
            fused_upsampled = F.interpolate(
                fused_vis.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            fused_upsampled = fused_vis
        
        return error_upsampled.cpu().numpy(), fused_upsampled.cpu().numpy()


def load_normal_sample():
    """Load a normal sample for comparison."""
    normal_folder = "../preprocess/output/Normal"
    normal_files = list(Path(normal_folder).glob("*.npz"))
    
    if normal_files:
        # Load first normal sample
        normal_file = normal_files[0]
        data = np.load(normal_file)
        
        for key in data.keys():
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 4:  # RGBD format
                return arr[:, :, :3]  # Return RGB only
    
    return None


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


def get_anomaly_bounding_boxes(binary_mask, min_area=100):
    """Get bounding boxes around anomaly regions."""
    # Find contours of anomaly regions
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # Filter small regions
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return bboxes


def create_complete_visualization(original_rgbd, fused_features, anomaly_map, binary_mask, threshold, object_mask, save_path, filename, reconstructed_features=None):
    """Create complete visualization with all images including bounding boxes."""
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Prepare images
    rgb = original_rgbd[:, :, :3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min()) if rgb.max() > rgb.min() else rgb
    depth = original_rgbd[:, :, 3]

    # 1. Original RGB
    axes[0].imshow(rgb_norm)
    axes[0].axis('off')

    # 2. Fused Features
    im1 = axes[1].imshow(fused_features, cmap='viridis')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Depth Map
    im2 = axes[2].imshow(depth, cmap='plasma')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Reconstruction Features
    if reconstructed_features is not None:
        recon_vis = reconstructed_features.mean(axis=0) if reconstructed_features.ndim == 3 else reconstructed_features
        recon_vis = (recon_vis - recon_vis.min()) / (recon_vis.max() - recon_vis.min())
        axes[3].imshow(recon_vis, cmap='gray')
        axes[3].axis('off')
        axes[3].set_title("Reconstruction")
    else:
        axes[3].set_visible(False)

    # 5. Overlay anomaly heatmap on original
    anomaly_map_smooth = cv2.GaussianBlur(anomaly_map, (7, 7), 1.5)
    heatmap = cv2.applyColorMap((anomaly_map_smooth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = rgb_norm.copy()
    alpha = 0.5
    if overlay.shape[-1] == 3 and heatmap.shape[-1] == 3:
        overlay = (1 - alpha) * overlay + alpha * heatmap
    axes[4].imshow(overlay, interpolation='bilinear')
    axes[4].axis('off')

    fig.suptitle(f'Anomaly Detection: {filename}', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\nStatistics for {filename}:")
    print(f"  Max anomaly score: {np.max(anomaly_map):.6f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Anomaly pixels: {np.sum(binary_mask):,}/{binary_mask.size:,}")
    print(f"  Anomaly ratio: {np.mean(binary_mask):.4f}")


def test_anomaly_detection():
    """Test anomaly detection on anomaly dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model with trained weights from 175 epochs
    model = AnomalyDetectionModel().to(device)
    model_path = "anomaly_detection_model_final_400.pth"
    
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
    output_dir = Path("anomaly_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Test each anomaly file
    all_scores = []
    all_ratios = []
    
    for i, npz_file in enumerate(anomaly_files[:2]):  # Test first 2 files (reduce for OOM)
        print(f"\nProcessing {npz_file.name}...")
        
        # Load image
        rgbd_tensor, original_rgbd = load_rgbd_image(npz_file)
        if rgbd_tensor is None:
            print(f"Failed to load {npz_file.name}")
            continue
        
        # Get reconstruction and anomaly map
        original_size = original_rgbd.shape[:2]
        anomaly_map, fused_features, reconstructed_features = get_reconstruction_and_anomaly(model, rgbd_tensor, device, original_size)
        
        # Apply object-focused thresholding
        binary_mask, threshold, object_mask = apply_object_threshold(anomaly_map, original_rgbd)
        
        # Create visualization with bounding boxes
        save_path = output_dir / f"enhanced_{npz_file.stem}.png"
        create_complete_visualization(
            original_rgbd, fused_features, anomaly_map, binary_mask, threshold, object_mask, 
            save_path, npz_file.name, reconstructed_features=reconstructed_features
        )
        
        # Compute metrics
        image_score = np.max(anomaly_map)
        anomaly_ratio = np.mean(binary_mask)
        object_scores = anomaly_map[object_mask] if np.any(object_mask) else []
        background_scores = anomaly_map[~object_mask] if np.any(~object_mask) else []
        
        all_scores.append(image_score)
        all_ratios.append(anomaly_ratio)

        # Release GPU memory after each image
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Print results with safe formatting
        print(f"Results for {npz_file.name}:")
        print(f"  Image score (max): {image_score:.6f}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Anomaly ratio: {anomaly_ratio:.4f}")
        
        if len(object_scores) > 0:
            print(f"  Object region - Mean: {np.mean(object_scores):.4f}, Max: {np.max(object_scores):.4f}")
        else:
            print(f"  Object region - Mean: N/A, Max: N/A")
            
        if len(background_scores) > 0:
            print(f"  Background region - Mean: {np.mean(background_scores):.4f}")
        else:
            print(f"  Background region - Mean: N/A")
            
        print(f"  Object pixels: {np.sum(object_mask):,}/{object_mask.size:,}")
    
    # Summary statistics
    print(f"\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Files processed: {len(all_scores)}")
    print(f"Average image score: {np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")
    print(f"Average anomaly ratio: {np.mean(all_ratios):.4f} ± {np.std(all_ratios):.4f}")
    print(f"Score range: [{np.min(all_scores):.6f}, {np.max(all_scores):.6f}]")
    print(f"Ratio range: [{np.min(all_ratios):.4f}, {np.max(all_ratios):.4f}]")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score distribution
    axes[0].hist(all_scores, bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[0].set_xlabel('Image Anomaly Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Anomaly Scores')
    axes[0].axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.4f}')
    axes[0].legend()

    # Ratio distribution
    axes[1].hist(all_ratios, bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Anomaly Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Anomaly Ratios')
    axes[1].axvline(np.mean(all_ratios), color='orange', linestyle='--', label=f'Mean: {np.mean(all_ratios):.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create final plot view with all processed images
    print(f"\n{'='*60}")
    print("CREATING FINAL PLOT VIEW WITH ANOMALY HEATMAP OVERLAY")
    print(f"{'='*60}")

    # Get a representative sample for final view
    sample_files = anomaly_files[:3]  # First 3 files for final view

    fig, axes = plt.subplots(len(sample_files), 4, figsize=(24, 5*len(sample_files)))
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)

    for idx, npz_file in enumerate(sample_files):
        print(f"Adding to final view: {npz_file.name}")

        # Load and process
        rgbd_tensor, original_rgbd = load_rgbd_image(npz_file)
        if rgbd_tensor is None:
            continue

        original_size = original_rgbd.shape[:2]
        anomaly_map, fused_features = get_reconstruction_and_anomaly(model, rgbd_tensor, device, original_size)
        binary_mask, threshold, object_mask = apply_object_threshold(anomaly_map, original_rgbd)

        # Prepare images
        rgb = original_rgbd[:, :, :3]
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min()) if rgb.max() > rgb.min() else rgb
        depth = original_rgbd[:, :, 3]

        # Plot row for this image
        row = axes[idx] if len(sample_files) > 1 else axes

        # 1. Original
        row[0].imshow(rgb_norm)
        row[0].axis('off')

        # 2. Fused Features
        im1 = row[1].imshow(fused_features, cmap='viridis')
        row[1].axis('off')
        plt.colorbar(im1, ax=row[1], fraction=0.046, pad=0.04)

        # 3. Depth Map
        im2 = row[2].imshow(depth, cmap='plasma')
        row[2].axis('off')
        plt.colorbar(im2, ax=row[2], fraction=0.046, pad=0.04)

        # 4. Overlay anomaly heatmap on original
        overlay = rgb_norm.copy()
        heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        alpha = 0.5
        if overlay.shape[-1] == 3 and heatmap.shape[-1] == 3:
            overlay = (1 - alpha) * overlay + alpha * heatmap
        row[3].imshow(overlay)
        row[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'final_plot_view.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nFinal plot view saved to: {output_dir / 'final_plot_view.png'}")
    print(f"\nResults saved in: {output_dir}")
    print("Results have been improved with:")
    print("- Overlay anomaly heatmap directly on original image")
    print("- No bounding boxes, no normal sample")
    print("- All text in English")
    print(f"Model used: 400 epochs trained ResNet50-based architecture")


if __name__ == "__main__":
    test_anomaly_detection()
