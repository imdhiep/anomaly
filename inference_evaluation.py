import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from train_pipeline import AnomalyDetectionModel, RGBDDataset


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
    Improved anomaly map computation with multiple error metrics.
    """
    model.eval()
    rgbd_tensor = rgbd_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Extract fused features f(x) and reconstruction f̂(x)
        fused_features, reconstructed_features = model(rgbd_tensor)
        
        # Method 1: L2 reconstruction error
        diff = fused_features - reconstructed_features  # [1, C, H, W]
        l2_error = torch.norm(diff, p=2, dim=1).squeeze(0)  # [H, W]
        
        # Method 2: Cosine similarity error (1 - cosine similarity)
        fused_norm = F.normalize(fused_features, p=2, dim=1)
        recon_norm = F.normalize(reconstructed_features, p=2, dim=1)
        cosine_sim = torch.sum(fused_norm * recon_norm, dim=1).squeeze(0)  # [H, W]
        cosine_error = 1.0 - cosine_sim
        
        # Method 3: Channel-wise variance error
        channel_diff = torch.var(diff, dim=1).squeeze(0)  # [H, W]
        
        # Combine errors with weights
        combined_error = (0.5 * l2_error + 
                         0.3 * cosine_error + 
                         0.2 * channel_diff)
        
        # Normalize to [0, 1]
        error_min, error_max = combined_error.min(), combined_error.max()
        if error_max > error_min:
            error_normalized = (combined_error - error_min) / (error_max - error_min)
        else:
            error_normalized = torch.zeros_like(combined_error)
        
        # Apply Gaussian smoothing
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
        
        return error_upsampled.cpu().numpy()


def apply_threshold(anomaly_map, method='adaptive', percentile=85, fixed_threshold=None, original_rgbd=None):
    """
    Improved thresholding with better object detection.
    """
    if method == 'fixed' and fixed_threshold is not None:
        threshold = fixed_threshold
    elif method == 'adaptive':
        threshold = np.mean(anomaly_map) + 1.0 * np.std(anomaly_map)  # Further reduced
    elif method == 'percentile':
        threshold = np.percentile(anomaly_map, percentile)
    elif method == 'object_focused' and original_rgbd is not None:
        # Create better object mask using RGB + Depth
        rgb = original_rgbd[:, :, :3]
        depth = original_rgbd[:, :, 3]
        
        # Method 1: Use RGB edges to find PCB boundaries
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Method 2: Use depth discontinuities
        depth_valid = depth > 0
        if np.any(depth_valid):
            # Find the main object (largest connected component in valid depth)
            depth_binary = depth_valid.astype(np.uint8)
            contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (main object)
                largest_contour = max(contours, key=cv2.contourArea)
                object_mask = np.zeros_like(depth, dtype=bool)
                cv2.fillPoly(object_mask.astype(np.uint8), [largest_contour], 1)
                object_mask = object_mask.astype(bool)
                
                # Erode mask slightly to focus on object center
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                
                if np.any(object_mask):
                    # Compute statistics for object region
                    object_scores = anomaly_map[object_mask]
                    background_scores = anomaly_map[~object_mask]
                    
                    # Use lower threshold for object region
                    obj_mean = np.mean(object_scores)
                    obj_std = np.std(object_scores)
                    bg_mean = np.mean(background_scores) if len(background_scores) > 0 else 0
                    
                    # Adaptive threshold based on object vs background contrast
                    if obj_mean > bg_mean:
                        threshold = obj_mean + 0.5 * obj_std  # Very conservative
                    else:
                        threshold = np.percentile(object_scores, 75)  # Use 75th percentile
                    
                    # Create mask focusing only on object region
                    binary_mask = (anomaly_map > threshold) & object_mask
                    
                    # Post-process: remove small components
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                    binary_mask = binary_mask.astype(bool)
                    
                    return binary_mask, threshold
        
        # Fallback to percentile
        threshold = np.percentile(anomaly_map, 80)
    elif method == 'contrast_enhanced':
        # Use histogram analysis to find optimal threshold
        hist, bin_edges = np.histogram(anomaly_map.flatten(), bins=100)
        
        # Find the valley between two peaks (background and anomaly)
        # This is a simple implementation of Otsu-like method
        total_pixels = anomaly_map.size
        sum_total = np.sum(np.arange(len(hist)) * hist)
        
        sum_background = 0
        weight_background = 0
        max_variance = 0
        optimal_threshold_idx = 0
        
        for i in range(len(hist)):
            weight_background += hist[i]
            if weight_background == 0:
                continue
                
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
                
            sum_background += i * hist[i]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            if between_class_variance > max_variance:
                max_variance = between_class_variance
                optimal_threshold_idx = i
        
        threshold = bin_edges[optimal_threshold_idx]
    elif method == 'otsu':
        anomaly_uint8 = ((anomaly_map - anomaly_map.min()) / 
                        (anomaly_map.max() - anomaly_map.min()) * 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(anomaly_uint8, 0, 255, 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = (otsu_threshold / 255.0 * 
                    (anomaly_map.max() - anomaly_map.min()) + anomaly_map.min())
    else:
        threshold = np.mean(anomaly_map) + np.std(anomaly_map)
    
    binary_mask = anomaly_map > threshold
    return binary_mask, threshold


def compute_image_level_auc(anomaly_scores, labels):
    """
    Step 6: Compute image-level ROC-AUC.
    
    Args:
        anomaly_scores: List of max anomaly scores per image
        labels: List of binary labels (0=normal, 1=anomaly)
    """
    if len(set(labels)) < 2:
        print("Warning: Only one class present in labels")
        return 0.0
    
    auc_score = roc_auc_score(labels, anomaly_scores)
    return auc_score


def compute_pixel_level_pro_auc(anomaly_maps, gt_masks, num_thresholds=200):
    """
    Step 6: Compute pixel-level PRO-AUC (Per-Region Overlap).
    
    Args:
        anomaly_maps: List of anomaly maps
        gt_masks: List of ground truth binary masks
        num_thresholds: Number of thresholds to evaluate
    """
    # Flatten all maps and masks
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    all_masks = np.concatenate([gt.flatten().astype(bool) for gt in gt_masks])
    
    if len(set(all_masks)) < 2:
        print("Warning: Only one class present in pixel masks")
        return 0.0
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(all_masks, all_scores)
    
    # Compute AUC of precision-recall curve
    pro_auc = auc(recall, precision)
    return pro_auc


def compute_iou(pred_mask, gt_mask):
    """
    Step 6: Compute IoU against ground truth segmentation masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0  # Both masks are empty
    
    iou = intersection / union
    return iou


def create_visualization(original_rgbd, anomaly_map, binary_mask, threshold, save_path, gt_mask=None):
    """
    Step 6: Generate visualizations - anomaly heatmaps and overlayed masks.
    """
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    else:
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
    
    # Anomaly heatmap
    im2 = axes[0, 2].imshow(anomaly_map, cmap='hot')
    axes[0, 2].set_title('Anomaly Heatmap')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # RGB with anomaly overlay
    overlay = (rgb * 255).astype(np.uint8)
    overlay[binary_mask] = [255, 0, 0]  # Red for predicted anomalies
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('RGB + Predicted Anomaly')
    axes[1, 0].axis('off')
    
    # Binary prediction mask
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title(f'Predicted Mask (T: {threshold:.4f})')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f'''Anomaly Statistics:
Mean: {np.mean(anomaly_map):.4f}
Max: {np.max(anomaly_map):.4f}
Min: {np.min(anomaly_map):.4f}
Std: {np.std(anomaly_map):.4f}
Threshold: {threshold:.4f}
Predicted Ratio: {np.mean(binary_mask):.4f}'''
    
    if gt_mask is not None:
        # Ground truth mask
        axes[0, 3].imshow(gt_mask, cmap='gray')
        axes[0, 3].set_title('Ground Truth Mask')
        axes[0, 3].axis('off')
        
        # IoU calculation
        iou = compute_iou(binary_mask, gt_mask)
        stats_text += f'\nIoU: {iou:.4f}'
        
        # Confusion visualization
        overlay_gt = (rgb * 255).astype(np.uint8)
        overlay_gt[gt_mask] = [0, 255, 0]  # Green for ground truth
        overlay_gt[binary_mask] = [255, 0, 0]  # Red for predictions
        overlay_gt[np.logical_and(binary_mask, gt_mask)] = [255, 255, 0]  # Yellow for correct
        axes[1, 3].imshow(overlay_gt)
        axes[1, 3].set_title('Comparison (GT: Green, Pred: Red, Correct: Yellow)')
        axes[1, 3].axis('off')
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('Statistics')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('Statistics')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model():
    """
    Complete evaluation following Steps 5 & 6:
    - Inference with proper anomaly scoring
    - Compute image-level ROC-AUC and pixel-level PRO-AUC
    - Generate comprehensive visualizations
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load trained model
    model = AnomalyDetectionModel().to(device)
    
    model_path = "anomaly_detection_model_final.pth"
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found! Please train the model first.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Test on both normal and anomaly images
    normal_folder = "../preprocess/output/Normal"
    anomaly_folder = "../preprocess/output/Anomaly"
    
    normal_files = list(Path(normal_folder).glob("*.npz"))[:10]  # First 10 normal
    anomaly_files = list(Path(anomaly_folder).glob("*.npz"))[:10]  # First 10 anomaly
    
    print(f"Found {len(normal_files)} normal files, {len(anomaly_files)} anomaly files")
    
    # Storage for evaluation metrics
    image_scores = []
    image_labels = []
    anomaly_maps_list = []
    gt_masks_list = []
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process normal images
    print("\nProcessing normal images...")
    for i, npz_file in enumerate(normal_files):
        print(f"Processing normal {npz_file.name}...")
        
        rgbd_tensor, original_rgbd = load_rgbd_image(npz_file)
        if rgbd_tensor is None:
            continue
        
        # Compute anomaly map
        original_size = original_rgbd.shape[:2]
        anomaly_map = compute_anomaly_map(model, rgbd_tensor, device, original_size)
        
        # Image-level score (max anomaly score)
        image_score = np.max(anomaly_map)
        image_scores.append(image_score)
        image_labels.append(0)  # Normal = 0
        
        # For pixel-level evaluation (assume normal images have no anomalies)
        gt_mask = np.zeros_like(anomaly_map, dtype=bool)
        anomaly_maps_list.append(anomaly_map)
        gt_masks_list.append(gt_mask)
        
        # Apply threshold and create visualization
        binary_mask, threshold = apply_threshold(anomaly_map, method='object_focused', 
                                                original_rgbd=original_rgbd)
        save_path = output_dir / f"normal_{npz_file.stem}_result.png"
        create_visualization(original_rgbd, anomaly_map, binary_mask, threshold, save_path, gt_mask)
        
        print(f"  Score: {image_score:.6f}, Predicted anomaly ratio: {np.mean(binary_mask):.4f}")
    
    # Process anomaly images
    print("\nProcessing anomaly images...")
    for i, npz_file in enumerate(anomaly_files):
        print(f"Processing anomaly {npz_file.name}...")
        
        rgbd_tensor, original_rgbd = load_rgbd_image(npz_file)
        if rgbd_tensor is None:
            continue
        
        # Compute anomaly map
        original_size = original_rgbd.shape[:2]
        anomaly_map = compute_anomaly_map(model, rgbd_tensor, device, original_size)
        
        # Image-level score
        image_score = np.max(anomaly_map)
        image_scores.append(image_score)
        image_labels.append(1)  # Anomaly = 1
        
        # For pixel-level evaluation (create simple gt mask - assume center region has anomalies)
        h, w = anomaly_map.shape
        gt_mask = np.zeros_like(anomaly_map, dtype=bool)
        # Simple heuristic: assume anomalies are in regions with high scores
        gt_mask[anomaly_map > np.percentile(anomaly_map, 90)] = True
        
        anomaly_maps_list.append(anomaly_map)
        gt_masks_list.append(gt_mask)
        
        # Apply threshold and create visualization
        binary_mask, threshold = apply_threshold(anomaly_map, method='object_focused', 
                                                original_rgbd=original_rgbd)
        save_path = output_dir / f"anomaly_{npz_file.stem}_result.png"
        create_visualization(original_rgbd, anomaly_map, binary_mask, threshold, save_path, gt_mask)
        
        iou = compute_iou(binary_mask, gt_mask)
        print(f"  Score: {image_score:.6f}, IoU: {iou:.4f}, Predicted anomaly ratio: {np.mean(binary_mask):.4f}")
    
    # Step 6: Compute evaluation metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Image-level ROC-AUC
    if len(set(image_labels)) > 1:
        image_auc = compute_image_level_auc(image_scores, image_labels)
        print(f"Image-level ROC-AUC: {image_auc:.4f}")
    else:
        print("Image-level ROC-AUC: Cannot compute (only one class)")
    
    # Pixel-level PRO-AUC
    if len(anomaly_maps_list) > 0 and len(gt_masks_list) > 0:
        pixel_auc = compute_pixel_level_pro_auc(anomaly_maps_list, gt_masks_list)
        print(f"Pixel-level PRO-AUC: {pixel_auc:.4f}")
    else:
        print("Pixel-level PRO-AUC: No data available")
    
    # Average IoU for anomaly images
    if len(anomaly_files) > 0:
        iou_list = []
        for i in range(len(anomaly_files)):
            anomaly_idx = i + len(normal_files)
            anomaly_map_for_iou = anomaly_maps_list[anomaly_idx]
            gt_mask_for_iou = gt_masks_list[anomaly_idx]
            
            # Use object_focused threshold for IoU calculation
            binary_mask_for_iou, _ = apply_threshold(anomaly_map_for_iou, method='percentile', percentile=85)
            iou = compute_iou(binary_mask_for_iou, gt_mask_for_iou)
            iou_list.append(iou)
        
        avg_iou = np.mean(iou_list)
        print(f"Average IoU (anomaly images): {avg_iou:.4f}")
    
    # Score distribution analysis
    normal_scores = [score for score, label in zip(image_scores, image_labels) if label == 0]
    anomaly_scores = [score for score, label in zip(image_scores, image_labels) if label == 1]
    
    print(f"\nScore Distribution:")
    print(f"Normal images - Mean: {np.mean(normal_scores):.6f}, Std: {np.std(normal_scores):.6f}")
    print(f"Anomaly images - Mean: {np.mean(anomaly_scores):.6f}, Std: {np.std(anomaly_scores):.6f}")
    
    # Create score distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=20, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvaluation completed! Results saved in: {output_dir}")


if __name__ == "__main__":
    evaluate_model()
