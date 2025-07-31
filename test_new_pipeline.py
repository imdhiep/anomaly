import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from train_new_pipeline import RGBDAnomalyDetector, load_rgbd_data

def load_trained_model(model_path: str, device: str = 'cuda') -> RGBDAnomalyDetector:
    """Load trained RGB-D anomaly detector."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = RGBDAnomalyDetector(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.threshold = torch.tensor(checkpoint.get('threshold', 0.0)).to(device)
    model.to(device)
    model.eval()
    
    print(f"Loaded model with threshold: {model.threshold.item():.6f}")
    
    return model

def test_on_anomaly_samples(model: RGBDAnomalyDetector, 
                           anomaly_folder: str,
                           output_folder: str = "new_anomaly_results",
                           max_samples: int = 5,
                           device: str = 'cuda'):
    """Test the trained model on anomaly samples."""
    
    # Load anomaly samples
    anomaly_data = load_rgbd_data(anomaly_folder, max_samples=max_samples)
    
    if len(anomaly_data) == 0:
        print("No anomaly data found!")
        return
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, rgbd_sample in enumerate(anomaly_data):
            print(f"\nProcessing anomaly sample {i+1}/{len(anomaly_data)}...")
            
            # Add batch dimension and move to device
            rgbd_batch = rgbd_sample.unsqueeze(0).to(device)
            
            # Forward pass
            fused_features, reconstructed_features, anomaly_map = model(rgbd_batch)
            
            # Get binary anomaly mask
            binary_mask = model.predict_anomaly(rgbd_batch)
            
            # Convert to numpy for visualization
            rgbd_np = rgbd_sample.cpu().numpy()
            anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
            binary_mask_np = binary_mask.squeeze().cpu().numpy()
            
            # Create comprehensive visualization
            create_comprehensive_visualization(
                rgbd_np, 
                anomaly_map_np, 
                binary_mask_np,
                fused_features.squeeze().cpu().numpy(),
                reconstructed_features.squeeze().cpu().numpy(),
                save_path=output_path / f"anomaly_result_{i+1}.png"
            )
            
            print(f"Mean anomaly score: {anomaly_map_np.mean():.6f}")
            print(f"Max anomaly score: {anomaly_map_np.max():.6f}")
            print(f"Anomaly threshold: {model.threshold.item():.6f}")
            print(f"Anomalous pixels: {binary_mask_np.sum()} / {binary_mask_np.size}")

def create_comprehensive_visualization(rgbd_data: np.ndarray,
                                     anomaly_map: np.ndarray,
                                     binary_mask: np.ndarray,
                                     fused_features: np.ndarray,
                                     reconstructed_features: np.ndarray,
                                     save_path: Path):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original RGB
    rgb_img = rgbd_data[:3].transpose(1, 2, 0)  # (H, W, 3)
    rgb_img = np.clip(rgb_img, 0, 1)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Depth map
    depth_img = rgbd_data[3]  # (H, W)
    im1 = axes[0, 1].imshow(depth_img, cmap='plasma')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Anomaly map (continuous)
    im2 = axes[0, 2].imshow(anomaly_map, cmap='hot')
    axes[0, 2].set_title('Anomaly Score Map')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Binary anomaly mask
    axes[0, 3].imshow(binary_mask, cmap='Reds')
    axes[0, 3].set_title('Binary Anomaly Mask')
    axes[0, 3].axis('off')
    
    # RGB with anomaly overlay
    overlay = rgb_img.copy()
    anomaly_pixels = binary_mask > 0
    overlay[anomaly_pixels] = [1, 0, 0]  # Red overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('RGB + Anomaly Overlay')
    axes[1, 0].axis('off')
    
    # Anomaly score histogram
    axes[1, 1].hist(anomaly_map.flatten(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].axvline(binary_mask.max() * 0.5, color='red', linestyle='--', label='Threshold')
    axes[1, 1].set_title('Anomaly Score Distribution')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Feature reconstruction quality
    reconstruction_error = np.mean((fused_features - reconstructed_features) ** 2, axis=0)
    im3 = axes[1, 2].imshow(reconstruction_error, cmap='viridis')
    axes[1, 2].set_title('Reconstruction Error Map')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    # Statistics
    mean_score = np.mean(anomaly_map)
    max_score = np.max(anomaly_map)
    min_score = np.min(anomaly_map)
    std_score = np.std(anomaly_map)
    anomaly_ratio = np.sum(binary_mask) / binary_mask.size
    
    stats_text = f'''Statistics:
Mean Score: {mean_score:.4f}
Max Score: {max_score:.4f}
Min Score: {min_score:.4f}
Std Score: {std_score:.4f}
Anomaly Ratio: {anomaly_ratio:.4f}
Anomalous Pixels: {np.sum(binary_mask):.0f}'''
    
    axes[1, 3].text(0.1, 0.5, stats_text, transform=axes[1, 3].transAxes, 
                   fontsize=10, verticalalignment='center')
    axes[1, 3].set_title('Statistics')
    axes[1, 3].axis('off')
    
    # Feature analysis
    # Sample feature channels for visualization
    n_channels = min(8, fused_features.shape[0])
    sample_indices = np.linspace(0, fused_features.shape[0]-1, n_channels, dtype=int)
    
    # Original vs reconstructed features
    for idx, ch_idx in enumerate(sample_indices[:4]):
        if idx < 2:
            ax = axes[2, idx*2]
            im = ax.imshow(fused_features[ch_idx], cmap='coolwarm', aspect='auto')
            ax.set_title(f'Original Feature {ch_idx}')
            ax.axis('off')
            
            ax = axes[2, idx*2 + 1]
            im = ax.imshow(reconstructed_features[ch_idx], cmap='coolwarm', aspect='auto')
            ax.set_title(f'Reconstructed Feature {ch_idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to: {save_path}")

def compare_with_normal_samples(model: RGBDAnomalyDetector,
                               normal_folder: str,
                               anomaly_folder: str,
                               device: str = 'cuda'):
    """Compare anomaly scores between normal and anomaly samples."""
    
    # Load samples
    normal_data = load_rgbd_data(normal_folder, max_samples=20)
    anomaly_data = load_rgbd_data(anomaly_folder, max_samples=20)
    
    if len(normal_data) == 0 or len(anomaly_data) == 0:
        print("Insufficient data for comparison!")
        return
    
    model.eval()
    
    normal_scores = []
    anomaly_scores = []
    
    # Process normal samples
    print("Processing normal samples...")
    with torch.no_grad():
        for rgbd_sample in normal_data:
            rgbd_batch = rgbd_sample.unsqueeze(0).to(device)
            _, _, anomaly_map = model(rgbd_batch)
            normal_scores.append(anomaly_map.mean().item())
    
    # Process anomaly samples
    print("Processing anomaly samples...")
    with torch.no_grad():
        for rgbd_sample in anomaly_data:
            rgbd_batch = rgbd_sample.unsqueeze(0).to(device)
            _, _, anomaly_map = model(rgbd_batch)
            anomaly_scores.append(anomaly_map.mean().item())
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=20, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(model.threshold.item(), color='black', linestyle='--', label='Threshold')
    plt.xlabel('Mean Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
    plt.axhline(model.threshold.item(), color='black', linestyle='--', label='Threshold')
    plt.ylabel('Mean Anomaly Score')
    plt.title('Anomaly Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\nComparison Results:")
    print(f"Normal samples - Mean: {np.mean(normal_scores):.6f}, Std: {np.std(normal_scores):.6f}")
    print(f"Anomaly samples - Mean: {np.mean(anomaly_scores):.6f}, Std: {np.std(anomaly_scores):.6f}")
    print(f"Threshold: {model.threshold.item():.6f}")
    
    # Classification accuracy
    normal_correct = sum(score < model.threshold.item() for score in normal_scores)
    anomaly_correct = sum(score >= model.threshold.item() for score in anomaly_scores)
    total_correct = normal_correct + anomaly_correct
    total_samples = len(normal_scores) + len(anomaly_scores)
    
    print(f"Classification Accuracy: {total_correct}/{total_samples} = {total_correct/total_samples:.2%}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = "rgbd_anomaly_detector.pth"
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found! Please train the model first.")
        exit(1)
    
    model = load_trained_model(model_path, device)
    
    # Test on anomaly samples
    anomaly_folder = "../preprocess/output/Anomaly"
    test_on_anomaly_samples(model, anomaly_folder, device=device)
    
    # Compare with normal samples
    normal_folder = "../preprocess/output/Normal"
    compare_with_normal_samples(model, normal_folder, anomaly_folder, device=device)
    
    print("\nInference complete!")
