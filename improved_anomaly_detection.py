import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import os

class RGBDResNet50Extractor(nn.Module):
    """ResNet50 feature extractor modified for RGBD input"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Replace first conv layer for 4-channel input (RGBD)
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from pretrained RGB channels
        with torch.no_grad():
            # Copy RGB weights
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            # Initialize depth channel with average of RGB channels
            new_conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Build the feature extractor
        self.conv1 = new_conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels  
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Freeze backbone
        self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all ResNet parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract multi-scale features"""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Multi-scale features
        f1 = self.layer1(x)    # 1/4 resolution, 64 channels
        f2 = self.layer2(f1)   # 1/8 resolution, 128 channels
        f3 = self.layer3(f2)   # 1/16 resolution, 256 channels
        f4 = self.layer4(f3)   # 1/32 resolution, 512 channels
        
        return f1, f2, f3, f4


class MultiScaleFusion(nn.Module):
    """Fuse multi-scale features into single representation"""
    
    def __init__(self, target_size=(64, 64), smooth_features=True):
        super().__init__()
        self.target_size = target_size
        self.smooth_features = smooth_features
        
        if smooth_features:
            # 1x1 conv layers for feature smoothing
            self.smooth1 = nn.Conv2d(64, 64, kernel_size=1)
            self.smooth2 = nn.Conv2d(128, 128, kernel_size=1)
            self.smooth3 = nn.Conv2d(256, 256, kernel_size=1)
            self.smooth4 = nn.Conv2d(512, 512, kernel_size=1)
        
        # Total fused channels: 64 + 128 + 256 + 512 = 960
        self.total_channels = 64 + 128 + 256 + 512
    
    def forward(self, f1, f2, f3, f4):
        """Fuse multi-scale features"""
        # Upsample all features to target size
        f1_up = F.interpolate(f1, size=self.target_size, mode='bilinear', align_corners=False)
        f2_up = F.interpolate(f2, size=self.target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=self.target_size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Apply smoothing if enabled
        if self.smooth_features:
            f1_up = self.smooth1(f1_up)
            f2_up = self.smooth2(f2_up)
            f3_up = self.smooth3(f3_up)
            f4_up = self.smooth4(f4_up)
        
        # Concatenate along channel dimension
        fused = torch.cat([f1_up, f2_up, f3_up, f4_up], dim=1)
        
        return fused


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder with only 1x1 convolutions"""
    
    def __init__(self, input_channels=960, latent_channels=128):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        
        # Encoder: series of 1x1 convolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, input_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class AnomalyDetectionModel(nn.Module):
    """Complete anomaly detection model"""
    
    def __init__(self, target_size=(64, 64), latent_channels=128):
        super().__init__()
        
        self.feature_extractor = RGBDResNet50Extractor(pretrained=True)
        self.fusion = MultiScaleFusion(target_size=target_size)
        self.autoencoder = ConvolutionalAutoencoder(
            input_channels=self.fusion.total_channels,
            latent_channels=latent_channels
        )
        self.target_size = target_size
    
    def forward(self, x):
        """Forward pass"""
        # Extract multi-scale features
        f1, f2, f3, f4 = self.feature_extractor(x)
        
        # Fuse features
        fused_features = self.fusion(f1, f2, f3, f4)
        
        # Reconstruct through autoencoder
        reconstructed = self.autoencoder(fused_features)
        
        return fused_features, reconstructed
    
    def compute_anomaly_map(self, x, original_size=None):
        """Compute anomaly map for input"""
        fused_features, reconstructed = self(x)
        
        # Compute per-location L2 error
        error = torch.norm(fused_features - reconstructed, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Upsample to original image size if specified
        if original_size is not None:
            error = F.interpolate(error, size=original_size, mode='bilinear', align_corners=False)
        
        return error


class RGBDDataset(Dataset):
    """Dataset for RGBD anomaly detection"""
    
    def __init__(self, data_folder, image_size=(256, 256)):
        self.data_folder = Path(data_folder)
        self.image_size = image_size
        self.npz_files = list(self.data_folder.glob("*.npz"))
        
        print(f"Found {len(self.npz_files)} files in {data_folder}")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        
        # Load RGBD data
        data = np.load(npz_file)
        
        # Find RGBD array
        rgbd_array = None
        for key in data.keys():
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 4:  # (H, W, 4)
                rgbd_array = arr
                break
        
        if rgbd_array is None:
            # Return zero tensor if no valid data
            return torch.zeros(4, *self.image_size)
        
        # Convert to tensor and resize
        rgbd_tensor = torch.from_numpy(rgbd_array).float()  # (H, W, 4)
        rgbd_tensor = rgbd_tensor.permute(2, 0, 1)  # (4, H, W)
        
        # Resize to target size
        rgbd_tensor = rgbd_tensor.unsqueeze(0)  # (1, 4, H, W)
        rgbd_tensor = F.interpolate(rgbd_tensor, size=self.image_size, mode='bilinear', align_corners=False)
        rgbd_tensor = rgbd_tensor.squeeze(0)  # (4, H, W)
        
        # Normalize channels
        # RGB channels (0-255) -> [0, 1]
        rgbd_tensor[:3] = rgbd_tensor[:3] / 255.0
        # Depth channel - normalize to [0, 1] range
        depth_channel = rgbd_tensor[3:4]
        if depth_channel.max() > depth_channel.min():
            depth_channel = (depth_channel - depth_channel.min()) / (depth_channel.max() - depth_channel.min())
        rgbd_tensor[3:4] = depth_channel
        
        return rgbd_tensor


def train_anomaly_model(model, train_loader, num_epochs=700, learning_rate=1e-4, device='cuda'):
    """Train the anomaly detection model"""
    
    model = model.to(device)
    model.train()
    
    # Only optimize autoencoder parameters (backbone is frozen)
    optimizer = optim.Adam(model.autoencoder.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_data in progress_bar:
            batch_data = batch_data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            fused_features, reconstructed = model(batch_data)
            
            # Compute loss
            loss = criterion(reconstructed, fused_features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f'model_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_losses


def test_anomaly_detection(model, test_data_path, device='cuda', threshold_percentile=95):
    """Test anomaly detection on anomalous samples"""
    
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    test_dataset = RGBDDataset(test_data_path, image_size=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create output directory
    output_dir = Path("improved_anomaly_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Testing on {len(test_dataset)} samples...")
    
    all_scores = []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            if idx >= 10:  # Test first 10 samples
                break
                
            batch_data = batch_data.to(device)
            
            # Compute anomaly map
            anomaly_map = model.compute_anomaly_map(batch_data, original_size=(256, 256))
            
            # Move to CPU for processing
            anomaly_map = anomaly_map.squeeze().cpu().numpy()  # (256, 256)
            rgbd_data = batch_data.squeeze().cpu().numpy()     # (4, 256, 256)
            
            # Store scores for threshold calculation
            all_scores.extend(anomaly_map.flatten())
            
            # Create visualization
            create_improved_visualization(rgbd_data, anomaly_map, 
                                        output_dir / f"anomaly_result_{idx:03d}.png",
                                        threshold_percentile)
    
    print(f"Results saved in {output_dir}")
    
    # Calculate and print statistics
    all_scores = np.array(all_scores)
    threshold = np.percentile(all_scores, threshold_percentile)
    
    print(f"Anomaly score statistics:")
    print(f"  Mean: {np.mean(all_scores):.6f}")
    print(f"  Std: {np.std(all_scores):.6f}")
    print(f"  Min: {np.min(all_scores):.6f}")
    print(f"  Max: {np.max(all_scores):.6f}")
    print(f"  {threshold_percentile}th percentile threshold: {threshold:.6f}")


def create_improved_visualization(rgbd_data, anomaly_map, save_path, threshold_percentile=95):
    """Create comprehensive visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract RGB and depth
    rgb_img = rgbd_data[:3].transpose(1, 2, 0)  # (H, W, 3)
    depth_img = rgbd_data[3]  # (H, W)
    
    # Original RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Depth map
    im1 = axes[0, 1].imshow(depth_img, cmap='plasma')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Anomaly map
    im2 = axes[0, 2].imshow(anomaly_map, cmap='hot')
    axes[0, 2].set_title('Anomaly Map')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # RGB with anomaly overlay
    overlay = rgb_img.copy()
    threshold = np.percentile(anomaly_map, threshold_percentile)
    anomaly_mask = anomaly_map > threshold
    
    overlay[anomaly_mask] = [1.0, 0.0, 0.0]  # Red overlay
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('RGB + Anomaly Overlay')
    axes[1, 0].axis('off')
    
    # Anomaly histogram
    axes[1, 1].hist(anomaly_map.flatten(), bins=50, alpha=0.7)
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'{threshold_percentile}th percentile')
    axes[1, 1].set_title('Anomaly Score Distribution')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # Statistics
    stats_text = f'''Statistics:
Mean: {np.mean(anomaly_map):.4f}
Std: {np.std(anomaly_map):.4f}
Min: {np.min(anomaly_map):.4f}
Max: {np.max(anomaly_map):.4f}
Threshold: {threshold:.4f}'''
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training and testing pipeline"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = AnomalyDetectionModel(target_size=(64, 64), latent_channels=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} total parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create training dataset
    normal_data_path = "../preprocess/output/Normal"
    train_dataset = RGBDDataset(normal_data_path, image_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Train model
    print("Starting training...")
    train_losses = train_anomaly_model(
        model=model,
        train_loader=train_loader,
        num_epochs=50,  # Start with fewer epochs for testing
        learning_rate=1e-4,
        device=device
    )
    
    # Save final model
    torch.save(model.state_dict(), 'improved_anomaly_model.pth')
    print("Model saved as improved_anomaly_model.pth")
    
    # Test on anomalous data
    print("Testing on anomalous samples...")
    anomaly_data_path = "../preprocess/output/Anomaly"
    test_anomaly_detection(model, anomaly_data_path, device=device)


if __name__ == "__main__":
    main()
