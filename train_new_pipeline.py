import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.models.feature_embedding import ResNet50RGBD, MultiScaleFeatureFusion, ConvolutionalAutoencoder

class RGBDAnomalyDetector(nn.Module):
    """Complete RGB-D Anomaly Detection pipeline using ResNet50 + Convolutional Autoencoder."""
    
    def __init__(self, 
                 feature_channels: dict = None,
                 target_size: tuple = (128, 128),
                 hidden_channels: list = None,
                 latent_channels: int = 64):
        super(RGBDAnomalyDetector, self).__init__()
        
        # Default feature channels for ResNet50 layers
        if feature_channels is None:
            feature_channels = {
                'layer1': 256,
                'layer2': 512, 
                'layer3': 1024,
                'layer4': 2048
            }
        
        if hidden_channels is None:
            hidden_channels = [512, 256, 128]
        
        self.feature_channels = feature_channels
        self.target_size = target_size
        
        # ResNet50 backbone for feature extraction
        self.backbone = ResNet50RGBD(pretrained=True)
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            feature_channels=feature_channels,
            target_size=target_size,
            use_conv_smooth=True
        )
        
        # Convolutional Autoencoder
        total_channels = sum(feature_channels.values())
        self.autoencoder = ConvolutionalAutoencoder(
            input_channels=total_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels
        )
        
        # Threshold for anomaly detection (learned during training)
        self.register_buffer('threshold', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the complete pipeline.
        
        Args:
            x: RGB-D input tensor of shape (B, 4, H, W)
            
        Returns:
            Tuple of (fused_features, reconstructed_features, anomaly_map)
        """
        # Extract multi-scale features
        layer_names = list(self.feature_channels.keys())
        feature_maps = self.backbone.extract_features(x, layer_names)
        
        # Fuse features
        fused_features = self.feature_fusion(feature_maps)
        
        # Reconstruct via autoencoder
        reconstructed_features = self.autoencoder(fused_features)
        
        # Compute per-pixel reconstruction error
        anomaly_map = torch.norm(fused_features - reconstructed_features, p=2, dim=1, keepdim=True)
        
        return fused_features, reconstructed_features, anomaly_map
    
    def compute_loss(self, fused_features: torch.Tensor, reconstructed_features: torch.Tensor) -> torch.Tensor:
        """Compute L2/MSE reconstruction loss."""
        return F.mse_loss(reconstructed_features, fused_features)
    
    def predict_anomaly(self, x: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """
        Predict binary anomaly mask.
        
        Args:
            x: RGB-D input tensor
            threshold: Anomaly threshold (uses learned threshold if None)
            
        Returns:
            Binary anomaly mask
        """
        with torch.no_grad():
            _, _, anomaly_map = self.forward(x)
            
            if threshold is None:
                threshold = self.threshold.item()
            
            # Upsample anomaly map to original image size
            original_size = (x.shape[2], x.shape[3])
            anomaly_map_upsampled = F.interpolate(
                anomaly_map, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Apply threshold
            binary_mask = (anomaly_map_upsampled > threshold).float()
            
            return binary_mask


def load_rgbd_data(folder_path: str, max_samples: int = None) -> list:
    """Load RGB-D data from preprocessed NPZ files."""
    folder_path = Path(folder_path)
    npz_files = list(folder_path.glob("*.npz"))
    
    if max_samples:
        npz_files = npz_files[:max_samples]
    
    rgbd_samples = []
    
    print(f"Loading {len(npz_files)} RGB-D samples...")
    
    for npz_file in tqdm(npz_files):
        try:
            data = np.load(npz_file)
            
            for key in data.keys():
                arr = data[key]
                
                if arr.ndim == 3 and arr.shape[-1] == 4:  # RGBD format (H, W, 4)
                    # Convert to (4, H, W) format
                    rgbd = arr.transpose(2, 0, 1)  # (4, H, W)
                    rgbd_tensor = torch.from_numpy(rgbd).float()
                    
                    # Normalize channels
                    # RGB channels (0-255) -> (0-1)
                    rgbd_tensor[:3] = rgbd_tensor[:3] / 255.0
                    # Depth channel - normalize by max value
                    if rgbd_tensor[3].max() > 0:
                        rgbd_tensor[3] = rgbd_tensor[3] / rgbd_tensor[3].max()
                    
                    rgbd_samples.append(rgbd_tensor)
                    break
                    
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(rgbd_samples)} RGB-D samples")
    return rgbd_samples


def train_anomaly_detector(model: RGBDAnomalyDetector, 
                          train_data: list,
                          val_data: list = None,
                          batch_size: int = 4,
                          num_epochs: int = 700,
                          learning_rate: float = 1e-4,
                          device: str = 'cuda',
                          save_path: str = 'rgbd_anomaly_detector.pth') -> dict:
    """
    Training loop for RGB-D anomaly detector.
    
    Args:
        model: RGBDAnomalyDetector model
        train_data: List of normal RGB-D samples
        val_data: List of validation RGB-D samples (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Adam learning rate
        device: Training device
        save_path: Path to save trained model
        
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    model.train()
    
    # Only train autoencoder parameters (backbone is frozen)
    optimizer = optim.Adam(model.autoencoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            
            # Stack batch and move to device
            batch_tensor = torch.stack(batch_data).to(device)
            
            # Forward pass
            fused_features, reconstructed_features, _ = model(batch_tensor)
            
            # Compute loss
            loss = model.compute_loss(fused_features, reconstructed_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_data:
            model.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch_data = val_data[i:i+batch_size]
                    batch_tensor = torch.stack(batch_data).to(device)
                    
                    fused_features, reconstructed_features, _ = model(batch_tensor)
                    loss = model.compute_loss(fused_features, reconstructed_features)
                    
                    epoch_val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = epoch_val_loss / val_batches
            history['val_loss'].append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, save_path)
        
        # Update learning rate
        scheduler.step()
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        if epoch % 10 == 0:
            if val_data:
                print(f'Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            else:
                print(f'Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    # Compute threshold on training data
    print("Computing anomaly threshold...")
    model.eval()
    all_anomaly_scores = []
    
    with torch.no_grad():
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_tensor = torch.stack(batch_data).to(device)
            
            _, _, anomaly_map = model(batch_tensor)
            all_anomaly_scores.append(anomaly_map.cpu())
    
    all_scores = torch.cat(all_anomaly_scores, dim=0)
    threshold = torch.quantile(all_scores, 0.95)  # 95th percentile
    model.threshold = threshold.to(device)
    
    print(f"Computed anomaly threshold: {threshold.item():.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold.item(),
        'history': history,
        'model_config': {
            'feature_channels': model.feature_channels,
            'target_size': model.target_size
        }
    }, save_path)
    
    print(f"Training complete. Model saved to {save_path}")
    
    return history


def plot_training_history(history: dict):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Training Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Learning rate
    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load training data (normal samples only)
    normal_folder = "../preprocess/output/Normal"
    train_data = load_rgbd_data(normal_folder, max_samples=200)  # Limit for memory
    
    if len(train_data) == 0:
        print("No training data found!")
        exit(1)
    
    # Split train/val
    val_split = 0.2
    val_size = int(len(train_data) * val_split)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create model
    model = RGBDAnomalyDetector()
    
    # Train model
    history = train_anomaly_detector(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=4,
        num_epochs=50,  # Reduced for testing
        learning_rate=1e-4,
        device=device,
        save_path='rgbd_anomaly_detector.pth'
    )
    
    # Plot training curves
    plot_training_history(history)
