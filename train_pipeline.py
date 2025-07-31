import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.models.feature_embedding import ResNet50RGBD, MultiScaleFeatureFusion, ConvolutionalAutoencoder


class RGBDDataset(Dataset):
    """Dataset for RGBD images."""
    
    def __init__(self, npz_folder):
        self.npz_files = list(Path(npz_folder).glob("*.npz"))
        print(f"Found {len(self.npz_files)} RGBD files")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        
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
                
                return rgbd
        
        # Fallback: return zeros if no valid data found
        return torch.zeros(4, 256, 256)


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


def train_model():
    """Training loop for the anomaly detection model."""
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 400,  # Increase training epochs
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_interval': 50,
        'log_interval': 10
    }
    
    print(f"Using device: {config['device']}")
    
    # Create dataset and dataloader
    dataset = RGBDDataset("../preprocess/output/Normal")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    if len(dataset) == 0:
        print("No training data found!")
        return
    
    # Initialize model
    model = AnomalyDetectionModel().to(config['device'])
    
    # Only optimize autoencoder parameters (feature extractor is frozen)
    optimizer = optim.Adam(model.autoencoder.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    
    print("Starting training...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, rgbd_batch in enumerate(progress_bar):
            rgbd_batch = rgbd_batch.to(config['device'])
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            original_features, reconstructed_features = model(rgbd_batch)
            
            # Compute reconstruction loss
            loss = criterion(reconstructed_features, original_features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        if (epoch + 1) % config['log_interval'] == 0:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save model checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses
            }
            torch.save(checkpoint, f'model_checkpoint_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'config': config
    }, 'anomaly_detection_model_final_400.pth')
    
    print("Training completed!")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, train_losses


if __name__ == "__main__":
    model, losses = train_model()
