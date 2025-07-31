import torch
import numpy as np
from src.core.inference import inference_pipeline

# Dummy classes for demonstration (replace with your actual implementations)
class DummyFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def extract_features(self, x, layer_names):
        # Return a dict of random tensors for each layer
        return {name: torch.randn(1, 8, 16, 16, 16) for name in layer_names}

class DummyFeatureEmbedding(torch.nn.Module):
    embed_dim = 32
    def __init__(self):
        super().__init__()
    def forward(self, patches):
        # Embed patches to random vectors
        return torch.randn(patches.shape[0], self.embed_dim)
    def __call__(self, patches):
        return self.forward(patches)

class DummyPatchSampler:
    def __call__(self, feature_map):
        # Sample random patches and coordinates
        N = 10
        patches = torch.randn(N, *feature_map.shape[1:])
        coordinates = torch.randint(0, 16, (N, 3))
        return patches, coordinates

class DummyAutoencoder(torch.nn.Module):
    def forward(self, x):
        # Return random reconstructions
        return x + torch.randn_like(x) * 0.1
    def __call__(self, x):
        return self.forward(x)

# Load npz file
npz_path = r"d:\final_year_project\private\detection\0001_rgbd.npz"
data = np.load(npz_path)

# Try to find the main key
main_key = list(data.keys())[0]
volume_np = data[main_key]

# Ensure shape is (C, D, H, W)
if volume_np.ndim == 4:
    volume = torch.from_numpy(volume_np).float()
else:
    raise ValueError(f"Expected 4D array, got shape {volume_np.shape}")

# Instantiate dummy models
feature_extractor = DummyFeatureExtractor()
embedding_module = DummyFeatureEmbedding()
patch_sampler = DummyPatchSampler()
autoencoder = DummyAutoencoder()

layer_names = ['layer1']  # Use your actual layer names
patch_size = (8, 8, 8)   # Use your actual patch size

score_map, metadata = inference_pipeline(
    volume,
    feature_extractor,
    embedding_module,
    patch_sampler,
    autoencoder,
    layer_names,
    patch_size,
    aggregation='mean',
    smoothing_sigma=None,
    device='cpu'
)

print("Anomaly score map shape:", score_map.shape)
print("Metadata:", metadata)
