"""
Visualization Module
===================

Provides visualization utilities for RGB-D data including side-by-side
comparisons, depth map colorization, and 3D point cloud visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, Union, List
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


class RGBDVisualizer:
    """
    Visualization utilities for RGB-D data.
    
    Provides methods for displaying RGB images, depth maps, and
    combined visualizations for analysis and debugging.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
    
    def visualize_side_by_side(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        if titles is None:
            titles = ['RGB Image', 'Depth Map']
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        if rgb_image.max() > 1.0:
            rgb_display = rgb_image / 255.0
        else:
            rgb_display = rgb_image
        axes[0].imshow(rgb_display)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        depth_colored = self.colorize_depth(depth_map)
        im = axes[1].imshow(depth_colored)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def colorize_depth(
        self,
        depth_map: np.ndarray,
        colormap: str = 'plasma',
        invert: bool = False
    ) -> np.ndarray:
        if invert:
            depth_map = 1.0 - depth_map
        colormap_fn = cm.get_cmap(colormap)
        colored_depth = colormap_fn(depth_map)
        if colored_depth.shape[-1] == 4:
            colored_depth = colored_depth[..., :3]
        return colored_depth
    
    def visualize_rgbd_volume(
        self,
        rgbd_volume: np.ndarray,
        slice_indices: Optional[List[int]] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        if rgbd_volume.shape[0] == 4:
            rgbd_volume = rgbd_volume.transpose(1, 2, 0)
        if slice_indices is None:
            slice_indices = [0, 1, 2, 3]
        if titles is None:
            titles = ['Red Channel', 'Green Channel', 'Blue Channel', 'Depth Channel']
        n_channels = len(slice_indices)
        fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
        if n_channels == 1:
            axes = [axes]
        for i, (idx, title) in enumerate(zip(slice_indices, titles)):
            channel = rgbd_volume[..., idx]
            if idx < 3:
                axes[i].imshow(channel, cmap='gray', vmin=0, vmax=1)
            else:
                colored_depth = self.colorize_depth(channel)
                im = axes[i].imshow(colored_depth)
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].set_title(title)
            axes[i].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def create_depth_histogram(
        self,
        depth_map: np.ndarray,
        bins: int = 50,
        title: str = "Depth Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=(8, 6))
        depth_values = depth_map.flatten()
        depth_values = depth_values[depth_values > 0]
        ax.hist(depth_values, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Depth Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        mean_depth = np.mean(depth_values)
        median_depth = np.median(depth_values)
        ax.axvline(mean_depth, color='red', linestyle='--', label=f'Mean: {mean_depth:.3f}')
        ax.axvline(median_depth, color='orange', linestyle='--', label=f'Median: {median_depth:.3f}')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def visualize_3d_pointcloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        downsample_factor: int = 10,
        max_depth: float = 1.0,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        height, width = depth_map.shape
        y_coords, x_coords = np.mgrid[0:height:downsample_factor, 0:width:downsample_factor]
        depth_values = depth_map[::downsample_factor, ::downsample_factor]
        rgb_values = rgb_image[::downsample_factor, ::downsample_factor]
        valid_mask = (depth_values > 0) & (depth_values <= max_depth)
        x_3d = x_coords[valid_mask]
        y_3d = y_coords[valid_mask]
        z_3d = depth_values[valid_mask] * 100
        colors = rgb_values[valid_mask]
        if colors.max() > 1.0:
            colors = colors / 255.0
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_3d, y_3d, z_3d, c=colors, s=1, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')
        ax.set_title('3D Point Cloud from RGB-D Data')
        ax.invert_yaxis()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def compare_depth_methods(
        self,
        rgb_image: np.ndarray,
        depth_maps: List[np.ndarray],
        method_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        n_methods = len(depth_maps)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        if rgb_image.max() > 1.0:
            rgb_display = rgb_image / 255.0
        else:
            rgb_display = rgb_image
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        for i, (depth_map, method_name) in enumerate(zip(depth_maps, method_names)):
            col_idx = i + 1
            colored_depth = self.colorize_depth(depth_map)
            im = axes[0, col_idx].imshow(colored_depth)
            axes[0, col_idx].set_title(f'{method_name} - Depth')
            axes[0, col_idx].axis('off')
            depth_values = depth_map.flatten()
            depth_values = depth_values[depth_values > 0]
            axes[1, col_idx].hist(depth_values, bins=30, alpha=0.7)
            axes[1, col_idx].set_title(f'{method_name} - Distribution')
            axes[1, col_idx].set_xlabel('Depth Value')
            axes[1, col_idx].set_ylabel('Frequency')
            axes[1, col_idx].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            return None
        else:
            return fig

if __name__ == "__main__":
    data = np.load('output/0001_rgbd.npz')
    print("Keys in npz:", data.files)
    visualizer = RGBDVisualizer()
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 3 and arr.shape[-1] == 4:
            rgb = arr[..., :3]
            depth = arr[..., 3]
            print(f"Visualizing {key} as RGB (left) and Depth (right)...")
            visualizer.visualize_side_by_side(rgb, depth)
        elif arr.ndim == 3 and arr.shape[0] == 4:
            print(f"Visualizing {key} as channels-first RGBD volume...")
            visualizer.visualize_rgbd_volume(arr)
        elif arr.ndim == 2:
            print(f"Visualizing {key} as grayscale image...")
            plt.imshow(arr, cmap='gray')
            plt.title(key)
            plt.axis('off')
            plt.show()
        else:
            print(f"{key} has unsupported shape for visualization.")
