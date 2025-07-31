# Detection Module

This directory contains the main codebase for anomaly detection using deep learning on RGBD images. The system is designed for industrial inspection and quality control tasks, leveraging a ResNet50 backbone, multi-scale feature fusion, and a convolutional autoencoder.

## Structure

- `src/` - Source code for core modules, models, and utilities.
- `tests/` - Integration and unit tests for the detection pipeline.
- `config/` - Configuration files, including `requirements.txt` for dependencies.
- `anomaly_detection_results/`, `anomaly_test_results/`, `evaluation_results/` - Output folders for results and visualizations.
- Model weights and checkpoints (e.g., `anomaly_detection_model_final_400.pth`).
- Training and test scripts (e.g., `train_pipeline.py`, `test_anomaly_quick.py`).
- Utility scripts for inference, visualization, and debugging.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r config/requirements.txt
   ```
2. **Train the model:**
   ```bash
   python train_pipeline.py
   ```
3. **Run anomaly detection:**
   ```bash
   python test_anomaly_quick.py
   ```

## Main Features
- Deep learning-based anomaly detection for RGBD images
- Multi-view visualizations: original, fused features, depth map, anomaly heatmap overlay
- Object-focused thresholding using depth information
- Modular codebase for easy extension and maintenance

## Results
- Outputs are saved in the `anomaly_test_results/` directory
- Visualizations include anomaly maps and overlays for easy interpretation

## Requirements
See `config/requirements.txt` for all required Python packages.

## Contact
For questions or support, please contact the development team.
