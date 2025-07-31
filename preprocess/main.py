"""
Main entry point for the 2D to 3D Preprocessing Pipeline.
Allows running the RGB to RGB-D conversion from the command line or as a script.
"""
import argparse
from core.rgb_to_rgbd import RGBToRGBDProcessor
from utils.visualization import RGBDVisualizer
import os
from pathlib import Path
import cv2


def main():

    parser = argparse.ArgumentParser(description="2D to 3D RGB-D Preprocessing Pipeline")
    parser.add_argument("input", type=str, nargs="?", default="input", help="Input image file or folder (default: ./input)")
    parser.add_argument("output", type=str, nargs="?", default="output", help="Output file or folder (default: ./output)")
    parser.add_argument("--filter", type=str, default="gaussian", choices=["gaussian", "median", "bilateral"], help="Filter type")
    parser.add_argument("--equalization", type=str, default="hsv", choices=["yuv", "hsv", "lab"], help="Equalization method")
    parser.add_argument("--depth-model", type=str, default="midas_small", choices=["midas_small", "midas_large", "dpt_hybrid"], help="Depth estimation model")
    parser.add_argument("--output-size", type=int, nargs=2, default=None, metavar=("W", "H"), help="Output size (width height)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--batch", action="store_true", help="Process a folder of images (default: True if using default input/output folders)")
    parser.add_argument("--folder", type=str, choices=["anomaly", "normal", "both"], default="both", help="Which folder to process (default: both)")
    args = parser.parse_args()

    # If using default input/output, set batch to True
    if (args.input == "input" and args.output == "output"):
        args.batch = True

    processor = RGBToRGBDProcessor(
        filter_type=args.filter,
        equalization_method=args.equalization,
        depth_model=args.depth_model,
        output_size=tuple(args.output_size) if args.output_size else None
    )

    if args.batch:
        # Batch processing
        script_dir = Path(__file__).parent
        input_dir_anomaly = (script_dir / args.input / "Anomaly").resolve()
        input_dir_normal = (script_dir / args.input / "Normal").resolve()
        output_dir_anomaly = (script_dir / args.output / "Anomaly").resolve()
        output_dir_normal = (script_dir / args.output / "Normal").resolve()

        os.makedirs(output_dir_anomaly, exist_ok=True)
        os.makedirs(output_dir_normal, exist_ok=True)

        print(f"[DEBUG] Looking for images in: {input_dir_anomaly}")
        jpgs_anomaly = list(input_dir_anomaly.glob("*.jpg")) + list(input_dir_anomaly.glob("*.JPG"))
        pngs_anomaly = list(input_dir_anomaly.glob("*.png")) + list(input_dir_anomaly.glob("*.PNG"))
        print(f"[DEBUG] Found .jpg/.JPG files in Anomaly: {[str(p) for p in jpgs_anomaly]}")
        print(f"[DEBUG] Found .png/.PNG files in Anomaly: {[str(p) for p in pngs_anomaly]}")

        print(f"[DEBUG] Looking for images in: {input_dir_normal}")
        jpgs_normal = list(input_dir_normal.glob("*.jpg")) + list(input_dir_normal.glob("*.JPG"))
        pngs_normal = list(input_dir_normal.glob("*.png")) + list(input_dir_normal.glob("*.PNG"))
        print(f"[DEBUG] Found .jpg/.JPG files in Normal: {[str(p) for p in jpgs_normal]}")
        print(f"[DEBUG] Found .png/.PNG files in Normal: {[str(p) for p in pngs_normal]}")

        input_paths_anomaly = jpgs_anomaly + pngs_anomaly
        input_paths_normal = jpgs_normal + pngs_normal

        if not input_paths_anomaly and not input_paths_normal:
            print(f"No images found in {input_dir_anomaly} or {input_dir_normal}. Please add .jpg or .png files to the input folders.")
            return

        if input_paths_anomaly and (args.folder == "anomaly" or args.folder == "both"):
            print(f"Processing {len(input_paths_anomaly)} images in Anomaly folder...")
            rgbd_batch_anomaly = processor.process_batch(input_paths_anomaly)
            for path, rgbd in zip(input_paths_anomaly, rgbd_batch_anomaly):
                out_path = output_dir_anomaly / f"{Path(path).stem}_rgbd.npz"
                processor.save_rgbd_volume(rgbd, out_path)
            print(f"Processed {len(rgbd_batch_anomaly)} images in Anomaly. Results saved to {output_dir_anomaly}")

        if input_paths_normal and (args.folder == "normal" or args.folder == "both"):
            print(f"Processing {len(input_paths_normal)} images in Normal folder...")
            rgbd_batch_normal = processor.process_batch(input_paths_normal)
            for path, rgbd in zip(input_paths_normal, rgbd_batch_normal):
                out_path = output_dir_normal / f"{Path(path).stem}_rgbd.npz"
                processor.save_rgbd_volume(rgbd, out_path)
            print(f"Processed {len(rgbd_batch_normal)} images in Normal. Results saved to {output_dir_normal}")
    else:
        # Single image
        script_dir = Path(__file__).parent
        input_path = (script_dir / args.input).resolve()
        output_path = (script_dir / args.output).resolve()
        rgbd = processor.process_image(str(input_path), visualize=args.visualize)
        processor.save_rgbd_volume(rgbd, str(output_path))
        print(f"Processed {input_path} -> {output_path}")
        if args.visualize:
            rgb = cv2.imread(str(input_path))
            visualizer = RGBDVisualizer()
            visualizer.visualize_side_by_side(rgb, rgbd[..., 3])

if __name__ == "__main__":
    main()
