#!/usr/bin/env python3
"""
Axelera Model Compilation Script

This script compiles TFLite models to Axelera Metis format using the Voyager SDK.
It handles quantization (if needed) and compilation for the Axelera AIPU.

Prerequisites:
- Axelera Voyager SDK installed
- TFLite source models downloaded (run convert_models.sh first)
- Calibration dataset (images for quantization)

Usage:
    python3 compile_axelera_models.py --model palm_detection_lite
    python3 compile_axelera_models.py --model hand_landmark_full --calib-dir ../../calib_dataset_kaggle/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from axelera import compiler
    print("[INFO] Axelera Voyager SDK found")
except ImportError as e:
    print(f"[ERROR] Failed to import Axelera Voyager SDK: {e}")
    print("Please install the Voyager SDK from: https://github.com/axelera-ai-hub/voyager-sdk")
    sys.exit(1)

# Model configurations
MODEL_CONFIGS = {
    "palm_detection_lite": {
        "source": "tflite_source/palm_detection_lite.tflite",
        "output": "palm_detection_lite.axmodel",
        "input_size": (192, 192),
        "type": "detection"
    },
    "palm_detection_full": {
        "source": "tflite_source/palm_detection_full.tflite",
        "output": "palm_detection_full.axmodel",
        "input_size": (192, 192),
        "type": "detection"
    },
    "hand_landmark_lite": {
        "source": "tflite_source/hand_landmark_lite.tflite",
        "output": "hand_landmark_lite.axmodel",
        "input_size": (224, 224),
        "type": "landmark"
    },
    "hand_landmark_full": {
        "source": "tflite_source/hand_landmark_full.tflite",
        "output": "hand_landmark_full.axmodel",
        "input_size": (224, 224),
        "type": "landmark"
    },
    "face_detection_short_range": {
        "source": "tflite_source/face_detection_short_range.tflite",
        "output": "face_detection_short_range.axmodel",
        "input_size": (128, 128),
        "type": "detection"
    },
    "face_detection_full_range": {
        "source": "tflite_source/face_detection_full_range.tflite",
        "output": "face_detection_full_range.axmodel",
        "input_size": (192, 192),
        "type": "detection"
    },
    "face_landmark": {
        "source": "tflite_source/face_landmark.tflite",
        "output": "face_landmark.axmodel",
        "input_size": (192, 192),
        "type": "landmark"
    },
    "pose_detection": {
        "source": "tflite_source/pose_detection.tflite",
        "output": "pose_detection.axmodel",
        "input_size": (224, 224),
        "type": "detection"
    },
    "pose_landmark_lite": {
        "source": "tflite_source/pose_landmark_lite.tflite",
        "output": "pose_landmark_lite.axmodel",
        "input_size": (256, 256),
        "type": "landmark"
    },
    "pose_landmark_full": {
        "source": "tflite_source/pose_landmark_full.tflite",
        "output": "pose_landmark_full.axmodel",
        "input_size": (256, 256),
        "type": "landmark"
    },
    "pose_landmark_heavy": {
        "source": "tflite_source/pose_landmark_heavy.tflite",
        "output": "pose_landmark_heavy.axmodel",
        "input_size": (256, 256),
        "type": "landmark"
    },
}


def load_calibration_images(calib_dir, input_size, num_images=100):
    """
    Load calibration images for quantization.

    Args:
        calib_dir: Directory containing calibration images
        input_size: Tuple of (height, width) for input resolution
        num_images: Number of calibration images to use

    Returns:
        Generator yielding preprocessed images
    """
    import cv2

    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(Path(calib_dir).glob(ext))

    if not image_files:
        print(f"[WARNING] No calibration images found in {calib_dir}")
        print("[INFO] Using synthetic calibration data")
        # Generate synthetic calibration data
        for i in range(num_images):
            img = np.random.rand(*input_size, 3).astype(np.float32)
            yield img
        return

    print(f"[INFO] Found {len(image_files)} calibration images")

    count = 0
    for img_path in image_files:
        if count >= num_images:
            break

        # Load and preprocess image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to input size
        img = cv2.resize(img, input_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        yield img
        count += 1

    print(f"[INFO] Loaded {count} calibration images")


def compile_model(model_name, calib_dir=None, num_cores=4):
    """
    Compile a TFLite model to Axelera format.

    Args:
        model_name: Name of the model to compile (key in MODEL_CONFIGS)
        calib_dir: Directory containing calibration images (optional)
        num_cores: Number of AIPU cores to use (1-4)
    """
    if model_name not in MODEL_CONFIGS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return False

    config = MODEL_CONFIGS[model_name]
    source_path = config["source"]
    output_path = config["output"]
    input_size = config["input_size"]

    print(f"\n{'='*60}")
    print(f"Compiling model: {model_name}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size}")
    print(f"{'='*60}\n")

    # Check if source model exists
    if not os.path.exists(source_path):
        print(f"[ERROR] Source model not found: {source_path}")
        print("Please run convert_models.sh first to download TFLite models")
        return False

    try:
        # Create compiler configuration
        compiler_config = compiler.CompilerConfig(
            aipu_cores=num_cores,
            resources=1.0,  # Use full memory
            ptq_scheme='symmetric'  # Post-training quantization scheme
        )

        print(f"[INFO] Compiler config: cores={num_cores}, resources=1.0, ptq_scheme=symmetric")

        # Load TFLite model
        print(f"[INFO] Loading TFLite model: {source_path}")

        # For TFLite models, we may need to quantize first if not already quantized
        # Check if calibration directory is provided
        if calib_dir and os.path.exists(calib_dir):
            print(f"[INFO] Using calibration data from: {calib_dir}")
            calib_dataset = load_calibration_images(calib_dir, input_size)
        else:
            print("[INFO] No calibration directory provided, using synthetic data")
            calib_dataset = load_calibration_images(".", input_size)

        # Quantize the model
        print("[INFO] Quantizing model...")
        quantized_model = compiler.quantize(
            model=source_path,
            calibration_dataset=calib_dataset,
            config=compiler_config
        )

        print("[INFO] Quantization complete")

        # Compile the quantized model
        print(f"[INFO] Compiling for Axelera Metis AIPU ({num_cores} cores)...")
        output_dir = os.path.dirname(output_path) or "."

        compiler.compile(
            model=quantized_model,
            config=compiler_config,
            output_dir=output_dir
        )

        print(f"[SUCCESS] Model compiled successfully: {output_path}")
        print(f"[INFO] Model ready for deployment on Axelera Metis hardware")

        return True

    except Exception as e:
        print(f"[ERROR] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compile TFLite models for Axelera Metis hardware"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to compile (e.g., palm_detection_lite)"
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Directory containing calibration images (optional)"
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of AIPU cores to use (1-4, default: 4)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"  - {model_name:30s} ({config['type']}, {config['input_size']})")
        print()
        return

    success = compile_model(args.model, args.calib_dir, args.num_cores)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
