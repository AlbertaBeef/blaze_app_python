#!/bin/bash

# Axelera Metis Model Conversion Script
# This script downloads TFLite models and converts them to Axelera-compiled format
# using the Voyager SDK compiler

# Prerequisites:
# - Axelera Voyager SDK installed (v1.5+)
# - Python 3.10+
# - axelera package available (from axelera import compiler, runtime)

echo "================================================"
echo "Axelera Metis Model Conversion Script"
echo "================================================"
echo ""

# Check if Voyager SDK is available
if ! python3 -c "from axelera import compiler" 2>/dev/null; then
    echo "ERROR: Axelera Voyager SDK not found!"
    echo "Please install the Voyager SDK from: https://github.com/axelera-ai-hub/voyager-sdk"
    echo "Follow the installation instructions for your platform (Ubuntu 22.04+ recommended)"
    exit 1
fi

echo "Voyager SDK detected. Proceeding with model download and compilation..."
echo ""

# Create temporary directory for TFLite models if it doesn't exist
mkdir -p tflite_source

# Download TFLite models from Google MediaPipe v0.10
# These will be used as source models for Axelera compilation

echo "Downloading TFLite source models..."
echo ""

# Palm Detection Models
echo "Downloading Palm Detection models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite -O tflite_source/palm_detection_lite.tflite
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite -O tflite_source/palm_detection_full.tflite

# Hand Landmark Models
echo "Downloading Hand Landmark models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite -O tflite_source/hand_landmark_lite.tflite
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite -O tflite_source/hand_landmark_full.tflite

# Face Detection Models
echo "Downloading Face Detection models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite -O tflite_source/face_detection_short_range.tflite
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite -O tflite_source/face_detection_full_range.tflite

# Face Landmark Models
echo "Downloading Face Landmark models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite -O tflite_source/face_landmark.tflite

# Pose Detection Models
echo "Downloading Pose Detection models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite -O tflite_source/pose_detection.tflite

# Pose Landmark Models
echo "Downloading Pose Landmark models..."
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite -O tflite_source/pose_landmark_lite.tflite
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite -O tflite_source/pose_landmark_full.tflite
wget -q --show-progress https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite -O tflite_source/pose_landmark_heavy.tflite

echo ""
echo "Download complete!"
echo ""

# Compile models using Axelera Voyager SDK
echo "================================================"
echo "Compiling models for Axelera Metis hardware..."
echo "================================================"
echo ""
echo "NOTE: Model compilation requires calibration data."
echo "Please use the Python compilation script: compile_axelera_models.py"
echo ""
echo "Usage:"
echo "  python3 compile_axelera_models.py --model <model_name>"
echo ""
echo "Example:"
echo "  python3 compile_axelera_models.py --model palm_detection_lite"
echo ""
echo "Available models:"
echo "  - palm_detection_lite"
echo "  - palm_detection_full"
echo "  - hand_landmark_lite"
echo "  - hand_landmark_full"
echo "  - face_detection_short_range"
echo "  - face_detection_full_range"
echo "  - face_landmark"
echo "  - pose_detection"
echo "  - pose_landmark_lite"
echo "  - pose_landmark_full"
echo "  - pose_landmark_heavy"
echo ""
echo "The compilation script will:"
echo "  1. Load the TFLite model"
echo "  2. Quantize using calibration data (if not already quantized)"
echo "  3. Compile for Axelera Metis AIPU"
echo "  4. Save to the models/ directory as .axmodel"
echo ""
echo "TFLite source models saved in: tflite_source/"
echo "Compiled Axelera models will be saved in: ./"
echo ""
echo "Script complete!"
