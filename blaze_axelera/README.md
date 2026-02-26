# Blaze Axelera - MediaPipe Models on Axelera Metis M.2

This directory contains the Axelera Metis M.2 AI accelerator implementation for MediaPipe Blaze models (palm/hand detection, face detection/landmarks, pose detection/landmarks).

## Overview

The Axelera Metis M.2 is a high-performance AI inference acceleration card featuring the Metis AIPU (AI Processing Unit). This implementation uses the Axelera Voyager SDK to run MediaPipe models on the Metis hardware.

## Prerequisites

### Hardware
- Axelera Metis M.2 AI Acceleration Card installed in your system
- USB 3.0 or PCIe connectivity
- Compatible host: Intel Core, AMD Ryzen, or Arm64 processors

### Software
- Ubuntu 22.04+ (Linux) or Windows with WSL2
- Python 3.10+
- Axelera Voyager SDK v1.5+

## Installation

### 1. Install Axelera Voyager SDK

Follow the official installation guide from Axelera:

```bash
# Clone the Voyager SDK repository
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk

# Follow the installation instructions in the SDK documentation
# Typically involves installing dependencies and the SDK package
```

### 2. Install Python Dependencies

```bash
pip install numpy opencv-python
```

### 3. Verify Axelera SDK Installation

```bash
python3 -c "from axelera import runtime, compiler; print('Axelera SDK installed successfully')"
```

## Model Preparation

The Axelera implementation requires models to be compiled for the Metis AIPU hardware. The compilation process involves:

1. **Download TFLite source models**
2. **Quantize models** (if not already quantized)
3. **Compile for Axelera AIPU**

### Step 1: Download TFLite Models

```bash
cd models
bash convert_models.sh
```

This downloads Google MediaPipe v0.10 TFLite models to `tflite_source/`.

### Step 2: Compile Models for Axelera

Compile individual models:

```bash
# List available models
python3 compile_axelera_models.py --list

# Compile a specific model (e.g., palm detection)
python3 compile_axelera_models.py --model palm_detection_lite

# Compile with custom calibration data
python3 compile_axelera_models.py --model hand_landmark_full --calib-dir ../../calib_dataset_kaggle/

# Compile with specific number of AIPU cores (1-4)
python3 compile_axelera_models.py --model pose_detection --num-cores 4
```

Compile all models:

```bash
# Compile all detection models
for model in palm_detection_lite palm_detection_full face_detection_short_range face_detection_full_range pose_detection; do
    python3 compile_axelera_models.py --model $model
done

# Compile all landmark models
for model in hand_landmark_lite hand_landmark_full face_landmark pose_landmark_lite pose_landmark_full pose_landmark_heavy; do
    python3 compile_axelera_models.py --model $model
done
```

### Model Compilation Options

- `--model`: Model name to compile (required)
- `--calib-dir`: Directory with calibration images for quantization (optional)
- `--num-cores`: Number of AIPU cores to use (1-4, default: 4)
- `--list`: List all available models

### Available Models

**Detection Models:**
- `palm_detection_lite` - Palm detection (192x192)
- `palm_detection_full` - Palm detection full (192x192)
- `face_detection_short_range` - Face detection short range (128x128)
- `face_detection_full_range` - Face detection full range (192x192)
- `pose_detection` - Pose detection (224x224)

**Landmark Models:**
- `hand_landmark_lite` - Hand landmarks lite (224x224)
- `hand_landmark_full` - Hand landmarks full (224x224)
- `face_landmark` - Face landmarks (192x192)
- `pose_landmark_lite` - Pose landmarks lite (256x256)
- `pose_landmark_full` - Pose landmarks full (256x256)
- `pose_landmark_heavy` - Pose landmarks heavy (256x256)

## Running the Application

### Using the Main Application

From the repository root:

```bash
# Run hand detection with Axelera backend
python3 blaze_detect_live.py --blaze hand --target blaze_axelera

# Run all models with Axelera backend
python3 blaze_detect_live.py --blaze hand,face,pose --target blaze_axelera

# List available Axelera pipelines
python3 blaze_detect_live.py --list --target blaze_axelera

# Run with profiling
python3 blaze_detect_live.py --blaze hand --target blaze_axelera --profileview --fps
```

### Pipeline Naming Convention

Axelera pipelines follow the naming pattern: `axl_<type>_<model_variant>`

Examples:
- `axl_hand_lite` - Hand detection + landmarks (lite models)
- `axl_hand_full` - Hand detection + landmarks (full models)
- `axl_face_short` - Face detection (short range) + landmarks
- `axl_face_full` - Face detection (full range) + landmarks
- `axl_pose_lite` - Pose detection + landmarks (lite)
- `axl_pose_full` - Pose detection + landmarks (full)
- `axl_pose_heavy` - Pose detection + landmarks (heavy)

## Architecture

### Implementation Structure

```
blaze_axelera/
├── blazedetector.py          # Detection model wrapper
├── blazelandmark.py          # Landmark model wrapper
├── models/
│   ├── convert_models.sh     # Download TFLite models
│   ├── compile_axelera_models.py  # Compile for Axelera
│   ├── tflite_source/        # TFLite source models
│   └── *.axmodel             # Compiled Axelera models
└── README.md                 # This file
```

### Key Components

**blazedetector.py:**
- Loads compiled Axelera models for detection (palm, face, pose)
- Uses `axelera.runtime` API for inference
- Inherits from `BlazeDetectorBase` in `blaze_common/blazebase.py`

**blazelandmark.py:**
- Loads compiled Axelera models for landmarks (hand, face, pose)
- Uses `axelera.runtime` API for inference
- Inherits from `BlazeLandmarkBase` in `blaze_common/blazebase.py`

### Axelera Runtime API

The implementation uses the Axelera Voyager SDK runtime API:

```python
from axelera import runtime

# Initialize device
device = runtime.Device()

# Load compiled model
model = runtime.Model(model_path, device)

# Run inference
outputs = model.run(input_data)
```

## Performance Optimization

### Multi-Core Processing

The Metis AIPU supports 1-4 cores. Compile models with more cores for higher throughput:

```bash
# Compile with 4 cores (maximum performance)
python3 compile_axelera_models.py --model palm_detection_lite --num-cores 4

# Compile with 2 cores (balanced)
python3 compile_axelera_models.py --model palm_detection_lite --num-cores 2
```

### Memory Optimization

The compilation process allows configuring memory resources (0.0-1.0). The default (1.0) uses full available memory for best performance.

## Troubleshooting

### Axelera SDK Not Found

```
ERROR: Failed to import Axelera Voyager SDK
```

**Solution:**
- Verify SDK installation: `python3 -c "from axelera import runtime"`
- Check SDK documentation: https://github.com/axelera-ai-hub/voyager-sdk
- Ensure Python environment has access to the SDK

### Model File Not Found

```
ERROR: Model path not found: models/palm_detection_lite.axmodel
```

**Solution:**
- Run `convert_models.sh` to download TFLite models
- Run `compile_axelera_models.py` to compile models
- Check that compiled `.axmodel` files exist in `models/`

### Compilation Fails

```
ERROR: Compilation failed
```

**Solution:**
- Check calibration data is available (or use synthetic data)
- Verify TFLite source model is valid
- Review Axelera SDK logs for specific errors
- Try compiling with fewer cores: `--num-cores 1`

### Device Not Detected

```
ERROR: No Axelera device found
```

**Solution:**
- Verify Metis M.2 card is properly installed
- Check USB 3.0 or PCIe connection
- Run Axelera device detection utility (from SDK)
- Check system logs: `dmesg | grep axelera`

## References

- [Axelera Metis M.2 Product Page](https://axelera.ai/ai-accelerators/metis-m2-ai-acceleration-card)
- [Axelera Voyager SDK (GitHub)](https://github.com/axelera-ai-hub/voyager-sdk)
- [Voyager SDK Documentation](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.5/docs/tutorials/quick_start_guide.md)
- [Axelera Community Forum](https://community.axelera.ai/)
- [Google MediaPipe Models](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md)

## Support

For Axelera-specific issues:
- Axelera Community: https://community.axelera.ai/
- Voyager SDK Issues: https://github.com/axelera-ai-hub/voyager-sdk/issues

For this implementation:
- Report issues in the main repository issue tracker
