# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python demonstration application for MediaPipe models (blazepalm/hand, blazeface, blazepose) with support for multiple AI inference frameworks. The repository enables side-by-side comparison of the same models running on different backends.

## Active Development

### Axelera Metis M.2 Implementation (blaze_axelera-dev-claude branch)

**Status**: Hybrid pipeline verified working on hardware
- Branch: `blaze_axelera-dev-claude`
- Created: 2026-02-26
- Virtual environment: `/home/abbeefai/.cache/axelera/venvs/644f17ff/bin/activate`

**Testing Results**:
- ✓ Hybrid pipeline working: TFLite palm detection + Axelera hand landmarks
- ✓ Pipeline `axl_hand_v0_10_full` tested on Metis M.2 hardware
- Only `hand_landmark_full` compiled so far; other models pending

**Next Steps**:
1. Compile remaining models in blaze_tutorial repo
2. Publish compiled models as GitHub release
3. Complete `blaze_axelera/models/get_axelera_models.sh` with release URL
4. Uncomment pure Axelera pipelines in `blaze_detect_live.py`

### DeepX M1 Implementation (blaze_deepx-dev-claude branch)

**Status**: Hybrid pipeline verified working on hardware
- Branch: `blaze_deepx-dev-claude`
- Created: 2026-02-28
- Virtual environment: `/media/abbeefai/TheExpanse/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate`
- DeepX SDK: `/media/abbeefai/TheExpanse/dx-all-suite/`

**Hardware**:
- DeepX M1 M.2 module, 3 NPU cores at 1000 MHz, LPDDR5 3.92 GiB
- Firmware: v2.5.0 (requires v2.4.0+ for dx_engine v1.1.4)
- PCIe Gen3 X4, device `/dev/dxrt0`
- Monitor: `dxrt-cli -s` (status), `dxrt-cli -m 1` (continuous)

**Testing Results**:
- ✓ Hybrid pipeline working: TFLite palm detection (CPU) + DeepX hand landmarks (NPU)
- ✓ Pipelines `dx_hand_v0_10_lite` and `dx_hand_v0_10_full` tested on M1 hardware
- Only `hand_landmark_lite` and `hand_landmark_full` compiled so far; other models pending

**Next Steps**:
1. Compile palm detection models with DX-COM for pure DeepX pipelines
2. Compile face and pose models
3. Complete `blaze_deepx/models/get_deepx_models.sh` with download URLs
4. Add pure DeepX pipelines to `blaze_detect_live.py`

## Architecture

### Multi-Framework Structure

The codebase uses a plugin-style architecture where each AI framework is implemented in its own directory:

- `blaze_tflite/` - TensorFlow Lite (Google MediaPipe v0.10 models)
- `blaze_tflite_quant/` - TensorFlow Lite with quantized models
- `blaze_tflite_qnn/` - TensorFlow Lite with Qualcomm QNN delegate
- `blaze_pytorch/` - PyTorch (zmurez/MediaPipePyTorch v0.07 models)
- `blaze_vitisai/` - AMD Vitis-AI 3.5
- `blaze_hailo/` - Hailo-8/8L accelerators
- `blaze_onnx/` - ONNX Runtime
- `blaze_rpp/` - AMD ROCm Performance Primitives
- `blaze_qairt/` - Qualcomm AI Engine Direct (QCS6490)
- `blaze_axelera/` - Axelera Metis M.2 AI accelerator (Voyager SDK)
- `blaze_deepx/` - DeepX M1 NPU accelerator (dx_engine SDK)

Each framework directory contains:
- `blazedetector.py` - Detection model wrapper (palm, face, pose detection)
- `blazelandmark.py` - Landmark model wrapper (hand, face, pose landmarks)
- `models/` - Model files and download scripts (`get_*.sh`)
- Optional `blaze_detect_live.py` - Framework-specific entry point

### Common Components

- `blaze_common/blazebase.py` - Base classes (`BlazeBase`, `BlazeDetectorBase`, `BlazeLandmarkBase`)
- `blaze_common/blazeconfig.py` - Model configurations and anchor generation for different model versions
- `blaze_common/visualization.py` - Drawing utilities for landmarks and bounding boxes
- `blaze_common/utils_linux.py` - Linux-specific utilities

### Main Application

`blaze_detect_live.py` (root directory) - Unified entry point that:
1. Dynamically imports available framework backends
2. Supports multiple pipelines running concurrently
3. Provides live camera input or test image processing
4. Handles keyboard controls and visualization

## Common Development Tasks

### Download Models

Before running, download models for the target framework. Navigate to the framework's models directory and run the download script:

```bash
cd blaze_tflite/models
source ./get_tflite_models.sh
cd ../..
```

Other framework model download scripts:
- `blaze_pytorch/models/get_pytorch_models.sh`
- `blaze_hailo/models/get_hailo8_models.sh` or `get_hailo8l_models.sh`
- `blaze_qairt/models/get_qcs6490_models.sh`
- `blaze_tflite_qnn/models/get_qcs6490_models.sh`
- `blaze_tflite_quant/models/get_tflite_quant_models.sh`
- `blaze_onnx/models/convert_models.sh`
- `blaze_rpp/models/convert_models.sh`
- `blaze_axelera/models/get_axelera_models.sh`
- `blaze_deepx/models/get_deepx_models.sh`
- `blaze_vitisai/models/` (varies by DPU architecture)

### Run the Application

Basic usage (auto-detects available frameworks):
```bash
python3 blaze_detect_live.py --blaze hand
```

Run specific frameworks:
```bash
python3 blaze_detect_live.py --blaze hand --target blaze_tflite,blaze_pytorch
```

Run all available pipelines for all supported targets:
```bash
python3 blaze_detect_live.py --blaze hand,face,pose
```

List available pipelines:
```bash
python3 blaze_detect_live.py --list
```

Common options:
- `--blaze hand|face|pose` - Select detection type(s)
- `--target <target_list>` - Select specific frameworks (default: all available)
- `--pipeline <pipeline_list>` - Select specific pipelines
- `--testimage` - Use test image instead of camera
- `--debug` - Enable debug output
- `--profileview` - Show profiling information
- `--fps` - Display FPS counter

### Runtime Keyboard Controls

When the application is running:
- `p` - Pause video
- `c` - Continue
- `s` - Step one frame
- `w` - Take a photo/screenshot
- `t` - Toggle between test image and live video
- `h` - Toggle horizontal mirror
- `a` - Toggle detection overlay
- `b` - Toggle ROI overlay
- `l` - Toggle landmarks overlay
- `d` - Toggle debug image
- `f` - Toggle FPS display
- `v` - Toggle verbose output
- `z` - Toggle profile log
- `y` - Toggle profile view

## Adding a New Framework

When adding support for a new AI framework:

1. Create a new directory: `blaze_<framework_name>/`

2. Implement `blazedetector.py`:
   - Inherit from `blaze_common.blazebase.BlazeDetectorBase`
   - Implement `load_model()` to load detection models
   - Implement `preprocess()` for framework-specific input preprocessing
   - Implement `predict()` for inference
   - Call `config_model()` with model type to set up anchors and configuration

3. Implement `blazelandmark.py`:
   - Inherit from `blaze_common.blazebase.BlazeLandmarkBase`
   - Implement `load_model()` to load landmark models
   - Implement `preprocess()` for framework-specific input preprocessing
   - Implement `predict()` for inference

4. Create `models/` directory with download/conversion scripts

5. Update `blaze_detect_live.py`:
   - Add import attempt in the `supported_targets` section
   - Add target to the `supported_targets` dictionary
   - Add instantiation logic in the pipeline creation section

6. Follow the naming convention: detector models output `(scores, boxes)`, landmark models output `(landmarks, flags/scores)`

## Model Configurations

The `blazeconfig.py` file contains anchor options and model configurations for different model versions:
- Palm detection: v0.06 (2944 anchors, 256x256) and v0.10 (2016 anchors, 192x192)
- Face detection: v0.06/v0.07 front/back, v0.10 short/full range (896 or 2304 anchors)
- Pose detection: v0.07 (896 anchors, 128x128) and v0.10 (2254 anchors, 224x224)

Model configuration parameters include:
- `num_anchors`, `num_coords`, `num_keypoints` - Model output dimensions
- `x_scale`, `y_scale`, `h_scale`, `w_scale` - Coordinate decoding scales
- `min_score_thresh` - Detection confidence threshold
- `min_suppression_threshold` - NMS IoU threshold
- `detection2roi_method` - ROI extraction method ('box' or 'alignment')
- `kp1`, `kp2`, `theta0`, `dscale`, `dy` - ROI transformation parameters

Use `get_model_config()` and `get_anchor_options()` to retrieve the correct configuration based on model type, input size, and number of anchors.

## Framework-Specific Dependencies

The codebase gracefully handles missing dependencies. Framework imports are wrapped in try-except blocks:

- **TFLite**: Prefers `ai_edge_litert.interpreter`, falls back to `tensorflow.lite` or `tflite_runtime.interpreter`
- **PyTorch**: Requires `torch`, auto-detects CUDA availability
- **Vitis-AI**: Requires `xir`, `vitis_ai_library`
- **Hailo**: Requires `hailo_platform`
- **QAIRT**: Requires QAIRT SDK with `QAIRT_SDK_ROOT` environment variable
- **TFLite QNN**: Requires Qualcomm QNN delegate libraries
- **Axelera**: Requires Voyager SDK v1.5+ (`axelera.runtime.objects`), Metis M.2 hardware
- **DeepX**: Requires `dx_engine` Python package from DeepX SDK, M1 firmware v2.4.0+, `/dev/dxrt0` device

When a framework is unavailable, it's marked as not supported and skipped during pipeline initialization.

## Profiling and Performance

The application includes built-in profiling capabilities:
- Use `--profileview` to display latency for each pipeline stage
- Use `--profilelog` to log profiling data
- Timing is captured for: preprocessing, inference (detector + landmark), and postprocessing
- Results can be compared across different frameworks and model versions

## Important Notes

- Each framework implementation maintains the same interface through base classes
- The root `blaze_detect_live.py` automatically detects and uses available frameworks
- Models must be downloaded separately before running (not included in git repository)
- The application supports running multiple pipelines simultaneously for benchmarking
- When modifying detection/landmark logic, update the base classes in `blaze_common/blazebase.py`
- QAIRT and TFLite QNN backends require Qualcomm platform-specific setup (QCS6490)
