# Knowledge base for using DeepX SW tools (2026/02/27)

## DeepX Software Tooling - Quick Reference

This document provides an integrated guide to using the DeepX (DEEPX) software toolchain, including the compiler, runtime, model zoo, streaming tools, and sample applications. It consolidates information from the DEEPX SDK documentation into a concise reference.

1) Overview of DeepX SDK Components

    DX-COM (NPU Compiler)
        Converts a pre-trained ONNX model and its configuration JSON into a hardware-optimized .dxnn binary.
        Outputs: a compiled .dxnn file containing the command set and weights for DEEPX NPUs.
        Command-line example:
            dx_com -m <MODEL_PATH> -c <CONFIG_PATH> -o <OUTPUT_DIR>
            Optional: --shrink to emit only essential data for the NPU.
        Important notes:
            ONNX file contains model structure and weights; JSON defines pre/post-processing and compilation parameters.
            Output may vary between runs due to internal optimization kernels.

    DX-RT (NPU Runtime)
        Runtime for executing .dxnn models on DEEPX hardware.
        Interfaces with the NPU via firmware and device drivers over PCIe.
        Provides C/C++ and Python APIs for application-level inference.
        Runtime capabilities:
            Model loading, I/O buffer management, inference execution, hardware monitoring.
        API exposure:
            C++ DX-RT API
            Python DX-RT API
            Core Runtime Library
            Device driver (Windows or Linux)

    DX ModelZoo
        Curated collection of pre-trained ONNX models, configuration JSONs, and pre-compiled .dxnn binaries.
        Includes benchmarking tools to compare INT8 quantized models on DEEPX NPUs vs FP32 on CPUs/GPUs.
        Useful for rapid prototyping and baseline comparisons.

    DX-Stream
        Custom GStreamer plugin for real-time streaming data integration in AI inference apps.
        Provides modular pipelines with preprocessing, inference, and post-processing stages.
        Use cases: video analytics, smart cameras, edge AI.

    DX-APP
        Sample application demonstrating how to run compiled models on DEEPX NPU via DX-RT.
        Includes ready-to-use code for object detection, face recognition, and image classification.
        Serves as a template for building your own DL-enabled applications.

2) Installation & Preparation

    Supported platforms
        DX-TRON (Model Viewer) installation noted as Windows-first; Linux/macOS TBD in some contexts.
        DX-RT runtime libraries and device drivers support Windows and Linux.

    Typical installation steps
        Acquire the relevant installer packages or pre-built binaries for:
            DX-COM (compiler)
            DX-RT (runtime)
            DX-ModelZoo tools
            DX-Stream and GStreamer components (for streaming pipelines)
        Install dependencies (e.g., system libraries, drivers). See 2.5–2.6 sections in the source for specific requirements.
        Verify versions via provided --version or info options where applicable.

    Quick validation flow
        Compile a sample ONNX model:
            dx_com -m sample/MobilenetV1.onnx -c sample/MobilenetV1.json -o output/mobilenetv1
        Run a simple inference via DX-RT (after modeling and runtime setup):
            Load the compiled .dxnn
            Execute inference and fetch results
        Optional: shrink output to minimize size:
            dx_com ... -o output/mobilenetv1 --shrink

3) Model Compilation with DX-COM

    Input requirements
        ONNX model file (single input model)
        JSON configuration file with:
            inputs: name and shape (batch size fixed to 1)
            calibration method and calibration_num
            dataset loader and preprocessing definitions
        Example input definition:
            "inputs": { "input.1": [1, 3, 512, 512] }

    JSON configuration highlights
        Calibration methods: ema or minmax
        Calibration_num: number of calibration steps (e.g., 100)
        default_loader: dataset path, file extensions, preprocessing steps

    Common errors during compilation (DX-COM)
        NotSupportError: unsupported features (e.g., multi-input, dynamic shapes)
        ConfigFileError: invalid/malformed JSON
        ConfigInputError: mismatch between ONNX inputs and config
        DatasetPathError: invalid dataset path
        NodeNotFoundError: unsupported ONNX node
        OSError / UbuntuVersionError / LDDVersionError: environment compatibility
        RamSizeError / DiskSizeError: resource limitations
        DataNotFoundError / OnnxFileNotFound: missing data or model

    Notes
        The dx_com command may produce different outputs even with the same ONNX in the same PC environment due to internal kernel behavior.
        Shrink option helps minimize the produced output, removing debug/intermediate files.

4) Runtime with DX-RT

    How DX-RT works
        Loads the .dxnn binary produced by DX-COM
        Handles I/O buffers and data transfer over PCIe to/from the DEEPX NPU
        Exposes APIs for application control of inference (C++ and Python)
        Provides a runtime environment and hardware monitoring

    Typical workflow
        Initialize runtime
        Load compiled model (.dxnn)
        Prepare input tensors and allocate output buffers
        Run inference
        Retrieve and process results
        Optionally run performance benchmarks or monitoring

    Example usage snippet (conceptual)
        C++:
            Use the DX-RT API to check devices, load model, and run inference
        Python:
            Use the Python DX-RT API to perform similar steps
        Reference code in docs provides a “Hello World” style example to verify setup.

    DX-RT CLI
        dxrt-cli tool for device status, firmware updates, and diagnostics
        Example usage:
            dxrt-cli --status
            dxrt-cli --fwupdate fw.bin
            dxrt-cli -m 1 (monitor with interval 1s)

5) DX ModelZoo, DX-Stream, and DX-APP in Practice

    DX ModelZoo
        Pick a pre-trained ONNX + .dxnn + JSON bundle
        Optionally re-compile with DX-COM for experimentation
        Use benchmarking tools to compare INT8 vs FP32 performance

    DX-Stream (GStreamer integration)
        Build video/image pipelines with preprocessing, inference, and postprocessing stages
        Integrate with DEEPX NPU inference to enable real-time vision tasks

    DX-APP (Sample application)
        Provides ready-to-run demos for object detection, face recognition, and image classification
        Serves as a template to develop your own AI apps using DX-RT

6) Model Viewer and DX-TRON

    DX-TRON (Model Viewer)
        Installation (Windows-centric in the referenced docs)
        Features:
            Load and visualize .dxnn models
            Visual representation of workloads (NPU vs CPU operations)
            Interactive exploration of model graph and operation details

    Quick notes
        DX-TRON is used to inspect compiled models and workloads, helpful for debugging and performance analysis.

7) Command-Line Interfaces and Examples

    DX-COM command (basic)
        dx_com -m <MODEL_PATH> -c <CONFIG_PATH> -o <OUTPUT_DIR>

    DX-COM with shrink
        dx_com -m <MODEL_PATH> -c <CONFIG_PATH> -o <OUTPUT_DIR> --shrink

    DX-RT CLI (firmware interface)
        dxrt-cli
        Examples:
            dxrt-cli --status
            dxrt-cli --fwupdate fw.bin
            dxrt-cli -m 1

    Sample inference by DX-APP
        Use the provided sample app and DX-RT APIs to load compiled models and run inference
        The DX-APP codebase includes common vision tasks and serves as a template for customization

8) Troubleshooting Summary

    If compilation fails
        Verify ONNX and JSON configurations
        Check input names and shapes match exactly
        Confirm proper dataset path and preprocessing steps
        Ensure system requirements (RAM, disk space, library versions) are met

    If runtime fails
        Confirm firmware and device driver status via dxrt-cli
        Ensure PCIe connectivity and correct device ID
        Check input/output buffer configurations and data formatting

    If outputs differ between runs
        Understand that DX-COM may produce slightly different optimized binaries due to internal kernel behavior
        Consider using --shrink to enforce consistent minimal outputs if reproducibility is critical

9) Quick Start Checklist

    Install DX-COM, DX-RT, and DX-ModelZoo tooling
    Prepare ONNX model and JSON config
    Run:
        dx_com -m <model.onnx> -c <config.json> -o <out_dir> [--shrink]
    Load the resulting .dxnn with DX-RT
    Run inference via C++ or Python API
    (Optional) Test with DX-Stream and GStreamer for streaming-based pipelines
    (Optional) Explore pre-built demos in DX-APP for quick validation


