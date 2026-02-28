# Knowledge base for using DeepX SW tools - Python inference examples (2026/02/27)

## DeepX SDK - Python Inference Examples

This version focuses on Python-based inference workflows using the DX-RT Python API (dx_engine). It includes synchronous, asynchronous, multi-input, and common preprocessing/postprocessing patterns with runnable code snippets.

1) Quickstart prerequisites

    Install Python package for inference (dx_engine) as described in the DEEPX Python examples:
        Install dx_engine from the dx_rt python_package:
            cd /your-dxrt-directory/python_package
            pip install .
    Basic usage:
        Load a compiled model (DXNN) via the InferenceEngine class
        Prepare input tensors aligned to the model requirements
        Run inference and handle outputs

Notes:

    Input/output tensor shapes follow the model’s requirements (e.g., [N, H, W, C] for some models).
    Input data often needs alignment (e.g., alignment to 64-byte boundaries). See examples for details.

2) Python API overview (dx_engine)

    Core class: InferenceEngine
        Initialization: ie = InferenceEngine("path/to/model.dxnn")
        Synchronous inference: outputs = ie.Run(input_tensor)
        Asynchronous inference: use callbacks and/or threading (examples below)
        Helpers: ie.input_size(), ie.output_dtype(), ie.GetInputSize(), etc. (refer to the actual API in your installed package)

    Typical data flow
        Create or load input tensor with the required size and dtype
        Optionally preprocess an image (resize, color format, normalization)
        Run inference
        Post-process outputs (decode boxes, apply softmax, etc.)

3) Simple synchronous Python inference (single input)

Example: image classification with a single input

Python
# 1) Import module
from dx_engine import InferenceEngine
import numpy as np
import cv2

# 2) Initialize with a compiled model
ie = InferenceEngine("assets/models/EfficientNetB0_4.dxnn")

# 3) Prepare a single input tensor
# Assume model expects [1, H, W, C] with uint8 data
input_size = ie.input_size()  # e.g., (224, 224, 3) or flat size
# If model expects flat uint8 input:
if isinstance(input_size, int):
    input_tensor = np.zeros((1, input_size), dtype=np.uint8)
else:
    H, W, C = input_size
    input_tensor = np.zeros((1, H, W, C), dtype=np.uint8)

# Optional: preprocess a real image
image = cv2.imread("images/imagenet_example.jpg", cv2.IMREAD_COLOR)
image_resized = cv2.resize(image, (W, H))
input_tensor[0] = image_resized  # adjust dtype/format as required

# 4) Run synchronous inference
ie_output = ie.Run(input_tensor)

# 5) Post-process (example for ArgMax-style top-1)
# If the model outputs a single class index directly:
top1 = int(ie_output[0][0])
print(f"Top-1 class index: {top1}")

Notes:

    Adjust input_tensor creation to match your model’s input layout (N, H, W, C vs flattened).
    If your model uses a preprocessing step (mean/std, color space), apply that before Run.

4) Simple asynchronous inference with callback (Python)

Example: using a callback to process results as they arrive

Python
from dx_engine import InferenceEngine
import numpy as np
import cv2

def on_inference_done(outputs):
    # Custom post-processing
    top1 = int(outputs[0][0])
    print(f"Async Top-1: {top1}")

ie = InferenceEngine("assets/models/EfficientNetB0_4.dxnn")

# Prepare input (same as synchronous)
image = cv2.imread("images/imagenet_example.jpg")
input_tensor = np.zeros((1, 224, 224, 3), dtype=np.uint8)
input_tensor[0] = cv2.resize(image, (224, 224))

# Start asynchronous inference
ie.RunAsync(input_tensor, callback=on_inference_done)

# Do other work here while inference runs
# ...

# Wait or join if needed
ie.wait_for_all()

Notes:

    The API exposes RunAsync with a callback. If your version uses a thread-based approach, adapt accordingly.

5) Asynchronous inference with multiple inputs (multi-input model)

If your model has multiple inputs, you can prepare a list/dict of inputs and feed them to the engine.

Python
# Example for a multi-input model: input_1 and input_2
ie = InferenceEngine("assets/models/MultiInputModel.dxnn")

# Create inputs
inp1 = np.zeros((1, 224, 224, 3), dtype=np.uint8)
inp2 = np.zeros((1, 10), dtype=np.float32)

# Fill with real data
# inp1[...] = preprocessed_image
# inp2[...] = some_vector

# Run (format depends on API; adapt to your dx_engine version)
ie_output = ie.Run([inp1, inp2])

# Post-process outputs as needed

Notes:

    The exact Run signature for multi-input models may be a list/tuple or a dict depending on the SDK version. Check your installed dx_engine API for the exact contract.

6) ImageNet-style preprocessing (Python example)

Illustrative preprocessing for a typical image classification workflow

Python
import cv2
import numpy as np
from dx_engine import InferenceEngine

ie = InferenceEngine("assets/models/EfficientNetB0_4.dxnn")

# Load and preprocess image
img = cv2.imread("images/imagenet_example.jpg", cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, (224, 224))
# Optional color space conversion (BGR to RGB)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# If the input must be contiguous flat bytes
input_tensor = img_rgb.astype(np.uint8)
# If needed, reshape to [1, H, W, C]
input_tensor = input_tensor.reshape((1, 224, 224, 3))

# Run
outputs = ie.Run(input_tensor)

# Post-process (example: ArgMax)
idx = int(outputs[0][0])
print(f"Predicted class: {idx}")

Notes:

    Many DX models expect input data aligned to certain byte boundaries; you can insert alignment steps if needed (e.g., padding along the channel dimension) as shown in some Python examples in the docs.
    You may need to normalize or scale inputs depending on the model training regime.

7) Yolov5S-style object detection (Python example)

This demonstrates end-to-end object detection with post-processing, highlighting the typical decode/post-process workflow in Python.

Python
import cv2
import numpy as np
from dx_engine import InferenceEngine

# Load YOLOv5S-like dxnn model
ie = InferenceEngine("./assets/models/YOLOV5S_3.dxnn")

# Prepare input image
image_src = cv2.imread("images/detection_sample.jpg", cv2.IMREAD_COLOR)
# Resize and convert as needed by your model
input_size = ie.input_size()  # e.g., 640x640 or (H, W, C)
H, W = (input_size if isinstance(input_size, tuple) else (640, 640))
image_resized = cv2.resize(image_src, (W, H))
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Align or pad to match network requirements if necessary
input_tensor = image_rgb.astype(np.uint8)
input_tensor = input_tensor.reshape((1, H, W, 3))

# Inference
ie_output = ie.Run(input_tensor)

# Decode (example placeholder; actual decoding depends on your model's output format)
# If there is a dedicated decode function, call it; otherwise implement decoding here
def dummy_decode(outputs):
    # Placeholder: extract top detections if your model outputs [N, num_classes + 5] etc.
    return outputs

detections = dummy_decode(ie_output)

# Print or render detections
for det in detections:
    print(det)

Notes:

    Yolov5S-style models often require channel alignment; ensure the output channels align to the 256-byte boundary (or as your model requires) and perform the decode using the project-provided helper (all_decode in the docs) if available.
    The exact decode logic depends on how your model outputs bounding boxes, scores, and class IDs.

8) ImageNet-style post-processing (softmax / top-k)

If your final layer provides logits for multiple classes, apply softmax and pick top-k

Python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Suppose outputs from DX engine are shape [1, num_classes]
logits = np.array([[1.2, 0.5, 2.1, -0.3]], dtype=np.float32)
probs = softmax(logits)
topk = np.argsort(-probs, axis=1)[0][:5]
print("Top-5 class indices:", topk)

Notes:

    If your outputs are already post-processed by the model with ArgMax, you may skip this step.

9) Troubleshooting quick tips

    Inference outputs differ across runs:
        This can happen due to internal kernel optimization differences in the DX-COM stage. If reproducibility matters, try the --shrink option during compilation or rely on a fixed workflow.
    Input/output shape mismatches:
        Verify model input shapes via ie.input_size() and ensure the input tensor matches exactly (including batch dimension).
    Performance tuning:
        Use asynchronous mode to overlap I/O and compute.
        Validate device status and firmware versions if you encounter runtime errors (dxrt-cli can help).

10) Quick start checklist (Python-focused)

    Install dx_engine Python package
    Prepare your ONNX model and corresponding JSON config (DX-COM workflow)
    Compile: dx_com -m model.onnx -c config.json -o output_dir
    Load the compiled model with Python InferenceEngine
    Prepare properly shaped input tensor (with any required preprocessing)
    Run synchronous or asynchronous inferences
    Implement appropriate post-processing for your task (classification, detection, etc.)
    Validate results and iterate with refinements

