import numpy as np

from blazebase import BlazeDetectorBase

bUseDeepXRuntime = False
try:
    from dx_engine import InferenceEngine
    bUseDeepXRuntime = True
except Exception as e:
    print(f"[BlazeDetector] Failed to import dx_engine: {e}")

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self, blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1

        if not bUseDeepXRuntime:
            raise ImportError("DeepX runtime (dx_engine) is not available. Please install the DX-RT Python package.")

        self.ie = None

    def load_model(self, model_path):
        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Model File: {model_path}")

        self.ie = InferenceEngine(model_path)

        # Determine input size from model
        input_size = self.ie.input_size()
        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Input Size: {input_size} (type={type(input_size).__name__})")

        if isinstance(input_size, (tuple, list)):
            if len(input_size) == 4:
                resolution = input_size[1]  # NHWC
            elif len(input_size) == 3:
                resolution = input_size[0]  # HWC
            else:
                resolution = input_size[0]
        elif isinstance(input_size, int):
            channels = 3
            pixels = input_size // channels
            resolution = int(np.sqrt(pixels))
        else:
            resolution = 192  # fallback for palm detection models

        self.x_scale = float(resolution)
        self.y_scale = float(resolution)
        self.h_scale = float(resolution)
        self.w_scale = float(resolution)

        self.in_shape = [1, resolution, resolution, 3]

        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Resolution: {resolution}")

        # Run a dummy inference to determine output shapes
        dummy_input = np.zeros((1, resolution, resolution, 3), dtype=np.uint8)
        dummy_output = self.ie.Run(dummy_input)
        dummy_outputs = [np.asarray(o) for o in dummy_output]

        if self.DEBUG:
            for j, o in enumerate(dummy_outputs):
                print(f"[BlazeDetector.load_model] Output[{j}]: shape={o.shape} dtype={o.dtype}")

        # Determine num_anchors from output shapes
        # Detection models output: [0] = regressors [1, num_anchors, num_coords], [1] = classifiers [1, num_anchors, 1]
        out_reg_shape = dummy_outputs[0].shape
        out_clf_shape = dummy_outputs[1].shape

        if len(out_clf_shape) >= 2:
            self.num_anchors = out_clf_shape[1] if len(out_clf_shape) == 3 else out_clf_shape[0]
        else:
            self.num_anchors = out_reg_shape[1] if len(out_reg_shape) == 3 else out_reg_shape[0]

        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Num Anchors: {self.num_anchors}")

        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [0, 1]."""
        x = x.astype(np.float32)
        x = (x / 255.0)
        return x

    def predict_on_image(self, img):
        img_expanded = np.expand_dims(img, axis=0)
        detections = self.predict_on_batch(img_expanded)
        if len(detections) > 0:
            return np.array(detections)[0]
        else:
            return []

    def predict_on_batch(self, x):
        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        assert x.shape[3] == 3
        assert x.shape[1] == int(self.y_scale)
        assert x.shape[2] == int(self.x_scale)

        out1, out2 = self.predict_core(x)

        assert out1.shape[0] == 1  # batch
        assert out1.shape[1] == self.num_anchors
        assert out1.shape[2] == 1

        assert out2.shape[0] == 1  # batch
        assert out2.shape[1] == self.num_anchors
        assert out2.shape[2] == self.num_coords

        start = timer()

        # Postprocess the raw predictions:
        detections = self._tensors_to_detections(out2, out1, self.anchors)

        # Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)
                if len(filtered_detections) > 0:
                    normalized_detections = np.array(filtered_detections)[0]

        self.profile_post = timer() - start

        return filtered_detections

    def predict_core(self, x):

        # 1. Preprocess the images into tensors:
        start = timer()
        x = self.preprocess(x)

        # Convert float32 [0,1] to uint8 [0,255] for DeepX inference
        x_uint8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        self.profile_pre = timer() - start

        # 2. Run the neural network:
        start = timer()
        ie_output = self.ie.Run(x_uint8)
        self.profile_model = timer() - start

        outputs = [np.asarray(o) for o in ie_output]

        # Detection models output: [0] = classifiers (scores), [1] = regressors (boxes)
        # or [0] = regressors, [1] = classifiers depending on model
        out1 = outputs[0]  # clf (scores)
        out2 = outputs[1]  # reg (boxes)

        # Ensure 3D shape [batch, num_anchors, ...]
        if len(out1.shape) == 2:
            out1 = np.expand_dims(out1, axis=0)
        if len(out2.shape) == 2:
            out2 = np.expand_dims(out2, axis=0)

        return out1, out2
