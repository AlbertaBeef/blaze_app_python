import numpy as np
import os

from blazebase import BlazeDetectorBase

bUseAxeleraRuntime = False
try:
    from axelera.runtime import objects as axr
    bUseAxeleraRuntime = True
except Exception as e:
    print(f"[BlazeDetector] Failed to import axelera.runtime: {e}")

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self, blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1

        if not bUseAxeleraRuntime:
            raise ImportError("Axelera runtime is not available. Please install the Voyager SDK.")

        # Runtime objects (initialized in load_model)
        self.ctx = None
        self.conn = None
        self.model = None
        self.model_instance = None
        self.input_info = None
        self.output_infos = None

    def load_model(self, model_path):
        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Model Path: {model_path}")

        # If model_path is a directory, look for model.json inside it
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "model.json")
        else:
            model_file = model_path

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Initialize Axelera runtime
        self.ctx = axr.Context()
        devices = self.ctx.list_devices()
        if not devices:
            raise RuntimeError("No Axelera devices found")

        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Device: {devices[0].name}")

        self.conn = self.ctx.device_connect(devices[0], num_sub_devices=1)
        self.model = self.ctx.load_model(model_file)
        self.model_instance = self.conn.load_model_instance(self.model)

        # Cache input/output tensor info
        self.input_info = self.model.inputs()[0]
        self.output_infos = self.model.outputs()

        if self.DEBUG:
            inp = self.input_info
            print(f"[BlazeDetector.load_model] Input: shape={inp.shape} unpadded={inp.unpadded_shape} scale={inp.scale} zp={inp.zero_point}")
            for i, out in enumerate(self.output_infos):
                print(f"[BlazeDetector.load_model] Output[{i}]: shape={out.shape} unpadded={out.unpadded_shape} scale={out.scale} zp={out.zero_point}")

        # Get num_anchors from unpadded output shape
        # Detection models output: regressors [1, num_anchors, num_coords], classificators [1, num_anchors, 1]
        out_reg_unpadded = self.output_infos[0].unpadded_shape
        self.num_anchors = out_reg_unpadded[1]

        # Resolution from unpadded input shape (NHWC)
        input_size = self.input_info.unpadded_shape[1]
        self.x_scale = float(input_size)
        self.y_scale = float(input_size)
        self.h_scale = float(input_size)
        self.w_scale = float(input_size)

        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Num Anchors: {self.num_anchors}")
            print(f"[BlazeDetector.load_model] Input Size: {input_size}")

        # Configure model with appropriate settings
        self.config_model(self.blaze_app)

    def _quantize_and_pad(self, x):
        """Quantize float32 input to int8 and pad for hardware."""
        inp = self.input_info
        quantized = np.round(x / inp.scale + inp.zero_point).clip(-128, 127).astype(np.int8)
        padded = np.pad(quantized, inp.padding, constant_values=inp.zero_point)
        return padded

    def _depad_and_dequantize(self, raw_output, info):
        """Remove padding and dequantize int8 output to float32."""
        depadded = raw_output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
        dequantized = (depadded.astype(np.float32) - info.zero_point) * info.scale
        return dequantized

    def preprocess(self, x):
        """Converts the image pixels to the range [0, 1]."""
        x = (x / 255.0)
        x = x.astype(np.float32)
        return x

    def predict(self, x):
        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        start = timer()

        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        # Quantize and pad input for Axelera hardware
        x_hw = self._quantize_and_pad(x)
        self.profile_pre = timer() - start

        # Allocate output buffers
        out_bufs = [np.zeros(o.shape, dtype=np.int8) for o in self.output_infos]

        # Run inference on Axelera Metis
        start = timer()
        self.model_instance.run([x_hw], out_bufs)
        self.profile_model = timer() - start

        start = timer()
        # Depad and dequantize outputs
        # Detection models output: [0] = regressors (boxes), [1] = classificators (scores)
        out_reg = self._depad_and_dequantize(out_bufs[0], self.output_infos[0])
        out_clf = self._depad_and_dequantize(out_bufs[1], self.output_infos[1])
        self.profile_post = timer() - start

        return out_reg, out_clf
