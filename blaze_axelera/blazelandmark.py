import numpy as np
import os

from blazebase import BlazeLandmarkBase

bUseAxeleraRuntime = False
try:
    from axelera.runtime import objects as axr
    bUseAxeleraRuntime = True
except Exception as e:
    print(f"[BlazeLandmark] Failed to import axelera.runtime: {e}")

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self, blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app

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
            print(f"[BlazeLandmark.load_model] Model Path: {model_path}")

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
            print(f"[BlazeLandmark.load_model] Device: {devices[0].name}")

        self.conn = self.ctx.device_connect(devices[0], num_sub_devices=1)
        self.model = self.ctx.load_model(model_file)
        self.model_instance = self.conn.load_model_instance(self.model)

        # Cache input/output tensor info
        self.input_info = self.model.inputs()[0]
        self.output_infos = self.model.outputs()

        if self.DEBUG:
            inp = self.input_info
            print(f"[BlazeLandmark.load_model] Input: shape={inp.shape} unpadded={inp.unpadded_shape} scale={inp.scale} zp={inp.zero_point}")
            for i, out in enumerate(self.output_infos):
                print(f"[BlazeLandmark.load_model] Output[{i}]: shape={out.shape} unpadded={out.unpadded_shape} scale={out.scale} zp={out.zero_point}")

        # Resolution from unpadded input shape (NHWC)
        self.resolution = self.input_info.unpadded_shape[1]

        if self.DEBUG:
            print(f"[BlazeLandmark.load_model] Resolution: {self.resolution}")

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
        # Input from extract_roi is already float32 in [0, 1] range
        return x

    def predict(self, x):

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        out1_list = []
        out2_list = []
        out3_list = []

        start = timer()
        x = self.preprocess(x)
        self.profile_pre += timer() - start

        nb_images = x.shape[0]
        for i in range(nb_images):

            start = timer()
            xi = np.expand_dims(x[i, :, :, :], axis=0)

            # Quantize and pad input for Axelera hardware
            xi_hw = self._quantize_and_pad(xi)
            self.profile_pre += timer() - start

            # Allocate output buffers
            out_bufs = [np.zeros(o.shape, dtype=np.int8) for o in self.output_infos]

            # Run inference on Axelera Metis
            start = timer()
            self.model_instance.run([xi_hw], out_bufs)
            self.profile_model += timer() - start

            start = timer()

            # Depad and dequantize outputs
            outputs = [self._depad_and_dequantize(buf, info)
                       for buf, info in zip(out_bufs, self.output_infos)]

            if self.blaze_app == "blazehandlandmark":
                # Output[0]: landmarks 3D (1,1,1,63), Output[1]: flag (1,1,1,1),
                # Output[2]: handedness (1,1,1,1), Output[3]: world landmarks
                out1 = outputs[1].reshape(1, 1)          # flag
                out2 = outputs[0].reshape(1, 21, -1)     # landmarks => [1,21,3]
                out2 = out2 / self.resolution
                out3 = outputs[2].reshape(1, 1)           # handedness
            elif self.blaze_app == "blazefacelandmark":
                out1 = outputs[1].reshape(1, 1)
                out2 = outputs[0].reshape(1, -1, 3)
                out2 = out2 / self.resolution
            elif self.blaze_app == "blazeposelandmark":
                out1 = outputs[1].reshape(1, 1)
                out2 = outputs[0].reshape(1, -1, 5)
                out2 = out2 / self.resolution

            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            if self.blaze_app == "blazehandlandmark":
                out3_list.append(out3.squeeze(0))
            self.profile_post += timer() - start

        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)
        if self.blaze_app == "blazehandlandmark":
            handedness_scores = np.asarray(out3_list)

        if self.blaze_app == "blazehandlandmark":
            return flag, landmarks, handedness_scores
        else:
            return flag, landmarks
