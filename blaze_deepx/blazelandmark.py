import numpy as np

from blazebase import BlazeLandmarkBase

bUseDeepXRuntime = False
try:
    from dx_engine import InferenceEngine
    bUseDeepXRuntime = True
except Exception as e:
    print(f"[BlazeLandmark] Failed to import dx_engine: {e}")

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self, blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app

        if not bUseDeepXRuntime:
            raise ImportError("DeepX runtime (dx_engine) is not available. Please install the DX-RT Python package.")

        self.ie = None

    def load_model(self, model_path):
        if self.DEBUG:
            print(f"[BlazeLandmark.load_model] Model File: {model_path}")

        self.ie = InferenceEngine(model_path)

        # Determine resolution from model input size
        input_size = self.ie.input_size()
        if self.DEBUG:
            print(f"[BlazeLandmark.load_model] Input Size: {input_size} (type={type(input_size).__name__})")

        if isinstance(input_size, (tuple, list)):
            if len(input_size) == 4:
                self.resolution = input_size[1]  # NHWC: (N, H, W, C)
            elif len(input_size) == 3:
                self.resolution = input_size[0]  # HWC: (H, W, C)
            else:
                self.resolution = input_size[0]
        elif isinstance(input_size, int):
            # Flat size - infer resolution assuming square RGB input
            channels = 3
            pixels = input_size // channels
            self.resolution = int(np.sqrt(pixels))
        else:
            self.resolution = 224  # fallback for hand landmark models

        if self.DEBUG:
            print(f"[BlazeLandmark.load_model] Resolution: {self.resolution}")

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
            xi = x[i, :, :, :]

            # Convert float32 [0,1] to uint8 [0,255] for DeepX inference
            xi_uint8 = (xi * 255.0).clip(0, 255).astype(np.uint8)
            xi_uint8 = np.expand_dims(xi_uint8, axis=0)  # Add batch dimension [1,H,W,C]
            self.profile_pre += timer() - start

            # Run inference on DeepX M1
            start = timer()
            ie_output = self.ie.Run(xi_uint8)
            self.profile_model += timer() - start

            start = timer()
            outputs = [np.asarray(o) for o in ie_output]

            if self.DEBUG and i == 0:
                for j, o in enumerate(outputs):
                    print(f"[BlazeLandmark.predict] Output[{j}]: shape={o.shape} dtype={o.dtype} min={o.min():.4f} max={o.max():.4f}")

            if self.blaze_app == "blazehandlandmark":
                # Expected output order (same as TFLite):
                # [0]: landmarks [1, 63] (21 keypoints * 3 coords)
                # [1]: flag [1, 1] (hand presence confidence)
                # [2]: handedness [1, 1] (left/right hand score)
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
