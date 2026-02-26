import numpy as np
import os

from blazebase import BlazeDetectorBase

bUseAxeleraRuntime = False
try:
    from axelera import runtime
    bUseAxeleraRuntime = True
except Exception as e:
    print(f"[BlazeDetector] Failed to import axelera.runtime: {e}")
    try:
        # Fallback: try importing InferenceStream for simpler API
        from axelera.pipeline import InferenceStream
        bUseAxeleraRuntime = True
    except Exception as e2:
        print(f"[BlazeDetector] Failed to import axelera.pipeline: {e2}")

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self, blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1
        self.model = None
        self.device = None

        if not bUseAxeleraRuntime:
            raise ImportError("Axelera runtime is not available. Please install the Voyager SDK.")

    def load_model(self, model_path):
        """
        Load a compiled Axelera model for the Metis hardware.

        Args:
            model_path: Path to the compiled model directory or .axmodel file
        """
        if self.DEBUG:
            print(f"[BlazeDetector.load_model] Model Path: {model_path}")

        # Check if model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        try:
            # Initialize Axelera device
            # The runtime API allows selecting available Metis devices
            self.device = runtime.Device()

            if self.DEBUG:
                print(f"[BlazeDetector.load_model] Axelera device initialized")
                print(f"[BlazeDetector.load_model] Device info: {self.device.get_info()}")

            # Load the compiled model onto the device
            self.model = runtime.Model(model_path, self.device)

            # Get model input/output information
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

            if self.DEBUG:
                print(f"[BlazeDetector.load_model] Number of Inputs: {len(self.input_details)}")
                for i, inp in enumerate(self.input_details):
                    print(f"[BlazeDetector.load_model] Input[{i}]: shape={inp['shape']}, dtype={inp['dtype']}")

                print(f"[BlazeDetector.load_model] Number of Outputs: {len(self.output_details)}")
                for i, out in enumerate(self.output_details):
                    print(f"[BlazeDetector.load_model] Output[{i}]: shape={out['shape']}, dtype={out['dtype']}")

            # Get input shape
            self.in_shape = self.input_details[0]['shape']
            self.out_reg_shape = self.output_details[0]['shape']
            self.out_clf_shape = self.output_details[1]['shape']

            # Extract model parameters from output shapes
            self.num_anchors = self.out_reg_shape[1]

            # Determine scales based on input size
            input_size = self.in_shape[1]  # Assuming square input
            self.x_scale = float(input_size)
            self.y_scale = float(input_size)
            self.h_scale = float(input_size)
            self.w_scale = float(input_size)

            if self.DEBUG:
                print(f"[BlazeDetector.load_model] Input Shape: {self.in_shape}")
                print(f"[BlazeDetector.load_model] Num Anchors: {self.num_anchors}")
                print(f"[BlazeDetector.load_model] Scales: x={self.x_scale}, y={self.y_scale}")

            # Configure model with appropriate settings
            self.config_model(self.blaze_app)

        except Exception as e:
            print(f"[BlazeDetector.load_model] Error loading model: {e}")
            raise

    def preprocess(self, x):
        """
        Converts the image pixels to the range [0, 1] for Axelera input.

        Args:
            x: Input image as numpy array

        Returns:
            Preprocessed image
        """
        x = (x / 255.0)
        x = x.astype(np.float32)
        return x

    def predict(self, x):
        """
        Run inference on the Axelera Metis device.

        Args:
            x: Preprocessed input image

        Returns:
            Tuple of (regression_output, classification_output)
        """
        if self.PROFILE:
            start = timer()

        # Ensure input has correct shape (batch, height, width, channels)
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        # Run inference on Axelera device
        try:
            outputs = self.model.run(x)

            # Extract regression and classification outputs
            # Typically: output[0] = regressors (boxes), output[1] = classificators (scores)
            out_reg = outputs[0]
            out_clf = outputs[1]

        except Exception as e:
            print(f"[BlazeDetector.predict] Inference error: {e}")
            raise

        if self.PROFILE:
            end = timer()
            self.profile_pre_nb_frames += 1
            self.profile_pre_time_ms += (end - start) * 1000
            if self.profile_pre_nb_frames == 1:
                self.profile_pre_time_first_ms = (end - start) * 1000

        return out_reg, out_clf
