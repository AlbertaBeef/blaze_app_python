import numpy as np
import os

from blazebase import BlazeLandmarkBase

bUseAxeleraRuntime = False
try:
    from axelera import runtime
    bUseAxeleraRuntime = True
except Exception as e:
    print(f"[BlazeLandmark] Failed to import axelera.runtime: {e}")
    try:
        # Fallback: try importing InferenceStream for simpler API
        from axelera.pipeline import InferenceStream
        bUseAxeleraRuntime = True
    except Exception as e2:
        print(f"[BlazeLandmark] Failed to import axelera.pipeline: {e2}")

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self, blaze_app="blazehand"):
        super(BlazeLandmark, self).__init__()

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
            print(f"[BlazeLandmark.load_model] Model Path: {model_path}")

        # Check if model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        try:
            # Initialize Axelera device (reuse if already created, or create new)
            self.device = runtime.Device()

            if self.DEBUG:
                print(f"[BlazeLandmark.load_model] Axelera device initialized")
                print(f"[BlazeLandmark.load_model] Device info: {self.device.get_info()}")

            # Load the compiled model onto the device
            self.model = runtime.Model(model_path, self.device)

            # Get model input/output information
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

            if self.DEBUG:
                print(f"[BlazeLandmark.load_model] Number of Inputs: {len(self.input_details)}")
                for i, inp in enumerate(self.input_details):
                    print(f"[BlazeLandmark.load_model] Input[{i}]: shape={inp['shape']}, dtype={inp['dtype']}")

                print(f"[BlazeLandmark.load_model] Number of Outputs: {len(self.output_details)}")
                for i, out in enumerate(self.output_details):
                    print(f"[BlazeLandmark.load_model] Output[{i}]: shape={out['shape']}, dtype={out['dtype']}")

            # Get input shape
            self.in_shape = self.input_details[0]['shape']

            # Determine resolution from input shape
            self.resolution = self.in_shape[1]  # Assuming square input

            if self.DEBUG:
                print(f"[BlazeLandmark.load_model] Input Shape: {self.in_shape}")
                print(f"[BlazeLandmark.load_model] Resolution: {self.resolution}")

        except Exception as e:
            print(f"[BlazeLandmark.load_model] Error loading model: {e}")
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
            x: Preprocessed input image(s)

        Returns:
            Tuple of (landmarks, flags/scores)
        """
        if self.PROFILE:
            start = timer()

        # Handle batched input
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        batch_size = x.shape[0]

        # Run inference for each item in batch
        # Axelera runtime may support batching; if not, process individually
        try:
            all_landmarks = []
            all_flags = []

            for i in range(batch_size):
                # Get single input
                input_img = x[i:i+1]

                # Run inference on Axelera device
                outputs = self.model.run(input_img)

                # Extract landmark and flag outputs
                # Output structure varies by model:
                # Hand: [landmarks_3d (63), handflag (1), handedness (1)]
                # Face: [landmarks_2d (1404), faceflag (1)]
                # Pose: [landmarks_3d (195), poseflag (1), segmentation, heatmap, world_landmarks]

                if self.blaze_app in ["blazehand", "blazehandlandmark"]:
                    # Hand landmark model
                    landmarks = outputs[0]  # Shape: (1, 63) for 21 landmarks * 3
                    flag = outputs[1]       # Shape: (1, 1)
                elif self.blaze_app in ["blazeface", "blazefacelandmark"]:
                    # Face landmark model
                    landmarks = outputs[0]  # Shape: (1, 1, 1, 1404) for 468 landmarks * 3
                    flag = outputs[1]       # Shape: (1, 1, 1, 1)
                elif self.blaze_app in ["blazepose", "blazeposelandmark"]:
                    # Pose landmark model
                    landmarks = outputs[0]  # Shape: (1, 195) for 39 landmarks * 5
                    flag = outputs[1]       # Shape: (1, 1)
                else:
                    # Default: assume first output is landmarks, second is flag
                    landmarks = outputs[0]
                    flag = outputs[1]

                all_landmarks.append(landmarks)
                all_flags.append(flag)

            # Stack results
            out_landmarks = np.concatenate(all_landmarks, axis=0)
            out_flags = np.concatenate(all_flags, axis=0)

        except Exception as e:
            print(f"[BlazeLandmark.predict] Inference error: {e}")
            raise

        if self.PROFILE:
            end = timer()
            self.profile_pre_nb_frames += 1
            self.profile_pre_time_ms += (end - start) * 1000
            if self.profile_pre_nb_frames == 1:
                self.profile_pre_time_first_ms = (end - start) * 1000

        return out_landmarks, out_flags
