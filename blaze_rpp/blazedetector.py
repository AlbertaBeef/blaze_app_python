import numpy as np
import sys
import math
from pathlib import Path
from datetime import datetime

from blazebase import BlazeDetectorBase

sys.path.append('/usr/local/rpp/lib')
import pyrt as trt

def rt_type_to_numpy_type(rt_type):
    if rt_type == trt.DataType.kFLOAT:
        return np.float32
    if rt_type == trt.DataType.kBF:
        return np.float16
    raise NotImplementedError(f"No supported data type, value: {rt_type}")

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1

        #self.int8 = True # INT8
        self.int8 = False # BF16

    def load_model(self, model_path):

        if self.DEBUG:
           print("[blaze_rpp.BlazeDetector.load_model] Model File : ",model_path)

        if self.DEBUG:
            #self.log = trt.Logger(trt.Logger.INTERNAL_ERROR)
            self.log = trt.Logger(trt.Logger.ERROR)
            #self.log = trt.Logger(trt.Logger.INFO)
            #self.log = trt.Logger(trt.Logger.VERBOSE)
        else:
            self.log = trt.Logger(trt.Logger.INTERNAL_ERROR)
            #self.log = trt.Logger(trt.Logger.ERROR)
        
        self.builder = trt.Builder(self.log)
        self.config = self.builder.createBuilderConfig()
        if self.int8:
            self.config.setFlag(trt.BuilderFlag.INT8)
            self.int8_calibrator = trt.Int8EntropyCalibrator()
            self.config.setInt8Calibrator(self.int8_calibrator)
        else:
            self.config.setFlag(trt.BuilderFlag.BF16)

        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Create Network")
        self.net = self.builder.createNetwork()

        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Create onnx parser.')
        self.parser = trt.OnnxParser(self.net, self.log)

        self.model_path = model_path
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Parsing model : ",self.model_path)
        model = open(self.model_path, "rb")
        if not self.parser.parse(model.read()):
            print("[blaze_rpp.BlazeDetector.load_model]    ERROR: Failed to parse the ONNX file:", self.model_path)
            for error in range(self.parser.num_errors):
                print(self.parser.get_error(error))

        self.binding_list = []
        self.binding_int_values = []
        self.input_names = []
        self.output_names = []
        self.input_dimensions = []
        self.output_dimensions = []
        self.input_dtypes = []
        self.output_dtypes = []
        self.input_bindings = []
        self.output_bindings = []

        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            print('[ERROR] Failed to build engine')
            return

        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Initialize IO buffers')
        for index in range(len(self.engine)):
            name = self.engine.get_binding_name(index)
            shape = self.engine.get_binding_shape(index).get()
            is_input = self.engine.binding_is_input(index)
            rt_dtype = self.engine.get_binding_dtype(index)
            dtype = rt_type_to_numpy_type(rt_dtype)

            bytes_size = math.prod(shape) * np.dtype(dtype).itemsize

            binding = trt.DeviceAllocation(bytes_size)
            if self.DEBUG:
                print(f"[blaze_rpp.BlazeLandmark.load_model]    io({index})")
                print("[blaze_rpp.BlazeDetector.load_model]       bytes_size = ",bytes_size)
                print("[blaze_rpp.BlazeDetector.load_model]       binding = ",binding)

            self.binding_list.append(binding)
            self.binding_int_values.append(int(binding))
            if is_input:
                self.input_names.append(name)
                self.input_dimensions.append(shape)
                self.input_dtypes.append(dtype)
                self.input_bindings.append(binding)
                input_dimension = shape
            else:
                self.output_names.append(name)
                self.output_dimensions.append(shape)
                self.output_dtypes.append(dtype)
                self.output_bindings.append(binding)

                
        self.num_inputs = len(self.input_names)
        self.num_output = len(self.output_names)
        
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Model Input/Output:")
            for index in range(self.num_inputs):
                print(f"[blaze_rpp.BlazeDetector.load_model]    input({index})")
                print("[blaze_rpp.BlazeDetector.load_model]       name = ",self.input_names[index])
                print("[blaze_rpp.BlazeDetector.load_model]       dimensions = ",self.input_dimensions[index])
                print("[blaze_rpp.BlazeDetector.load_model]       dtype = ",self.input_dtypes[index])
            for index in range(self.num_output):
                print(f"[blaze_rpp.BlazeDetector.load_model]    output({index})")
                print("[blaze_rpp.BlazeDetector.load_model]       name = ",self.output_names[index])
                print("[blaze_rpp.BlazeDetector.load_model]       dimensions = ",self.output_dimensions[index])
                print("[blaze_rpp.BlazeDetector.load_model]       dtype = ",self.output_dtypes[index])

        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Create execution context")
        self.context = self.engine.createExecutionContext()

        self.x_scale = self.input_dimensions[0][0]
        self.y_scale = self.input_dimensions[0][1]
        self.h_scale = self.input_dimensions[0][0]
        self.w_scale = self.input_dimensions[0][1]

        self.num_anchors = self.output_dimensions[0][0]
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        """Converts the image pixels to defined input scale."""
        x = x.astype(np.float32)
        x = (x / 255.0)

       
        return x

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        
        # Convert img.unsqueeze(0) to NumPy equivalent
        img_expanded = np.expand_dims(img, axis=0)

        # Call the predict_on_batch function
        detections = self.predict_on_batch(img_expanded)

        # Extract the first element from the predictions
        #return predictions[0]        
        if len(detections)>0:
            return np.array(detections)[0]
        else:
            return []


    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0
        
        assert x.shape[3] == 3
        assert x.shape[1] == self.y_scale
        assert x.shape[2] == self.x_scale

        # 1. Preprocess the images into tensors:
        start = timer()
        x = self.preprocess(x)
        
        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.predict_on_batch] Prepare Input')
        input_binding = self.input_bindings[0]
        input_data = np.array([x],dtype=self.input_dtypes[0])
        input_binding.copy_from_numpy(input_data.ravel())        

        self.profile_pre = timer()-start

        # 2. Run the neural network:
        start = timer()
        if self.DEBUG:
            #print('[blaze_rpp.BlazeDetector.predict_on_batch] Execute context with ',self.binding_int_values)
            for index, int_value in enumerate(self.binding_int_values):
            	print(f"[blaze_rpp.BlazeDetector.predict_on_batch], binding index: {index}, value: {hex(int_value)}")
            print(f"[blaze_rpp.BlazeDetector.predict_on_batch] Execute context with engine bindings: {len(self.engine)}")
        self.context.execute(1, self.binding_int_values)
        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.predict_on_batch] Finished inference')
        self.profile_model = timer()-start

        # 3. Extract outputs
        start = timer() 
        
        # classificators [1,anchors,1]
        out1 = self.output_bindings[1].numpy_float()
        out1 = out1.reshape(1, self.num_anchors, -1)

        # regressors     [1,anchors,18]
        out2 = self.output_bindings[0].numpy_float()
        out2 = out2.reshape(1, self.num_anchors, -1)
        
        #if self.DEBUG:
        #    print("[BlazeDetector.predict] Input   : ",x.shape, x.dtype)
        #    print("[BlazeDetector.predict] Input Min/Max: ",np.amin(x),np.amax(x))
        #    print("[BlazeDetector.predict] Output1 : ",out1.shape, out1.dtype)
        #    print("[BlazeDetector.predict] Output1 Min/Max: ",np.amin(out1),np.amax(out1))
        #    print("[BlazeDetector.predict] Output2 : ",out2.shape, out2.dtype)
        #    print("[BlazeDetector.predict] Output2 Min/Max: ",np.amin(out2),np.amax(out2))

        assert out1.shape[0] == 1 # batch
        assert out1.shape[1] == self.num_anchors
        assert out1.shape[2] == 1

        assert out2.shape[0] == 1 # batch
        assert out2.shape[1] == self.num_anchors
        assert out2.shape[2] == self.num_coords

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out2, out1, self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)
                if len(filtered_detections) > 0:
                    normalized_detections = np.array(filtered_detections)[0]

        self.profile_post = timer()-start

        return filtered_detections



