import numpy as np
import sys

from blazebase import BlazeDetectorBase

sys.path.append('/usr/local/rpp/lib')
import pyrt as trt

def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol

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
           
        self.log = trt.Logger(trt.Logger.INTERNAL_ERROR)
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

        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model]    name = ",self.net.name)
            print("[blaze_rpp.BlazeDetector.load_model]    num_inputs = ",self.net.num_inputs)
            print("[blaze_rpp.BlazeDetector.load_model]    num_layers = ",self.net.num_layers)
            print("[blaze_rpp.BlazeDetector.load_model]    num_outputs = ",self.net.num_outputs)
            print("[blaze_rpp.BlazeDetector.load_model]    IsInputProcDisabled() = ",self.net.IsInputProcDisabled())
            print("[blaze_rpp.BlazeDetector.load_model]    IsOutputProcDisabled() = ",self.net.IsOutputProcDisabled())

        self.num_inputs = self.net.num_inputs
        self.num_outputs = self.net.num_outputs
        self.num_layers = self.net.num_layers

        self.bindings = []
        self.input_names = []
        self.output_names = []
        self.input_dimensions = []
        self.output_dimensions = []
        self.input_bindings = []
        self.output_bindings = []

        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Initialize IO buffers')
        for i in range(self.num_inputs):
            inputx = self.net.get_input(i)
            if self.DEBUG:
                print(f"[blaze_rpp.BlazeDetector.load_model]    net.get_input({i})")
                print("[blaze_rpp.BlazeDetector.load_model]       name = ",inputx.name)
                print("[blaze_rpp.BlazeDetector.load_model]       dimensions = ",inputx.dimensions)
                print("[blaze_rpp.BlazeDetector.load_model]       dataType = ",inputx.dataType)
                print("[blaze_rpp.BlazeDetector.load_model]       isNetworkInput() = ",inputx.isNetworkInput())
                print("[blaze_rpp.BlazeDetector.load_model]       isNetworkOutput() = ",inputx.isNetworkOutput())
            input_dimension = inputx.dimensions
            input_size = volume(input_dimension) * 4
            if self.DEBUG:
                print("[blaze_rpp.BlazeDetector.load_model]       input_size = ",input_size)
            input_binding = trt.DeviceAllocation(input_size)
            #
            self.input_names.append(inputx.name)
            self.input_dimensions.append(inputx.dimensions)
            self.input_bindings.append(input_binding)
            self.bindings.append(int(input_binding))

        for i in range(self.num_outputs):
            outputx = self.net.get_output(i)
            if self.DEBUG:
                print(f"[blaze_rpp.BlazeDetector.load_model]    net.get_output({i})")
                print("[blaze_rpp.BlazeDetector.load_model]       name = ",outputx.name)
                print("[blaze_rpp.BlazeDetector.load_model]       dimensions = ",outputx.dimensions)
                print("[blaze_rpp.BlazeDetector.load_model]       dataType = ",outputx.dataType)
                print("[blaze_rpp.BlazeDetector.load_model]       isNetworkInput() = ",outputx.isNetworkInput())
                print("[blaze_rpp.BlazeDetector.load_model]       isNetworkOutput() = ",outputx.isNetworkOutput())
            output_dimension = outputx.dimensions
            output_size = volume(output_dimension) * 4
            if self.DEBUG:
                print("[blaze_rpp.BlazeDetector.load_model]       output_size = ",output_size)
            output_binding = trt.DeviceAllocation(output_size)
            #
            self.output_names.append(outputx.name)
            self.output_dimensions.append(outputx.dimensions)
            self.output_bindings.append(output_binding)
            self.bindings.append(int(output_binding))
    
        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            return

        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Create execution context")
        self.context = self.engine.createExecutionContext()

        if self.DEBUG:
            print('[blaze_rpp.BlazeDetector.load_model] Prepare Input')
        self.rng = np.random.default_rng()
        if self.DEBUG:
            print(f"[blaze_rpp.BlazeDetector.load_model]    Creating random test data")
        test_data = self.rng.random(input_dimension,dtype=np.float32)
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model]       test_data.shape = ",test_data.shape)
            print("[blaze_rpp.BlazeDetector.load_model]       test_data.dtype = ",test_data.dtype)
        input_binding = self.input_bindings[0]
        input_binding.copy_from_numpy(test_data)

        # Inference (warmup)
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Execute (warmup)")
        self.context.execute(1, self.bindings)

        self.in_shape = self.input_dimensions[0]
        if self.output_names[0] == "classifiers": # ... TBD ...
            self.out_reg_name = self.output_names[1]
            self.out_clf_name = self.output_names[0]
            self.out_reg_shape = self.output_dimensions[1]
            self.out_clf_shape = self.output_dimensions[0]
        else:
            self.out_reg_name = self.output_names[0]
            self.out_clf_name = self.output_names[1]
            self.out_reg_shape = self.output_dimensions[0]
            self.out_clf_shape = self.output_dimensions[1]
        if self.DEBUG:
           print("[blaze_rpp.BlazeDetector.load_model] Input Shape : ",self.in_shape)
           print("[blaze_rpp.BlazeDetector.load_model] Output1 Shape : ",self.out_reg_shape)
           print("blaze_rpp.[BlazeDetector.load_model] Output2 Shape : ",self.out_clf_shape)

        self.x_scale = self.in_shape[1]
        self.y_scale = self.in_shape[2]
        self.h_scale = self.in_shape[1]
        self.w_scale = self.in_shape[2]

        self.num_anchors = self.out_clf_shape[1]
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
        #self.interp_detector.set_tensor(self.in_idx, x)
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()
        input_name = self.session_inputs[0].name
        #output_names = [output.name for output in self.session_outputs]
        output_names = [self.out_clf_name,self.out_reg_name]
        result = self.session.run(output_names, {input_name: x})   
        self.profile_model = timer()-start

        out1 = result[0] # classificators [1,anchors,1]
        out2 = result[1] # regressors     [1,anchors,18]

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

        start = timer() 

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



