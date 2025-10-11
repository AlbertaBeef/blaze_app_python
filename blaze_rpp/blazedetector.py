import numpy as np
import sys

from blazebase import BlazeDetectorBase

sys.path.append('/usr/local/rpp/lib')
import pyrt as trt

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1

        self.int8 = True # INT8
        #self.int8 = False # BF16
        

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

        print("[blaze_rpp.BlazeDetector.load_model] Create Network")
        self.net = self.builder.createNetwork()
        print("[blaze_rpp.BlazeDetector.load_model]    net.name = ",self.net.name)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_inputs = ",self.net.num_inputs)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_layers = ",self.net.num_layers)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_outputs = ",self.net.num_outputs)
        print("[blaze_rpp.BlazeDetector.load_model]    net.IsInputProcDisabled() = ",self.net.IsInputProcDisabled())
        print("[blaze_rpp.BlazeDetector.load_model]    net.IsOutputProcDisabled() = ",self.net.IsOutputProcDisabled())
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0) = ",self.net.get_input(0))
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0) = ",self.net.get_output(0))

        print('[blaze_rpp.BlazeDetector.load_model] Create onnx parser.')
        self.parser = trt.OnnxParser(self.net, self.log)

        self.model_path = model_path
        #with open(self.model_path, "rb") as model:
        #    if not self.parser.parse(model.read()):
        #        print("[blaze_rpp.BlazeDetector.load_model] ERROR: Failed to parse the ONNX file:", self.model_path)
        #        for error in range(self.parser.num_errors):
        #            print(self.parser.get_error(error))
        model = open(self.model_path, "rb")
        if not self.parser.parse(model.read()):
            print("[blaze_rpp.BlazeDetector.load_model] ERROR: Failed to parse the ONNX file:", self.model_path)
            for error in range(self.parser.num_errors):
                print(self.parser.get_error(error))

        print("[blaze_rpp.BlazeDetector.load_model]    net.name = ",self.net.name)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_inputs = ",self.net.num_inputs)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_layers = ",self.net.num_layers)
        print("[blaze_rpp.BlazeDetector.load_model]    net.num_outputs = ",self.net.num_outputs)
        print("[blaze_rpp.BlazeDetector.load_model]    net.IsInputProcDisabled() = ",self.net.IsInputProcDisabled())
        print("[blaze_rpp.BlazeDetector.load_model]    net.IsOutputProcDisabled() = ",self.net.IsOutputProcDisabled())

        self.input0 = self.net.get_input(0)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0).name = ",self.input0.name)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0).dimensions = ",self.input0.dimensions)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0).dataType = ",self.input0.dataType)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0).isNetworkInput() = ",self.input0.isNetworkInput())
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_input(0).isNetworkOutput() = ",self.input0.isNetworkOutput())
        self.output0 = self.net.get_output(0)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0).name = ",self.output0.name)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0).dimensions = ",self.output0.dimensions)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0).dataType = ",self.output0.dataType)
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0).isNetworkInput() = ",self.output0.isNetworkInput())
        print("[blaze_rpp.BlazeDetector.load_model]    net.get_output(0).isNetworkOutput() = ",self.output0.isNetworkOutput())

        # parser.parse(model_name)
   
        self.input_dimension = self.net.get_input(0).dimensions
        self.output_dimension = self.net.get_output(0).dimensions
        print("[blaze_rpp.BlazeDetector.load_model]    input dimension : ",self.input_dimension)
        print("[blaze_rpp.BlazeDetector.load_model]    output dimension : ",self.output_dimension)

        print('[blaze_rpp.BlazeDetector.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            return

        print('[blaze_rpp.BlazeDetector.load_model] Initialize IO buffer')
        self.input_binding = trt.DeviceAllocation(volume(self.input_dimension) * 4)
        self.output_binding = trt.DeviceAllocation(volume(self.output_dimension) * 4)

        rng = np.random.default_rng()
        test_data = rng.random(input_dimension)
        input_binding.copy_from_numpy(test_data)

        self.bindings = [int(self.input_binding), int(self.output_binding)]

        self.context = self.engine.createExecutionContext()

        # warmup
        self.context.execute(1, self.bindings)


        # reading onnx model parameters
        self.session_inputs = self.session.get_inputs()
        self.session_outputs = self.session.get_outputs()
        self.num_inputs = len(self.session_inputs)
        self.num_outputs = len(self.session_outputs)
        if self.DEBUG:
           print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeDetector.load_model] Input[",i,"] Shape : ",self.session_inputs[i].shape," (",self.session_inputs[i].name,")")
           print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeDetector.load_model] Output[",i,"] Shape : ",self.session_outputs[i].shape," (",self.session_outputs[i].name,")")

        self.in_shape = self.session_inputs[0].shape
        if self.session_outputs[0].name == "classificators":
            self.out_reg_name = self.session_outputs[1].name
            self.out_clf_name = self.session_outputs[0].name
            self.out_reg_shape = self.session_outputs[1].shape
            self.out_clf_shape = self.session_outputs[0].shape
        else:
            self.out_reg_name = self.session_outputs[0].name
            self.out_clf_name = self.session_outputs[1].name
            self.out_reg_shape = self.session_outputs[0].shape
            self.out_clf_shape = self.session_outputs[1].shape
        if self.DEBUG:
           print("[BlazeDetector.load_model] Input Shape : ",self.in_shape)
           print("[BlazeDetector.load_model] Output1 Shape : ",self.out_reg_shape)
           print("[BlazeDetector.load_model] Output2 Shape : ",self.out_clf_shape)

        self.x_scale = self.in_shape[1]
        self.y_scale = self.in_shape[2]
        self.h_scale = self.in_shape[1]
        self.w_scale = self.in_shape[2]

        self.num_anchors = self.out_clf_shape[1]
        if self.DEBUG:
            print("[BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
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



