import numpy as np
import sys

from blazebase import BlazeLandmarkBase

sys.path.append('/usr/local/rpp/lib')
import pyrt as trt

def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app

        #self.int8 = True # INT8
        self.int8 = False # BF16

    def load_model(self, model_path):

        if self.DEBUG:
           print("[blaze_rpp.BlazeLandmark.load_model] Model File : ",model_path)
           
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
            print("[blaze_rpp.BlazeLandmark.load_model] Create Network")
        self.net = self.builder.createNetwork()

        if self.DEBUG:
            print('[blaze_rpp.BlazeLandmark.load_model] Create onnx parser.')
        self.parser = trt.OnnxParser(self.net, self.log)

        self.model_path = model_path
        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model] Parsing model : ",self.model_path)
        model = open(self.model_path, "rb")
        if not self.parser.parse(model.read()):
            print("[blaze_rpp.BlazeLandmark.load_model]    ERROR: Failed to parse the ONNX file:", self.model_path)
            for error in range(self.parser.num_errors):
                print(self.parser.get_error(error))

        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model]    name = ",self.net.name)
            print("[blaze_rpp.BlazeLandmark.load_model]    num_inputs = ",self.net.num_inputs)
            print("[blaze_rpp.BlazeLandmark.load_model]    num_layers = ",self.net.num_layers)
            print("[blaze_rpp.BlazeLandmark.load_model]    num_outputs = ",self.net.num_outputs)
            print("[blaze_rpp.BlazeLandmark.load_model]    IsInputProcDisabled() = ",self.net.IsInputProcDisabled())
            print("[blaze_rpp.BlazeLandmark.load_model]    IsOutputProcDisabled() = ",self.net.IsOutputProcDisabled())

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
            print('[blaze_rpp.BlazeLandmark.load_model] Initialize IO buffers')
        for i in range(self.num_inputs):
            inputx = self.net.get_input(i)
            if self.DEBUG:
                print(f"[blaze_rpp.BlazeLandmark.load_model]    net.get_input({i})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",inputx.name)
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",inputx.dimensions)
                print("[blaze_rpp.BlazeLandmark.load_model]       dataType = ",inputx.dataType)
                print("[blaze_rpp.BlazeLandmark.load_model]       isNetworkInput() = ",inputx.isNetworkInput())
                print("[blaze_rpp.BlazeLandmark.load_model]       isNetworkOutput() = ",inputx.isNetworkOutput())
            input_dimension = inputx.dimensions
            input_size = volume(input_dimension) * 4
            if self.DEBUG:
                print("[blaze_rpp.BlazeLandmark.load_model]       input_size = ",input_size)
            input_binding = trt.DeviceAllocation(input_size)
            #
            self.input_names.append(inputx.name)
            self.input_dimensions.append(inputx.dimensions)
            self.input_bindings.append(input_binding)
            self.bindings.append(int(input_binding))

        for i in range(self.num_outputs):
            outputx = self.net.get_output(i)
            if self.DEBUG:
                print(f"[blaze_rpp.BlazeLandmark.load_model]    net.get_output({i})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",outputx.name)
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",outputx.dimensions)
                print("[blaze_rpp.BlazeLandmark.load_model]       dataType = ",outputx.dataType)
                print("[blaze_rpp.BlazeLandmark.load_model]       isNetworkInput() = ",outputx.isNetworkInput())
                print("[blaze_rpp.BlazeLandmark.load_model]       isNetworkOutput() = ",outputx.isNetworkOutput())
            output_dimension = outputx.dimensions
            output_size = volume(output_dimension) * 4
            if self.DEBUG:
                print("[blaze_rpp.BlazeLandmark.load_model]       output_size = ",output_size)
            output_binding = trt.DeviceAllocation(output_size)
            #
            self.output_names.append(outputx.name)
            self.output_dimensions.append(outputx.dimensions)
            self.output_bindings.append(output_binding)
            self.bindings.append(int(output_binding))
    
        if self.DEBUG:
            print('[blaze_rpp.BlazeLandmark.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            return

        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model] Create execution context")
        self.context = self.engine.createExecutionContext()

        if self.DEBUG:
            print('[blaze_rpp.BlazeLandmark.load_model] Prepare Input')
        self.rng = np.random.default_rng()
        if self.DEBUG:
            print(f"[blaze_rpp.BlazeLandmark.load_model]    Creating random test data")
        test_data = self.rng.random(input_dimension,dtype=np.float32)
        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model]       test_data.shape = ",test_data.shape)
            print("[blaze_rpp.BlazeLandmark.load_model]       test_data.dtype = ",test_data.dtype)
        input_binding = self.input_bindings[0]
        input_binding.copy_from_numpy(test_data)

        # Inference (warmup)
        if self.DEBUG:
            print("[blaze_rpp.BlazeDetector.load_model] Execute (warmup)")
        self.context.execute(1, self.bindings)

        self.in_shape = self.input_dimensions[0]
        self.out_landmark_shape = self.output_dimensions[0]
        self.out_flag_shape = self.output_dimensions[1]
        if self.DEBUG:
           print("[blaze_rpp.BlazeLandmark.load_model] Input Shape : ",self.in_shape)
           print("[blaze_rpp.BlazeLandmark.load_model] Output1 Shape : ",self.out_landmark_shape)
           print("[blaze_rpp.BlazeLandmark.load_model] Output2 Shape : ",self.out_flag_shape)

        self.resolution = self.in_shape[1]

    def preprocess(self, x):
        # image was already pre-processed by extract_roi in blaze_common/blazebase.py
        # format = RGB
        # dtype = float32
        # range = 0.0 - 1.0
        x = x * 128
        return x

    def predict(self, x):

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        out1_list = []
        out2_list = []
        out3_list = []

        #print("[BlazeLandmark] x ",x.shape,x.dtype)
        start = timer()        
        x = self.preprocess(x)
        self.profile_pre += timer()-start
                
        nb_images = x.shape[0]
        for i in range(nb_images):

            start = timer()
            xi = np.expand_dims(x[i,:,:,:], axis=0)
            #print("[BlazeLandmark] xi ",xi.shape,xi.dtype)

            # 1. Preprocess the images into tensors:
            #self.interp_landmark.set_tensor(self.in_idx, xi)
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            
            #print('Prepare Input')
            input_binding = self.input_bindings[0]
            input_data = np.array([xi],dtype=np.float32)
            input_binding.copy_from_numpy(input_data)

            # Inference
            #print("Inference")
            self.context.execute(1, self.bindings)
                
            self.profile_model += timer()-start

            start = timer()  

            if self.blaze_app == "blazehandlandmark":
                out2 = self.output_bindings[3].numpy_float()
                out3 = self.output_bindings[1].numpy_float()
                out1 = self.output_bindings[2].numpy_float()
                #
                out2 = out2.reshape(21,-1) # 42 => [21,2] / 63 => [21,3]
                #out2 = out2/self.resolution
            elif self.blaze_app == "blazefacelandmark":
                out2 = self.output_bindings[0].numpy_float()
                out1 = self.output_bindings[1].numpy_float()
                #
                out1 = out1.reshape(1,1)
                out2 = out2.reshape(-1,3) # 1404 => [356,2]
                #out2 = out2/self.resolution            
            elif self.blaze_app == "blazeposelandmark":
                out2 = self.output_bindings[0].numpy_float()
                out1 = self.output_bindings[1].numpy_float()
                #
                out2 = out2.reshape(-1,5) # 195 => [39,5]
                #out2 = out2/self.resolution

            if self.DEBUG:
                print("[blaze_rpp.BlazeLandmark.predict] out1 ",out1.shape,out1.dtype, out1)
                print("[blaze_rpp.BlazeLandmark.predict] out2 ",out2.shape,out2.dtype, out2)
                if self.blaze_app == "blazehandlandmark":
                    print("[blaze_rpp.BlazeLandmark.predict] out3 ",out3.shape,out3.dtype, out3)

            out1_list.append(out1)
            out2_list.append(out2)
            if self.blaze_app == "blazehandlandmark":
                out3_list.append(out3)
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        
        if self.blaze_app == "blazehandlandmark":
            handedness_scores = np.asarray(out3_list)

        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.predict] flag ",flag.shape,flag.dtype)
            print("[blaze_rpp.BlazeLandmark.predict] landmarks ",landmarks.shape,landmarks.dtype)
            if self.blaze_app == "blazehandlandmark":
                print("[blaze_rpp.BlazeLandmark.predict] handedness_scores ",handedness_scores.shape,handedness_scores.dtype)

        if self.blaze_app == "blazehandlandmark":
            return flag,landmarks,handedness_scores
        else:
            return flag,landmarks
