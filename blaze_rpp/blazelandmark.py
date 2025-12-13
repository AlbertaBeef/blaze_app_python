import numpy as np
import sys
import math
from pathlib import Path
from datetime import datetime

from blazebase import BlazeLandmarkBase

sys.path.append('/usr/local/rpp/lib')
import pyrt as trt

def rt_type_to_numpy_type(rt_type):
    if rt_type == trt.DataType.kFLOAT:
        return np.float32
    if rt_type == trt.DataType.kBF:
        return np.float16
    raise NotImplementedError(f"No supported data type, value: {rt_type}")

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
            print('[blaze_rpp.BlazeLandmark.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            print('[ERROR] Failed to build engine')
            return

        if self.DEBUG:
            print('[blaze_rpp.BlazeLandmark.load_model] Initialize IO buffers')
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
                print("[blaze_rpp.BlazeLandmark.load_model]       bytes_size = ",bytes_size)
                print("[blaze_rpp.BlazeLandmark.load_model]       binding = ",binding)

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
            print("[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:")
            for index in range(self.num_inputs):
                print(f"[blaze_rpp.BlazeLandmark.load_model]    input({index})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",self.input_names[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",self.input_dimensions[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dtype = ",self.input_dtypes[index])
            for index in range(self.num_output):
                print(f"[blaze_rpp.BlazeLandmark.load_model]    output({index})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",self.output_names[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",self.output_dimensions[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dtype = ",self.output_dtypes[index])

        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model] Create execution context")
        self.context = self.engine.createExecutionContext()

        self.resolution = self.input_dimensions[0][1]

    def preprocess(self, x):
        # image was already pre-processed by extract_roi in blaze_common/blazebase.py
        # format = RGB
        # dtype = float32
        # range = 0.0 - 1.0
        return x

    def predict(self, x):

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        np.set_printoptions(precision=6, suppress=True)

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
            input_binding = self.input_bindings[0]
            input_data = np.array([xi],dtype=self.input_dtypes[0])
            input_binding.copy_from_numpy(input_data.ravel())
            self.profile_pre += timer()-start

            # 2. Run the neural network:
            start = timer()
            if self.DEBUG:
                #print('[blaze_rpp.BlazeLandmark.predict] Execute context with ',self.binding_int_values)
                for index, int_value in enumerate(self.binding_int_values):
                    print(f"[blaze_rpp.BlazeLandmark.predict], binding index: {index}, value: {hex(int_value)}")
                print(f"[blaze_rpp.BlazeLandmark.predict] Execute context with engine bindings: {len(self.engine)}")

            self.context.execute(1, self.binding_int_values)
            if self.DEBUG:
                print('[blaze_rpp.blazehandlandmark.predict_on_batch] Finished inference')

            self.profile_model += timer()-start

            # 3. Extract outputs
            start = timer()

            if self.blaze_app == "blazehandlandmark" and self.resolution == 256:
                #[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:           
                #[blaze_rpp.BlazeLandmark.load_model]    input(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  input_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (256, 256, 3)
                #[blaze_rpp.BlazeLandmark.load_model]    output(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  ld_21_3d
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (63,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(1)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  output_handflag
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(2)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  output_handedness
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                
                # output_handflag
                out1 = self.output_bindings[1].numpy_float()
                
                # ld_21_3d
                out2 = self.output_bindings[0].numpy_float()
                out2 = out2.reshape(21,-1) # 63 => [21,3]
                out2 = out2/self.resolution

                # output_handedness
                out3 = self.output_bindings[2].numpy_float()

            if self.blaze_app == "blazehandlandmark" and self.resolution == 224:
                #[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:           
                #[blaze_rpp.BlazeLandmark.load_model]    input(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  input_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (224, 224, 3)
                #[blaze_rpp.BlazeLandmark.load_model]    output(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(1)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_2
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(2)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (63,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(3)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_3
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (63,)            
                
                # Identity_1
                out1 = self.output_bindings[0].numpy_float()

                # Identity
                out2 = self.output_bindings[2].numpy_float()
                out2 = out2.reshape(21, -1)  # 63 => [21,3]
                out2 = out2/self.resolution

                # Identity_2
                out3 = self.output_bindings[1].numpy_float()

            elif self.blaze_app == "blazefacelandmark":
                #[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:
                #[blaze_rpp.BlazeLandmark.load_model]    input(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  input_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (192, 192, 3)
                #[blaze_rpp.BlazeLandmark.load_model]    output(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  conv2d_31
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1, 1, 1)
                #[blaze_rpp.BlazeLandmark.load_model]    output(1)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  conv2d_21
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1, 1, 1404)

                # conv2d_31
                out1 = self.output_bindings[0].numpy_float()
                out1 = out1.reshape(1,1)
            
                # conv2d_21
                out2 = self.output_bindings[1].numpy_float()
                out2 = out2.reshape(-1,3) # 1404 => [468,3]
                out2 = out2/self.resolution            

            elif self.blaze_app == "blazeposelandmark" and self.output_names[1] == "Identity":
                #[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:
                #[blaze_rpp.BlazeLandmark.load_model]    input(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  input_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (256, 256, 3)
                #[blaze_rpp.BlazeLandmark.load_model]    output(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_4
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (117,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(1)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (195,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(2)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(3)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_3
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (64, 64, 39)
                #[blaze_rpp.BlazeLandmark.load_model]    output(4)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_2
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (256, 256, 1)

                # Identity_1
                out1 = self.output_bindings[2].numpy_float()
                out1 = out1.reshape(1,1)
                
                # Identity
                out2 = self.output_bindings[1].numpy_float()
                out2 = out2.reshape(-1,5) # 195 => [39,5]
                out2 = out2/self.resolution
                
            elif self.blaze_app == "blazeposelandmark" and self.output_names[2] == "Identity":
                #[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:
                #[blaze_rpp.BlazeLandmark.load_model]    input(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  input_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (256, 256, 3)
                #[blaze_rpp.BlazeLandmark.load_model]    output(0)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_2
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (256, 256, 1)
                #[blaze_rpp.BlazeLandmark.load_model]    output(1)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_4
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (117,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(2)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (195,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(3)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_1
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (1,)
                #[blaze_rpp.BlazeLandmark.load_model]    output(4)
                #[blaze_rpp.BlazeLandmark.load_model]       name =  Identity_3
                #[blaze_rpp.BlazeLandmark.load_model]       dimensions =  (64, 64, 39)

                # Identity_1
                out1 = self.output_bindings[3].numpy_float()
                out1 = out1.reshape(1,1)
                
                # Identity
                out2 = self.output_bindings[2].numpy_float()
                out2 = out2.reshape(-1,5) # 195 => [39,5]
                out2 = out2/self.resolution
                

            #if self.DEBUG:
            #    print("[blaze_rpp.BlazeLandmark.predict] out1 (condifence)",out1.shape,out1.dtype, out1)
            #    print("[blaze_rpp.BlazeLandmark.predict] out2 (landmarks)",out2.shape,out2.dtype, out2)
            #    if self.blaze_app == "blazehandlandmark":
            #        print("[blaze_rpp.BlazeLandmark.predict] out3 (handedness)",out3.shape,out3.dtype, out3)
            #        #print("[blaze_rpp.BlazeLandmark.predict] out4 (mini hand)",out4.shape,out4.dtype, out4)

            out1_list.append(out1)
            out2_list.append(out2)
            if self.blaze_app == "blazehandlandmark":
                out3_list.append(out3)
            self.profile_post += timer()-start

        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)
        if self.blaze_app == "blazehandlandmark":
            handedness_scores = np.asarray(out3_list)

        #if self.DEBUG:
        #    print("[blaze_rpp.BlazeLandmark.predict] flag ",flag.shape,flag.dtype)
        #    print("[blaze_rpp.BlazeLandmark.predict] landmarks ",landmarks.shape,landmarks.dtype)
        #    if self.blaze_app == "blazehandlandmark":
        #        print("[blaze_rpp.BlazeLandmark.predict] handedness_scores ",handedness_scores.shape,handedness_scores.dtype)

        if self.blaze_app == "blazehandlandmark":
            return flag,landmarks,handedness_scores
        else:
            return flag,landmarks
