import numpy as np
import sys
import math
from pathlib import Path
from datetime import datetime

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

        self.log = trt.Logger(trt.Logger.ERROR)
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
            print('[blaze_rpp.BlazeLandmark.load_model] Build IEngine')
        self.engine = self.builder.build_EngineWithConfig(self.net, self.config)
        if self.engine is None:
            print('[ERROR] Failed to build engine')
            return

        if self.DEBUG:
            print('[blaze_rpp.BlazeLandmark.load_model] Initialize IO buffers')
        self.output_index_mapping = {}
        for index in range(len(self.engine)):
            name = self.engine.get_binding_name(index)
            shape = self.engine.get_binding_shape(index).get()
            is_input = self.engine.binding_is_input(index)

            bytes_size = volume(shape) * 4
            binding = trt.DeviceAllocation(bytes_size)

            init_data = np.zeros(shape, dtype=np.float32)
            binding.copy_from_numpy(init_data)

            self.bindings.append(int(binding))
            if is_input:
                self.input_names.append(name)
                self.input_dimensions.append(shape)
                self.input_bindings.append(binding)
                input_dimension = shape
            else:
                self.output_names.append(name)
                self.output_dimensions.append(shape)
                self.output_bindings.append(binding)

                output_index = index - 1
                self.output_index_mapping[output_index] = name

        if self.DEBUG:
            print("[blaze_rpp.BlazeLandmark.load_model] Model Input/Output:")
            for index in range(len(self.input_names)):
                print(f"[blaze_rpp.BlazeLandmark.load_model]    input({index})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",self.input_names[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",self.input_dimensions[index])
            for index in range(len(self.output_names)):
                print(f"[blaze_rpp.BlazeLandmark.load_model]    output({index})")
                print("[blaze_rpp.BlazeLandmark.load_model]       name = ",self.output_names[index])
                print("[blaze_rpp.BlazeLandmark.load_model]       dimensions = ",self.output_dimensions[index])

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
            input_data = np.array([xi],dtype=np.float32)
            input_binding.copy_from_numpy(input_data.ravel())
            self.profile_pre += timer()-start

            # 2. Run the neural network:
            start = timer()
            self.context.execute(1, self.bindings)
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
