import numpy as np

from blazebase import BlazeLandmarkBase

import xir
import vitis_ai_library

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        # Create graph runner
        self.g = xir.Graph.deserialize(model_path)
        self.runner = vitis_ai_library.GraphRunner.create_graph_runner(self.g)

        self.input_tensor_buffers  = self.runner.get_inputs()
        self.output_tensor_buffers = self.runner.get_outputs()

        # Get input scaling
        self.input_fixpos = self.input_tensor_buffers[0].get_tensor().get_attr("fix_point")
        self.input_scale = 2**self.input_fixpos

       # Get input/output tensors dimensions
        self.num_inputs = len(self.input_tensor_buffers)
        self.num_outputs = len(self.output_tensor_buffers)
        if self.DEBUG:
           print("[BlazeLandmark.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeLandmark.load_model] Input[",i,"] Shape : ",tuple(self.input_tensor_buffers[i].get_tensor().dims))
           print("[BlazeLandmark.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeLandmark.load_model] Output[",i,"] Shape : ",tuple(self.output_tensor_buffers[i].get_tensor().dims))

        self.inputShape = tuple(self.input_tensor_buffers[0].get_tensor().dims)
        self.outputShape1 = tuple(self.output_tensor_buffers[0].get_tensor().dims)
        self.outputShape2 = tuple(self.output_tensor_buffers[1].get_tensor().dims)
        self.outputShape3 = tuple(self.output_tensor_buffers[2].get_tensor().dims)
        self.batchSize = self.inputShape[0]

        #if self.DEBUG:
        #   print("[BlazeLandmark.load_model] Input Shape : ",self.inputShape)
        #   print("[BlazeLandmark.load_model] Output1 Shape : ",self.outputShape1)
        #   print("[BlazeLandmark.load_model] Output2 Shape : ",self.outputShape2)
        #   print("[BlazeLandmark.load_model] Output3 Shape : ",self.outputShape3)


        self.resolution = self.inputShape[1]

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

        out1_list = []
        out2_list = []
        out3_list = []

        #print("[BlazeLandmark] x ",x.shape,x.dtype)
        start = timer()        
        x = self.preprocess(x)
        self.profile_pre += timer()-start

        
        nb_images = x.shape[0]
        for i in range(nb_images):

            # 1. Preprocess the images into tensors:
            start = timer()
            input_data = np.asarray(self.input_tensor_buffers[0])
            input_data[0] = x[i,:,:,:]
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            """ Execute model on DPU """
            job_id = self.runner.execute_async(self.input_tensor_buffers, self.output_tensor_buffers)
            self.runner.wait(job_id)
            self.profile_model += timer()-start

            start = timer()  

            if self.blaze_app == "blazehandlandmark":
               out1 = np.asarray(self.output_tensor_buffers[0]) 
               #handedness = np.asarray(self.output_tensor_buffers[1]) 
               out2 = np.asarray(self.output_tensor_buffers[2]) 
            elif self.blaze_app == "blazefacelandmark":
               out1 = np.asarray(self.output_tensor_buffers[0]) 
               out2 = np.asarray(self.output_tensor_buffers[1]) 
            elif self.blaze_app == "blazeposelandmark":
               out1 = np.asarray(self.output_tensor_buffers[0]) 
               out2 = np.asarray(self.output_tensor_buffers[1]) 

            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        

        #if self.DEBUG:
        #    print("[BlazeLandmark] flag ",flag.shape,flag.dtype)
        #    print("[BlazeLandmark] landmarks ",landmarks.shape,landmarks.dtype)

        return flag,landmarks
