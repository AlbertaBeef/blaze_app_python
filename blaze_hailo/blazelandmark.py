import numpy as np

from blazebase import BlazeLandmarkBase

# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_4_Inference_Tutorial.html

from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            VDevice)

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        # Setting VDevice params to disable the HailoRT service feature
        self.params = VDevice.create_params()
        self.params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = VDevice(params=self.params)

        # Loading compiled HEFs to device:
        self.hef = HEF(model_path)

        # Get the "network groups" (connectivity groups, aka. "different networks") information from the .hef
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False,
                                                             format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=True,
                                                               format_type=FormatType.UINT8)

        # Define dataset params
        self.input_vstream_infos = self.hef.get_input_vstream_infos()
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        if self.DEBUG:
           print("[BlazeLandmark.load_model] Input VStream Infos : ",self.input_vstream_infos)
           print("[BlazeLandmark.load_model] Output VStream Infos : ",self.output_vstream_infos)

        # Get input/output tensors dimensions
        self.num_inputs = len(self.input_vstream_infos)
        self.num_outputs = len(self.output_vstream_infos)
        if self.DEBUG:
           print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeDetector.load_model] Input[",i,"] Shape : ",tuple(self.input_vstream_infos[i].shape))
           print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeDetector.load_model] Output[",i,"] Shape : ",tuple(self.output_vstream_infos[i].shape))

        self.inputShape = self.input_vstream_infos[0].shape
        self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
        self.outputShape2 = tuple(self.output_vstream_infos[1].shape)

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Input Shape : ",self.inputShape)
           print("[BlazeLandmark.load_model] Output1 Shape : ",self.outputShape1)
           print("[BlazeLandmark.load_model] Output2 Shape : ",self.outputShape2)

        self.resolution = self.inputShape[1]

    def predict(self, x):

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0

        out1_list = []
        out2_list = []
        out3_list = []

        #print("[BlazeLandmark] x ",x.shape,x.dtype)
        
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
