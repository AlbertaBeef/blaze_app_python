import numpy as np

from blazebase import BlazeLandmarkBase


# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_4_Inference_Tutorial.html

from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            Device, VDevice)


# Reference : https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/main/runtime/python/model_scheduler_inference/hailo_inference_scheduler.py

import os
import psutil

# ----------------------------------------------------------- #
# --------------- Hailo Scheduler service functions ---------- #

def check_if_service_enabled(process_name):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            if process_name.lower() in proc.name().lower():
                print('HailoRT Scheduler service is enabled!')
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print('HailoRT Scheduler service is disabled. Enabling service...')
    os.system('sudo systemctl disable hailort.service --now  && sudo systemctl daemon-reload && sudo systemctl enable hailort.service --now')
    

def create_vdevice_params():
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    if False: #if args.use_multi_process:
        params.group_id = "SHARED"
    return params


from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app
        
        #check_if_service_enabled('hailort_service')
        
        self.params = VDevice.create_params()
        
        # Setting VDevice params to disable the HailoRT service feature
        self.params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        #[HailoRT] [error] CHECK_AS_EXPECTED failed - Failed to create vdevice. there are not enough free devices. requested: 1, found: 0
        #[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)
        
        #self.params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        #self.params.group_id = "SHARED"
        #[HailoRT] [error] multi_process_service requires service compilation with HAILO_BUILD_SERVICE
        #Traceback (most recent call last):
        #  File "/usr/lib/python3.9/site-packages/hailo_platform/pyhailort/pyhailort.py", line 2626, in _open_vdevice
        #    self._vdevice = _pyhailort.VDevice.create(self._params, device_ids)
        #hailo_platform.pyhailort._pyhailort.HailoRTStatusException: 6



    def load_model(self, model_path):

        if self.DEBUG:
            print("[BlazeLandmark.load_model] Model File : ",model_path)
            #[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_lite.hef
           
        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        #self.target = VDevice(params=self.params)
        self.devices = Device.scan()
        if self.DEBUG:
            print("[BlazeLandmark.load_model] Hailo Devices : ",self.devices)
            #[BlazeLandmark.load_model] Hailo Devices :  ['0000:01:00.0']
        
        # Loading compiled HEFs to device:
        self.hef = HEF(model_path)

        # The target is used as a context manager ("with" statement) to ensure it's released on time.
        with VDevice(device_ids=self.devices) as target:
            if self.DEBUG:
                print("[BlazeLandmark.load_model] Hailo target : ",target)
                #[BlazeLandmark.load_model] Hailo target :  <hailo_platform.pyhailort.pyhailort.VDevice object at 0xffff95c62d90>
        
            # Get the "network groups" (connectivity groups, aka. "different networks") information from the .hef
            self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
            if self.DEBUG:
                print("[BlazeLandmark.load_model] Hailo configure_params : ",self.configure_params)
                #[BlazeLandmark.load_model] Hailo configure_params :  {'hand_landmark_lite': <hailo_platform.pyhailort._pyhailort.ConfigureParams object at 0xffff96276870>}
            self.network_groups = target.configure(self.hef, self.configure_params)
            if self.DEBUG:
                print("[BlazeLandmark.load_model] Hailo network_groups : ",self.network_groups)
                #[BlazeLandmark.load_model] Hailo network_groups :  [<hailo_platform.pyhailort.pyhailort.ConfiguredNetwork object at 0xffff8c2e9eb0>]
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()

            # Create input and output virtual streams params
            # Quantized argument signifies whether or not the incoming data is already quantized.
            # Data is quantized by HailoRT if and only if quantized == False .
            #self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            #self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8)
            self.input_vstreams_params = InputVStreamParams.make(self.network_group)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

            # Define dataset params
            self.input_vstream_infos = self.hef.get_input_vstream_infos()
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            if self.DEBUG:
                print("[BlazeLandmark.load_model] Input VStream Infos : ",self.input_vstream_infos)
                print("[BlazeLandmark.load_model] Output VStream Infos : ",self.output_vstream_infos)
                #[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_lite/input_layer1")]
                #[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_lite/fc1"), VStreamInfo("hand_landmark_lite/fc3"), VStreamInfo("hand_landmark_lite/fc2"), VStreamInfo("hand_landmark_lite/fc4")]

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
                #[BlazeDetector.load_model] Number of Inputs :  1
                #[BlazeDetector.load_model] Input[ 0 ] Shape :  (224, 224, 3)
                #[BlazeDetector.load_model] Number of Outputs :  4
                #[BlazeDetector.load_model] Output[ 0 ] Shape :  (63,)
                #[BlazeDetector.load_model] Output[ 1 ] Shape :  (63,)
                #[BlazeDetector.load_model] Output[ 2 ] Shape :  (1,)
                #[BlazeDetector.load_model] Output[ 3 ] Shape :  (1,)

            self.inputShape = self.input_vstream_infos[0].shape
            self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
            self.outputShape2 = tuple(self.output_vstream_infos[1].shape)

            if self.DEBUG:
                print("[BlazeLandmark.load_model] Input Shape : ",self.inputShape)
                print("[BlazeLandmark.load_model] Output1 Shape : ",self.outputShape1)
                print("[BlazeLandmark.load_model] Output2 Shape : ",self.outputShape2)
                #[BlazeLandmark.load_model] Input Shape :  (224, 224, 3)
                #[BlazeLandmark.load_model] Output1 Shape :  (63,)
                #[BlazeLandmark.load_model] Output2 Shape :  (63,)

        self.resolution = self.inputShape[1]
        if self.DEBUG:
            print("[BlazeLandmark.load_model] Input Resolution : ",self.resolution)

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
            xi = np.expand_dims(x[i,:,:,:], axis=0)
            xi = xi * 255.0
            input_data = {self.input_vstream_infos[0].name: xi}
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            """ Execute model on Hailo-8 """
            # The target is used as a context manager ("with" statement) to ensure it's released on time.
            with VDevice(device_ids=self.devices) as target:
            
                # Get the "network groups" (connectivity groups, aka. "different networks") information from the .hef
                self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
                self.network_groups = target.configure(self.hef, self.configure_params)
                self.network_group = self.network_groups[0]
                self.network_group_params = self.network_group.create_params()

                # Create input and output virtual streams params
                # Quantized argument signifies whether or not the incoming data is already quantized.
                # Data is quantized by HailoRT if and only if quantized == False .
                self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False,
                                                                     format_type=FormatType.FLOAT32)
                self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=True,
                                                                       format_type=FormatType.UINT8)
                                                                       #format_type=FormatType.FLOAT32)
             
                with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                    with self.network_group.activate(self.network_group_params):
                        infer_results = infer_pipeline.infer(input_data)
            self.profile_model += timer()-start

            start = timer()  

            if self.blaze_app == "blazehandlandmark":
                out1 = infer_results[self.output_vstream_infos[2].name]
                handedness = infer_results[self.output_vstream_infos[3].name] 
                out2 = infer_results[self.output_vstream_infos[0].name]
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2]
                out2 = out2/self.resolution
            elif self.blaze_app == "blazefacelandmark":
                out1 = infer_results[self.output_vstream_infos[0].name]
                out2 = infer_results[self.output_vstream_infos[1].name]
                out2 = out2.reshape(1,-1,3) # 1404 => [1,356,2]
                out2 = out2/self.resolution                 
            elif self.blaze_app == "blazeposelandmark":
                out1 = infer_results[self.output_vstream_infos[0].name]
                out2 = infer_results[self.output_vstream_infos[1].name]
                out2 = out2.reshape(1,-1,5) # 195 => [1,39,5]
                out2 = out2/self.resolution  
                
            #if self.DEBUG:
            #    print("[BlazeLandmark] out1 : ",out1.shape,out1)
            #    #print("[BlazeLandmark] handedness : ",handedness.shape,handedness)                              
            #    print("[BlazeLandmark] out2 : ",out1.shape,out2)                              

            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        

        #if self.DEBUG:
        #    print("[BlazeLandmark] flag ",flag.shape,flag.dtype)
        #    print("[BlazeLandmark] landmarks ",landmarks.shape,landmarks.dtype)

        return flag,landmarks
