import numpy as np
import cv2

from blazebase import BlazeLandmarkBase


# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_4_Inference_Tutorial.html

from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            Device, VDevice)


from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    
    #def __init__(self,blaze_app="blazehandlandmark"):
    def __init__(self,blaze_app,hailo_infer):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app
        self.hailo_infer = hailo_infer        

    def load_model(self, model_path):

        if self.DEBUG:
            print("[BlazeLandmark.load_model] Model File : ",model_path)

        self.hef_id = self.hailo_infer.load_model(model_path)           
        if self.DEBUG:
            print("[BlazeLandmark.load_model] HEF Id : ",self.hef_id)

        self.hef = self.hailo_infer.hef_list[self.hef_id]
        self.network_group = self.hailo_infer.network_group_list[self.hef_id]
        self.network_group_params = self.hailo_infer.network_group_params_list[self.hef_id]
        self.input_vstreams_params = self.hailo_infer.input_vstreams_params_list[self.hef_id]
        self.output_vstreams_params = self.hailo_infer.output_vstreams_params_list[self.hef_id]
 
        if True:
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
                print("[BlazeLandmark.load_model] Number of Inputs : ",self.num_inputs)
                for i in range(self.num_inputs):
                    print("[BlazeLandmark.load_model] Input[",i,"] Shape : ",tuple(self.input_vstream_infos[i].shape))
                print("[BlazeLandmark.load_model] Number of Outputs : ",self.num_outputs)
                for i in range(self.num_outputs):
                    print("[BlazeLandmark.load_model] Output[",i,"] Shape : ",tuple(self.output_vstream_infos[i].shape))

            if self.blaze_app == "blazehandlandmark":
                self.inputShape = self.input_vstream_infos[0].shape
                if self.inputShape[1] == 224: # hand_landmark_v0_07
                    self.outputShape1 = tuple((1,1))
                    self.outputShape2 = tuple((1,63))
                else: # hand_landmark_lite/hand_landmark_full
                    self.outputShape1 = tuple(self.output_vstream_infos[2].shape)
                    self.outputShape2 = tuple(self.output_vstream_infos[0].shape)

            if self.blaze_app == "blazefacelandmark":
                self.inputShape = self.input_vstream_infos[0].shape
                self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
                self.outputShape2 = tuple(self.output_vstream_infos[1].shape)

            if self.blaze_app == "blazeposelandmark":
                self.inputShape = self.input_vstream_infos[0].shape
                self.outputShape1 = tuple(self.output_vstream_infos[0].shape)
                self.outputShape2 = tuple(self.output_vstream_infos[1].shape)

            if self.DEBUG:
                print("[BlazeLandmark.load_model] Input Shape : ",self.inputShape)
                print("[BlazeLandmark.load_model] Output1 Shape : ",self.outputShape1)
                print("[BlazeLandmark.load_model] Output2 Shape : ",self.outputShape2)

        self.resolution = self.inputShape[1]
        if self.DEBUG:
            print("[BlazeLandmark.load_model] Input Resolution : ",self.resolution)

    def preprocess(self, x):
        # image was already pre-processed by extract_roi in blaze_common/blazebase.py
        # format = RGB
        # dtype = float32
        # range = 0.0 - 1.0
        
        # Need to convert back to unsigned 8 bit for hailo implementation
        x = x * 255.0
        x = x.astype(np.uint8)
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
            image_input = np.expand_dims(x[i,:,:,:], axis=0)
            input_data = {self.input_vstream_infos[0].name: image_input}
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            """ Execute model on Hailo-8 """
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
            self.profile_model += timer()-start

            start = timer()  

            if self.blaze_app == "blazehandlandmark" and self.resolution == 256:
                #[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_v0_07.hef
                #[BlazeLandmark.load_model] HEF Id :  0
                #[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_v0_07/input_layer1")]
                #[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_v0_07/conv48"), VStreamInfo("hand_landmark_v0_07/conv47"), VStreamInfo("hand_landmark_v0_07/conv46")]
                #[BlazeLandmark.load_model] Number of Inputs :  1
                #[BlazeLandmark.load_model] Input[ 0 ] Shape :  (256, 256, 3)
                #[BlazeLandmark.load_model] Number of Outputs :  3
                #[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 63)
                #[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1)
                #[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1, 1, 1)
                #[BlazeLandmark.load_model] Input Shape :  (256, 256, 3)
                #[BlazeLandmark.load_model] Output1 Shape :  (1, 1, 1)
                #[BlazeLandmark.load_model] Output2 Shape :  (1, 1, 63)
                #[BlazeLandmark.load_model] Input Resolution :  256
                out1 = infer_results[self.output_vstream_infos[1].name]
                out1 = out1.reshape(1,1)
                handedness = infer_results[self.output_vstream_infos[2].name] 
                out2 = infer_results[self.output_vstream_infos[0].name]
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2] | 63 => [1,21,3]
                out2 = out2/self.resolution
            elif self.blaze_app == "blazehandlandmark" and self.resolution == 224:
                #[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_lite.hef
                #[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_lite/input_layer1")]
                #[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_lite/fc1"), VStreamInfo("hand_landmark_lite/fc4"), VStreamInfo("hand_landmark_lite/fc3"), VStreamInfo("hand_landmark_lite/fc2")]
                #[BlazeLandmark.load_model] Number of Inputs :  1
                #[BlazeLandmark.load_model] Input[ 0 ] Shape :  (224, 224, 3)
                #[BlazeLandmark.load_model] Number of Outputs :  4
                #[BlazeLandmark.load_model] Output[ 0 ] Shape :  (63,)
                #[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1,)
                #[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1,)
                #[BlazeLandmark.load_model] Output[ 3 ] Shape :  (63,)
                out1 = infer_results[self.output_vstream_infos[2].name]
                handedness = infer_results[self.output_vstream_infos[3].name] 
                out2 = infer_results[self.output_vstream_infos[0].name]
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2] | 63 => [1,21,3]
                out2 = out2/self.resolution
            elif self.blaze_app == "blazefacelandmark":
                #[BlazeLandmark.load_model] Model File :  blaze_hailo/models/face_landmark.hef
                #[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("face_landmark/input_layer1")]
                #[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("face_landmark/conv23"), VStreamInfo("face_landmark/conv25")]
                #[BlazeLandmark.load_model] Number of Inputs :  1
                #[BlazeLandmark.load_model] Input[ 0 ] Shape :  (192, 192, 3)
                #[BlazeLandmark.load_model] Number of Outputs :  2
                #[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 1)
                #[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1404)
                #[BlazeLandmark.load_model] Input Shape :  (192, 192, 3)
                out1 = infer_results[self.output_vstream_infos[0].name]
                out2 = infer_results[self.output_vstream_infos[1].name]
                out2 = out2.reshape(1,-1,3) # 1404 => [1,356,3]
                out2 = out2/self.resolution                 
            elif self.blaze_app == "blazeposelandmark":
                #[BlazeLandmark.load_model] Model File :  blaze_hailo/models/pose_landmark_lite.hef
                #[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("pose_landmark_lite/input_layer1")]
                #[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("pose_landmark_lite/conv46"), VStreamInfo("pose_landmark_lite/conv45"), VStreamInfo("pose_landmark_lite/conv54"), VStreamInfo("pose_landmark_lite/conv48"), VStreamInfo("pose_landmark_lite/conv47")]
                #[BlazeLandmark.load_model] Number of Inputs :  1
                #[BlazeLandmark.load_model] Input[ 0 ] Shape :  (256, 256, 3)
                #[BlazeLandmark.load_model] Number of Outputs :  5
                #[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 195)
                #[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1)
                #[BlazeLandmark.load_model] Output[ 2 ] Shape :  (256, 256, 1)
                #[BlazeLandmark.load_model] Output[ 3 ] Shape :  (64, 64, 39)
                #[BlazeLandmark.load_model] Output[ 4 ] Shape :  (1, 1, 117)            
                out1 = infer_results[self.output_vstream_infos[1].name]
                out2 = infer_results[self.output_vstream_infos[0].name]
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
