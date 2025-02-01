import numpy as np
import cv2

from blazebase import BlazeLandmarkBase


# References : 
#    https://developer.memryx.com/api/accelerator/python.html
#    https://developer.memryx.com/api/dfp.html

from memryx import Dfp,SyncAccl

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    
    #def __init__(self,blaze_app="blazehandlandmark"):
    def __init__(self,blaze_app,mx3_accl):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app
        self.accl = mx3_accl        

    def load_model(self, model_path):

        model_name,model_id = model_path.split(":")
        self.model_name = model_name
        self.model_id   = int(model_id)
        if self.DEBUG:
            print("[BlazeLandmark.load_model] Model File : ",self.model_name)
            print("[BlazeLandmark.load_model] Model ID   : ",self.model_id)

        self.dfp  = Dfp(self.model_name)
        self.num_models = len(self.dfp.models)
        if self.DEBUG:
            print("[BlazeLandmark.load_model] num_models : ",self.num_models)
            
            print("[BlazeLandmark.load_model] dfp.input_ports : ",self.dfp.input_ports)
            print("[BlazeLandmark.load_model] dfp.output_ports : ",self.dfp.output_ports)
            print("[BlazeLandmark.load_model] dfp.input_shapes : ",self.dfp.input_shapes)
            print("[BlazeLandmark.load_model] dfp.output_shapes : ",self.dfp.output_shapes)


        #self.accl = SyncAccl(self.model_path)


        if True:
            self.inport_assignment = self.accl.inport_assignment(model_idx=self.model_id)
            self.outport_assignment = self.accl.outport_assignment(model_idx=self.model_id)
            self.inport_indices = [*self.inport_assignment]
            self.outport_indices = [*self.outport_assignment]
            if self.DEBUG:
                print("[BlazeLandmark.load_model] self.inport_assignment : ",self.inport_assignment)
                print("[BlazeLandmark.load_model] self.outport_assignment : ",self.outport_assignment)
                print("[BlazeLandmark.load_model] self.inport_indices : ",self.inport_indices)
                print("[BlazeLandmark.load_model] self.outport_indices : ",self.outport_indices)


            self.input_shapes = [self.dfp.input_shapes[i] for i in self.inport_indices]
            self.output_shapes = [self.dfp.output_shapes[i] for i in self.outport_indices]
            if self.DEBUG:
                print("[BlazeLandmark.load_model] self.input_shapes : ",self.input_shapes)
                print("[BlazeLandmark.load_model] self.output_shapes : ",self.output_shapes)

            self.inputShape = self.input_shapes[0]

            if self.blaze_app == "blazehandlandmark":
                if self.inputShape[1] == 224: # hand_landmark_v0_07
                    self.outputShape1 = tuple((1,1))
                    self.outputShape2 = tuple((1,63))
                else: # hand_landmark_lite/hand_landmark_full
                    self.outputShape1 = tuple(self.output_shapes[2])
                    self.outputShape2 = tuple(self.output_shapes[0])
                    #self.outputShape1 = tuple((1,1))
                    #self.outputShape2 = tuple((1,63))

            if self.blaze_app == "blazefacelandmark":
                self.outputShape1 = tuple(self.output_shapes[0])
                self.outputShape2 = tuple(self.output_shapes[1])
                #self.outputShape1 = tuple((1,1))
                #self.outputShape2 = tuple((1,1404))

            if self.blaze_app == "blazeposelandmark":
                self.outputShape1 = tuple(self.output_shapes[0])
                self.outputShape2 = tuple(self.output_shapes[1])
                #self.outputShape1 = tuple((1,1))
                #self.outputShape2 = tuple((1,195))

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
        #if self.DEBUG:
        #    print("[BlazeLandmark.preprocess] x Shape : ",x.shape)

        """Change NHWC ordering to HWNC """
        x = np.transpose(x,[1, 2, 0, 3])
        x = np.array(x).astype(np.float32)

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
        
        #nb_images = x.shape[0]
        """HWNC ordering"""
        nb_images = x.shape[2]
        for i in range(nb_images):

            # 1. Preprocess the images into tensors:
            start = timer()
            #input_data = np.expand_dims(x[i,:,:,:], axis=0)
            """HWNC ordering"""
            input_data = np.expand_dims(x[:,:,i,:], axis=2)
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            """ Execute model on Hailo-8 """
            infer_results = self.accl.run(input_data,model_idx=self.model_id)
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
                out1 = infer_results[1]
                out1 = out1.reshape(1,1)
                handedness = infer_results[2] 
                out2 = infer_results[0]
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
                out1 = infer_results[2]
                handedness = infer_results[3] 
                out2 = infer_results[0]
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
                out1 = infer_results[0]
                out2 = infer_results[1]
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
                out1 = infer_results[1]
                out2 = infer_results[0]
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
