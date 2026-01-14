import numpy as np
import os
import sys

from blazebase import BlazeLandmarkBase

from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)

qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
if not qnn_sdk_root:
    print("Error: QNN_SDK_ROOT environment variable is not set.")
    #sys.exit(1)

qnn_dir = os.path.join(qnn_sdk_root, "lib/aarch64-oe-linux-gcc11.2")

# blazelandmark_qairt class which inherited from the class QNNContext.
class blazelandmark_qairt(QNNContext):
    def Inference(self, input_data):
        output_data = super().Inference([input_data])       
        return output_data

#os.environ["QNN_LOG_LEVEL"] = "FATAL"
#os.environ["QNN_SILENT"] = "1"


from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        # Config AppBuilder environment.
        QNNConfig.Config(qnn_dir, Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)

        # Instance for blazelandmark
        self.interp_landmark = blazelandmark_qairt("blazelandmark_qairt", model_path)

        if "hand_landmark_v0_07" in model_path:
            self.resolution = 256
        if "hand_landmark_full" in model_path:
            self.resolution = 224
        if "hand_landmark_lite" in model_path:
            self.resolution = 224
        if "face_landmark" in model_path:
            self.resolution = 192
        if "pose_landmark_heavy" in model_path:
            self.resolution = 256
        if "pose_landmark_full" in model_path:
            self.resolution = 256
        if "pose_landmark_lite" in model_path:
            self.resolution = 256


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

            start = timer()
            xi = np.expand_dims(x[i,:,:,:], axis=0)
            #print("[BlazeLandmark] xi ",xi.shape,xi.dtype)

            # 1. Preprocess the images into tensors:
            # ...
            self.profile_pre += timer()-start

            # 2. Run the neural network:
            start = timer()  

            # Burst the HTP.
            #PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

            # Run the inference.
            output_data = self.interp_landmark.Inference([xi])

            # Reset the HTP.
            #PerfProfile.RelPerfProfileGlobal()

            if self.DEBUG:
                for i in range(len(output_data)):
                    print(f"[BlazeLandmark.predict] output_data[{i}] = {output_data[i].shape}")
                # [BlazeLandmark.predict] output_data[0] = (63,)
                # [BlazeLandmark.predict] output_data[1] = (1,)
                # [BlazeLandmark.predict] output_data[2] = (1,)
                # [BlazeLandmark.predict] output_data[3] = (63,)

            out1 = output_data[1]
            out2 = output_data[0]
            if self.blaze_app == "blazehandlandmark":
                out3 = output_data[2]

            self.profile_model += timer()-start

            start = timer()

            if self.blaze_app == "blazehandlandmark":
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2] / 63 => [1,21,3]
            elif self.blaze_app == "blazefacelandmark":
                out1 = out1.reshape(1,1)
                out2 = out2.reshape(1,-1,3) # 1404 => [1,356,2]
            elif self.blaze_app == "blazeposelandmark":
                if out2.shape[0] == 124:
                    out2 = out2.reshape(1,-1,4) # v0.07 upper : 124 => [1,31,4]
                else:
                    out2 = out2.reshape(1,-1,5) # v0.10 full  : 195 => [1,39,5]
    
            out2 = out2/self.resolution
                
            if self.DEBUG:
                print("q[BlazeLandmark] Input   : ",x.shape, x.dtype) #, x)
                print("q[BlazeLandmark] Input Min/Max: ",np.amin(x),np.amax(x))
                print("q[BlazeLandmark] Output1 : ",out1.shape, out1.dtype) #, out1)
                print("q[BlazeLandmark] Output1 Min/Max: ",np.amin(out1),np.amax(out1))
                print("q[BlazeLandmark] Output2 : ",out2.shape, out2.dtype) #, out2)
                print("q[BlazeLandmark] Output2 Min/Max: ",np.amin(out2),np.amax(out2))
                if self.blaze_app == "blazehandlandmark":
                    print("q[BlazeLandmark] Output3 : ",out3.shape, out3.dtype) #, out3)
                    print("q[BlazeLandmark] Output3 Min/Max: ",np.amin(out3),np.amax(out3))

            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            if self.blaze_app == "blazehandlandmark":
                out3_list.append(out3.squeeze(0))
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        
        if self.blaze_app == "blazehandlandmark":
            handedness_scores = np.asarray(out3_list)

        if self.DEBUG:
            print("q[BlazeLandmark] flag ",flag.shape,flag.dtype)
            print("q[BlazeLandmark] flag Min/Max: ",np.amin(flag),np.amax(flag))
            print("q[BlazeLandmark] landmarks ",landmarks.shape,landmarks.dtype)
            print("q[BlazeLandmark] landmarks Min/Max: ",np.amin(landmarks),np.amax(landmarks))
            
        if self.blaze_app == "blazehandlandmark":
            return flag,landmarks,handedness_scores
        else:
            return flag,landmarks
