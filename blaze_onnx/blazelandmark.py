import numpy as np

from blazebase import BlazeLandmarkBase

import onnxruntime

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        self.session = onnxruntime.InferenceSession(model_path)        

        # reading onnx model parameters
        self.session_inputs = self.session.get_inputs()
        self.session_outputs = self.session.get_outputs()
        self.num_inputs = len(self.session_inputs)
        self.num_outputs = len(self.session_outputs)
        if self.DEBUG:
           print("[BlazeLandmark.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeLandmark.load_model] Input[",i,"] Shape : ",self.session_inputs[i].shape," (",self.session_inputs[i].name,")")
           print("[BlazeLandmark.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeLandmark.load_model] Output[",i,"] Shape : ",self.session_outputs[i].shape," (",self.session_outputs[i].name,")")
                
        self.resolution = self.session_inputs[0].shape[1]

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
            self.profile_pre += timer()-start

            # 2. Run the neural network:
            start = timer()  
            #self.interp_landmark.invoke()
            input_name = self.session_inputs[0].name
            output_names = [output.name for output in self.session_outputs]
            result = self.session.run(output_names, {input_name: xi})   
            self.profile_model += timer()-start

            start = timer()

            if self.blaze_app == "blazehandlandmark":
                # [BlazeHandLandmark.load_model] Model File :  ./models/hand_landmark_lite.tflite
                # [BlazeHandLandmark.load_model] Number of Inputs :  1
                # [BlazeHandLandmark.load_model] Input[ 0 ] Shape :  [  1 224 224   3]  ( input_1 )
                # [BlazeHandLandmark.load_model] Number of Outputs :  4
                # [BlazeHandLandmark.load_model] Output[ 0 ] Shape :  [ 1 63]  ( Identity ) => out2 (landmarks)
                # [BlazeHandLandmark.load_model] Output[ 1 ] Shape :  [1 1]  ( Identity_1 ) => out1 (confidence)
                # [BlazeHandLandmark.load_model] Output[ 2 ] Shape :  [1 1]  ( Identity_2 ) => out3 (handedness)
                # [BlazeHandLandmark.load_model] Output[ 3 ] Shape :  [ 1 63]  ( Identity_3 ) => tiny hands without offset         
                out1 = result[1]
                out2 = result[0]
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2] / 63 => [1,21,3]
                out2 = out2/self.resolution
                out3 = result[2]
                #out4 = result[3]
            elif self.blaze_app == "blazefacelandmark":
                out1 = result[1]
                out1 = out1.reshape(1,1)
                out2 = result[0]
                out2 = out2.reshape(1,-1,3) # 1404 => [1,356,2]
                out2 = out2/self.resolution            
            elif self.blaze_app == "blazeposelandmark":
                out1 = result[1]
                out2 = result[0]
                out2 = out2.reshape(1,-1,5) # 195 => [1,39,5]
                out2 = out2/self.resolution

            #if self.DEBUG:
            #    print("[blaze_onnx.BlazeLandmark.predict] out1 (condifence)",out1.shape,out1.dtype, out1)
            #    print("[blaze_onnx.BlazeLandmark.predict] out2 (landmarks)",out2.shape,out2.dtype, out2*self.resolution)
            #    if self.blaze_app == "blazehandlandmark":
            #        print("[blaze_onnx.BlazeLandmark.predict] out3 (handedness)",out3.shape,out3.dtype, out3)
            #        #print("[blaze_onnx.BlazeLandmark.predict] out4 (mini hand)",out4.shape,out4.dtype, out4)

            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            if self.blaze_app == "blazehandlandmark":
                out3_list.append(out3.squeeze(0))
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        
        if self.blaze_app == "blazehandlandmark":
            handedness_scores = np.asarray(out3_list)

        #if self.DEBUG:
        #    print("[BlazeLandmark.predict] flag ",flag.shape,flag.dtype)
        #    print("[BlazeLandmark.predict] landmarks ",landmarks.shape,landmarks.dtype)

        if self.blaze_app == "blazehandlandmark":
            return flag,landmarks,handedness_scores
        else:
            return flag,landmarks
