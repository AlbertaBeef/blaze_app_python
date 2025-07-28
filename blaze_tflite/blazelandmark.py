import numpy as np

from blazebase import BlazeLandmarkBase

#import tensorflow as tf
bUseTfliteRuntime = False
try:
    import tensorflow as tf
    #import tensorflow.lite
    import tf.contrib
except:
    from tflite_runtime.interpreter import Interpreter
    bUseTfliteRuntime = True

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        if bUseTfliteRuntime:
            self.interp_landmark = Interpreter(model_path)
        else:
            #self.interp_landmark = tf.lite.Interpreter(model_path)
            self.interp_landmark = tf.contrib.lite.Interpreter(model_path)

        self.interp_landmark.allocate_tensors()

        # reading tflite model paramteres
        self.input_details = self.interp_landmark.get_input_details()
        self.output_details = self.interp_landmark.get_output_details()
        self.num_inputs = len(self.input_details)
        self.num_outputs = len(self.output_details)       
        if self.DEBUG:
           print("[BlazeLandmark.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeLandmark.load_model] Input[",i,"] Details : ",self.input_details[i])
               print("[BlazeLandmark.load_model] Input[",i,"] Shape : ",self.input_details[i]['shape']," (",self.input_details[i]['name'],") Quantization : ",self.input_details[i]['quantization'])          
           print("[BlazeLandmark.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeLandmark.load_model] Output[",i,"] Details : ",self.output_details[i])
               print("[BlazeLandmark.load_model] Output[",i,"] Shape : ",self.output_details[i]['shape']," (",self.output_details[i]['name'],") Quantization : ",self.output_details[i]['quantization'])          
                
        self.in_idx = self.input_details[0]['index']
        self.out_landmark_idx = self.output_details[0]['index']
        self.out_flag_idx = self.output_details[1]['index']

        self.in_shape = self.input_details[0]['shape']
        self.out_landmark_shape = self.output_details[0]['shape']
        self.out_flag_shape = self.output_details[1]['shape']
        #if self.DEBUG:
        #   print("[BlazeLandmark.load_model] Input Shape : ",self.in_shape)
        #   print("[BlazeLandmark.load_model] Output1 Shape : ",self.out_landmark_shape)
        #   print("[BlazeLandmark.load_model] Output2 Shape : ",self.out_flag_shape)

        self.resolution = self.in_shape[1]

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
        #out3_list = []

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
            self.interp_landmark.set_tensor(self.in_idx, xi)
            self.profile_pre += timer()-start
                               
            # 2. Run the neural network:
            start = timer()  
            self.interp_landmark.invoke()
            self.profile_model += timer()-start

            start = timer()  

            if self.blaze_app == "blazehandlandmark":
                out1 = np.asarray(self.interp_landmark.get_tensor(self.out_flag_idx))
                out2 = np.asarray(self.interp_landmark.get_tensor(self.out_landmark_idx))
                out2 = out2.reshape(1,21,-1) # 42 => [1,21,2] / 63 => [1,21,3]
                out2 = out2/self.resolution
                #out3 = np.zeros(out1.shape,out1.dtype) # tflite model not returning handedness
            elif self.blaze_app == "blazefacelandmark":
                out1 = np.asarray(self.interp_landmark.get_tensor(self.out_flag_idx))
                out1 = out1.reshape(1,1)
                out2 = np.asarray(self.interp_landmark.get_tensor(self.out_landmark_idx))
                out2 = out2.reshape(1,-1,3) # 1404 => [1,356,2]
                out2 = out2/self.resolution            
            elif self.blaze_app == "blazeposelandmark":
                out1 = np.asarray(self.interp_landmark.get_tensor(self.out_flag_idx))
                out2 = np.asarray(self.interp_landmark.get_tensor(self.out_landmark_idx))
                if out2.shape[1] == 124:
                    out2 = out2.reshape(1,-1,4) # v0.07 upper : 124 => [1,31,4]
                else:
                    out2 = out2.reshape(1,-1,5) # v0.10 full  : 195 => [1,39,5]
                out2 = out2/self.resolution
                #out3 = np.asarray(self.interp_poselandmark.get_tensor(self.out_seg_idx))


            out1_list.append(out1.squeeze(0))
            out2_list.append(out2.squeeze(0))
            #out3_list.append(out3.squeeze(0))
            self.profile_post += timer()-start


        flag = np.asarray(out1_list)
        landmarks = np.asarray(out2_list)        

        if self.DEBUG:
            print("[BlazeLandmark] flag ",flag.shape,flag.dtype)
            print("[BlazeLandmark] flag Min/Max: ",np.amin(flag),np.amax(flag))
            print("[BlazeLandmark] landmarks ",landmarks.shape,landmarks.dtype)
            print("[BlazeLandmark] landmarks Min/Max: ",np.amin(landmarks),np.amax(landmarks))
        return flag,landmarks
