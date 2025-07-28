import numpy as np
import torch

from blazebase import BlazeLandmarkBase

from blazehand_landmark import BlazeHandLandmark
from blazeface_landmark import BlazeFaceLandmark
from blazepose_landmark import BlazePoseLandmark

from timeit import default_timer as timer

class BlazeLandmark(BlazeLandmarkBase):
    def __init__(self,blaze_app="blazehandlandmark"):
        super(BlazeLandmark, self).__init__()

        self.blaze_app = blaze_app
        
        if torch.cuda.is_available():
            self.gpu_device = "cuda:0"
            self.gpu_name   = torch.cuda.get_device_name(0)
        else:
            self.gpu_device = "cpu"
            self.gpu_name   = ""
        torch.set_grad_enabled(False)
        if self.DEBUG:
           print("[BlazeLandmark] GPU : ",self.gpu_device,self.gpu_name)

            
    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        if self.blaze_app == "blazehandlandmark":
            self.model = BlazeHandLandmark().to(self.gpu_device)
            self.model.load_weights(model_path)            
            self.resolution = 256
        elif self.blaze_app == "blazefacelandmark":
            self.model = BlazeFaceLandmark().to(self.gpu_device)
            self.model.load_weights(model_path)            
            self.resolution = 192
        elif self.blaze_app == "blazeposelandmark":
            self.model = BlazePoseLandmark().to(self.gpu_device)
            self.model.load_weights(model_path)            
            self.resolution = 256

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Resolution : ",self.resolution)

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
        
        assert x.shape[3] == 3
        assert x.shape[1] == self.resolution
        assert x.shape[2] == self.resolution
        
        start = timer()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
        x = x.to(self.gpu_device)            
        x = self.preprocess(x)
        self.profile_pre = timer()-start

        #if self.DEBUG:
        #   print("[BlazeLandmark.predict] Input Shape : ",x.shape)

        # 2. Run the neural network:
        start = timer()        
        with torch.no_grad():
            out = self.model.__call__(x)
        self.profile_model = timer()-start
        
        #if self.DEBUG:
        #   print("[BlazeLandmark.predict] Number of Outputs : ",len(out))
        #   for i in range(len(out)):
        #       print("[BlazeLandmark.predict] Output[",i,"] shape : ",out[i].shape)

        start = timer()
        
        if self.blaze_app == "blazehandlandmark":
            flag = out[0].cpu().numpy()
            landmarks = out[2].cpu().numpy()
            handedness = out[1].cpu().numpy()
        elif self.blaze_app == "blazefacelandmark":
            flag = out[0].cpu().numpy()
            landmarks = out[1].cpu().numpy()
        elif self.blaze_app == "blazeposelandmark":
            flag = out[0].cpu().numpy()
            landmarks = out[1].cpu().numpy()
        
        #if self.DEBUG:
        #   print("[BlazeLandmark.predict] flag Shape : ",flag.shape)
        #   print("[BlazeLandmark.predict] landmarks Shape : ",landmarks.shape)
          

        #if self.DEBUG:
        #    print("[BlazeLandmark] flag ",flag.shape,flag.dtype)
        #    print("[BlazeLandmark] landmarks ",landmarks.shape,landmarks.dtype)

        self.profile_post = timer()-start

        if self.blaze_app == "blazehandlandmark":        
            return flag,landmarks,handedness
        else:
            return flag,landmarks
        
        return flag,landmarks
