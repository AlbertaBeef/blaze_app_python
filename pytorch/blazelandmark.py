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
        
            
    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Model File : ",model_path)
           
        if self.blaze_app == "blazehandlandmark":
            self.model = BlazeHandLandmark()
            self.model.load_weights(model_path)            
            self.resolution = 256
        elif self.blaze_app == "blazefacelandmark":
            self.model = BlazeFaceLandmark()
            self.model.load_weights(model_path)            
            self.resolution = 192
        elif self.blaze_app == "blazeposelandmark":
            self.model = BlazePoseLandmark()
            self.model.load_weights(model_path)            
            self.resolution = 256

        if self.DEBUG:
           print("[BlazeLandmark.load_model] Resolution : ",self.resolution)

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
        
        return flag,landmarks
