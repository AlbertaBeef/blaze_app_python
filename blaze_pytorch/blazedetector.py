import numpy as np
import torch

from blazebase import BlazeDetectorBase

from blazepalm import BlazePalm
from blazeface import BlazeFace
from blazepose import BlazePose

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1
        
        if torch.cuda.is_available():
            self.gpu_device = "cuda:0"
            self.gpu_name   = torch.cuda.get_device_name(0)
        else:
            self.gpu_device = "cpu"
            self.gpu_name   = ""
        torch.set_grad_enabled(False)
        if self.DEBUG:
           print("[BlazeDetector] GPU : ",self.gpu_device,self.gpu_name)


    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeDetector.load_model] Model File : ",model_path)
           
        if self.blaze_app == "blazepalm":
            self.num_anchors = 2944
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.model = BlazePalm().to(self.gpu_device)
            self.model.load_weights(model_path)
            self.model.eval()
        elif self.blaze_app == "blazeface":
            self.num_anchors = 896
            if "back" in model_path:
                self.back_model = True
                self.x_scale = 256.0
                self.y_scale = 256.0
                self.h_scale = 256.0
                self.w_scale = 256.0
            else:
                self.back_model = False
                self.x_scale = 128.0
                self.y_scale = 128.0
                self.h_scale = 128.0
                self.w_scale = 128.0
            self.model = BlazeFace(self.back_model).to(self.gpu_device)
            self.model.load_weights(model_path)
            self.model.eval()
        elif self.blaze_app == "blazepose":
            self.num_anchors = 896 
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.model = BlazePose().to(self.gpu_device)
            self.model.load_weights(model_path)
            self.model.eval()

        if self.DEBUG:
            print("[BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        """Converts the image pixels to defined input scale."""
        x = (x / 255.0)
        x = x.astype(np.float32)
       
        return x

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        
        # Convert img.unsqueeze(0) to NumPy equivalent
        img_expanded = np.expand_dims(img, axis=0)

        # Call the predict_on_batch function
        detections = self.predict_on_batch(img_expanded)

        # Extract the first element from the predictions
        #return predictions[0]        
        if len(detections)>0:
            return np.array(detections)[0]
        else:
            return []


    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """

        self.profile_pre = 0.0
        self.profile_model = 0.0
        self.profile_post = 0.0
        
        assert x.shape[3] == 3
        assert x.shape[1] == self.y_scale
        assert x.shape[2] == self.x_scale

        # 1. Preprocess the images into tensors:
        start = timer()
        x = self.preprocess(x)
        self.profile_pre = timer()-start
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
        x = x.to(self.gpu_device)

        #if self.DEBUG:
        #   print("[BlazeDetector.predict_on_batch] Input Shape : ",x.shape)

        # 2. Run the neural network:
        start = timer()
        with torch.no_grad():
            out = self.model.__call__(x)
        self.profile_model = timer()-start

        #if self.DEBUG:
        #   print("[BlazeDetector.predict_on_batch] Number of Outputs : ",len(out))
        #   for i in range(len(out)):
        #       print("[BlazeDetector.predict_on_batch] Output[",i,"] shape : ",out[i].shape)

        start = timer()
        
        out2 = out[0].cpu().numpy()
        out1 = out[1].cpu().numpy()
        
        #if self.DEBUG:
        #   print("[BlazeDetector.predict_on_batch] Output1 Shape : ",out1.shape)
        #   print("[BlazeDetector.predict_on_batch] Output2 Shape : ",out2.shape)
        
        #if self.DEBUG:
        #    print("[BlazeDetector.predict_on_batch] out1.shape = ",out1.shape)
        #    print("[BlazeDetector.predict_on_batch] out2.shape = ",out2.shape)

        assert out1.shape[0] == 1 # batch
        assert out1.shape[1] == self.num_anchors
        assert out1.shape[2] == 1

        assert out2.shape[0] == 1 # batch
        assert out2.shape[1] == self.num_anchors
        assert out2.shape[2] == self.num_coords

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out2, out1, self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)
                if len(filtered_detections) > 0:
                    normalized_detections = np.array(filtered_detections)[0]

        self.profile_post = timer()-start

        return filtered_detections



