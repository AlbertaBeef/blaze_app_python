import numpy as np
import os
import sys

from blazebase import BlazeDetectorBase

from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)

qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
if not qnn_sdk_root:
    print("Error: QNN_SDK_ROOT environment variable is not set.")
    #sys.exit(1)

qnn_dir = os.path.join(qnn_sdk_root, "lib/aarch64-oe-linux-gcc11.2")

# blazedetector_qairt class which inherited from the class QNNContext.
class blazedetector_qairt(QNNContext):
    def Inference(self, input_data):
        output_data = super().Inference([input_data])            
        return output_data

#os.environ["QNN_LOG_LEVEL"] = "FATAL"
#os.environ["QNN_SILENT"] = "1"


from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1
        

    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeDetector.load_model] Model File : ",model_path)

        # Config AppBuilder environment.
        QNNConfig.Config(qnn_dir, Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)

        # Instance for blazedetector
        self.interp_detector = blazedetector_qairt("blazedetector_qairt", model_path)
        
        print("blazedetector_qairt = ",self.interp_detector)

        # the following model info was obtained via  blaze_tflite_qnn in verbose mode
        if "palm_detection_v0_07" in model_path:
                self.x_scale = 256
                self.num_anchors = 2944 
        if "palm_detection_full" in model_path:
                self.x_scale = 192
                self.num_anchors = 2016
        if "palm_detection_lite" in model_path:
                self.x_scale = 192
                self.num_anchors = 2016
        if "face_detection_short_range" in model_path:
                self.x_scale = 128
                self.num_anchors = 896
        if "face_detection_full_range" in model_path:
                self.x_scale = 192
                self.num_anchors = 2304 
        if "pose_detection" in model_path:
                self.x_scale = 224
                self.num_anchors = 2254

        self.y_scale = self.x_scale
        self.h_scale = self.x_scale
        self.w_scale = self.x_scale
                
        if self.DEBUG:
            print("[BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [0, 1]."""
        x = x.astype(np.float32)
        x = (x / 255.0)
                       
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
        
        out1, out2 = self.predict_core( x )
        
        assert out1.shape[0] == 1 # batch
        assert out1.shape[1] == self.num_anchors
        assert out1.shape[2] == 1

        assert out2.shape[0] == 1 # batch
        assert out2.shape[1] == self.num_anchors
        assert out2.shape[2] == self.num_coords

        start = timer()                

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



    def predict_core(self, x):

        # 1. Preprocess the images into tensors:
        start = timer()
        x = self.preprocess(x)
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()

        # Burst the HTP.
        #PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

        # Run the inference.
        output_data = self.interp_detector.Inference([x])

        # Reset the HTP.
        #PerfProfile.RelPerfProfileGlobal()

        self.profile_model = timer()-start

        if self.DEBUG:
            for i in range(len(output_data)):
                print(f"[BlazeDetector.predict_core] output_data[{i}] = {output_data[i].shape}")
            # [BlazeDetector.predict_core] output_data[0] = (36288,)
            # [BlazeDetector.predict_core] output_data[0] = (2016,)
        
        out1 = output_data[1]
        out2 = output_data[0]

        out1 = out1.reshape(1,self.num_anchors,1)
        out2 = out2.reshape(1,self.num_anchors,-1)
        
        if self.DEBUG:
            print("q[BlazeDetector] Input   : ",x.shape, x.dtype) #, x)
            print("q[BlazeDetector] Input Min/Max: ",np.amin(x),np.amax(x))
            print("q[BlazeDetector] Output1 : ",out1.shape, out1.dtype) #, out1)
            print("q[BlazeDetector] Output1 Min/Max: ",np.amin(out1),np.amax(out1))
            print("q[BlazeDetector] Output2 : ",out2.shape, out2.dtype) #, out2)
            print("q[BlazeDetector] Output2 Min/Max: ",np.amin(out2),np.amax(out2))


        return out1, out2
