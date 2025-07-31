import numpy as np

from blazebase import BlazeDetectorBase

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

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1
        

    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeDetector.load_model] Model File : ",model_path)
           
        if bUseTfliteRuntime:
            self.interp_detector = Interpreter(model_path)
        else:
            #self.interp_detector = tf.lite.Interpreter(model_path)
            self.interp_detector = tf.contrib.lite.Interpreter(model_path)
        self.interp_detector.allocate_tensors()

        # reading tflite model paramteres
        self.input_details = self.interp_detector.get_input_details()
        self.output_details = self.interp_detector.get_output_details()
        self.num_inputs = len(self.input_details)
        self.num_outputs = len(self.output_details)
        if self.DEBUG:
           print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeDetector.load_model] Input[",i,"] Details : ",self.input_details[i])
               print("[BlazeDetector.load_model] Input[",i,"] Shape : ",self.input_details[i]['shape']," (",self.input_details[i]['name'],") Quantization : ",self.input_details[i]['quantization'])
           print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeDetector.load_model] Output[",i,"] Details : ",self.output_details[i])
               print("[BlazeDetector.load_model] Output[",i,"] Shape : ",self.output_details[i]['shape']," (",self.output_details[i]['name'],") Quantization : ",self.output_details[i]['quantization'])

        self.in_idx = self.input_details[0]['index']
        self.out_reg_idx = self.output_details[0]['index']
        self.out_clf_idx = self.output_details[1]['index']

        self.in_shape = self.input_details[0]['shape']
        self.out_reg_shape = self.output_details[0]['shape']
        self.out_clf_shape = self.output_details[1]['shape']
        #if self.DEBUG:
        #   print("[BlazeDetector.load_model] Input Shape : ",self.in_shape)
        #   print("[BlazeDetector.load_model] Output1 Shape : ",self.out_reg_shape)
        #   print("[BlazeDetector.load_model] Output2 Shape : ",self.out_clf_shape)

        self.x_scale = self.in_shape[1]
        self.y_scale = self.in_shape[2]
        self.h_scale = self.in_shape[1]
        self.w_scale = self.in_shape[2]

        self.num_anchors = self.out_clf_shape[1]
        if self.DEBUG:
            print("[BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        """Converts the image pixels to defined input scale."""
        x = x.astype(np.float32)
        x = (x / 255.0)
        #print("[BlazeDetector.preprocess] min|max = ",np.amin(x),np.amax(x))
       
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
        
        out1, out2 = self.predict_core(x)

        #print("[BlazeDetector.predict_on_batch] out1 min|max = ",np.amin(out1),np.amax(out1))
        #print("[BlazeDetector.predict_on_batch] out2 min|max = ",np.amin(out2),np.amax(out2))

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
        self.interp_detector.set_tensor(self.in_idx, x)
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()
        self.interp_detector.invoke()
        self.profile_model = timer()-start

        """
        out_clf shape is [number of anchors]
        it is the classification score if there is a hand for each anchor box
        """
        out1 = self.interp_detector.get_tensor(self.out_clf_idx)
        """
        out_reg shape is [number of anchors, 18]
        Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
        Second dimension 4 - 18 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        """
        out2 = self.interp_detector.get_tensor(self.out_reg_idx)

        if self.DEBUG:
            print("[BlazeDetector] Input   : ",x.shape, x.dtype) #, x)
            print("[BlazeDetector] Input Min/Max: ",np.amin(x),np.amax(x))
            print("[BlazeDetector] Output1 : ",out1.shape, out1.dtype) #, out1)
            print("[BlazeDetector] Output1 Min/Max: ",np.amin(out1),np.amax(out1))
            print("[BlazeDetector] Output2 : ",out2.shape, out2.dtype) #, out2)
            print("[BlazeDetector] Output2 Min/Max: ",np.amin(out2),np.amax(out2))

        return out1, out2
