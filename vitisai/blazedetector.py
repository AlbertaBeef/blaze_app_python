import numpy as np

from blazebase import BlazeDetectorBase

import xir
import vitis_ai_library

from timeit import default_timer as timer

class BlazeDetector(BlazeDetectorBase):
    def __init__(self,blaze_app="blazepalm"):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.batch_size = 1
        

    def load_model(self, model_path):

        if self.DEBUG:
           print("[BlazeDetector.load_model] Model File : ",model_path)
           
        # Create graph runner
        self.g = xir.Graph.deserialize(model_path)
        self.runner = vitis_ai_library.GraphRunner.create_graph_runner(self.g)

        self.input_tensor_buffers  = self.runner.get_inputs()
        self.output_tensor_buffers = self.runner.get_outputs()

        # Get input scaling
        self.input_fixpos = self.input_tensor_buffers[0].get_tensor().get_attr("fix_point")
        self.input_scale = 2**self.input_fixpos

        # Get input/output tensors dimensions
        self.num_inputs = len(self.input_tensor_buffers)
        self.num_outputs = len(self.output_tensor_buffers)
        if self.DEBUG:
           print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
           for i in range(self.num_inputs):
               print("[BlazeDetector.load_model] Input[",i,"] Shape : ",tuple(self.input_tensor_buffers[i].get_tensor().dims))
           print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
           for i in range(self.num_outputs):
               print("[BlazeDetector.load_model] Output[",i,"] Shape : ",tuple(self.output_tensor_buffers[i].get_tensor().dims))

        self.inputShape = tuple(self.input_tensor_buffers[0].get_tensor().dims)
        self.outputShape1 = tuple(self.output_tensor_buffers[0].get_tensor().dims)
        self.outputShape2 = tuple(self.output_tensor_buffers[1].get_tensor().dims)
        self.batch_size = self.inputShape[0]
    
        #if self.DEBUG:
        #   print("[BlazeDetector.load_model] Input Shape : ",self.inputShape)
        #   print("[BlazeDetector.load_model] Output1 Shape : ",self.outputShape1)
        #   print("[BlazeDetector.load_model] Output2 Shape : ",self.outputShape2)

        self.x_scale = self.inputShape[1]
        self.y_scale = self.inputShape[2]
        self.h_scale = self.inputShape[1]
        self.w_scale = self.inputShape[2]

        self.num_anchors = self.outputShape2[1]

        if self.DEBUG:
            print("[BlazeDetector.load_model] Num Anchors : ",self.num_anchors)
           
        self.config_model(self.blaze_app)

    def preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        """Converts the image pixels to defined input scale."""
        x = (x / 255.0) * self.input_scale
        x = x.astype(np.int8)

        # Reformat from x of size (256,256,3) to model_x of size (1,256,256,3)
        model_x = []
        model_x.append( x )
        model_x = np.array(model_x)
        
        return model_x

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
        input_data = np.asarray(self.input_tensor_buffers[0])
        input_data[0] = x
        output_size = int(self.input_tensor_buffers[0].get_tensor().get_element_num() / self.batch_size)
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()
        """ Execute model on DPU """
        job_id = self.runner.execute_async(self.input_tensor_buffers, self.output_tensor_buffers)
        self.runner.wait(job_id)
        self.profile_model = timer()-start

        start = timer()                
        output_size=[1,1]
        for i in range(2):
            output_size[i] = int(self.output_tensor_buffers[i].get_tensor().get_element_num() / self.batch_size)

        out1 = np.asarray(self.output_tensor_buffers[0]) #.reshape(-1,1)
        out2 = np.asarray(self.output_tensor_buffers[1]) #.reshape(-1,18)

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



