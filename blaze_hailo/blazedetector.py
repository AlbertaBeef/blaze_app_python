import numpy as np
import cv2

from blazebase import BlazeDetectorBase

# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_4_Inference_Tutorial.html

from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, InputVStreams, OutputVStreamParams, OutputVStreams,
                            Device, VDevice)

                           
from timeit import default_timer as timer


class BlazeDetector(BlazeDetectorBase):

    #def __init__(self,blaze_app="blazepalm"):
    def __init__(self,blaze_app,hailo_infer):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.hailo_infer = hailo_infer
        
        self.batch_size = 1
        
        

    def load_model(self, model_path):

        if self.DEBUG:
            print("[BlazeDetector.load_model] Model File : ",model_path)
            #[BlazeDetector.load_model] Model File :  blaze_hailo/models/palm_detection_lite.hef
           
        self.hef_id = self.hailo_infer.load_model(model_path)           
        if self.DEBUG:
            print("[BlazeDetector.load_model] HEF Id : ",self.hef_id)

        self.hef = self.hailo_infer.hef_list[self.hef_id]
        self.network_group = self.hailo_infer.network_group_list[self.hef_id]
        self.network_group_params = self.hailo_infer.network_group_params_list[self.hef_id]
        self.input_vstreams_params = self.hailo_infer.input_vstreams_params_list[self.hef_id]
        self.output_vstreams_params = self.hailo_infer.output_vstreams_params_list[self.hef_id]
         
        if True:
            # Define dataset params
            self.input_vstream_infos = self.hef.get_input_vstream_infos()
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            if self.DEBUG:
                print("[BlazeDetector.load_model] Input VStream Infos : ",self.input_vstream_infos)
                print("[BlazeDetector.load_model] Output VStream Infos : ",self.output_vstream_infos)
                #[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("palm_detection_lite/input_layer1")]
                #[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("palm_detection_lite/conv24"), VStreamInfo("palm_detection_lite/conv29"), VStreamInfo("palm_detection_lite/conv25"), VStreamInfo("palm_detection_lite/conv30")]
        
            # Get input/output tensors dimensions
            self.num_inputs = len(self.input_vstream_infos)
            self.num_outputs = len(self.output_vstream_infos)
            if self.DEBUG:
                print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
                for i in range(self.num_inputs):
                    print("[BlazeDetector.load_model] Input[",i,"] Shape : ",tuple(self.input_vstream_infos[i].shape)," Name : ",self.input_vstream_infos[i].name)
                print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
                for i in range(self.num_outputs):
                    print("[BlazeDetector.load_model] Output[",i,"] Shape : ",tuple(self.output_vstream_infos[i].shape)," Name : ",self.output_vstream_infos[i].name)
                #[BlazeDetector.load_model] Number of Inputs :  1
                #[BlazeDetector.load_model] Input[ 0 ] Shape :  (192, 192, 3)  Name :  palm_detection_lite/input_layer1
                #[BlazeDetector.load_model] Number of Outputs :  4
                #[BlazeDetector.load_model] Output[ 0 ] Shape :  (12, 12, 6)  Name :  palm_detection_lite/conv24
                #[BlazeDetector.load_model] Output[ 1 ] Shape :  (24, 24, 2)  Name :  palm_detection_lite/conv29
                #[BlazeDetector.load_model] Output[ 2 ] Shape :  (12, 12, 108)  Name :  palm_detection_lite/conv25
                #[BlazeDetector.load_model] Output[ 3 ] Shape :  (24, 24, 36)  Name :  palm_detection_lite/conv30
        
            self.inputShape = tuple(self.input_vstream_infos[0].shape)

            ### palm_detection_v0_07
            # Conv__533 [1x6x8x8]   =transpose=> [1x8x8x6]   =reshape=> [1x384x1]  \\
            # Conv__544 [1x2x16x16] =transpose=> [1x16x16x2] =reshape=> [1x512x1]    => [1x2944x1]
            # Conv__551 [1x2x32x32] =transpose=> [1x32x23x2] =reshape=> [1x2048x1]  //
            #
            # Conv__532 [1x108x8x8]  =transpose=> [1x8x8x108]  =reshape=> [1x384x18]   \\
            # Conv__543 [1x36x16x16] =transpose=> [1x16x16x36] =reshape=> [1x512x18]    => [1x2944x18]
            # Conv__550 [1x36x32x32] =transpose=> [1x32x32x36] =reshape=> [1x2048x1]  //
            if self.blaze_app == "blazepalm" and self.num_outputs == 6:

                self.outputShape1 = tuple((1,2944,1))
                self.outputShape2 = tuple((1,2944,18))
            
                if self.DEBUG:
                    print("[BlazeDetector.load_model] Input Shape : ",self.inputShape)
                    print("[BlazeDetector.load_model] Output1 Shape : ",self.outputShape1)
                    print("[BlazeDetector.load_model] Output2 Shape : ",self.outputShape2)

            ### palm_detection_lite/full
            # Conv__410 [1x2x24x24] =transpose=> [1x24x24x2] =reshape=> [1x1152x1] \\
            #                                                                        => [1x2016x1]
            # Conv__412 [1x6x12x12] =transpose=> [1x12x12x6] =reshape=> [1x864x1]  //
            #
            # Conv__409 [1x36x24x24]  =transpose=> [1x24x24x36]  =reshape=> [1x1152x18] \\
            #                                                                             => [1x2016x18]
            # Conv__411 [1x108x12x12] =transpose=> [1x12x12x108] =reshape=> [1x864x18]  //
            if self.blaze_app == "blazepalm" and self.num_outputs == 4:

                self.outputShape1 = tuple((1,2016,1))
                self.outputShape2 = tuple((1,2016,18))
            
                if self.DEBUG:
                    print("[BlazeDetector.load_model] Input Shape : ",self.inputShape)
                    print("[BlazeDetector.load_model] Output1 Shape : ",self.outputShape1)
                    print("[BlazeDetector.load_model] Output2 Shape : ",self.outputShape2)
            
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
        #x = (x / 255.0)
        #x = x.astype(np.float32)
        
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.uint8)

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
        input_data = {self.input_vstream_infos[0].name: x}
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()
        """ Execute model on Hailo-8 """
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
        self.profile_model = timer()-start

        #if self.DEBUG:
        #    print("[BlazeDetector] infer_results : ",infer_results)

        start = timer()                

        #out1 = np.asarray(self.output_tensor_buffers[0]) #.reshape(-1,1)
        #out2 = np.asarray(self.output_tensor_buffers[1]) #.reshape(-1,18)

        ### palm_detection_v0_07
        # Conv__533 [1x6x8x8]   =transpose=> [1x8x8x6]   =reshape=> [1x384x1]  \\
        # Conv__544 [1x2x16x16] =transpose=> [1x16x16x2] =reshape=> [1x512x1]    => [1x2944x1]
        # Conv__551 [1x2x32x32] =transpose=> [1x32x23x2] =reshape=> [1x2048x1]  //
        #
        # Conv__532 [1x108x8x8]  =transpose=> [1x8x8x108]  =reshape=> [1x384x18]   \\
        # Conv__543 [1x36x16x16] =transpose=> [1x16x16x36] =reshape=> [1x512x18]    => [1x2944x18]
        # Conv__550 [1x36x32x32] =transpose=> [1x32x32x36] =reshape=> [1x2048x18]  //
        if self.blaze_app == "blazepalm" and self.num_outputs == 6:
            conv_1_6_8_8 = infer_results[self.output_vstream_infos[0].name]
            conv_1_2_16_16 = infer_results[self.output_vstream_infos[1].name]
            conv_1_2_32_32 = infer_results[self.output_vstream_infos[2].name]
            
            transpose_1_8_8_6 = np.transpose(conv_1_6_8_8,[0,2,3,1])
            transport_16_16_2 = np.transpose(conv_1_2_16_16,[0,2,3,1])
            transport_32_32_2 = np.transpose(conv_1_2_32_32,[0,2,3,1])
            
            reshape_1_384_1 = transpose_1_8_8_6.reshape(1,384,1)
            reshape_1_512_1 = transport_16_16_2.reshape(1,512,1)
            reshape_1_2048_1 = transport_32_32_2.reshape(1,2048,1)

            concat_1_2944_1 = np.concatenate((reshape_1_2048_1,reshape_1_512_1,reshape_1_384_1),axis=1)

            out1 = concat_1_2944_1.astype(np.float32)

            if self.DEBUG:
                print("[BlazeDetector.load_model] Output1 : ",out1.shape,out1.dtype)
            
            conv_1_108_8_8 = infer_results[self.output_vstream_infos[3].name]
            conv_1_36_16_16 = infer_results[self.output_vstream_infos[4].name]
            conv_1_36_32_32 = infer_results[self.output_vstream_infos[5].name]

            transpose_1_8_8_108 = np.transpose(conv_1_108_8_8,[0,2,3,1])
            transport_16_16_36 = np.transpose(conv_1_36_16_16,[0,2,3,1])
            transport_32_32_36 = np.transpose(conv_1_36_32_32,[0,2,3,1])

            reshape_1_384_18 = transpose_1_8_8_108.reshape(1,384,18)
            reshape_1_512_18 = transport_16_16_36.reshape(1,512,18)
            reshape_1_2048_18 = transport_32_32_36.reshape(1,2048,18)
            
            concat_1_2944_18 = np.concatenate((reshape_1_2048_18,reshape_1_512_18,reshape_1_384_18),axis=1)
            
            out2 = concat_1_2944_18.astype(np.float32)

            if self.DEBUG:
                print("[BlazeDetector.load_model] Output2 : ",out2.shape,out2.dtype)
            
        ### palm_detection_lite/full
        # Conv__410 [1x2x24x24] =transpose=> [1x24x24x2] =reshape=> [1x1152x1] \\
        #                                                                        => [1x2016x1]
        # Conv__412 [1x6x12x12] =transpose=> [1x12x12x6] =reshape=> [1x864x1]  //
        #
        # Conv__409 [1x36x24x24]  =transpose=> [1x24x24x36]  =reshape=> [1x1152x18] \\
        #                                                                             => [1x2016x18]
        # Conv__411 [1x108x12x12] =transpose=> [1x12x12x108] =reshape=> [1x864x18]  //
        if self.blaze_app == "blazepalm" and self.num_outputs == 4:
            transpose_1_24_24_2 = infer_results[self.output_vstream_infos[1].name]
            transport_12_12_6 = infer_results[self.output_vstream_infos[0].name]
            
            reshape_1_1152_1 = transpose_1_24_24_2.reshape(1,1152,1)
            reshape_1_864_1 = transport_12_12_6.reshape(1,864,1)

            concat_1_2016_1 = np.concatenate((reshape_1_1152_1,reshape_1_864_1),axis=1)

            out1 = concat_1_2016_1.astype(np.float32)

            transpose_1_24_24_36 = infer_results[self.output_vstream_infos[3].name]
            transport_12_12_108 = infer_results[self.output_vstream_infos[2].name]
            
            reshape_1_1152_18 = transpose_1_24_24_36.reshape(1,1152,18)
            reshape_1_864_18 = transport_12_12_108.reshape(1,864,18)

            concat_1_2016_18 = np.concatenate((reshape_1_1152_18,reshape_1_864_18),axis=1)

            out2 = concat_1_2016_18.astype(np.float32)
            
        #if self.DEBUG:
        #    print("[BlazeDetector.load_model] Input   : ",x.shape, x.dtype, x)
        #    print("[BlazeDetector.load_model] Input Min/Max: ",np.amin(x),np.amax(x))
        #    print("[BlazeDetector.load_model] Output1 : ",out1.shape, out1.dtype, out1)
        #    print("[BlazeDetector.load_model] Output1 Min/Max: ",np.amin(out1),np.amax(out1))
        #    print("[BlazeDetector.load_model] Output2 : ",out2.shape, out2.dtype, out2)
        #    print("[BlazeDetector.load_model] Output2 Min/Max: ",np.amin(out2),np.amax(out2))

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



