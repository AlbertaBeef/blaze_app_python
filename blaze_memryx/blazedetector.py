import numpy as np
import cv2

from blazebase import BlazeDetectorBase

# References : 
#    https://developer.memryx.com/api/accelerator/python.html
#    https://developer.memryx.com/api/accelerator/python.html#accl.AsyncAccl.inport_assignment
#    https://developer.memryx.com/api/accelerator/python.html#accl.AsyncAccl.outport_assignment
#    https://developer.memryx.com/tutorials/basic_inf/classification_accl.html

from memryx import Dfp,SyncAccl
                           
from timeit import default_timer as timer


class BlazeDetector(BlazeDetectorBase):

    #def __init__(self,blaze_app="blazepalm"):
    def __init__(self,blaze_app,memryx_context):
        super(BlazeDetector, self).__init__()

        self.blaze_app = blaze_app
        self.memryx_context = memryx_context
        
        self.batch_size = 1
        
        

    def load_model(self, model_path):

        self.model_path = model_path
        if self.DEBUG:
            print("[BlazeDetector.load_model] Model File : ",self.model_path)
           
        self.dfp  = Dfp(self.model_path)
        self.accl = SyncAccl(self.model_path)

        if True:

            self.inport_assignment = self.accl.inport_assignment(0)
            self.outport_assignment = self.accl.outport_assignment(0)
            self.input_shapes = self.dfp.input_shapes
            self.output_shapes = self.dfp.output_shapes
            if self.DEBUG:
                print("[BlazeDetector.load_model] inport_assignment : ",self.inport_assignment)
                print("[BlazeDetector.load_model] outport_assignment : ",self.outport_assignment)
                print("[BlazeDetector.load_model] input_shapes : ",self.input_shapes)
                print("[BlazeDetector.load_model] output_shapes : ",self.output_shapes)

            # Get input/output tensors dimensions
            self.num_inputs = len(self.input_shapes)
            self.num_outputs = len(self.output_shapes)
            if self.DEBUG:
                print("[BlazeDetector.load_model] Number of Inputs : ",self.num_inputs)
                for i in range(self.num_inputs):
                    print("[BlazeDetector.load_model] Input[",i,"] Shape : ",tuple(self.input_shapes[i])," Name : ",self.inport_assignment[i])
                print("[BlazeDetector.load_model] Number of Outputs : ",self.num_outputs)
                for i in range(self.num_outputs):
                    print("[BlazeDetector.load_model] Output[",i,"] Shape : ",tuple(self.output_shapes[i])," Name : ",self.outport_assignment[i])

            self.inputShape = self.input_shapes[0]

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

            # face_detection_short_range
            if self.blaze_app == "blazeface" and self.num_outputs == 4:
                self.outputShape1 = tuple((1,896,1))
                self.outputShape2 = tuple((1,896,16))
            
            # face_detection_full_range
            if self.blaze_app == "blazeface" and self.num_outputs == 2:
                self.outputShape1 = tuple((1,2304,1))
                self.outputShape2 = tuple((1,2304,16))

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
        """Change NHWC ordering to HWNC """
        x = np.transpose(x,[1, 2, 0, 3])
        """Converts the image pixels to the range [-1, 1]."""
        x = x.astype(np.float32)
        x = (x / 255.0)
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype(np.float32)
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
        input_data = self.preprocess(x)
        self.profile_pre = timer()-start
                               
        # 2. Run the neural network:
        start = timer()
        """ Execute model on MX3 """
        infer_results = self.accl.run(input_data)
        self.profile_model = timer()-start

        if False: #self.DEBUG:
            #print("[BlazeDetector.predict_on_batch] inference results : ",infer_results)
            for i in range(len(infer_results)):
                print("[BlazeDetector.predict_on_batch] Output[",i,"] Name  : ",self.outport_assignment[i])
                print("[BlazeDetector.predict_on_batch] Output[",i,"] Shape : ",tuple(infer_results[i].shape))
                #[BlazeDetector.predict_on_batch] Output[ 0 ] Name  :  model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1
                #[BlazeDetector.predict_on_batch] Output[ 0 ] Shape :  (12, 12, 6)  Name : 
                #[BlazeDetector.predict_on_batch] Output[ 1 ] Name  :  model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1
                #[BlazeDetector.predict_on_batch] Output[ 1 ] Shape :  (24, 24, 2)  Name : 
                #[BlazeDetector.predict_on_batch] Output[ 2 ] Name  :  model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1
                #[BlazeDetector.predict_on_batch] Output[ 2 ] Shape :  (12, 12, 108)  Name : 
                #[BlazeDetector.predict_on_batch] Output[ 3 ] Name  :  model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1
                #[BlazeDetector.predict_on_batch] Output[ 3 ] Shape :  (24, 24, 36) 

        start = timer()                


        ### palm_detection_v0_07
        # palm_detection_v0_07/conv47 [1x8x8x6]   =reshape=> [1x384x1]  \\
        # palm_detection_v0_07/conv44 [1x16x16x2] =reshape=> [1x512x1]    => [1x2944x1]
        # palm_detection_v0_07/conv47 [1x32x23x2] =reshape=> [1x2048x1]  //
        #
        # palm_detection_v0_07/conv42 [1x8x8x108]  =reshape=> [1x384x18]   \\
        # palm_detection_v0_07/conv45 [1x16x16x36] =reshape=> [1x512x18]    => [1x2944x18]
        # palm_detection_v0_07/conv48 [1x32x32x36] =reshape=> [1x2048x18]  //
        if self.blaze_app == "blazepalm" and self.num_outputs == 6:
            transpose_1_8_8_6 = infer_results[2]
            transport_16_16_2 = infer_results[1]
            transport_32_32_2 = infer_results[0]
            
            reshape_1_384_1 = transpose_1_8_8_6.reshape(1,384,1)
            reshape_1_512_1 = transport_16_16_2.reshape(1,512,1)
            reshape_1_2048_1 = transport_32_32_2.reshape(1,2048,1)

            concat_1_2944_1 = np.concatenate((reshape_1_2048_1,reshape_1_512_1,reshape_1_384_1),axis=1)

            out1 = concat_1_2944_1.astype(np.float32)

            #if self.DEBUG:
            #    print("[BlazeDetector.load_model] Output1 : ",out1.shape,out1.dtype)
            
            transpose_1_8_8_108 = infer_results[5]
            transport_16_16_36 = infer_results[4]
            transport_32_32_36 = infer_results[3]

            reshape_1_384_18 = transpose_1_8_8_108.reshape(1,384,18)
            reshape_1_512_18 = transport_16_16_36.reshape(1,512,18)
            reshape_1_2048_18 = transport_32_32_36.reshape(1,2048,18)
            
            concat_1_2944_18 = np.concatenate((reshape_1_2048_18,reshape_1_512_18,reshape_1_384_18),axis=1)
            
            out2 = concat_1_2944_18.astype(np.float32)

            #if self.DEBUG:
            #    print("[BlazeDetector.load_model] Output2 : ",out2.shape,out2.dtype)
            
        ### palm_detection_lite/full
        # palm_detection_lite/conv29 [1x24x24x2] =reshape=> [1x1152x1] \\
        #                                                                        => [1x2016x1]
        # palm_detection_lite/conv24 [1x12x12x6] =reshape=> [1x864x1]  //
        #
        # palm_detection_lite/conv30 [1x24x24x36]  =reshape=> [1x1152x18] \\
        #                                                                             => [1x2016x18]
        # palm_detection_lite/conv25 [1x12x12x108] =reshape=> [1x864x18]  //
        if self.blaze_app == "blazepalm" and self.num_outputs == 4:
            conv_1_24_24_2 = infer_results[1]
            conv_12_12_6 = infer_results[0]
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] conv_1_24_24_2 Shape : ",tuple(conv_1_24_24_2.shape))
                print("[BlazeDetector.predict_on_batch] conv_12_12_6 Shape : ",tuple(conv_12_12_6.shape))
                #[BlazeDetector.predict_on_batch] conv_1_24_24_2 Shape :  (24, 24, 2)
                #[BlazeDetector.predict_on_batch] conv_12_12_6 Shape :  (12, 12, 6)
            
            reshape_1_1152_1 = conv_1_24_24_2.reshape(1,1152,1)
            reshape_1_864_1 = conv_12_12_6.reshape(1,864,1)
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] reshape_1_1152_1 Shape : ",tuple(reshape_1_1152_1.shape))
                print("[BlazeDetector.predict_on_batch] reshape_1_864_1 Shape : ",tuple(reshape_1_864_1.shape))
                #[BlazeDetector.predict_on_batch] reshape_1_1152_1 Shape :  (1, 1152, 1)
                #[BlazeDetector.predict_on_batch] reshape_1_864_1 Shape :  (1, 864, 1)

            concat_1_2016_1 = np.concatenate((reshape_1_1152_1,reshape_1_864_1),axis=1)
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] concat_1_2016_1 Shape : ",tuple(concat_1_2016_1.shape))
                #[BlazeDetector.predict_on_batch] concat_1_2016_1 Shape :  (1, 2016, 1)

            out1 = concat_1_2016_1.astype(np.float32)

            conv_1_24_24_36 = infer_results[3]
            conv_12_12_108 = infer_results[2]
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] conv_1_24_24_36 Shape : ",tuple(conv_1_24_24_36.shape))
                print("[BlazeDetector.predict_on_batch] conv_12_12_108 Shape : ",tuple(conv_12_12_108.shape))
                #[BlazeDetector.predict_on_batch] conv_1_24_24_36 Shape :  (24, 24, 36)
                #[BlazeDetector.predict_on_batch] conv_12_12_108 Shape :  (12, 12, 108)
            
            reshape_1_1152_18 = conv_1_24_24_36.reshape(1,1152,18)
            reshape_1_864_18 = conv_12_12_108.reshape(1,864,18)
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] reshape_1_1152_18 Shape : ",tuple(reshape_1_1152_18.shape))
                print("[BlazeDetector.predict_on_batch] reshape_1_864_18 Shape : ",tuple(reshape_1_864_18.shape))
                #[BlazeDetector.predict_on_batch] reshape_1_1152_18 Shape :  (1, 1152, 18)
                #[BlazeDetector.predict_on_batch] reshape_1_864_18 Shape :  (1, 864, 18)

            concat_1_2016_18 = np.concatenate((reshape_1_1152_18,reshape_1_864_18),axis=1)
            if False: #self.DEBUG:
                print("[BlazeDetector.predict_on_batch] concat_1_2016_18 Shape : ",tuple(concat_1_2016_18.shape))
                #[BlazeDetector.predict_on_batch] concat_1_2016_18 Shape :  (1, 2016, 18)

            out2 = concat_1_2016_18.astype(np.float32)

        ### face_detection_short_range
        #[BlazeDetector.load_model] Model File :  blaze_hailo/models/face_detection_short_range.hef
        #[BlazeDetector.load_model] HEF Id :  0
        #[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("face_detection_short_range/input_layer1")]
        #[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("face_detection_short_range/conv21"), VStreamInfo("face_detection_short_range/conv14"), VStreamInfo("face_detection_short_range/conv20"), VStreamInfo("face_detection_short_range/conv13")]
        #[BlazeDetector.load_model] Number of Inputs :  1
        #[BlazeDetector.load_model] Input[ 0 ] Shape :  (128, 128, 3)  Name :  face_detection_short_range/input_layer1
        #[BlazeDetector.load_model] Number of Outputs :  4
        #[BlazeDetector.load_model] Output[ 0 ] Shape :  (8, 8, 96)  Name :  face_detection_short_range/conv21
        #[BlazeDetector.load_model] Output[ 1 ] Shape :  (16, 16, 32)  Name :  face_detection_short_range/conv14
        #[BlazeDetector.load_model] Output[ 2 ] Shape :  (8, 8, 6)  Name :  face_detection_short_range/conv20
        #[BlazeDetector.load_model] Output[ 3 ] Shape :  (16, 16, 2)  Name :  face_detection_short_range/conv13
        if self.blaze_app == "blazeface" and self.num_outputs == 4:
            transpose_1_16_16_2 = infer_results[3]
            transport_1_8_8_6 = infer_results[2]
            
            reshape_1_512_1 = transpose_1_16_16_2.reshape(1,512,1)
            reshape_1_384_1 = transport_1_8_8_6.reshape(1,384,1)

            concat_1_896_1 = np.concatenate((reshape_1_512_1,reshape_1_384_1),axis=1)

            out1 = concat_1_896_1.astype(np.float32)

            transpose_1_16_16_32 = infer_results[1]
            transport_8_8_96 = infer_results[0]
            
            reshape_1_512_16 = transpose_1_16_16_32.reshape(1,512,16)
            reshape_1_384_16 = transport_8_8_96.reshape(1,384,16)

            concat_1_896_16 = np.concatenate((reshape_1_512_16,reshape_1_384_16),axis=1)

            out2 = concat_1_896_16.astype(np.float32)

        ### face_detection_full_range
        #[BlazeDetector.load_model] Model File :  blaze_hailo/models/face_detection_full_range.hef
        #[BlazeDetector.load_model] HEF Id :  0
        #[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("face_detection_full_range/input_layer1")]
        #[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("face_detection_full_range/conv49"), VStreamInfo("face_detection_full_range/conv48")]
        #[BlazeDetector.load_model] Number of Inputs :  1
        #[BlazeDetector.load_model] Input[ 0 ] Shape :  (192, 192, 3)  Name :  face_detection_full_range/input_layer1
        #[BlazeDetector.load_model] Number of Outputs :  2
        #[BlazeDetector.load_model] Output[ 0 ] Shape :  (48, 48, 16)  Name :  face_detection_full_range/conv49
        #[BlazeDetector.load_model] Output[ 1 ] Shape :  (48, 48, 1)  Name :  face_detection_full_range/conv48
        if self.blaze_app == "blazeface" and self.num_outputs == 2:
            transpose_1_48_48_1 = infer_results[1]
            transpose_1_48_48_16 = infer_results[0]

            reshape_1_2304_1 = transpose_1_48_48_1.reshape(1,2304,1)
            reshape_1_2304_16 = transpose_1_48_48_16.reshape(1,2304,16)
        
            out1 = reshape_1_2304_1.astype(np.float32)
            out2 = reshape_1_2304_16.astype(np.float32)
           
            
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



