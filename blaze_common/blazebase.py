import cv2
import numpy as np

from matplotlib import pyplot as plt

from blazeconfig import get_model_config, get_anchor_options, generate_anchors

"""
Based on code from :
https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
https://github.com/axinc-ai/ailia-models/blob/master/pose_estimation/blazepose/blazepose_utils.py
"""

def display_shape_type(pre_msg,m_msg,m):
    print(pre_msg,m_msg,"shape=",m.shape,"dtype=",m.dtype)



class BlazeBase():
    """ Base class for media pipe models. """

    def __init__(self):
        self.DEBUG = False
        self.DEBUG_USE_MODEL_REF_OUTPUT = False
        self.model_ref_output1 = ""
        self.model_ref_output2 = ""
        self.DEBUG_DUMP_DATA = False
        
    def set_debug(self, debug=True):
        self.DEBUG = debug    

    def set_model_ref_output(self, model_ref_output1, model_ref_output2):
        self.DEBUG_USE_MODEL_REF_OUTPUT = True
        self.model_ref_output1 = model_ref_output1
        self.model_ref_output2 = model_ref_output2    

    def set_dump_data(self, debug=True):
        self.DEBUG_DUMP_DATA = debug    

    def set_profile(self, profile=True):
        self.PROFILE = profile    



class BlazeLandmarkBase(BlazeBase):
    """ Base class for landmark models. """

    def __init__(self):
        super(BlazeLandmarkBase, self).__init__()
        
    def extract_roi(self, frame, xc, yc, theta, scale):

        # Assuming scale is a NumPy array of size [N]
        scaleN = scale.reshape(-1, 1, 1).astype(np.float32)

        # Define points
        points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)

        # Element-wise multiplication
        points = points * scaleN / 2
        points = points.astype(np.float32)

        R = np.zeros((theta.shape[0],2,2),dtype=np.float32)
        for i in range (theta.shape[0]):
            R[i,:,:] = [[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]

        center = np.column_stack((xc, yc))
        center = np.expand_dims(center, axis=-1)

        points = np.matmul(R, points) + center
        points = points.astype(np.float32)

        res = self.resolution
        points1 = np.array([[0, 0], [0, res-1], [res-1, 0]], dtype=np.float32)

        affines = []
        imgs = []
        for i in range(points.shape[0]):
            pts = points[i,:,:3].T
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (res, res))  # No borderValue in NumPy
            img = img.astype('float32') / 255.0
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affines.append(affine)
        if imgs:
            imgs = np.stack(imgs).astype('float32')
            affines = np.stack(affines).astype('float32')
        else:
            imgs = np.zeros((0, 3, res, res), dtype='float32')
            affines = np.zeros((0, 2, 3), dtype='float32')

        return imgs, affines, points


    def denormalize_landmarks(self, landmarks, affines):
        landmarks[:,:,:2] *= self.resolution
        for i in range(len(landmarks)):
            landmark, affine = landmarks[i], affines[i]
            landmark = (affine[:,:2] @ landmark[:,:2].T + affine[:,2:]).T
            landmarks[i,:,:2] = landmark
        return landmarks



class BlazeDetectorBase(BlazeBase):
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    """

    def __init__(self):
        super(BlazeDetectorBase, self).__init__()
    
    def config_model(self,blaze_app):
        # Get anchor options
        self.anchor_options = get_anchor_options(blaze_app,self.x_scale,self.y_scale,self.num_anchors)
        if self.DEBUG:
           print("[BlazeDetectorBase.config_model] Anchor Options : ",self.anchor_options)
           
        # Generate anchors
        self.anchors = generate_anchors( self.anchor_options )
        if self.DEBUG:
           print("[BlazeDetectorBase.config_model] Anchors Shape : ",self.anchors.shape)

        # Get model config
        self.config = get_model_config(blaze_app,self.x_scale,self.y_scale,self.num_anchors)
        if self.DEBUG:
           print("[BlazeDetectorBase.config_model] Model Config : ",self.config)
           
        # Set model config
        self.num_classes = self.config["num_classes"]
        self.num_anchors = self.config["num_anchors"]
        self.num_coords = self.config["num_coords"]
        self.score_clipping_thresh = self.config["score_clipping_thresh"]
        #self.back_model = ...
        self.x_scale = self.config["x_scale"]
        self.y_scale = self.config["y_scale"]
        self.h_scale = self.config["h_scale"]
        self.w_scale = self.config["w_scale"]
        self.min_score_thresh = self.config["min_score_thresh"]
        self.min_suppression_threshold = self.config["min_suppression_threshold"]
        self.num_keypoints = self.config["num_keypoints"]
        #
        self.detection2roi_method = self.config["detection2roi_method"]
        self.kp1 = self.config["kp1"]
        self.kp2 = self.config["kp2"]
        self.theta0 = self.config["theta0"]
        self.dscale = self.config["dscale"]
        self.dy = self.config["dy"]

    def resize_pad(self,img):
        """ resize and pad images to be input to the detectors

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio.

        Returns:
            img: HxW
            scale: scale factor between original image and 256x256 image
            pad: pixels of padding in the original image
        """

        size0 = img.shape
        if size0[0] >= size0[1]:
            h1 = int(self.h_scale)
            w1 = int(self.w_scale * size0[1] // size0[0])
            padh = 0
            padw = int(self.w_scale - w1)
            scale = size0[1] / w1
        else:
            h1 = int(self.h_scale * size0[0] // size0[1])
            w1 = int(self.w_scale)
            padh = int(self.h_scale - h1)
            padw = 0
            scale = size0[0] / h1
        padh1 = padh//2
        padh2 = padh//2 + padh % 2
        padw1 = padw//2
        padw2 = padw//2 + padw % 2
        img = cv2.resize(img, (w1, h1))
        img = np.pad(img, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
        pad = (int(padh1 * scale), int(padw1 * scale))
        return img, scale, pad

    def denormalize_detections(self, detections, scale, pad):
        """ maps detection coordinates from [0,1] to image coordinates

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio. This function maps the
        normalized coordinates back to the original image coordinates.

        Inputs:
            detections: nxm tensor. n is the number of detections.
                m is 4+2*k where the first 4 valuse are the bounding
                box coordinates and k is the number of additional
                keypoints output by the detector.
            scale: scalar that was used to resize the image
            pad: padding in the x and y dimensions
 
        """
        detections[:, 0] = detections[:, 0] * scale * self.x_scale - pad[0]
        detections[:, 1] = detections[:, 1] * scale * self.x_scale - pad[1]
        detections[:, 2] = detections[:, 2] * scale * self.x_scale - pad[0]
        detections[:, 3] = detections[:, 3] * scale * self.x_scale - pad[1]

        detections[:, 4::2] = detections[:, 4::2] * scale * self.x_scale - pad[1]
        detections[:, 5::2] = detections[:, 5::2] * scale * self.x_scale - pad[0]
        return detections
        
    def detection2roi(self, detection):
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """
        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = np.sqrt(((xc-x1)**2 + (yc-y1)**2)) * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        theta = np.arctan2(y0-y1, x0-x1) - self.theta0

        return xc, yc, scale, theta



    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is an array of shape (b, 896, 12)
        containing the bounding box regressor predictions, as well as an array 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" arrays into proper detections.
        Returns a list of (num_detections, 13) arrays, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """        
        detection_boxes = self._decode_boxes(raw_box_tensor, self.anchors)

        thresh = self.score_clipping_thresh
        clipped_score_tensor = np.clip(raw_score_tensor,-thresh,thresh)
        
        #print("[BlazeBase._tensors_to_detections] raw_score_tensor min|max = ",np.amin(raw_score_tensor),np.amax(raw_score_tensor))        
        #print("[BlazeBase._tensors_to_detections] clipped_score_tensor thresh=",thresh," min|max = ",np.amin(clipped_score_tensor),np.amax(clipped_score_tensor))        

        detection_scores = 1/(1 + np.exp(-clipped_score_tensor))
        detection_scores = np.squeeze(detection_scores, axis=-1)        

        #print("[BlazeBase._tensors_to_detections] detection_scores min|max = ",np.amin(detection_scores),np.amax(detection_scores))        

        self.detection_scores = detection_scores.copy()
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]

            scores = detection_scores[i, mask[i]]
            scores = np.expand_dims(scores,axis=-1) 

            boxes_scores = np.concatenate((boxes,scores),axis=-1)
            output_detections.append(boxes_scores)

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = np.zeros( raw_boxes.shape )

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(self.num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: 
           return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        # argsort() returns ascending order, therefore read the array from end
        remaining = np.argsort(detections[:, self.num_coords])[::-1]    

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                weighted = np.sum(coordinates * scores, axis=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0),
    )
    min_xy = np.maximum(
        np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0),
    )
    inter = np.clip((max_xy - min_xy), 0, None)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = np.repeat(
        np.expand_dims( (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]), 
            axis=1
        ),
        inter.shape[1],
        axis=1
    )  # [A,B]
    area_b = np.repeat(
        np.expand_dims(
            (box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]),
            axis=0
        ),
        inter.shape[0],
        axis=0
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

