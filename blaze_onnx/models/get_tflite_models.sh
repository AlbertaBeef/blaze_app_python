
# MetalWhale
# reference : https://github.com/metalwhale/hand_tracking

wget https://raw.githubusercontent.com/metalwhale/hand_tracking/master/models/palm_detection_without_custom_op.tflite
# [BlazePalm.load_model] Model File :  ./models/palm_detection_without_custom_op.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazePalm.load_model] Number of Inputs :  1
# [BlazePalm.load_model] Input[ 0 ] Shape :  [  1 256 256   3]  ( input_1 )
# [BlazePalm.load_model] Number of Outputs :  2
# [BlazePalm.load_model] Output[ 0 ] Shape :  [   1 2944   18]  ( regressors/concat )
# [BlazePalm.load_model] Output[ 1 ] Shape :  [   1 2944    1]  ( classificators/concat )
# [BlazePalm.load_model] Num Anchors :  2944
# [BlazePalm.load_model] Anchors Shape :  (2944, 4)
# [BlazePalm.load_model] Min Score Threshold :  0.7

#wget https://raw.githubusercontent.com/metalwhale/hand_tracking/master/models/hand_landmark.tflite
# [BlazeHandLandmark.load_model] Model File :  ./models/hand_landmark.tflite
# [BlazeHandLandmark.load_model] Number of Inputs :  1
# [BlazeHandLandmark.load_model] Input[ 0 ] Shape :  [  1 256 256   3]  ( input_1 )
# [BlazeHandLandmark.load_model] Number of Outputs :  2
# [BlazeHandLandmark.load_model] Output[ 0 ] Shape :  [ 1 42]  ( ld_21_2d )
# [BlazeHandLandmark.load_model] Output[ 1 ] Shape :  [1 1]  ( output_handflag )


# Google MediaPipe v0.07 models
# reference : https://github.com/google/mediapipe/blob/master/docs/solutions/models.md

#wget https://raw.githubusercontent.com/google/mediapipe/v0.6.9/mediapipe/models/palm_detection.tflite -O palm_detection_v0_06.tflite
wget https://raw.githubusercontent.com/google/mediapipe/v0.7.11/mediapipe/models/palm_detection.tflite -O palm_detection_v0_07.tflite
# [BlazeDetector.load_model] Model File :  models/palm_detection_v0_07.tflite
# RuntimeError: Encountered unresolved custom op: Convolution2DTransposeBias.

wget https://raw.githubusercontent.com/google/mediapipe/v0.7.11/mediapipe/models/hand_landmark.tflite -O hand_landmark_v0_07.tflite
# [BlazeLandmark.load_model] Model File :  models/hand_landmark_v0_07.tflite
# [BlazeLandmark.load_model] Number of Inputs :  1
# [BlazeLandmark.load_model] Input[ 0 ] Shape :  [  1 256 256   3]  ( input_1 )
# [BlazeLandmark.load_model] Number of Outputs :  3
# [BlazeLandmark.load_model] Output[ 0 ] Shape :  [ 1 63]  ( ld_21_3d )
# [BlazeLandmark.load_model] Output[ 1 ] Shape :  [1 1]  ( output_handflag )
# [BlazeLandmark.load_model] Output[ 2 ] Shape :  [1 1]  ( output_handedness )

wget https://raw.githubusercontent.com/google/mediapipe/v0.7.11/mediapipe/models/face_detection_front.tflite -O face_detection_front_v0_07.tflite
# [BlazeDetector.load_model] Model File :  models/face_detection_front_v0_07.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazeDetector.load_model] Number of Inputs :  1
# [BlazeDetector.load_model] Input[ 0 ] Shape :  [  1 128 128   3]  ( input )
# [BlazeDetector.load_model] Number of Outputs :  2
# [BlazeDetector.load_model] Output[ 0 ] Shape :  [  1 896  16]  ( regressors )
# [BlazeDetector.load_model] Output[ 1 ] Shape :  [  1 896   1]  ( classificators )
# [BlazeDetector.load_model] Num Anchors :  896
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 128, 'input_size_width': 128, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 128.0, 'y_scale': 128.0, 'h_scale': 128.0, 'w_scale': 128.0, 'min_score_thresh': 0.75, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}

wget https://raw.githubusercontent.com/google/mediapipe/v0.7.11/mediapipe/models/face_detection_back.tflite -O face_detection_back_v0_07.tflite
# [BlazeDetector.load_model] Model File :  models/face_detection_back_v0_07.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazeDetector.load_model] Number of Inputs :  1
# [BlazeDetector.load_model] Input[ 0 ] Shape :  [  1 256 256   3]  ( input )
# [BlazeDetector.load_model] Number of Outputs :  2
# [BlazeDetector.load_model] Output[ 0 ] Shape :  [  1 896  16]  ( regressors )
# [BlazeDetector.load_model] Output[ 1 ] Shape :  [  1 896   1]  ( classificators )
# [BlazeDetector.load_model] Num Anchors :  896
# [BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.15625, 'max_scale': 0.75, 'input_size_height': 256, 'input_size_width': 256, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [16, 32, 32, 32], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
# [BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
# [BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 256.0, 'y_scale': 256.0, 'h_scale': 256.0, 'w_scale': 256.0, 'min_score_thresh': 0.65, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}

wget https://raw.githubusercontent.com/google/mediapipe/v0.7.11/mediapipe/models/face_landmark.tflite -O face_landmark_v0_07.tflite
# [BlazeLandmark.load_model] Model File :  models/face_landmark_v0_07.tflite
# [BlazeLandmark.load_model] Number of Inputs :  1
# [BlazeLandmark.load_model] Input[ 0 ] Shape :  [  1 192 192   3]  ( input_1 )
# [BlazeLandmark.load_model] Number of Outputs :  2
# [BlazeLandmark.load_model] Output[ 0 ] Shape :  [   1    1    1 1404]  ( conv2d_20 )
# [BlazeLandmark.load_model] Output[ 1 ] Shape :  [1 1 1 1]  ( conv2d_30 )


# pose_detection ?
# pose_landmark ?


# Google MediaPipe v0.10 models
# reference : https://github.com/google/mediapipe/blob/master/docs/solutions/models.md

wget https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite
wget https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite
# [BlazePalm.load_model] Model File :  ./models/palm_detection_lite.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazePalm.load_model] Number of Inputs :  1
# [BlazePalm.load_model] Input[ 0 ] Shape :  [  1 192 192   3]  ( input_1 )
# [BlazePalm.load_model] Number of Outputs :  2
# [BlazePalm.load_model] Output[ 0 ] Shape :  [   1 2016   18]  ( Identity )
# [BlazePalm.load_model] Output[ 1 ] Shape :  [   1 2016    1]  ( Identity_1 )
# [BlazePalm.load_model] Num Anchors :  2016
# [BlazePalm.load_model] Anchors Shape :  (2016, 4)
# [BlazePalm.load_model] Min Score Threshold :  0.5


wget https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite
wget https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite
# [BlazeHandLandmark.load_model] Model File :  ./models/hand_landmark_lite.tflite
# [BlazeHandLandmark.load_model] Number of Inputs :  1
# [BlazeHandLandmark.load_model] Input[ 0 ] Shape :  [  1 224 224   3]  ( input_1 )
# [BlazeHandLandmark.load_model] Number of Outputs :  4
# [BlazeHandLandmark.load_model] Output[ 0 ] Shape :  [ 1 63]  ( Identity )
# [BlazeHandLandmark.load_model] Output[ 1 ] Shape :  [1 1]  ( Identity_1 )
# [BlazeHandLandmark.load_model] Output[ 2 ] Shape :  [1 1]  ( Identity_2 )
# [BlazeHandLandmark.load_model] Output[ 3 ] Shape :  [ 1 63]  ( Identity_3 )

wget https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
# [BlazeFace.load_model] Model File :  ./models/face_detection_short_range.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazeFace.load_model] Number of Inputs :  1
# [BlazeFace.load_model] Input[ 0 ] Shape :  [  1 128 128   3]  ( input )
# [BlazeFace.load_model] Number of Outputs :  2
# [BlazeFace.load_model] Output[ 0 ] Shape :  [  1 896  16]  ( regressors )
# [BlazeFace.load_model] Output[ 1 ] Shape :  [  1 896   1]  ( classificators )
# [BlazeFace.load_model] Num Anchors :  896
# [BlazeFace.load_model] Min Score Threshold :  0.75

wget https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite
wget https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite
# [BlazeFace.load_model] Model File :  ./models/face_detection_full_range.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazeFace.load_model] Number of Inputs :  1
# [BlazeFace.load_model] Input[ 0 ] Shape :  [  1 192 192   3]  ( input )
# [BlazeFace.load_model] Number of Outputs :  2
# [BlazeFace.load_model] Output[ 0 ] Shape :  [   1 2304   16]  ( reshaped_regressor_face_4 )
# [BlazeFace.load_model] Output[ 1 ] Shape :  [   1 2304    1]  ( reshaped_classifier_face_4 )
# [BlazeFace.load_model] Num Anchors :  2304
# [BlazeFace.load_model] Min Score Threshold :  0.6

wget https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
# [BlazeFaceLandmark.load_model] Model File :  ./models/face_landmark.tflite
# [BlazeFaceLandmark.load_model] Number of Inputs :  1
# [BlazeFaceLandmark.load_model] Input[ 0 ] Shape :  [  1 192 192   3]  ( input_1 )
# [BlazeFaceLandmark.load_model] Number of Outputs :  2
# [BlazeFaceLandmark.load_model] Output[ 0 ] Shape :  [   1    1    1 1404]  ( conv2d_21 )
# [BlazeFaceLandmark.load_model] Output[ 1 ] Shape :  [1 1 1 1]  ( conv2d_31 )

#wget https://storage.googleapis.com/mediapipe-assets/face_landmark_with_attention.tflite
# [BlazeFaceLandmark.load_model] Model File :  ./face_landmark_with_attention.tflite
# RuntimeError: Encountered unresolved custom op: Landmarks2TransformMatrix.

wget https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite
# [BlazePose.load_model] Model File :  ./models/pose_detection.tflite
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# [BlazePose.load_model] Number of Inputs :  1
# [BlazePose.load_model] Input[ 0 ] Shape :  [  1 224 224   3]  ( input_1 )
# [BlazePose.load_model] Number of Outputs :  2
# [BlazePose.load_model] Output[ 0 ] Shape :  [   1 2254   12]  ( Identity )
# [BlazePose.load_model] Output[ 1 ] Shape :  [   1 2254    1]  ( Identity_1 )
# [BlazePose.load_model] Num Anchors :  2254
# [BlazePose.load_model] Anchors Shape :  (2254, 4)
# [BlazePose.load_model] Min Score Threshold :  0.5

wget https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite
wget https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite
wget https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite
# [BlazePoseLandmark.load_model] Model File :  ./models/pose_landmark_lite.tflite
# [BlazePoseLandmark.load_model] Number of Inputs :  1
# [BlazePoseLandmark.load_model] Input[ 0 ] Shape :  [  1 256 256   3]  ( input_1 )
# [BlazePoseLandmark.load_model] Number of Outputs :  5
# [BlazePoseLandmark.load_model] Output[ 0 ] Shape :  [  1 195]  ( Identity )
# [BlazePoseLandmark.load_model] Output[ 1 ] Shape :  [1 1]  ( Identity_1 )
# [BlazePoseLandmark.load_model] Output[ 2 ] Shape :  [  1 256 256   1]  ( Identity_2 )
# [BlazePoseLandmark.load_model] Output[ 3 ] Shape :  [ 1 64 64 39]  ( Identity_3 )
# [BlazePoseLandmark.load_model] Output[ 4 ] Shape :  [  1 117]  ( Identity_4 )


