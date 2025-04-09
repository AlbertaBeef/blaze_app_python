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

