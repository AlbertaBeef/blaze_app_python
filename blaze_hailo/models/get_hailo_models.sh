# Get models from github.com/AlbertaBeef/blaze_tutorial
wget https://github.com/AlbertaBeef/blaze_app_python/releases/download/version1/blaze_hailo_models.zip

#[BlazeDetector.load_model] Model File :  blaze_hailo/models/palm_detection_v0_07.hef
#[BlazeDetector.load_model] HEF Id :  0
#[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("palm_detection_v0_07/input_layer1")]
#[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("palm_detection_v0_07/conv47"), VStreamInfo("palm_detection_v0_07/conv44"), VStreamInfo("palm_detection_v0_07/conv41"), VStreamInfo("palm_detection_v0_07/conv48"), VStreamInfo("palm_detection_v0_07/conv45"), VStreamInfo("palm_detection_v0_07/conv42")]
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (256, 256, 3)  Name :  palm_detection_v0_07/input_layer1
#[BlazeDetector.load_model] Number of Outputs :  6
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (32, 32, 2)  Name :  palm_detection_v0_07/conv47
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (16, 16, 2)  Name :  palm_detection_v0_07/conv44
#[BlazeDetector.load_model] Output[ 2 ] Shape :  (8, 8, 6)  Name :  palm_detection_v0_07/conv41
#[BlazeDetector.load_model] Output[ 3 ] Shape :  (32, 32, 36)  Name :  palm_detection_v0_07/conv48
#[BlazeDetector.load_model] Output[ 4 ] Shape :  (16, 16, 36)  Name :  palm_detection_v0_07/conv45
#[BlazeDetector.load_model] Output[ 5 ] Shape :  (8, 8, 108)  Name :  palm_detection_v0_07/conv42
#[BlazeDetector.load_model] Input Shape :  (256, 256, 3)
#[BlazeDetector.load_model] Output1 Shape :  (1, 2944, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 2944, 18)
#[BlazeDetector.load_model] Num Anchors :  2944
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 5, 'min_scale': 0.1171875, 'max_scale': 0.75, 'input_size_height': 256, 'input_size_width': 256, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 32, 32, 32], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2944, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2944, 'num_coords': 18, 'score_clipping_thresh': 100.0, 'x_scale': 256.0, 'y_scale': 256.0, 'h_scale': 256.0, 'w_scale': 256.0, 'min_score_thresh': 0.7, 'min_suppression_threshold': 0.3, 'num_keypoints': 7, 'detection2roi_method': 'box', 'kp1': 0, 'kp2': 2, 'theta0': 1.5707963267948966, 'dscale': 2.6, 'dy': -0.5}


#[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_v0_07.hef
#[BlazeLandmark.load_model] HEF Id :  0
#[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_v0_07/input_layer1")]
#[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_v0_07/conv48"), VStreamInfo("hand_landmark_v0_07/conv47"), VStreamInfo("hand_landmark_v0_07/conv46")]
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (256, 256, 3)
#[BlazeLandmark.load_model] Number of Outputs :  3
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 63)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Input Shape :  (256, 256, 3)
#[BlazeLandmark.load_model] Output1 Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Output2 Shape :  (1, 1, 63)
#[BlazeLandmark.load_model] Input Resolution :  256

#[BlazeDetector.load_model] Model File :  blaze_hailo/models/palm_detection_lite.hef
#[BlazeDetector.load_model] HEF Id :  0
#[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("palm_detection_lite/input_layer1")]
#[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("palm_detection_lite/conv24"), VStreamInfo("palm_detection_lite/conv29"), VStreamInfo("palm_detection_lite/conv25"), VStreamInfo("palm_detection_lite/conv30")]
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (192, 192, 3)  Name :  palm_detection_lite/input_layer1
#[BlazeDetector.load_model] Number of Outputs :  4
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (12, 12, 6)  Name :  palm_detection_lite/conv24
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (24, 24, 2)  Name :  palm_detection_lite/conv29
#[BlazeDetector.load_model] Output[ 2 ] Shape :  (12, 12, 108)  Name :  palm_detection_lite/conv25
#[BlazeDetector.load_model] Output[ 3 ] Shape :  (24, 24, 36)  Name :  palm_detection_lite/conv30
#[BlazeDetector.load_model] Input Shape :  (192, 192, 3)
#[BlazeDetector.load_model] Output1 Shape :  (1, 2016, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 2016, 18)
#[BlazeDetector.load_model] Num Anchors :  2016
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 192, 'input_size_width': 192, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2016, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2016, 'num_coords': 18, 'score_clipping_thresh': 100.0, 'x_scale': 192.0, 'y_scale': 192.0, 'h_scale': 192.0, 'w_scale': 192.0, 'min_score_thresh': 0.5, 'min_suppression_threshold': 0.3, 'num_keypoints': 7, 'detection2roi_method': 'box', 'kp1': 0, 'kp2': 2, 'theta0': 1.5707963267948966, 'dscale': 2.6, 'dy': -0.5}

#[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_lite.hef
#[BlazeLandmark.load_model] HEF Id :  1
#[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_lite/input_layer1")]
#[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_lite/fc1"), VStreamInfo("hand_landmark_lite/fc4"), VStreamInfo("hand_landmark_lite/fc3"), VStreamInfo("hand_landmark_lite/fc2")]
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (224, 224, 3)
#[BlazeLandmark.load_model] Number of Outputs :  4
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (63,)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 3 ] Shape :  (63,)
#[BlazeLandmark.load_model] Input Shape :  (224, 224, 3)
#[BlazeLandmark.load_model] Output1 Shape :  (1,)
#[BlazeLandmark.load_model] Output2 Shape :  (63,)
#[BlazeLandmark.load_model] Input Resolution :  224

#[BlazeLandmark.load_model] Model File :  blaze_hailo/models/hand_landmark_full.hef
#[BlazeLandmark.load_model] HEF Id :  1
#[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("hand_landmark_full/input_layer1")]
#[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("hand_landmark_full/fc1"), VStreamInfo("hand_landmark_full/fc4"), VStreamInfo("hand_landmark_full/fc3"), VStreamInfo("hand_landmark_full/fc2")]
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (224, 224, 3)
#[BlazeLandmark.load_model] Number of Outputs :  4
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (63,)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 3 ] Shape :  (63,)
#[BlazeLandmark.load_model] Input Shape :  (224, 224, 3)
#[BlazeLandmark.load_model] Output1 Shape :  (1,)
#[BlazeLandmark.load_model] Output2 Shape :  (63,)
#[BlazeLandmark.load_model] Input Resolution :  224

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
#[BlazeDetector.load_model] Input Shape :  (128, 128, 3)
#[BlazeDetector.load_model] Output1 Shape :  (1, 896, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 896, 16)
#[BlazeDetector.load_model] Num Anchors :  896
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 128, 'input_size_width': 128, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 128.0, 'y_scale': 128.0, 'h_scale': 128.0, 'w_scale': 128.0, 'min_score_thresh': 0.75, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}

#[BlazeDetector.load_model] Model File :  blaze_hailo/models/face_detection_full_range.hef
#[BlazeDetector.load_model] HEF Id :  0
#[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("face_detection_full_range/input_layer1")]
#[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("face_detection_full_range/conv49"), VStreamInfo("face_detection_full_range/conv48")]
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (192, 192, 3)  Name :  face_detection_full_range/input_layer1
#[BlazeDetector.load_model] Number of Outputs :  2
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (48, 48, 16)  Name :  face_detection_full_range/conv49
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (48, 48, 1)  Name :  face_detection_full_range/conv48
#[BlazeDetector.load_model] Input Shape :  (192, 192, 3)
#[BlazeDetector.load_model] Output1 Shape :  (1, 2304, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 2304, 16)
#[BlazeDetector.load_model] Num Anchors :  2304
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 1, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 192, 'input_size_width': 192, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [4], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 0.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2304, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2304, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 192.0, 'y_scale': 192.0, 'h_scale': 192.0, 'w_scale': 192.0, 'min_score_thresh': 0.6, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}

#[BlazeLandmark.load_model] Model File :  blaze_hailo/models/face_landmark.hef
#[BlazeLandmark.load_model] HEF Id :  1
#[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("face_landmark/input_layer1")]
#[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("face_landmark/conv23"), VStreamInfo("face_landmark/conv25")]
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (192, 192, 3)
#[BlazeLandmark.load_model] Number of Outputs :  2
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1404)
#[BlazeLandmark.load_model] Input Shape :  (192, 192, 3)
#[BlazeLandmark.load_model] Output1 Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Output2 Shape :  (1, 1, 1404)
#[BlazeLandmark.load_model] Input Resolution :  192


#[BlazeLandmark.load_model] Model File :  blaze_hailo/models/pose_landmark_lite.hef
#[BlazeLandmark.load_model] HEF Id :  0
#[BlazeLandmark.load_model] Input VStream Infos :  [VStreamInfo("pose_landmark_lite/input_layer1")]
#[BlazeLandmark.load_model] Output VStream Infos :  [VStreamInfo("pose_landmark_lite/conv46"), VStreamInfo("pose_landmark_lite/conv45"), VStreamInfo("pose_landmark_lite/conv54"), VStreamInfo("pose_landmark_lite/conv48"), VStreamInfo("pose_landmark_lite/conv47")]
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (256, 256, 3)
#[BlazeLandmark.load_model] Number of Outputs :  5
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1, 1, 195)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Output[ 2 ] Shape :  (256, 256, 1)
#[BlazeLandmark.load_model] Output[ 3 ] Shape :  (64, 64, 39)
#[BlazeLandmark.load_model] Output[ 4 ] Shape :  (1, 1, 117)
#[BlazeLandmark.load_model] Input Shape :  (256, 256, 3)
#[BlazeLandmark.load_model] Output1 Shape :  (1, 1, 195)
#[BlazeLandmark.load_model] Output2 Shape :  (1, 1, 1)
#[BlazeLandmark.load_model] Input Resolution :  256

