
#[BlazeDetector.load_model] Model File :  models/palm_detection_v0_07.hef
#[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("palm_detection_v0_07/input_layer1")]
#[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("palm_detection_v0_07/conv42"), VStreamInfo("palm_detection_v0_07/conv45"), VStreamInfo("palm_detection_v0_07/conv48"), VStreamInfo("palm_detection_v0_07/conv41"), VStreamInfo("palm_detection_v0_07/conv44"), VStreamInfo("palm_detection_v0_07/conv47")]
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (256, 256, 3)  Name :  palm_detection_v0_07/input_layer1
#[BlazeDetector.load_model] Number of Outputs :  6
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (8, 8, 6)  Name :  palm_detection_v0_07/conv42
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (16, 16, 2)  Name :  palm_detection_v0_07/conv45
#[BlazeDetector.load_model] Output[ 2 ] Shape :  (32, 32, 2)  Name :  palm_detection_v0_07/conv48
#[BlazeDetector.load_model] Output[ 3 ] Shape :  (8, 8, 108)  Name :  palm_detection_v0_07/conv41
#[BlazeDetector.load_model] Output[ 4 ] Shape :  (16, 16, 36)  Name :  palm_detection_v0_07/conv44
#[BlazeDetector.load_model] Output[ 5 ] Shape :  (32, 32, 36)  Name :  palm_detection_v0_07/conv47
#[BlazeDetector.load_model] Input Shape :  (256, 256, 3)
#[BlazeDetector.load_model] conv533 Shape :  (8, 8, 6)
#[BlazeDetector.load_model] conv544 Shape :  (16, 16, 2)
#[BlazeDetector.load_model] conv551 Shape :  (32, 32, 2)
#[BlazeDetector.load_model] conv532 Shape :  (8, 8, 108)
#[BlazeDetector.load_model] conv543 Shape :  (16, 16, 36)
#[BlazeDetector.load_model] conv550 Shape :  (32, 32, 36)
#[BlazeDetector.load_model] Output1 Shape :  (1, 2944, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 2944, 18)
#[BlazeDetector.load_model] Num Anchors :  2944
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 5, 'min_scale': 0.1171875, 'max_scale': 0.75, 'input_size_height': 256, 'input_size_width': 256, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 32, 32, 32], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2944, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2944, 'num_coords': 18, 'score_clipping_thresh': 100.0, 'x_scale': 256.0, 'y_scale': 256.0, 'h_scale': 256.0, 'w_scale': 256.0, 'min_score_thresh': 0.7, 'min_suppression_threshold': 0.3, 'num_keypoints': 7, 'detection2roi_method': 'box', 'kp1': 0, 'kp2': 2, 'theta0': 1.5707963267948966, 'dscale': 2.6, 'dy': -0.5}
#[BlazeLandmark.load_model] Model File :  models/hand_landmark_v0_07.hef
#[HailoRT] [error] CHECK_AS_EXPECTED failed - Failed to create vdevice. there are not enough free devices. requested: 1, found: 0
#[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)
#[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)
#[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)
#[HailoRT] [error] CHECK_EXPECTED failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)


#[BlazeDetector.load_model] Model File :  models/palm_detection_lite.hef
#[BlazeDetector.load_model] Input VStream Infos :  [VStreamInfo("palm_detection_lite/input_layer1")]
#[BlazeDetector.load_model] Output VStream Infos :  [VStreamInfo("palm_detection_lite/conv30"), VStreamInfo("palm_detection_lite/conv25"), VStreamInfo("palm_detection_lite/conv29"), VStreamInfo("palm_detection_lite/conv24")]
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (192, 192, 3)
#[BlazeDetector.load_model] Number of Outputs :  4
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (24, 24, 2)
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (12, 12, 6)
#[BlazeDetector.load_model] Output[ 2 ] Shape :  (24, 24, 36)
#[BlazeDetector.load_model] Output[ 3 ] Shape :  (12, 12, 108)
#[BlazeDetector.load_model] Input Shape :  (192, 192, 3)
#[BlazeDetector.load_model] conv410 Shape :  (24, 24, 2)
#[BlazeDetector.load_model] conv412 Shape :  (12, 12, 6)
#[BlazeDetector.load_model] conv409 Shape :  (24, 24, 36)
#[BlazeDetector.load_model] conv411 Shape :  (12, 12, 108)
#[BlazeDetector.load_model] Output1 Shape :  (1, 2016, 1)
#[BlazeDetector.load_model] Output2 Shape :  (1, 2016, 18)
#[BlazeDetector.load_model] Num Anchors :  2016
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 192, 'input_size_width': 192, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2016, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2016, 'num_coords': 18, 'score_clipping_thresh': 100.0, 'x_scale': 192.0, 'y_scale': 192.0, 'h_scale': 192.0, 'w_scale': 192.0, 'min_score_thresh': 0.5, 'min_suppression_threshold': 0.3, 'num_keypoints': 7, 'detection2roi_method': 'box', 'kp1': 0, 'kp2': 2, 'theta0': 1.5707963267948966, 'dscale': 2.6, 'dy': -0.5}


