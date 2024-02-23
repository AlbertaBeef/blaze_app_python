

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

