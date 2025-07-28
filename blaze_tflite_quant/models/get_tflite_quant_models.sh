wget https://avtinc.sharepoint.com/:u:/t/ET-Downloads/EZy_0ELgBfpOnwh8RwTOFz4Bjx7S9KE8DmulIBTq5bcC7Q?e=oxwr9h -O pose_detection_128x128_full_integer_quant.tflite
# [BlazeDetector.load_model] Model File :  models/pose_detection_128x128_full_integer_quant.tflite
# [BlazeDetector.load_model] Number of Inputs :  1
# [BlazeDetector.load_model] Input[ 0 ] Shape :  [  1 128 128   3]  ( input )
# [BlazeDetector.load_model] Number of Outputs :  2
# [BlazeDetector.load_model] Output[ 0 ] Shape :  [  1 896   1]  ( Identity )
# [BlazeDetector.load_model] Output[ 1 ] Shape :  [  1 896  12]  ( Identity_1 )
# [BlazeDetector.load_model] Num Anchors :  896
# [BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 128, 'input_size_width': 128, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
# [BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
# [BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 12, 'score_clipping_thresh': 100.0, 'x_scale': 128.0, 'y_scale': 128.0, 'h_scale': 128.0, 'w_scale': 128.0, 'min_score_thresh': 0.5, 'min_suppression_threshold': 0.3, 'num_keypoints': 4, 'detection2roi_method': 'alignment', 'kp1': 2, 'kp2': 3, 'theta0': 1.5707963267948966, 'dscale': 1.5, 'dy': 0.0}

wget https://avtinc.sharepoint.com/:u:/t/ET-Downloads/ER6plyKdK_VMkpVguTaFtJ8BEbKL0KpOKvMjKI6-RkDBQQ?e=z2W863 -O pose_landmark_full_quant.tflite
# [BlazeLandmark.load_model] Model File :  models/pose_landmark_full_quant.tflite



# references : 
#    https://github.com/PINTO0309/PINTO_model_zoo/tree/main/053_BlazePose
#       https://google.github.io/mediapipe/solutions/models.html#pose
#       https://github.com/PINTO0309/tflite2tensorflow


curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/053_BlazePose/resources.tar.gz" -o resources.tar.gz
curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/053_BlazePose/resources_full.tar.gz" -o resources.tar.gz


