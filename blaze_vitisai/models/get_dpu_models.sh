# TODO : get models from github.com/AlbertaBeef/blaze_tutorial

#[BlazeDetector.load_model] Model File :  models/BlazePalm/B512/BlazePalm.xmodel
#[BlazeDetector.load_model] Input Scale :  128  (fixpos= 7 )
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (1, 256, 256, 3)
#[BlazeDetector.load_model] Number of Outputs :  2
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (1, 2944, 1)
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (1, 2944, 18)
#[BlazeDetector.load_model] Num Anchors :  2944
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 5, 'min_scale': 0.1171875, 'max_scale': 0.75, 'input_size_height': 256, 'input_size_width': 256, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 32, 32, 32], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (2944, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 2944, 'num_coords': 18, 'score_clipping_thresh': 100.0, 'x_scale': 256.0, 'y_scale': 256.0, 'h_scale': 256.0, 'w_scale': 256.0, 'min_score_thresh': 0.7, 'min_suppression_threshold': 0.3, 'num_keypoints': 7, 'detection2roi_method': 'box', 'kp1': 0, 'kp2': 2, 'theta0': 1.5707963267948966, 'dscale': 2.6, 'dy': -0.5}


#[BlazeLandmark.load_model] Model File :  models/BlazeHandLandmark/B512/BlazeHandLandmark.xmodel
#[BlazeLandmark.load_model] Input Scale :  128  (fixpos= 7 )
#[BlazeLandmark.load_model] Number of Inputs :  1
#[BlazeLandmark.load_model] Input[ 0 ] Shape :  (1, 256, 256, 3)
#[BlazeLandmark.load_model] Number of Outputs :  3
#[BlazeLandmark.load_model] Output[ 0 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 1 ] Shape :  (1,)
#[BlazeLandmark.load_model] Output[ 2 ] Shape :  (1, 21, 3)


#[BlazeDetector.load_model] Model File :  models/BlazeFace/B512/BlazeFace.xmodel
#[BlazeDetector.load_model] Input Scale :  128  (fixpos= 7 )
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (1, 128, 128, 3)
#[BlazeDetector.load_model] Number of Outputs :  2
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (1, 896, 1)
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (1, 896, 16)
#[BlazeDetector.load_model] Num Anchors :  896
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.1484375, 'max_scale': 0.75, 'input_size_height': 128, 'input_size_width': 128, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [8, 16, 16, 16], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 128.0, 'y_scale': 128.0, 'h_scale': 128.0, 'w_scale': 128.0, 'min_score_thresh': 0.75, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}
#[BlazeLandmark.load_model] Model File :  models/BlazeFaceLandmark/B512/BlazeFaceLandmark.xmodel
#WARNING: Logging before InitGoogleLogging() is written to STDERR
#F0312 16:00:08.870633   991 op_imp.cpp:110] [UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]  lib=libvart_op_imp_prelu.so;error=libvart_op_imp_prelu.so: cannot open shared object file: No such file or directory;op=xir::Op{name = BlazeFaceLandmark__BlazeFaceLandmark_Sequential_backbone1__PReLU_1__ret_9, type = prelu}
#*** Check failure stack trace: ***
#Aborted


#[BlazeDetector.load_model] Model File :  blaze_vitisai/models/BlazeFaceBack/B512/BlazeFaceBack.xmodel
#[BlazeDetector.load_model] Input Scale :  128  (fixpos= 7 )
#[BlazeDetector.load_model] Number of Inputs :  1
#[BlazeDetector.load_model] Input[ 0 ] Shape :  (1, 256, 256, 3)
#[BlazeDetector.load_model] Number of Outputs :  2
#[BlazeDetector.load_model] Output[ 0 ] Shape :  (1, 896, 1)
#[BlazeDetector.load_model] Output[ 1 ] Shape :  (1, 896, 16)
#[BlazeDetector.load_model] Num Anchors :  896
#[BlazeDetectorBase.config_model] Anchor Options :  {'num_layers': 4, 'min_scale': 0.15625, 'max_scale': 0.75, 'input_size_height': 256, 'input_size_width': 256, 'anchor_offset_x': 0.5, 'anchor_offset_y': 0.5, 'strides': [16, 32, 32, 32], 'aspect_ratios': [1.0], 'reduce_boxes_in_lowest_layer': False, 'interpolated_scale_aspect_ratio': 1.0, 'fixed_anchor_size': True}
#[BlazeDetectorBase.config_model] Anchors Shape :  (896, 4)
#[BlazeDetectorBase.config_model] Model Config :  {'num_classes': 1, 'num_anchors': 896, 'num_coords': 16, 'score_clipping_thresh': 100.0, 'x_scale': 256.0, 'y_scale': 256.0, 'h_scale': 256.0, 'w_scale': 256.0, 'min_score_thresh': 0.65, 'min_suppression_threshold': 0.3, 'num_keypoints': 6, 'detection2roi_method': 'box', 'kp1': 1, 'kp2': 0, 'theta0': 0.0, 'dscale': 1.5, 'dy': 0.0}
#[BlazeLandmark.load_model] Model File :  blaze_vitisai/models/BlazeFaceLandmark/B512/BlazeFaceLandmark.xmodel
#WARNING: Logging before InitGoogleLogging() is written to STDERR
#F0312 16:05:40.381615  1093 op_imp.cpp:110] [UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]  lib=libvart_op_imp_prelu.so;error=libvart_op_imp_prelu.so: cannot open shared object file: No such file or directory;op=xir::Op{name = BlazeFaceLandmark__BlazeFaceLandmark_Sequential_backbone1__PReLU_1__ret_9, type = prelu}
#*** Check failure stack trace: ***
#Aborted


#[BlazeLandmark.load_model] Model File :  blaze_vitisai/models/BlazeFaceLandmark/B512/BlazeFaceLandmark.xmodel
#WARNING: Logging before InitGoogleLogging() is written to STDERR
#F0312 16:11:32.505277  1163 op_imp.cpp:110] [UNILOG][FATAL][VAILIB_CPU_RUNNER_OPEN_LIB_ERROR][dlopen can not open lib!]  lib=libvart_op_imp_prelu.so;error=libvart_op_imp_prelu.so: cannot open shared object file: No such file or directory;op=xir::Op{name = BlazeFaceLandmark__BlazeFaceLandmark_Sequential_backbone1__PReLU_1__ret_9, type = prelu}
#*** Check failure stack trace: ***
#Aborted


