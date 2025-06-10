import numpy as np

# Anchor options
# reference : https://github.com/hollance/BlazeFace-PyTorch/blob/master/Anchors.ipynb


# reference : https://github.com/google/mediapipe/blob/v0.6.9/mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt
palm_detect_v0_06_anchor_options = {
    "num_layers": 5,
    "min_scale": 0.1171875,
    "max_scale": 0.75,
    "input_size_height": 256,
    "input_size_width": 256,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
palm_detect_v0_06_model_config = {    
    "num_classes": 1,
    "num_anchors": 2944,
    "num_coords": 18,
    "score_clipping_thresh": 100.0,
    "x_scale": 256.0,
    "y_scale": 256.0,
    "h_scale": 256.0,
    "w_scale": 256.0,
    "min_score_thresh": 0.7,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 7,

    "detection2roi_method": 'box',
    "kp1": 0,
    "kp2": 2,
    "theta0": np.pi/2,
    "dscale": 2.6,
    "dy": -0.5,
}

# reference : https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/palm_detection/palm_detection_gpu.pbtxt
palm_detect_v0_10_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
palm_detect_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2016,
    "num_coords": 18,
    "score_clipping_thresh": 100.0,
    "x_scale": 192.0,
    "y_scale": 192.0,
    "h_scale": 192.0,
    "w_scale": 192.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 7,

    "detection2roi_method": 'box',
    "kp1": 0,
    "kp2": 2,
    "theta0": np.pi/2,
    "dscale": 2.6,
    "dy": -0.5,
}

# reference : https://github.com/google/mediapipe/blob/v0.6.9/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
face_front_v0_06_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
face_front_v0_06_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.75,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,

    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}
# reference : https://github.com/google/mediapipe/blob/v0.7.12/mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
face_back_v0_07_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.15625,
    "max_scale": 0.75,
    "input_size_height": 256,
    "input_size_width": 256,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
face_back_v0_07_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 256.0,
    "y_scale": 256.0,
    "h_scale": 256.0,
    "w_scale": 256.0,
    "min_score_thresh": 0.65,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,

    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

# reference : https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_detection/face_detection_short_range.pbtxt
#             https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_detection/face_detection.pbtxt
face_short_range_v0_10_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
face_short_range_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,

    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

# reference : https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_detection/face_detection_full_range.pbtxt
#             https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/face_detection/face_detection.pbtxt
face_full_range_v0_10_anchor_options = {
    "num_layers": 1,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [4],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 0.0,
    "fixed_anchor_size": True,
}
face_full_range_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2304,
    "num_coords": 16,
    "score_clipping_thresh": 100.0,
    "x_scale": 192.0,
    "y_scale": 192.0,
    "h_scale": 192.0,
    "w_scale": 192.0,
    "min_score_thresh": 0.6,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 6,

    "detection2roi_method": 'box',
    "kp1": 1,
    "kp2": 0,
    "theta0": 0.,
    "dscale": 1.5,
    "dy": 0.,
}

# reference : https://github.com/google/mediapipe/blob/v0.7.12/mediapipe/modules/pose_detection/pose_detection_gpu.pbtxt
pose_detect_v0_07_anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
pose_detect_v0_07_model_config = {    
    "num_classes": 1,
    "num_anchors": 896,
    "num_coords": 12,
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 4,

    "detection2roi_method": 'alignment',
    "kp1": 2,
    "kp2": 3,
    "theta0": 90 * np.pi / 180,
    "dscale": 1.5,
    "dy": 0.,
}

# reference : https://github.com/google/mediapipe/blob/v0.10.9/mediapipe/modules/pose_detection/pose_detection_gpu.pbtxt
pose_detect_v0_10_anchor_options = {
    "num_layers": 5,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 224,
    "input_size_width": 224,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}
pose_detect_v0_10_model_config = {    
    "num_classes": 1,
    "num_anchors": 2254,
    "num_coords": 12,
    "score_clipping_thresh": 100.0,
    "x_scale": 224.0,
    "y_scale": 224.0,
    "h_scale": 224.0,
    "w_scale": 224.0,
    "min_score_thresh": 0.5,
    "min_suppression_threshold": 0.3,
    "num_keypoints": 4,

    "detection2roi_method": 'alignment',
    "kp1": 2,
    "kp2": 3,
    "theta0": 90 * np.pi / 180,
    "dscale": 2.5,
    "dy": 0.,
}


def get_model_config( model_type, input_width, input_height, num_anchors ):
    if model_type == "blazepalm":
        if num_anchors == 2944 and input_width == 256:
            model_config = palm_detect_v0_06_model_config
        elif num_anchors == 2016 and input_width == 192:
            model_config = palm_detect_v0_10_model_config
        else:
            print("[BlazeConfig.get_model_config] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    elif model_type == "blazeface":
        if num_anchors == 896 and input_width == 128:
            model_config = face_front_v0_06_model_config
        elif num_anchors == 896 and input_width == 256:
            model_config = face_back_v0_07_model_config
            #model_config = face_short_range_v0_10_model_config
        elif num_anchors == 2304 and input_width == 192:
            model_config = face_full_range_v0_10_model_config
        else:
            print("[BlazeConfig.get_model_config] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    elif model_type == "blazepose":       
        if num_anchors == 896 and input_width == 128:
            model_config = pose_detect_v0_07_model_config
        elif num_anchors == 2254 and input_width == 224:
            model_config = pose_detect_v0_10_model_config
        else:
            print("[BlazeConfig.get_model_config] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    else:
            print("[BlazeConfig.get_model_config] ERROR : Unsupported Model Type : ",model_type)
        
    return model_config  
    
def get_anchor_options( model_type, input_width, input_height, num_anchors ):
    if model_type == "blazepalm":
        if num_anchors == 2944 and input_width == 256:
            anchor_options = palm_detect_v0_06_anchor_options
        elif num_anchors == 2016 and input_width == 192:
            anchor_options = palm_detect_v0_10_anchor_options
        else:
            print("[BlazeConfig.get_anchor_options] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    elif model_type == "blazeface":
        if num_anchors == 896 and input_width == 128:
            anchor_options = face_front_v0_06_anchor_options
        elif num_anchors == 896 and input_width == 256:
            anchor_options = face_back_v0_07_anchor_options
            #anchor_options = face_short_range_v0_10_anchor_options
        elif num_anchors == 2304 and input_width == 192:
            anchor_options = face_full_range_v0_10_anchor_options
        else:
            print("[BlazeConfig.get_anchor_options] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    elif model_type == "blazepose":       
        if num_anchors == 896 and input_width == 128:
            anchor_options = pose_detect_v0_07_anchor_options
        elif num_anchors == 2254 and input_width == 224:
            anchor_options = pose_detect_v0_10_anchor_options
        else:
            print("[BlazeConfig.get_anchor_options] ERROR : Unsupported Number of Anchors : ",num_anchors,"(",model_type," ",input_width,"x",input_height,")")
    else:
            print("[BlazeConfig.get_anchor_options] ERROR : Unsupported Model Type : ",model_type)
        
    return anchor_options  


# Based on :
# reference : https://github.com/vidursatija/BlazePalm/blob/master/ML/genarchors.py
# reference : https://github.com/hollance/BlazeFace-PyTorch/blob/master/Anchors.ipynb
# Added case for num_strides==1
def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (max_scale + min_scale) * 0.5
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)

# This is a literal translation of ssd_anchors_calculator.cc
# reference : https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
def generate_anchors(options):
    strides_size = len(options["strides"])
    assert options["num_layers"] == strides_size

    anchors = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and \
              (options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
            scale = calculate_scale(options["min_scale"],
                                    options["max_scale"],
                                    last_same_stride_layer,
                                    strides_size)

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)                
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                                     else calculate_scale(options["min_scale"],
                                                          options["max_scale"],
                                                          last_same_stride_layer + 1,
                                                          strides_size)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)            
            
        stride = options["strides"][layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options["fixed_anchor_size"]:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    anchors = np.asarray(anchors)

    return anchors  
