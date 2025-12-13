#model_palm_detector_v0_07=("palm_detection_v0_07","palm_detection_v0_07.tflite")
model_palm_detector_v0_07=("palm_detection_v0_07","palm_detection_without_custom_op.tflite")
model_hand_landmark_v0_07=("hand_landmark_v0_07","hand_landmark_v0_07.tflite")
model_palm_detector_v0_10_lite=("palm_detection_lite","palm_detection_lite.tflite")
model_palm_detector_v0_10_full=("palm_detection_full","palm_detection_full.tflite")
model_hand_landmark_v0_10_lite=("hand_landmark_lite","hand_landmark_lite.tflite")
model_hand_landmark_v0_10_full=("hand_landmark_full","hand_landmark_full.tflite")
model_face_detector_v0_07_front=("face_detection_front_v0_07","face_detection_front_v0_07.tflite")
model_face_detector_v0_07_back=("face_detection_back_v0_07","face_detection_back_v0_07.tflite")
model_face_landmark_v0_07=("face_landmark_v0_07","face_landmark_v0_07.tflite")
model_face_detector_v0_10_full=("face_detection_full_range","face_detection_full_range.tflite")
model_face_detector_v0_10_short=("face_detection_short_range","face_detection_short_range.tflite")
model_face_landmark_v0_10=("face_landmark","face_landmark.tflite")
model_pose_detector_v0_07=("pose_detection_v0_07","pose_detection_v0_07.tflite")
model_pose_landmark_v0_07_upper=("pose_landmark_v0_07_upper_body","pose_landmark_v0_07_upper_body.tflite")
model_pose_detector_v0_10=("pose_detection","pose_detection.tflite")
model_pose_landmark_v0_10_lite=("pose_landmark_lite","pose_landmark_lite.tflite")
model_pose_landmark_v0_10_full=("pose_landmark_full","pose_landmark_full.tflite")
model_pose_landmark_v0_10_heavy=("pose_landmark_heavy","pose_landmark_heavy.tflite")
model_list=(
	model_palm_detector_v0_07[@]
	model_hand_landmark_v0_07[@]
	
	model_palm_detector_v0_10_lite[@]
	model_palm_detector_v0_10_full[@]
	model_hand_landmark_v0_10_lite[@]
	model_hand_landmark_v0_10_full[@]
	
	model_face_detector_v0_07_front[@]
	model_face_detector_v0_07_back[@]
	model_face_landmark_v0_07[@]
	
	model_face_detector_v0_10_full[@]
	model_face_detector_v0_10_short[@]
	model_face_landmark_v0_10[@]
	
	model_pose_detector_v0_07[@]
	model_pose_landmark_v0_07_upper[@]

	model_pose_detector_v0_10	
	model_pose_landmark_v0_10_lite[@]
	model_pose_landmark_v0_10_full[@]
	model_pose_landmark_v0_10_heavy[@]
)
model_count=${#model_list[@]}
#echo $model_count


# Convert to ONNX

for ((i=0; i<$model_count; i++))
do
	model=${!model_list[i]}
	model_array=(${model//,/ })
	model_name=${model_array[0]}
	model_file=${model_array[1]}

	echo onnxsim ../../blaze_onnx/models/${model_name}.onnx ${model_name}_sim.onnx
	onnxsim ../../blaze_onnx/models/${model_name}.onnx ${model_name}_sim.onnx
	
done
