# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

FP=${1:-FP32}

#source /opt/intel/openvino_2021/bin/setupvars.sh

model_palm_detector_v0_07=("palm_detection_v0_07","palm_detection_v0_07.tflite")
model_palm_detector_v0_10_lite=("palm_detection_lite","palm_detection_lite.tflite")
model_palm_detector_v0_10_full=("palm_detection_full","palm_detection_full.tflite")
model_hand_landmark_v0_07=("hand_landmark_v0_07","hand_landmark_v0_07.tflite")
model_hand_landmark_v0_10_lite=("hand_landmark_lite","hand_landmark_lite.tflite")
model_hand_landmark_v0_10_full=("hand_landmark_full","hand_landmark_full.tflite")
model_list=(
	model_palm_detector_v0_07[@]
	model_palm_detector_v0_10_lite[@]
	model_palm_detector_v0_10_full[@]
	model_hand_landmark_v0_07[@]
	model_hand_landmark_v0_10_lite[@]
	model_hand_landmark_v0_10_full[@]
)
model_count=${#model_list[@]}
#echo $model_count


# Convert to TensorFlow-Keras

for ((i=0; i<$model_count; i++))
do
	model=${!model_list[i]}
	model_array=(${model//,/ })
	model_name=${model_array[0]}
	model_file=${model_array[1]}

	echo tflite2tensorflow \
	  --model_path ${model_file} \
	  --model_output_path ${model_name} \
	  --flatc_path ../../flatc \
	  --schema_path ../../schema.fbs \
	  --output_pb

	tflite2tensorflow \
	  --model_path ${model_file} \
	  --model_output_path ${model_name} \
	  --flatc_path ../../flatc \
	  --schema_path ../../schema.fbs \
	  --output_pb

	echo tflite2tensorflow \
	  --model_path ${model_file} \
	  --model_output_path ${model_name} \
	  --flatc_path ../../flatc \
	  --schema_path ../../schema.fbs \
	  --output_onnx

	tflite2tensorflow \
	  --model_path ${model_file} \
	  --model_output_path ${model_name} \
	  --flatc_path ../../flatc \
	  --schema_path ../../schema.fbs \
	  --output_onnx

	#cp ${model_name}/model_float32.pb ${model_name}.pb
done
