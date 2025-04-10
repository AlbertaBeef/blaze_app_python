# MediaPipe_on_IMX93
Run the MediaPipe Models on MaaXBoard OSM93.

clone the blaze_app_python repository here: 
```
git clone https://github.com/zebular13/blaze_app_python/tree/imx 
```
This is a python implementation of the MediaPipe framework.

Download the models: 

```
cd blaze_app_python
cd blaze_tflite_imx
cd models
./get_tflite_models.sh
```

This includes the float32 pose, hand and face models (check the script for details) as well as the quantized pose and face models. 

Once the models have been downloaded, you can run the script in the ```blaze_tflite_imx``` folder:
```
python3 blaze_detect_live_gstreamer.py -b face -N -q
```
### Launch Arguments

| -Argument | --Argument    | Description                               | 
| :-------: | :-----------: | :---------------------------------------- | 
|  -i       | --input       | Video input device. Default is auto-detect (first usbcam) |
|  -I       | --image       | Use 'womand_hands.jpg' image as input. Default is usbcam  |
|  -b       | --blaze       | Command seperated list of targets  (hand, face, pose).  Default is 'hand, face, pose'      |
|  -t       | --target      | Command seperated list of targets (blaze_tflite, blaze_pytorch, blaze_vitisai).  Default is 'blaze_tflite,blaze_pytorch,blaze_vitisai,blaze_hailo'      |
|  -p       | --pipeline    | Command seperated list of pipelines (Use --list to get list of targets). Default is 'all'  |
|  -l       | --list        | List pipelines                            |
|  -d       | --debug       | Enable Debug Mode.  Default is off        |
|  -w       | --withoutview | Disable Output Viewing.  Default is on    |
|  -z       | --profilelog  | Enable Profile Log Mode.  Default is off  |
|  -Z       | --profileview | Enable Profile View Mode.  Default is off |
|  -f       | --fps         | Enable FPS Display.  Default is off       |
|  -N       | --npu         | Enable NPU.  Default is off       |
|  -q       | --quant       | Select for quantized models      |

## Tutorials


To learn how to get started on the MaaXBoard OSM93, please refer to the following Getting Started Guide on Hackster:
http://avnet.me/MLonOSM93
![MaaXBoardOSM93GettingStarted](https://github.com/user-attachments/assets/eb8b7a1f-c78f-4d42-8537-0e6f54ef508b)


To learn how to quantize your own models (necessary for deployment on the i.MX93's ethos-U65 NPU) please refer to Part 1: Accelerating AI on the MaaXBoard OSM93 – Quantization: [avnet.me/acceleratingAIonOSM93-part1](http://avnet.me/acceleratingAIonOSM93-part1)
![MaaXBoardOSM93GettingStarted-dark](https://github.com/user-attachments/assets/b39084a6-18ee-47f5-ba1d-8cb23ee14663)


To learn how to convert models to vela, check out [Part 2: Accelerating AI on the MaaXBoard OSM93 – Vela Conversion](http://avnet.me/acceleratingAIonOSM93-part2)
![MaaXBoardOSM93VelaConvert](https://github.com/user-attachments/assets/22518602-14f8-4cd9-ad77-4b8113f52895)


Finally, learn how to convert your image pipeline to NNStreamer in [Part 3: Accelerating AI on the MaaXBoard OSM93 – NNStreamer] (https://www.hackster.io/monica/accelerating-ai-on-maaxboard-osm93-camera-pipeline-822c68)
![MaaXBoardOSM93-NNstreamer](https://github.com/user-attachments/assets/36d09101-051c-43ba-8ce0-01db4bc35ca7)

