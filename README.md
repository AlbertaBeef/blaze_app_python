# Overview

Python demonstration code for mediapipe models (blazepalm/hand, blazeface, blazepose).
![](images/blaze_detect_live_hand_face_pose_animation.gif)

## Instructions

Perform the following steps to execute on your platform, using tflite as example.

Replace "tflite" with other targets when applicable.

1. Clone repository

    $ git clone https://github.com/AlbertaBeef/blaze_app_python

    $ cd blaze_app_python

2. Select target

    $ cd tflite

3. Download models

    $ cd models
   
    $ source ./get_tflite_models.sh
   
    $ cd ..

5. Launch Application

    For hand landmarks :

    $ python3 blaze_detect_live.py --blaze hand

    Replace "hand" with "face" for face landmarks, or "pose" for body landmarks.

### Launch Arguments

| -Argument | --Argument    | Description                               | 
| :-------: | :-----------: | :---------------------------------------- | 
|  -i       | --image       | Use 'womand_hands.jpg' image as input. Default is usbcam      |
|  -b       | --blaze       | Blaze Application : hand, face, pose      |
|  -m       | --model1      | Detection Model (optional)                |
|  -n       | --model2      | Landmark Model (optional)                 |
|  -l       | --list        | List pipelines                            |
|  -d       | --debug       | Enable Debug Mode.  Default is off        |
|  -w       | --withoutview | Disable Output Viewing.  Default is on    |
|  -z       | --profile     | Enable Profile Mode.  Default is off      |

### Run-Time Options

| Key | Description                               | 
| :-: | :---------------------------------------- | 
|  p  | pause video ...                           |
|  c  | continue ...                              |
|  s  | step one frame at a time ...              |
|  w  | take a photo ...                          |
|  t  | toggle between image and live video       |
|  d  | toggle debug image on/off                 |
|  e  | toggle scores image on/off                |
|  f  | toggle FPS display on/off                 |
|  v  | toggle verbose on/off                     |
|  z  | toggle profiling on/off                   |
   

## Supported Frameworks

The blaze_app_python demo provides a common code base for comparing models inferenced from the following frameworks:
- TFLite : [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md)
- PyTorch : [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch)
- Vitis-AI 3.5 : [AlbertaBeef/blaze_tutorial/vitis-ai](https://github.com/AlbertaBeef/blaze_tutorial/tree/2023.1/vitis-ai)
- Hailo-8 (coming soon) : [AlbertaBeef/blaze_tutorial/hailo-8](https://github.com/AlbertaBeef/blaze_tutorial/tree/2023.1/hailo-8)


## Supported Models

The following specific models are supported.

<details>
 <summary><b>TFLite models</b></summary>

Detection Models

| Version | Model                    | Input Size  |  Scores   |   Boxes   | Comments             | Source               | Model Link                                | 
| :-----: | :----------------------: | :---------: | :-------: | :-------: | :------------------- | :------------------- | :---------------------------------------- | 
|  v0.06  | Palm                     |   256\*256  |   2944    |  2944x18  | BBox + 7 keypoints   | [metalwhale/hand_tracking](https://.github.com/metalwhale/hand_tracking) | [palm_detection_without_custom_op.tflite](https://raw.githubusercontent.com/metalwhale/hand_tracking/master/models/palm_detection_without_custom_op.tflite)  |
|  v0.10  | Palm (Full)              |   192\*192  |   2016    |  2016x18  | BBox + 7 keypoints   | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [palm_detection_lite.tflite](https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite) | 
|  v0.10  | Palm (Full)              |   192\*192  |   2016    |  2016x18  | BBox + 7 keypoints   | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [palm_detection_full.tflite](https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite) | 
|  v0.10  | Face (Short Range)       |   128\*128  |    896    |   896x16  | BBox + 6 keypoints   | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [face_detection_short_range.tflite](https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite) | 
|  v0.10  | Face (Full Range)        |   128\*128  |    896    |   896x16  | BBox + 6 keypoints   | [mgoogle/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [face_detection_full_range.tflite](https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite) | 
|  v0.10  | Face (Full Range Sparse) |   128\*128  |    896    |   896x16  | BBox + 6 keypoints   | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [face_detection_full_range_sparse.tflite](https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite) | 
|  v0.10  | Pose                     |   224\*224  |   2254    |  2254x12  | BBox + 4 keypoints   | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [pose_detection.tflite](https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite) | 

Landmarks Models

| Version | Model        | Input Size  |  Flag  | Landmarks | Comments                                 | Source               | Model Link                                | 
| :-----: | :----------: | :---------: | :----: | :-------: | :--------------------------------------- | :------------------- | :---------------------------------------- | 
|  v0.06  | Hand         |   256\*256  |    1   |    21x3   | 21 keypoints (x,y,score)                 | [metalwhale/hand_tracking](https://.github.com/metalwhale/hand_tracking) | [hand_landmark.tflite](https://raw.githubusercontent.com/metalwhale/hand_tracking/master/models/hand_landmark.tflite)  |
|  v0.10  | Hand (Lite)  |   224\*224  |    1   |  2016x18  | 21 keypoints (x,y,score)                 | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [hand_landmark_lite.tflite](https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite) | 
|  v0.10  | Hand (Full)  |   224\*224  |    1   |  2016x18  | 21 keypoints (x,y,score)                 | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md)| [hand_landmark_full.tflite](https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite) | 
|  v0.10  | Face         |   192\*192  |    1   |   896x16  | 468 keypoints (x,y,score)                | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [face_landmark.tflite](https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite) | 
|  v0.10  | Pose (Full Body, Lite)  |   256\*256  |    1   |    39x5   | 39 keypoints (x,y,z,visibility,presence) | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [pose_landmark_lite.tflite](https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite) | 
|  v0.10  | Pose (Full Body, Full)  |   256\*256  |    1   |    39x5   | 39 keypoints (x,y,z,visibility,presence) | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [pose_landmark_full.tflite](https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite) | 
|  v0.10  | Pose (Full Body, Heavy) |   256\*256  |    1   |    39x5   | 39 keypoints (x,y,z,visibility,presence) | [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md) | [pose_landmark_heavy.tflite](https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite) | 


</details>

<details>
 <summary><b>PyTorch models</b></summary>

Detection Models

| Version | Type                | Input Size  |  Scores   |   Boxes   | Comments             | Source               | Model Link                                | 
| :-----: | :-----------------: | :---------: | :-------: | :-------: | :------------------- | :------------------- | :---------------------------------------- | 
|  v0.06  | Palm                |   256\*256  |   2944    |  2944x18  | BBox + 7 keypoints   | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazepalm.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazepalm.pth)  |
|  v0.06  | Face (Front Camera) |   128\*128  |    896    |   896x16  | BBox + 6 keypoints   | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazeface.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazeface.pth) | 
|  v0.06  | Face (Back Camera)  |   256\*256  |    896    |   896x16  | BBox + 6 keypoints   | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazefaceback.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazefaceback.pth) | 
|  v0.06  | Pose                |   128\*128  |    896    |   896x12  | BBox + 4 keypoints   | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazepose.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazepose.pth) | 

Landmarks Models

| Version | Type              | Input Size  |  Flag  | Landmarks  | Comments             | Source               | Model Link                                | 
| :-----: | :---------------: | :---------: | :----: | :-------: | :------------------- | :------------------- | :---------------------------------------- | 
|  v0.06  | Hand              |   256\*256  |    1   |    21x3  | 21 keypoints (x,y,score)  | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazehand_landmark.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazehand_landmark.pth) |
|  v0.06  | Face              |   192\*192  |    1   |   896x16  | 468 keypoints (x,y,score)   | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazeface_landmark.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazeface_landmark.pth) | 
|  v0.06  | Pose (Upper Body) |   256\*256  |    1   |    39x5   | 31 keypoints (x,y,z,visibility,presence) | [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch) | [blazepose_landmark.pth](https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/blazepose_landmark.pth)| 


</details>

## Profiling

Profiling Results on various platforms

<details>
 <summary><b>Workstation (HP Z4 G4) with AMD Radeon Pro W7900 GPU</b></summary>

Latency.
![](images/blaze_detect_live_workstation01_profiling_latency.tif)

Frame Rate (estimated from total latency)
![](images/blaze_detect_live_workstation01_profiling_fps.tif)

</details>


<details>
 <summary><b>Laptop (ZBook)</b></summary>

Latency.
![](images/blaze_detect_live_laptop01_profiling_latency.tif)

Frame Rate (estimated from total latency)
![](images/blaze_detect_live_laptop01_profiling_fps.tif)

</details>


<details>
 <summary><b>ZUBoard with B512 DPU</b></summary>

Latency.
![](images/blaze_detect_live_zuboardn01_profiling_latency.tif)

Frame Rate (estimated from total latency)
![](images/blaze_detect_live_zuboard01_profiling_fps.tif)

</details>


## References

The blaze_app_python demo is based on the following work:
- Google MediaPipe models : [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md)
- BlazePalm : [vidursatija/BlazePalm](https://github.com/vidursatija/BlazePalm)
- BlazeFace-PyTorch : [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)
- MediaPipePyTorch : [zmurez/MediaPipePytorch](https://github.com/zmurez/MediaPipePyTorch)
- Blaze Tutorial : [AlbertaBeef/blaze_tutorial](https://github.com/AlbertaBeef/blaze_tutorial/tree/2023.1)

