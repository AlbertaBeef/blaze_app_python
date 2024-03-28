'''
Copyright 2024 Avnet Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
#
# Blaze Demo Application (live with USB camera)
#
# References:
#   https://www.github.com/AlbertaBeef/blaze_app_python
#   https://www.github.com/AlbertaBeef/blaze_tutorial/tree/2023.1
#
# Dependencies:
#   TFLite
#      tensorflow
#    or
#      tflite_runtime
#   PyTorch
#      torch
#   Vitis-AI 3.5
#      xir
#      vitis_ai_library
#   Hailo
#      hailo_platform
#   plots
#      pyplotly
#      kaleido
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

from ctypes import *
from typing import List
import pathlib
#import threading
import time
import sys
import argparse
import glob
import subprocess
import re
import sys

from datetime import datetime
import plotly.graph_objects as go

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('blaze_common/'))
sys.path.append(os.path.abspath('blaze_tflite/'))
sys.path.append(os.path.abspath('blaze_pytorch/'))
sys.path.append(os.path.abspath('blaze_vitisai/'))
sys.path.append(os.path.abspath('blaze_hailo/'))

supported_targets = {
    "blaze_tflite": False,
    "blaze_pytorch": False,
    "blaze_vitisai": False,
    "blaze_hailo": False
}
try:
    from blaze_tflite.blazedetector import BlazeDetector as BlazeDetector_tflite
    from blaze_tflite.blazelandmark import BlazeLandmark as BlazeLandmark_tflite
    print("[INFO] blaze_tflite supported ...")
    supported_targets["blaze_tflite"] = True
except:
    print("[INFO] blaze_tflite NOT supported ...")

try:
    from blaze_pytorch.blazedetector import BlazeDetector as BlazeDetector_pytorch
    from blaze_pytorch.blazelandmark import BlazeLandmark as BlazeLandmark_pytorch
    print("[INFO] blaze_pytorch supported ...")
    supported_targets["blaze_pytorch"] = True
except:
    print("[INFO] blaze_pytorch NOT supported ...")

try:
    from blaze_vitisai.blazedetector import BlazeDetector as BlazeDetector_vitisai
    from blaze_vitisai.blazelandmark import BlazeLandmark as BlazeLandmark_vitisai
    print("[INFO] blaze_vitisai supported ...")
    supported_targets["blaze_vitisai"] = True
    
    def detect_dpu_architecture():
        proc = subprocess.run(['xdputil','query'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if 'DPU Arch' in line:
                #         "DPU Arch":"DPUCZDX8G_ISA0_B128_01000020E2012208",
                #dpu_arch = re.search('DPUCZDX8G_ISA0_(.+?)_', line).group(1)  
                #                 "DPU Arch":"DPUCZDX8G_ISA1_B2304",
                #dpu_arch = re.search('DPUCZDX8G_ISA1_(.+?)', line).group(1)
                #         "DPU Arch":"DPUCZDX8G_ISA1_B512_0101000016010200",
                dpu_arch = re.search('DPUCZDX8G_ISA1_(.+?)_', line).group(1) 
                return dpu_arch
            
    dpu_arch = detect_dpu_architecture()
    print("[INFO] DPU Architecture : ",dpu_arch)            
except:
    print("[INFO] blaze_vitisai NOT supported ...")
    dpu_arch = "B?"    

try:
    from blaze_hailo.hailo_inference import HailoInference
    hailo_infer = HailoInference()
    from blaze_hailo.blazedetector import BlazeDetector as BlazeDetector_hailo
    from blaze_hailo.blazelandmark import BlazeLandmark as BlazeLandmark_hailo
    print("[INFO] blaze_hailo supported ...")
    supported_targets["blaze_hailo"] = True
except:
    print("[INFO] blaze_hailo NOT supported ...")

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

from timeit import default_timer as timer

def get_media_dev_by_name(src):
    devices = glob.glob("/dev/media*")
    for dev in sorted(devices):
        proc = subprocess.run(['media-ctl','-d',dev,'-p'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

def get_video_dev_by_name(src):
    devices = glob.glob("/dev/video*")
    for dev in sorted(devices):
        proc = subprocess.run(['v4l2-ctl','-d',dev,'-D'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev


# Parameters (tweaked for video)
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input'      , type=str, default="", help="Video input device. Default is auto-detect (first usbcam)")
ap.add_argument('-I', '--image'      , default=False, action='store_true', help="Use 'womand_hands.jpg' image as input. Default is usbcam")
ap.add_argument('-b', '--blaze'      , type=str,  default="hand,face,pose", help="Command seperated list of targets  (hand, face, pose).  Default is 'hand, face, pose'")
ap.add_argument('-t', '--target'     , type=str,  default="blaze_tflite,blaze_pytorch,blaze_vitisai,blaze_hailo", help="Command seperated list of targets (blaze_tflite, blaze_pytorch, blaze_vitisai).  Default is 'blaze_tflite,blaze_pytorch,blaze_vitisai,blaze_hailo'")
ap.add_argument('-p', '--pipeline'   , type=str,  default="all", help="Command seperated list of pipelines (Use --list to get list of targets). Default is 'all'")
ap.add_argument('-l', '--list'       , default=False, action='store_true', help="List pipelines.")
ap.add_argument('-d', '--debug'      , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-z', '--profilelog' , default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
ap.add_argument('-Z', '--profileview', default=False, action='store_true', help="Enable Profile View (Latency). Default is off")
ap.add_argument('-f', '--fps'        , default=False, action='store_true', help="Enable FPS display. Default is off")

args = ap.parse_args()  
  
print('Command line options:')
print(' --input       : ', args.input)
print(' --image       : ', args.image)
print(' --blaze       : ', args.blaze)
print(' --target      : ', args.target)
print(' --pipeline    : ', args.pipeline)
print(' --list        : ', args.list)
print(' --debug       : ', args.debug)
print(' --withoutview : ', args.withoutview)
print(' --profilelog  : ', args.profilelog)
print(' --profileview  : ', args.profileview)
print(' --fps         : ', args.fps)


blaze_pipelines = [
#   { "blaze": "hand", "pipeline": "tfl_hand_v0_07"       , "model1": "blaze_tflite/models/palm_detection_v0_07.tflite",             "model2": "blaze_tflite/models/hand_landmark_v0_07.tflite" },
    { "blaze": "hand", "pipeline": "tfl_hand_v0_07"       , "model1": "blaze_tflite/models/palm_detection_without_custom_op.tflite", "model2": "blaze_tflite/models/hand_landmark_v0_07.tflite" },
    { "blaze": "hand", "pipeline": "tfl_hand_v0_10_lite"  , "model1": "blaze_tflite/models/palm_detection_lite.tflite",              "model2": "blaze_tflite/models/hand_landmark_lite.tflite" },
    { "blaze": "hand", "pipeline": "tfl_hand_v0_10_full"  , "model1": "blaze_tflite/models/palm_detection_full.tflite",              "model2": "blaze_tflite/models/hand_landmark_full.tflite" },
    { "blaze": "hand", "pipeline": "pyt_hand_v0_07"       , "model1": "blaze_pytorch/models/blazepalm.pth",                          "model2": "blaze_pytorch/models/blazehand_landmark.pth" },
    { "blaze": "hand", "pipeline": "vai_hand_v0_07"       , "model1": "blaze_vitisai/models/BlazePalm/"+dpu_arch+"/BlazePalm.xmodel","model2": "blaze_vitisai/models/BlazeHandLandmark/"+dpu_arch+"/BlazeHandLandmark.xmodel" },
    { "blaze": "hand", "pipeline": "hai_hand_v0_07_2c"    , "model1": "blaze_hailo/models/palm_detection_v0_07_2c.hef",              "model2": "blaze_hailo/models/hand_landmark_v0_07.hef" },
    { "blaze": "hand", "pipeline": "hai_hand_v0_07_1c"    , "model1": "blaze_hailo/models/palm_detection_v0_07_1c.hef",              "model2": "blaze_hailo/models/hand_landmark_v0_07.hef" },
    { "blaze": "hand", "pipeline": "hai_hand_v0_10_lite"  , "model1": "blaze_hailo/models/palm_detection_lite.hef",                  "model2": "blaze_hailo/models/hand_landmark_lite.hef" },
    { "blaze": "hand", "pipeline": "hai_hand_v0_10_full"  , "model1": "blaze_hailo/models/palm_detection_full.hef",                  "model2": "blaze_hailo/models/hand_landmark_full.hef" },
    { "blaze": "face", "pipeline": "tfl_face_v0_07_front" , "model1": "blaze_tflite/models/face_detection_front_v0_07.tflite",       "model2": "blaze_tflite/models/face_landmark_v0_07.tflite" },
    { "blaze": "face", "pipeline": "tfl_face_v0_07_back"  , "model1": "blaze_tflite/models/face_detection_back_v0_07.tflite",        "model2": "blaze_tflite/models/face_landmark_v0_07.tflite" },
    { "blaze": "face", "pipeline": "tfl_face_v0_10_short" , "model1": "blaze_tflite/models/face_detection_short_range.tflite",       "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "pipeline": "tfl_face_v0_10_full"  , "model1": "blaze_tflite/models/face_detection_full_range.tflite",        "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "pipeline": "tfl_face_v0_10_sparse", "model1": "blaze_tflite/models/face_detection_full_range_sparse.tflite", "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "pipeline": "pyt_face_v0_07_front" , "model1": "blaze_pytorch/models/blazeface.pth",                          "model2": "blaze_pytorch/models/blazeface_landmark.pth" },
    { "blaze": "face", "pipeline": "pyt_face_v0_07_back"  , "model1": "blaze_pytorch/models/blazefaceback.pth",                      "model2": "blaze_pytorch/models/blazeface_landmark.pth" },
#    { "blaze": "face", "pipeline": "vai_face_v0_07_front" , "model1": "blaze_vitisai/models/BlazeFace/"+dpu_arch+"/BlazeFace.xmodel","model2": "blaze_vitisai/models/BlazeFaceLandmark/"+dpu_arch+"/BlazeFaceLandmark.xmodel" },
#    { "blaze": "face", "pipeline": "vai_face_v0_07_back"  , "model1": "blaze_vitisai/models/BlazeFaceBack/"+dpu_arch+"/BlazeFaceBack.xmodel","model2": "blaze_vitisai/models/BlazeFaceLandmark/"+dpu_arch+"/BlazeFaceLandmark.xmodel" },
    { "blaze": "face", "pipeline": "hai_face_v0_10_short" , "model1": "blaze_hailo/models/face_detection_short_range.hef",           "model2": "blaze_hailo/models/face_landmark.hef" },
    { "blaze": "face", "pipeline": "hai_face_v0_10_full"  , "model1": "blaze_hailo/models/face_detection_full_range.hef",            "model2": "blaze_hailo/models/face_landmark.hef" },
    { "blaze": "pose", "pipeline": "tfl_pose_v0_10_lite"  , "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_lite.tflite" },
    { "blaze": "pose", "pipeline": "tfl_pose_v0_10_full"  , "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_full.tflite" },
    { "blaze": "pose", "pipeline": "tfl_pose_v0_10_heavy" , "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_heavy.tflite" },
    { "blaze": "pose", "pipeline": "pyt_pose_v0_06"       , "model1": "blaze_pytorch/models/blazepose.pth",                          "model2": "blaze_pytorch/models/blazepose_landmark.pth" },
    { "blaze": "pose", "pipeline": "hai_pose_v0_10_lite"  , "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_hailo/models/pose_landmark_lite.hef" }
]
nb_blaze_pipelines = len(blaze_pipelines)

if args.list:
   print("")
   print("List of target pipelines:")
   for i in range(nb_blaze_pipelines):
      print("%02d %s %s"%(i,
         blaze_pipelines[i]["pipeline"].ljust(25),
         blaze_pipelines[i]["model1"])
         )
      print("%s %s"%("".ljust(2+1+25),
         blaze_pipelines[i]["model2"])
         )
   print("")
   exit()


bInputImage = False
bInputVideo = False
bInputCamera = True

if os.path.exists(args.input):
    print("[INFO] Input exists : ",args.input)
    file_name, file_extension = os.path.splitext(args.input)
    file_extension = file_extension.lower()
    print("[INFO] Input type : ",file_extension)
    if file_extension == ".jpg" or file_extension == ".png" or file_extension == ".tif":
        bInputImage = True
        bInputVideo = False
        bInputCamera = False
    if file_extension == ".mov" or file_extension == ".mp4":
        bInputImage = False
        bInputVideo = True
        bInputCamera = False

if bInputCamera == True:
    print("[INFO] Searching for USB camera ...")
    dev_video = get_video_dev_by_name("uvcvideo")
    dev_media = get_media_dev_by_name("uvcvideo")
    print(dev_video)
    print(dev_media)

    if dev_video == None:
        input_video = 0
    elif args.input != "":
        input_video = args.input 
    else:
        input_video = dev_video  

    # Open video
    cap = cv2.VideoCapture(input_video)
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    #frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    #frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] input : camera",input_video," (",frame_width,",",frame_height,")")

if bInputVideo == True:
    # Open video file
    cap = cv2.VideoCapture(args.input)
    frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] input : video ",args.input," (",frame_width,",",frame_height,")")

if bInputImage == True:
    image = cv2.imread(args.input)
    frame_height,frame_width,_ = image.shape
    print("[INFO] input : image ",args.input," (",frame_width,",",frame_height,")")

output_dir = './captured-images'

profile_csv = './blaze_detect_live.csv'
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :",profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :",profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist


nb_active_pipelines = 0
for i in range(nb_blaze_pipelines):
    blaze    = blaze_pipelines[i]["blaze"]
    pipeline = blaze_pipelines[i]["pipeline"]
    model1   = blaze_pipelines[i]["model1"]
    model2   = blaze_pipelines[i]["model2"]
   
    blaze_pipelines[i]["supported"] = False # until proven otherwise
    blaze_pipelines[i]["selected"] = False # until proven otherwise
    
    target1 = re.search('(.+?)/', model1).group(1) 
    target2 = re.search('(.+?)/', model2).group(1)
    
    if blaze in args.blaze and target1 in args.target and target2 in args.target and (pipeline in args.pipeline or args.pipeline == "all"):
        blaze_pipelines[i]["selected"] = True
        if args.debug:
            print("[blaze_detect_live] Pipeline ",pipeline," selected.")
    
    if supported_targets[target1]==True and supported_targets[target2]==True and blaze_pipelines[i]["selected"] == True:
        if blaze=="hand":
            detector_type = "blazepalm"
            landmark_type = "blazehandlandmark"
        elif blaze=="face":
            detector_type = "blazeface"
            landmark_type = "blazefacelandmark"
        elif blaze=="pose":
            detector_type = "blazepose"
            landmark_type = "blazeposelandmark"
        else:
            print("[ERROR] Invalid Blaze application : ",blaze,".  MUST be one of hand,face,pose.")

        if target1=="blaze_tflite":
            blaze_detector = BlazeDetector_tflite(detector_type)
        elif target1=="blaze_pytorch":
            blaze_detector = BlazeDetector_pytorch(detector_type)
        elif target1=="blaze_vitisai":
            blaze_detector = BlazeDetector_vitisai(detector_type)
        elif target1=="blaze_hailo":
            blaze_detector = BlazeDetector_hailo(detector_type,hailo_infer)
        else:
            print("[ERROR] Invalid target : ",target1,".  MUST be a valid blaze_* directory.")
        blaze_detector.set_debug(debug=args.debug)
        blaze_detector.display_scores(debug=False)
        blaze_detector.load_model(model1)
 
        if target2=="blaze_tflite":
            blaze_landmark = BlazeLandmark_tflite(landmark_type)
        elif target2=="blaze_pytorch":
            blaze_landmark = BlazeLandmark_pytorch(landmark_type)
        elif target2=="blaze_vitisai":
            blaze_landmark = BlazeLandmark_vitisai(landmark_type)
        elif target2=="blaze_hailo":
            blaze_landmark = BlazeLandmark_hailo(landmark_type,hailo_infer)
        else:
            print("[ERROR] Invalid target : ",target1,".  MUST be a valid blaze_* directory.")
        blaze_landmark.set_debug(debug=args.debug)
        blaze_landmark.load_model(model2)
       
        blaze_pipelines[i]["supported"]     = True
        blaze_pipelines[i]["detector_type"] = detector_type
        blaze_pipelines[i]["detector"]      = blaze_detector
        blaze_pipelines[i]["landmark_type"] = landmark_type
        blaze_pipelines[i]["landmark"]      = blaze_landmark

        if args.debug:
            print("[blaze_detect_live] Pipeline ",pipeline," supported and initialized.")

        nb_active_pipelines += 1
        

if nb_active_pipelines == 0:
    print("[ERROR] no pipelines selected !")
    exit()        
        
print("================================================================")
print("Blaze Detect Live Demo")
print("================================================================")
print("\tPress ESC to quit ...")
print("----------------------------------------------------------------")
print("\tPress 'p' to pause video ...")
print("\tPress 'c' to continue ...")
print("\tPress 's' to step one frame at a time ...")
print("\tPress 'w' to take a photo ...")
print("----------------------------------------------------------------")
print("\tPress 't' to toggle between image and live video")
print("\tPress 'd' to toggle debug image on/off")
print("\tPress 'e' to toggle scores image on/off")
print("\tPress 'f' to toggle FPS display on/off")
print("\tPress 'v' to toggle verbose on/off")
print("\tPress 'z' to toggle profiling log on/off")
print("\tPress 'Z' to toggle profiling view on/off")
print("================================================================")

bStep = False
bPause = False
bWrite = False
bUseImage = args.image
bShowDebugImage = False
bShowScores = False
bShowFPS = args.fps
bVerbose = args.debug
bViewOutput = not args.withoutview
bProfileLog = args.profilelog
bProfileView = args.profileview

def ignore(x):
    pass

for pipeline_id in range(nb_blaze_pipelines):

    if blaze_pipelines[pipeline_id]["supported"] and blaze_pipelines[pipeline_id]["selected"]:

        blaze_detector_type = blaze_pipelines[pipeline_id]["detector_type"]
        blaze_landmark_type = blaze_pipelines[pipeline_id]["landmark_type"]
        blaze_title = blaze_pipelines[pipeline_id]["pipeline"]
                
        app_main_title = blaze_title+" Demo"
        app_ctrl_title = blaze_title+" Demo"
        
        if bViewOutput:
            cv2.namedWindow(app_main_title)

            thresh_min_score = blaze_detector.min_score_thresh
            thresh_min_score_prev = thresh_min_score
            cv2.createTrackbar('threshMinScore', app_ctrl_title, int(thresh_min_score*100), 100, ignore)

image = []
output = []

frame_count = 0

# init the real-time FPS counter
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)

while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    frame_count = frame_count + 1

    if bUseImage:
        frame = cv2.imread('woman_hands.jpg')
        if not (type(frame) is np.ndarray):
            print("[ERROR] cv2.imread('woman_hands.jpg') FAILED !")
            break;
    elif bInputImage:
        frame = cv2.imread(args.input)
        if not (type(frame) is np.ndarray):
            print("[ERROR] cv2.imread(",args.input,") FAILED !")
            break;
    else:
        flag, frame = cap.read()
        if not flag:
            print("[ERROR] cap.read() FAILEd !")
            break

    if bProfileLog or bProfileView:
        prof_title          = ['']*nb_blaze_pipelines
        prof_resize         = np.zeros(nb_blaze_pipelines)
        prof_detector_pre   = np.zeros(nb_blaze_pipelines)
        prof_detector_model = np.zeros(nb_blaze_pipelines)
        prof_detector_post  = np.zeros(nb_blaze_pipelines)
        prof_extract_roi    = np.zeros(nb_blaze_pipelines)
        prof_landmark_pre   = np.zeros(nb_blaze_pipelines)
        prof_landmark_model = np.zeros(nb_blaze_pipelines)
        prof_landmark_post  = np.zeros(nb_blaze_pipelines)
        prof_annotate       = np.zeros(nb_blaze_pipelines)
        #
        prof_total          = np.zeros(nb_blaze_pipelines)
        prof_fps            = np.zeros(nb_blaze_pipelines)
    
    for pipeline_id in range(nb_blaze_pipelines):

        if blaze_pipelines[pipeline_id]["supported"] and blaze_pipelines[pipeline_id]["selected"]:

            image = frame.copy()
            
            blaze_detector_type = blaze_pipelines[pipeline_id]["detector_type"]
            blaze_landmark_type = blaze_pipelines[pipeline_id]["landmark_type"]
            blaze_title = blaze_pipelines[pipeline_id]["pipeline"]
            blaze_detector = blaze_pipelines[pipeline_id]["detector"]
            blaze_landmark = blaze_pipelines[pipeline_id]["landmark"]
            
            app_main_title = blaze_title+" Demo"
            app_ctrl_title = blaze_title+" Demo"
            app_debug_title = blaze_title+" Debug"
            
            # Get trackbar values
            if bViewOutput:
                thresh_min_score = cv2.getTrackbarPos('threshMinScore', app_ctrl_title)
                if thresh_min_score < 10:
                    thresh_min_score = 10
                    cv2.setTrackbarPos('threshMinScore', app_ctrl_title,thresh_min_score)
                thresh_min_score = thresh_min_score*(1/100)
                if thresh_min_score != thresh_min_score_prev:
                    blaze_detector.min_score_thresh = thresh_min_score
                    thresh_min_score_prev = thresh_min_score
                
                
            #image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            
            # BlazePalm pipeline
            
            start = timer()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img1,scale1,pad1=blaze_detector.resize_pad(image)
            profile_resize = timer()-start

            if bShowDebugImage:
                # show the resized input image
                debug_img = img1.astype(np.float32)/255.0
                debug_img = cv2.resize(debug_img,(blaze_landmark.resolution,blaze_landmark.resolution))
            
            normalized_detections = blaze_detector.predict_on_image(img1)
            if len(normalized_detections) > 0:
  
                start = timer()          
                detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector.detection2roi(detections)
                roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
                profile_extract = timer()-start

                flags, normalized_landmarks = blaze_landmark.predict(roi_img)

                if bShowDebugImage:
                    # show the ROIs
                    for i in range(roi_img.shape[0]):
                        #roi_landmarks = np.expand_dims(normalized_landmarks[i,:,:].copy(), axis=0)
                        roi_landmarks = normalized_landmarks[i,:,:].copy()
                        roi_landmarks = roi_landmarks*blaze_landmark.resolution
                        if blaze_landmark_type == "blazehandlandmark":
                            draw_landmarks(roi_img[i], roi_landmarks[:,:2], HAND_CONNECTIONS, size=2)
                        elif blaze_landmark_type == "blazefacelandmark":
                            draw_landmarks(roi_img[i], roi_landmarks[:,:2], FACE_CONNECTIONS, size=1)                                    
                        elif blaze_landmark_type == "blazeposelandmark":
                            if roi_landmarks.shape[1] > 33:
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                            else:
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                
                        debug_img = cv2.hconcat([debug_img,roi_img[i]])

                start = timer() 
                landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    #if True: #flag>.5:
                    if blaze_landmark_type == "blazehandlandmark":
                        draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)
                    elif blaze_landmark_type == "blazefacelandmark":
                        draw_landmarks(output, landmark[:,:2], FACE_CONNECTIONS, size=1)                                    
                    elif blaze_landmark_type == "blazeposelandmark":
                        if landmarks.shape[1] > 33:
                            draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                        else:
                            draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                
                   
                draw_roi(output,roi_box)
                draw_detections(output,detections)
                profile_annotate = timer()-start

            if bShowDebugImage:
                if debug_img.shape[0] == debug_img.shape[1]:
                    zero_img = np.full_like(debug_img,0.0)
                    debug_img = cv2.hconcat([debug_img,zero_img])
                debug_img = cv2.cvtColor(debug_img,cv2.COLOR_RGB2BGR)
                cv2.imshow(app_debug_title, debug_img)
                
            # display real-time FPS counter (if valid)
            if rt_fps_valid == True and bShowFPS:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)

            if bViewOutput:                
                # show the output image
                cv2.imshow(app_main_title, output)

            # Profiling
            if bProfileLog or bProfileView:
               prof_title[pipeline_id] = blaze_title
               prof_resize[pipeline_id]         = profile_resize
               prof_detector_pre[pipeline_id]   = blaze_detector.profile_pre
               prof_detector_model[pipeline_id] = blaze_detector.profile_model
               prof_detector_post[pipeline_id]  = blaze_detector.profile_post
               if len(normalized_detections) > 0:
                   prof_extract_roi[pipeline_id]    = profile_extract
                   prof_landmark_pre[pipeline_id]   = blaze_landmark.profile_pre
                   prof_landmark_model[pipeline_id] = blaze_landmark.profile_model
                   prof_landmark_post[pipeline_id]  = blaze_landmark.profile_post
                   prof_annotate[pipeline_id]       = profile_annotate
               #
               prof_total[pipeline_id] = profile_resize + \
                                         blaze_detector.profile_pre + \
                                         blaze_detector.profile_model + \
                                         blaze_detector.profile_post
               if len(normalized_detections) > 0:
                   prof_total[pipeline_id] += profile_extract + \
                                              blaze_landmark.profile_pre + \
                                              blaze_landmark.profile_model + \
                                              blaze_landmark.profile_post + \
                                              profile_annotate
               prof_fps[pipeline_id] = 1.0 / prof_total[pipeline_id]
            if bWrite:
                filename = ("blaze_detect_live_frame%04d_%s_input.tif"%(frame_count,blaze_title))
                print("Capturing ",filename," ...")
                input_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir,filename),input_img)

                filename = ("blaze_detect_live_frame%04d_%s_detection.tif"%(frame_count,blaze_title))
                print("Capturing ",filename," ...")
                cv2.imwrite(os.path.join(output_dir,filename),output)
        
                if bShowDebugImage:
                    filename = ("blaze_detect_live_frame%04d_%s_debug.tif"%(frame_count,blaze_title))
                    print("Capturing ",filename," ...")
                    cv2.imwrite(os.path.join(output_dir,filename),debug_img)

            if False:
               if len(normalized_detections) == 0:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)]"%(
                       profile_resize+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post
                       ))
               else:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)] Extract[(%001.06f)] Landmark[(%001.06f) (%001.06f) (%001.06f)]  Annotate[(%001.06f)]"%(
                       profile_resize+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post,
                       profile_extract,                       
                       blaze_landmark.profile_pre, blaze_landmark.profile_model, blaze_landmark.profile_post,
                       profile_annotate
                       ))
            
    if bProfileLog:            
        timestamp = datetime.now()
        for pipeline_id in range(nb_blaze_pipelines):
            if blaze_pipelines[pipeline_id]["supported"] and blaze_pipelines[pipeline_id]["selected"]:
                csv_str = \
                    str(timestamp)+","+\
                    str(user)+","+\
                    str(host)+","+\
                    blaze_pipelines[pipeline_id]["pipeline"]+","+\
                    str(prof_resize[pipeline_id])+","+\
                    str(prof_detector_pre[pipeline_id])+","+\
                    str(prof_detector_model[pipeline_id])+","+\
                    str(prof_detector_post[pipeline_id])+","+\
                    str(prof_extract_roi[pipeline_id])+","+\
                    str(prof_landmark_pre[pipeline_id])+","+\
                    str(prof_landmark_model[pipeline_id])+","+\
                    str(prof_landmark_post[pipeline_id])+","+\
                    str(prof_annotate[pipeline_id])+","+\
                    str(prof_total[pipeline_id])+","+\
                    str(prof_fps[pipeline_id])+"\n"
                #print("[LOG] ",csv_str)
                f_profile_csv.write(csv_str)
    
    
    if bProfileView:
        #
        # Latency
        #
        #prof_resize
        #prof_detector_pre
        #prof_detector_model
        #prof_detector_post
        #prof_extract_roi
        #prof_landmark_pre
        #prof_landmark_model
        #prof_landmark_post
        #prof_annotate

        # Create stacked bar chart
        fig = go.Figure(data=[
            go.Bar(name='resize'         , y=prof_title, x=prof_resize        , orientation='h'),
            go.Bar(name='detector[pre]'  , y=prof_title, x=prof_detector_pre  , orientation='h'),
            go.Bar(name='detector[model]', y=prof_title, x=prof_detector_model, orientation='h'),
            go.Bar(name='detector[post]' , y=prof_title, x=prof_detector_post , orientation='h'),
            go.Bar(name='extract_roi'    , y=prof_title, x=prof_extract_roi   , orientation='h'),
            go.Bar(name='landmark[pre]'  , y=prof_title, x=prof_landmark_pre  , orientation='h'),
            go.Bar(name='landmark[model]', y=prof_title, x=prof_landmark_model, orientation='h'),
            go.Bar(name='landmark[post]' , y=prof_title, x=prof_landmark_post , orientation='h'),
            go.Bar(name='annotate'       , y=prof_title, x=prof_annotate      , orientation='h')
        ])

        # Change the layout
        profile_latency_title = 'Latency (sec)'
        fig.update_layout(title=profile_latency_title,
                          xaxis_title='Latency',
                          yaxis_title='Pipeline',
                          legend_title="Component:",
                          #legend_traceorder="reversed",
                          barmode='stack')
                          #barmode='group')
                      
        # Show the plot
        #fig.show()   
        
        # Convert chart to image
        img_bytes = fig.to_image(format="png")
        img = np.array(bytearray(img_bytes), dtype=np.uint8)
        profile_latency_img = cv2.imdecode(img, -1)

        # Display or process the image using OpenCV or any other library
        cv2.imshow(profile_latency_title, profile_latency_img)                         

        if bWrite:
            filename = ("blaze_detect_live_frame%04d_profiling_latency.png"%(frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),profile_latency_img)

        #
        # FPS
        #
        #prof_total
        #prof_fps

        # Create stacked bar chart
        fig = go.Figure(data=[
            #go.Bar(name='latency' , y=prof_title, x=prof_total, orientation='h'),
            go.Bar(name='FPS'     , y=prof_title, x=prof_fps  , orientation='h')
        ])

        # Change the layout
        profile_fps_title = 'Performance (FPS)'
        fig.update_layout(title=profile_fps_title,
                          xaxis_title='FPS',
                          yaxis_title='Pipeline',
                          legend_title="Component:",
                          #legend_traceorder="reversed",
                          barmode='group')
                      
        # Show the plot
        #fig.show()   
        
        # Convert chart image
        img_bytes = fig.to_image(format="png")
        img = np.array(bytearray(img_bytes), dtype=np.uint8)
        profile_fps_img = cv2.imdecode(img, -1)

        # Display or process the image using OpenCV or any other library
        cv2.imshow(profile_fps_title, profile_fps_img)                         

        if bWrite:
            filename = ("blaze_detect_live_frame%04d_profiling_fps.png"%(frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),profile_fps_img)
            

    if bStep == True:
        key = cv2.waitKey(0)
    elif bPause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(1)

    #print(key)
    
    bWrite = False
    if key == 119: # 'w'
        bWrite = True

    if key == 115: # 's'
        bStep = True    
    
    if key == 112: # 'p'
        bPause = not bPause

    if key == 99: # 'c'
        bStep = False
        bPause = False
        
    if key == 116: # 't'
        bUseImage = not bUseImage  

    if key == 100: # 'd'
        bShowDebugImage = not bShowDebugImage  
        if not bShowDebugImage:
            for pipeline_id in range(nb_blaze_pipelines):
                if blaze_pipelines[pipeline_id]["supported"] and blaze_pipelines[pipeline_id]["selected"]:
                    blaze_title = blaze_pipelines[pipeline_id]["pipeline"]
                    app_debug_title = blaze_title+" Debug"
                    cv2.destroyWindow(app_debug_title)
           
    if key == 101: # 'e'
        bShowScores = not bShowScores
        blaze_detector.display_scores(debug=bShowScores)
        if not bShowScores:
           cv2.destroyWindow("Detection Scores (sigmoid)")

    if key == 102: # 'f'
        bShowFPS = not bShowFPS

    if key == 118: # 'v'
        bVerbose = not bVerbose
        for pipeline_id in range(nb_blaze_pipelines):
            if blaze_pipelines[pipeline_id]["supported"] and blaze_pipelines[pipeline_id]["selected"]:
                blaze_detector = blaze_pipelines[pipeline_id]["detector"]
                blaze_landmark = blaze_pipelines[pipeline_id]["landmark"]
                
                blaze_detector.set_debug(debug=bVerbose) 
                blaze_landmark.set_debug(debug=bVerbose)

    if key == 122: # 'z'
        bProfileLog = not bProfileLog

    if key == 90: # 'Z'
        bProfileView = not bProfileView 
        blaze_detector.set_profile(profile=bProfileView) 
        blaze_landmark.set_profile(profile=bProfileView)
        if not bProfileView:
            cv2.destroyWindow(profile_latency_title)
            cv2.destroyWindow(profile_fps_title)

    if key == 27 or key == 113: # ESC or 'q':
        break

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = 1
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #print("[INFO] ",rt_fps_message)
        rt_fps_count = 0

# Cleanup
f_profile_csv.close()
cv2.destroyAllWindows()
