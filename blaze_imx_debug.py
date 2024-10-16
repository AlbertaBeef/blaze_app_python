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

import matplotlib.pyplot as plt

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('blaze_common/'))
sys.path.append(os.path.abspath('blaze_tflite/'))
sys.path.append(os.path.abspath('blaze_tflite_quant/'))
sys.path.append(os.path.abspath('blaze_pytorch/'))
sys.path.append(os.path.abspath('blaze_vitisai/'))
sys.path.append(os.path.abspath('blaze_hailo/'))

from blaze_tflite.blazedetector import BlazeDetector as BlazeDetector_tflite
from blaze_tflite.blazelandmark import BlazeLandmark as BlazeLandmark_tflite
print("[INFO] blaze_tflite supported ...")

from blaze_tflite_quant.blazedetector import BlazeDetector as BlazeDetector_tflite_quant
from blaze_tflite_quant.blazelandmark import BlazeLandmark as BlazeLandmark_tflite_quant
print("[INFO] blaze_tflite_quant supported ...")

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
                # Start by looking for following format :
                #         "DPU Arch":"DPUCZDX8G_ISA0_B128_01000020E2012208",
                dpu_re_search = re.search('DPUCZDX8G_ISA0_(.+?)_', line)
                if dpu_re_search == None:
                    # else continue looking for following format :
                    #     "DPU Arch":"DPUCZDX8G_ISA1_B512_0101000016010200",
                    dpu_re_search = re.search('DPUCZDX8G_ISA1_(.+?)_', line)
                if dpu_re_search == None:
                    # else continue looking for following format :
                    #     "DPU Arch":"DPUCZDX8G_ISA1_B2304",
                    dpu_re_search = re.search('DPUCZDX8G_ISA1_(.+?)"', line)
                if dpu_re_search == None:
                    # else continue looking for following format :
                    #     "DPU Arch":"DPUCV2DX8G_ISA1_C20B1",
                    dpu_re_search = re.search('DPUCV2DX8G_ISA1_(.+?)"', line)
                dpu_arch = dpu_re_search.group(1)
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
ap.add_argument('-b', '--blaze',  type=str, default="hand", help="Application (hand, face, pose).  Default is hand")
ap.add_argument('-m', '--model1', type=str, help='Path of blazepalm model. Default is models/palm_detection_without_custom_op.tflite')
ap.add_argument('-n', '--model2', type=str, help='Path of blazehandlardmark model. Default is models/hand_landmark.tflite')
ap.add_argument('-d', '--debug'      , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-f', '--fps'        , default=False, action='store_true', help="Enable FPS display. Default is off")

args = ap.parse_args()  
  
print('Command line options:')
print(' --input       : ', args.input)
print(' --image       : ', args.image)
print(' --blaze       : ', args.blaze)
print(' --model1      : ', args.model1)
print(' --model2      : ', args.model2)
print(' --debug       : ', args.debug)
print(' --withoutview : ', args.withoutview)
print(' --fps         : ', args.fps)

nb_blaze_pipelines = 1
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

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist



default_detector_model = "blaze_tflite/models/pose_detection.tflite"
default_landmark_model = "blaze_tflite/models/pose_landmark_full.tflite"

debug_detector_model = "blaze_tflite_quant/models/pose_detection_full_quant.tflite"
debug_landmark_model = "blaze_tflite_quant/models/pose_landmark_full_quant.tflite"

blaze_detector_type = "blazepose"
blaze_landmark_type = "blazeposelandmark"
blaze_title = "BlazePoseLandmark"

if args.model1 == None:
   args.model1 = default_detector_model
if args.model2 == None:
   args.model2 = default_landmark_model

blaze_detector = BlazeDetector_tflite(blaze_detector_type)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(default_detector_model)

blaze_landmark = BlazeLandmark_tflite(blaze_landmark_type)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(default_landmark_model)

blaze_detector2 = BlazeDetector_tflite_quant(blaze_detector_type)
blaze_detector2.set_debug(debug=args.debug)
blaze_detector2.display_scores(debug=False)
blaze_detector2.load_model(debug_detector_model)

blaze_landmark2 = BlazeLandmark_tflite_quant(blaze_landmark_type)
blaze_landmark2.set_debug(debug=args.debug)
blaze_landmark2.load_model(debug_landmark_model)


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

def ignore(x):
    pass

if bViewOutput:
    app_main_title = blaze_title+" Reference"
    app_ctrl_title = blaze_title+" Reference"
    app_debug_title = blaze_title+" Reference ROIs"
    cv2.namedWindow(app_main_title)

    thresh_min_score = blaze_detector.min_score_thresh
    thresh_min_score_prev = thresh_min_score
    cv2.createTrackbar('threshMinScore', app_ctrl_title, int(thresh_min_score*100), 100, ignore)

    thresh_nms = blaze_detector.min_suppression_threshold
    thresh_nms_prev = thresh_nms
    cv2.createTrackbar('threshNMS', app_ctrl_title, int(thresh_nms*100), 100, ignore)

    app_main_title2 = blaze_title+" Debug"
    app_ctrl_title2 = blaze_title+" Debug"
    app_debug_title2 = blaze_title+" Debug ROIs"
    cv2.namedWindow(app_main_title2)

    thresh_min_score2 = blaze_detector2.min_score_thresh
    thresh_min_score2_prev = thresh_min_score2
    cv2.createTrackbar('threshMinScore', app_ctrl_title2, int(thresh_min_score2*100), 100, ignore)

    thresh_nms2 = blaze_detector2.min_suppression_threshold
    thresh_nms2_prev = thresh_nms2
    cv2.createTrackbar('threshNMS', app_ctrl_title2, int(thresh_nms2*100), 100, ignore)

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

    if True:    
        pipeline_id = 0
        if True:
            image = frame.copy()
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

                thresh_nms = cv2.getTrackbarPos('threshNMS', app_ctrl_title)
                if thresh_nms < 10:
                    thresh_nms = 10
                    cv2.setTrackbarPos('threshNMS', app_ctrl_title,thresh_nms)
                thresh_nms = thresh_nms*(1/100)
                if thresh_nms != thresh_nms_prev:
                    blaze_detector.min_suppression_threshold = thresh_nms
                    thresh_nms_prev = thresh_nms

                thresh_min_score2 = cv2.getTrackbarPos('threshMinScore', app_ctrl_title2)
                if thresh_min_score2 < 10:
                    thresh_min_score2 = 10
                    cv2.setTrackbarPos('threshMinScore', app_ctrl_title2,thresh_min_score2)
                thresh_min_score2 = thresh_min_score2*(1/100)
                if thresh_min_score2 != thresh_min_score2_prev:
                    blaze_detector2.min_score_thresh = thresh_min_score2
                    thresh_min_score2_prev = thresh_min_score2

                thresh_nms2 = cv2.getTrackbarPos('threshNMS', app_ctrl_title2)
                if thresh_nms2 < 10:
                    thresh_nms2 = 10
                    cv2.setTrackbarPos('threshNMS', app_ctrl_title2,thresh_nms2)
                thresh_nms2 = thresh_nms2*(1/100)
                if thresh_nms2 != thresh_nms2_prev:
                    blaze_detector2.min_suppression_threshold = thresh_nms2
                    thresh_nms2_prev = thresh_nms2                
                
            #image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            output2 = image.copy()
            
            # BlazePalm pipeline
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img1,scale1,pad1=blaze_detector.resize_pad(image)

            if bShowDebugImage:
                # show the resized input image
                debug_img = img1.astype(np.float32)/255.0
                debug_img = cv2.resize(debug_img2,(blaze_landmark.resolution,blaze_landmark.resolution))
                #
                debug_img2 = img1.astype(np.float32)/255.0
                debug_img2 = cv2.resize(debug_img2,(blaze_landmark2.resolution,blaze_landmark2.resolution))

            
            out1_reference,out2_reference = blaze_detector.predict_core(np.expand_dims(img1, axis=0))
            detection_boxes_reference = blaze_detector._decode_boxes(out2_reference, blaze_detector.anchors)
            #thresh = blaze_detector.score_clipping_thresh
            #clipped_score_tensor = np.clip(out1_reference,-thresh,thresh)
            #detection_scores = 1/(1 + np.exp(-clipped_score_tensor))
            detection_scores = 1/(1 + np.exp(-out1_reference))
            detection_scores_reference = np.squeeze(detection_scores, axis=-1)        

            out1_debug,out2_debug = blaze_detector2.predict_core(np.expand_dims(img1, axis=0))
            detection_boxes_debug = blaze_detector2._decode_boxes(out2_debug, blaze_detector2.anchors)
            #thresh = blaze_detector2.score_clipping_thresh
            #clipped_score_tensor = np.clip(out1_debug,-thresh,thresh)
            #detection_scores = 1/(1 + np.exp(-clipped_score_tensor))
            detection_scores = 1/(1 + np.exp(-out1_debug))
            detection_scores_debug = np.squeeze(detection_scores, axis=-1)        
            

            Y1 = out1_reference.reshape(-1)
            Y2 = out1_debug.reshape(-1)
            X12 = range(0,len(Y1))
            #
            Y3 = detection_scores_reference.reshape(-1)
            Y4 = detection_scores_debug.reshape(-1)
            X34 = range(0,len(Y3))
            #
            Y5 = out2_reference.reshape(-1)
            Y6 = out2_debug.reshape(-1)
            X56 = range(0,len(Y5))
            #
            figure, axis = plt.subplots(2, 3)
            axis[0, 0].plot(X12, Y1)
            axis[0, 0].set_title("Scores (reference)")
            axis[1, 0].plot(X12, Y2)
            axis[1, 0].set_title("Scores (debug)")
            axis[0, 1].plot(X34, Y3)
            axis[0, 1].set_title("Sigmoid (reference)")
            axis[1, 1].plot(X34, Y4)
            axis[1, 1].set_title("Sigmoid (debug)")
            axis[0, 2].plot(X56, Y5)
            axis[0, 2].set_title("BBoxes (reference)")
            axis[1, 2].plot(X56, Y6)
            axis[1, 2].set_title("BBoxes (debug)")
            plt.show()            
            
            normalized_detections = blaze_detector.predict_on_image(img1)
            if len(normalized_detections) > 0:
  
                detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector.detection2roi(detections)
                roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)

                flags, normalized_landmarks = blaze_landmark.predict(roi_img)

                if bShowDebugImage:
                    # show the ROIs
                    for i in range(roi_img.shape[0]):
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

            if bShowDebugImage:
                if debug_img.shape[0] == debug_img.shape[1]:
                    zero_img = np.full_like(debug_img,0.0)
                    debug_img = cv2.hconcat([debug_img,zero_img])
                debug_img = cv2.cvtColor(debug_img,cv2.COLOR_RGB2BGR)
                cv2.imshow(app_debug_title, debug_img)

            normalized_detections2 = blaze_detector2.predict_on_image(img1)
            if len(normalized_detections2) > 0:
  
                detections2 = blaze_detector2.denormalize_detections(normalized_detections2,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector2.detection2roi(detections2)
                roi_img2,roi_affine2,roi_box2 = blaze_landmark2.extract_roi(image,xc,yc,theta,scale)

                flags2, normalized_landmarks2 = blaze_landmark2.predict(roi_img2)

                if bShowDebugImage:
                    # show the ROIs
                    for i in range(roi_img.shape[0]):
                        roi_landmarks = normalized_landmarks2[i,:,:].copy()
                        roi_landmarks = roi_landmarks*blaze_landmark2.resolution
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

                landmarks2 = blaze_landmark.denormalize_landmarks(normalized_landmarks2, roi_affine2)

                for i in range(len(flags)):
                    landmark, flag = landmarks2[i], flags2[i]
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
                   
                draw_roi(output2,roi_box2)
                draw_detections(output2,detections2)

            if bShowDebugImage:
                if debug_img2.shape[0] == debug_img2.shape[1]:
                    zero_img2 = np.full_like(debug_img2,0.0)
                    debug_img2 = cv2.hconcat([debug_img2,zero_img2])
                debug_img2 = cv2.cvtColor(debug_img2,cv2.COLOR_RGB2BGR)
                cv2.imshow(app_debug_title2, debug_img2)
                
            # display real-time FPS counter (if valid)
            if rt_fps_valid == True and bShowFPS:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)

            if bViewOutput:                
                # show the output image
                cv2.imshow(app_main_title, output)
                cv2.imshow(app_main_title2, output2)

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
        blaze_detector.set_debug(debug=bVerbose) 
        blaze_landmark.set_debug(debug=bVerbose)
        blaze_detector2.set_debug(debug=bVerbose) 
        blaze_landmark2.set_debug(debug=bVerbose)

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
cv2.destroyAllWindows()
