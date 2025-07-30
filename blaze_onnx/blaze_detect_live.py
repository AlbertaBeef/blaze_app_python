'''
Copyright 2025 Tria Technologies Inc.
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
#   ONNX
#      onnxruntime
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

import getpass
import socket
user = getpass.getuser()
host = socket.gethostname()
user_host_descriptor = user+"@"+host
print("[INFO] user@hosthame : ",user_host_descriptor)

sys.path.append(os.path.abspath('../blaze_common/'))
from blazedetector import BlazeDetector
from blazelandmark import BlazeLandmark

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS
from visualization import draw_detection_scores
from visualization import draw_stacked_bar_chart, stacked_bar_latency_colors, stacked_bar_performance_colors
from utils_linux import get_media_dev_by_name, get_video_dev_by_name

from timeit import default_timer as timer


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
ap.add_argument('-I', '--testimage'  , default=False, action='store_true', help="Use test image as input (womand_hands.jpg). Default is usbcam")
ap.add_argument('-b', '--blaze',  type=str, default="hand", help="Application (hand, face, pose).  Default is hand")
ap.add_argument('-m', '--model1', type=str, help='Path of blazepalm model. Default is models/palm_detection_lite/model_float32.onnx')
ap.add_argument('-n', '--model2', type=str, help='Path of blazehandlardmark model. Default is models/hand_landmark_lite/model_float32.onnx')
ap.add_argument('-d', '--debug'      , default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-z', '--profilelog' , default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
ap.add_argument('-y', '--profileview', default=False, action='store_true', help="Enable Profile View (Latency). Default is off")
ap.add_argument('-f', '--fps'        , default=False, action='store_true', help="Enable FPS display. Default is off")

args = ap.parse_args()  
  
print('Command line options:')
print(' --input       : ', args.input)
print(' --testimage   : ', args.testimage)
print(' --blaze       : ', args.blaze)
print(' --model1      : ', args.model1)
print(' --model2      : ', args.model2)
print(' --debug       : ', args.debug)
print(' --withoutview : ', args.withoutview)
print(' --profilelog  : ', args.profilelog)
print(' --profileview : ', args.profileview)
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

profile_csv = './blaze_detect_live.csv'
if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :",profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :",profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,detector_qty,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist



if args.blaze == "hand":
   blaze_detector_type = "blazepalm"
   blaze_landmark_type = "blazehandlandmark"
   blaze_title = "BlazeHandLandmark"
   #default_detector_model='models/palm_detection_lite/model_float32.onnx'
   default_detector_model='models/palm_detection_v0_07/model_float32.onnx'
   default_landmark_model='models/hand_landmark_lite/model_float32.onnx'
#elif args.blaze == "face":
#   blaze_detector_type = "blazeface"
#   blaze_landmark_type = "blazefacelandmark"
#   blaze_title = "BlazeFaceLandmark"
#   default_detector_model='models/face_detection_short_range.tflite'
#   default_landmark_model='models/face_landmark.tflite'
#elif args.blaze == "pose":
#   blaze_detector_type = "blazepose"
#   blaze_landmark_type = "blazeposelandmark"
#   blaze_title = "BlazePoseLandmark"
#   default_detector_model='models/pose_detection.tflite'
#   default_landmark_model='models/pose_landmark_full.tflite'
else:
   print("[ERROR] Invalid Blaze application : ",args.blaze,".  MUST be one of hand,face,pose.")

if args.model1 == None:
   args.model1 = default_detector_model
if args.model2 == None:
   args.model2 = default_landmark_model

blaze_detector = BlazeDetector(blaze_detector_type)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.load_model(args.model1)

blaze_landmark = BlazeLandmark(blaze_landmark_type)
blaze_landmark.set_debug(debug=args.debug)
#blaze_landmark.load_model(args.model2)

thresh_min_score = blaze_detector.min_score_thresh
thresh_min_score_prev = thresh_min_score

thresh_nms = blaze_detector.min_suppression_threshold
thresh_nms_prev = thresh_nms

thresh_confidence = 0.5
thresh_confidence_prev = thresh_confidence

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
print("\tPress 't' to toggle between test image and live video")
print("\tPress 'h' to toggle horizontal mirror on input")
print("\tPress 'a' to toggle detection overlay on/off")
print("\tPress 'b' to toggle roi overlay on/off")
print("\tPress 'l' to toggle landmarks overlay on/off")
print("\tPress 'd' to toggle debug image on/off")
print("\tPress 'e' to toggle scores image on/off")
print("\tPress 'f' to toggle FPS display on/off")
print("\tPress 'v' to toggle verbose on/off")
print("\tPress 'z' to toggle profiling log on/off")
print("\tPress 'y' to toggle profiling view on/off")
print("================================================================")

bStep = False
bPause = False
bWrite = False

bUseImage = args.testimage
bMirrorImage = False
bShowDetection = True
bShowExtractROI = True
bShowLandmarks = True
bShowDebugImage = False
bShowScores = False
bShowFPS = args.fps
bVerbose = args.debug
bViewOutput = not args.withoutview
bProfileLog = args.profilelog
bProfileView = args.profileview

def ignore(x):
    pass

if bViewOutput:
    app_main_title = blaze_title+" Demo"
    app_ctrl_title = blaze_title+" Demo"
    app_debug_title = blaze_title+" Debug"
    cv2.namedWindow(app_main_title)

    cv2.createTrackbar('threshMinScore', app_ctrl_title, int(thresh_min_score*100), 100, ignore)
    print("[INFO] thresh_min_score=",thresh_min_score)

    cv2.createTrackbar('threshNMS', app_ctrl_title, int(thresh_nms*100), 100, ignore)
    print("[INFO] thresh_nms=",thresh_nms)

    cv2.createTrackbar('threshConfidence', app_ctrl_title, int(thresh_confidence*100), 100, ignore)
    print("[INFO] thresh_confidence=",thresh_confidence)

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
        frame = cv2.imread('../woman_hands.jpg')
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

    if bMirrorImage:
        # Mirror horizontally for selfie-mode
        frame = cv2.flip(frame, 1) 

    if bProfileLog or bProfileView:
        prof_title          = ['']*nb_blaze_pipelines
        prof_detector_qty   = np.zeros(nb_blaze_pipelines)
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
        #
        profile_latency_title     = "Latency (sec)"
        profile_performance_title = "Performance (FPS)"

    if True:    
        pipeline_id = 0
        if True:
            image = frame.copy()

            app_scores_title = blaze_title+" Detection Scores (sigmoid)" 

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
                    print("[INFO] thresh_min_score=",thresh_min_score)

                thresh_nms = cv2.getTrackbarPos('threshNMS', app_ctrl_title)
                if thresh_nms > 99:
                    thresh_nms = 99
                    cv2.setTrackbarPos('threshNMS', app_ctrl_title,thresh_nms)
                thresh_nms = thresh_nms*(1/100)
                if thresh_nms != thresh_nms_prev:
                    blaze_detector.min_suppression_threshold = thresh_nms
                    thresh_nms_prev = thresh_nms
                    print("[INFO] thresh_nms=",thresh_nms)

                thresh_confidence = cv2.getTrackbarPos('threshConfidence', app_ctrl_title)
                thresh_confidence = thresh_confidence*(1/100)
                if thresh_confidence != thresh_confidence_prev:
                    thresh_confidence_prev = thresh_confidence
                    print("[INFO] thresh_confidence=",thresh_confidence)


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
            
            if bShowScores:
                detection_scores_chart = draw_detection_scores( blaze_detector.detection_scores, blaze_detector.min_score_thresh );
                cv2.imshow(app_scores_title,detection_scores_chart);

            profile_detector_qty = len(normalized_detections)
            
            #if len(normalized_detections) > 0:
            if False:
  
                start = timer()          
                detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector.detection2roi(detections)
                roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
                profile_extract = timer()-start

                landmark_results = blaze_landmark.predict(roi_img)
                handedness_results = []
                if len(landmark_results) == 3:
                    flags, normalized_landmarks, handedness = landmark_results
                    # process handedness
                    for i in range(len(handedness)):
                        # mediapipe expects mirror image, need to invert result otherwise
                        if bMirrorImage == False:
                            handedness[i] = 1.0 - handedness[i]
                        # handedness 
                        if handedness[i] > 0.5:
                            # left => 1.0
                            handedness_results.append("left")
                        else:
                            # right => 0.0
                            handedness_results.append("right")
                else:
                    flags, normalized_landmarks = landmark_results

                if bShowDebugImage:
                    # show the ROIs
                    for i in range(roi_img.shape[0]):
                        #roi_landmarks = np.expand_dims(normalized_landmarks[i,:,:].copy(), axis=0)
                        roi_landmarks = normalized_landmarks[i,:,:].copy()
                        roi_landmarks = roi_landmarks*blaze_landmark.resolution
                        if blaze_landmark_type == "blazehandlandmark":
                            if len(handedness_results) == 0:
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], HAND_CONNECTIONS, size=2, color=(0, 255, 0)) # green (RGB format)
                            elif handedness_results[i] == "left":
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], HAND_CONNECTIONS, size=2, color=(0, 255, 0)) # green (RGB format)
                            else:
                                draw_landmarks(roi_img[i], roi_landmarks[:,:2], HAND_CONNECTIONS, size=2, color=(0, 161, 190)) # aqua (RGB format)
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

                if bShowLandmarks:
                    for i in range(len(flags)):
                        landmark, flag = landmarks[i], flags[i]
                        if flag > thresh_confidence:
                            if blaze_landmark_type == "blazehandlandmark":
                                if len(handedness_results) == 0:
                                    draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2, color=(0, 255, 0)) # green (BGR format)
                                elif handedness_results[i] == "left":                                    
                                    draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2, color=(0, 255, 0)) # green (BGR format)
                                else:
                                    draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2, color=(190, 161, 0)) # aqua (BGR format)
                            elif blaze_landmark_type == "blazefacelandmark":
                                draw_landmarks(output, landmark[:,:2], FACE_CONNECTIONS, size=1)                                    
                            elif blaze_landmark_type == "blazeposelandmark":
                                if landmarks.shape[1] > 33:
                                    draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                                else:
                                    draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)                

                if bShowExtractROI:
                    draw_roi(output,roi_box)
                if bShowDetection:
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
               prof_detector_qty[pipeline_id]   = profile_detector_qty
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
        pipeline_id = 0
        if True:
                csv_str = \
                    str(timestamp)+","+\
                    str(user)+","+\
                    str(host)+","+\
                    "blaze_onnx"+","+\
                    str(prof_detector_qty[pipeline_id])+","+\
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
        component_labels = [
            "resize",
            "detector[pre]",
            "detector[model]",
            "detector[post]",
            "extract_roi",
            "landmark[pre]",
            "landmark[model]",
            "landmark[post]",
            "annotate"
        ]
        component_idx = [i for i, s in enumerate(prof_title) if s]
        #print("[INFO] prof_title=",prof_title)
        #print("[INFO] component_idx=",component_idx)
        pipeline_titles = [prof_title[i] for i in component_idx]
        component_values=[
            prof_resize[component_idx],
            prof_detector_pre[component_idx],
            prof_detector_model[component_idx],
            prof_detector_post[component_idx],
            prof_extract_roi[component_idx],
            prof_landmark_pre[component_idx],
            prof_landmark_model[component_idx],
            prof_landmark_post[component_idx],
            prof_annotate[component_idx]
        ]
        profile_latency_chart = draw_stacked_bar_chart(
            pipeline_titles=pipeline_titles,
            component_labels=component_labels,
            component_values=component_values,
            component_colors=stacked_bar_latency_colors,
            chart_name=profile_latency_title
        )

        # Display or process the image using OpenCV or any other library
        cv2.imshow(profile_latency_title, profile_latency_chart)                         

        if bWrite:
            filename = ("blaze_detect_live_frame%04d_profiling_latency.png"%(frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),profile_latency_chart)

        #
        # FPS
        #

        component_labels = [
            "fps"
        ]
        component_values=[
            prof_fps[component_idx]
        ]        
        profile_performance_chart = draw_stacked_bar_chart(
            pipeline_titles=pipeline_titles,
            component_labels=component_labels,
            component_values=component_values,
            component_colors=stacked_bar_performance_colors,
            chart_name=profile_performance_title
        )

        # Display or process the image using OpenCV or any other library
        cv2.imshow(profile_performance_title, profile_performance_chart)                         

        if bWrite:
            filename = ("blaze_detect_live_frame%04d_profiling_performance.png"%(frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),profile_performance_chart)
            

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
        print("[INFO] bUseImage=",bUseImage)

    if key == 104: # 'h'
        bMirrorImage = not bMirrorImage  
        print("[INFO] bMirrorImage=",bMirrorImage)

    if key == 97: # 'a'
        bShowDetection = not bShowDetection  
        print("[INFO] bShowDetection=",bShowDetection)

    if key == 98: # 'b'
        bShowExtractROI = not bShowExtractROI  
        print("[INFO] bShowExtractROI=",bShowExtractROI)

    if key == 108: # 'l'
        bShowLandmarks = not bShowLandmarks  
        print("[INFO] bShowLandmarks=",bShowLandmarks)

    if key == 100: # 'd'
        bShowDebugImage = not bShowDebugImage  
        print("[INFO] bShowDebugImage=",bShowDebugImage)
        if not bShowDebugImage:
           cv2.destroyWindow(app_debug_title)
           
    if key == 101: # 'e'
        bShowScores = not bShowScores
        print("[INFO] bShowScores=",bShowScores)
        if not bShowScores:
           cv2.destroyWindow(app_scores_title)

    if key == 102: # 'f'
        bShowFPS = not bShowFPS
        print("[INFO] bShowFPS=",bShowFPS)

    if key == 118: # 'v'
        bVerbose = not bVerbose
        blaze_detector.set_debug(debug=bVerbose) 
        blaze_landmark.set_debug(debug=bVerbose)

    if key == 122: # 'z'
        bProfileLog = not bProfileLog
        print("[INFO] bProfileLog=",bProfileLog)

    if key == 121: # 'y'
        bProfileView = not bProfileView 
        print("[INFO] bProfileView=",bProfileView)
        blaze_detector.set_profile(profile=bProfileView) 
        blaze_landmark.set_profile(profile=bProfileView)
        if not bProfileView:
            cv2.destroyWindow(profile_latency_title)
            cv2.destroyWindow(profile_performance_title)

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
