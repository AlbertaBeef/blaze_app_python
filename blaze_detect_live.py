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

sys.path.append(os.path.abspath('blaze_common/'))
sys.path.append(os.path.abspath('blaze_tflite/'))
sys.path.append(os.path.abspath('blaze_pytorch/'))
sys.path.append(os.path.abspath('blaze_vitisai/'))

supported_targets = {
    "blaze_tflite": False,
    "blaze_pytorch": False,
    "blaze_vitisai": False
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
    print("[INFO] blaze_pytorch NOT found ...")

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

print("[INFO] Searching for USB camera ...")
dev_video = get_video_dev_by_name("uvcvideo")
dev_media = get_media_dev_by_name("uvcvideo")
print(dev_video)
print(dev_media)

#input_video = 0 
input_video = dev_video  
print("[INFO] Input Video : ",input_video)

output_dir = './captured-images'

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist

# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("camera",input_video," (",frame_width,",",frame_height,")")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--blaze',  type=str, default="hand,face,pose", help="Application (hand, face, pose).  Default is hand,face,port")
ap.add_argument('-d', '--debug',  type=bool, default=False, help = 'Debug mode. Default is off')
ap.add_argument('-p', '--profile',type=bool, default=False, help = 'Profile mode. Default is off')

args = ap.parse_args()  
  
print('Command line options:')
print(' --blaze   : ', args.blaze)
print(' --debug   : ', args.debug)
print(' --profile : ', args.profile)


blaze_models = [
    { "blaze": "hand", "model1": "blaze_tflite/models/palm_detection_without_custom_op.tflite", "model2": "blaze_tflite/models/hand_landmark.tflite" },
    { "blaze": "hand", "model1": "blaze_tflite/models/palm_detection_lite.tflite",              "model2": "blaze_tflite/models/hand_landmark_lite.tflite" },
    { "blaze": "hand", "model1": "blaze_tflite/models/palm_detection_full.tflite",              "model2": "blaze_tflite/models/hand_landmark_full.tflite" },
    { "blaze": "hand", "model1": "blaze_pytorch/models/blazepalm.pth",                          "model2": "blaze_pytorch/models/blazehand_landmark.pth" },
    { "blaze": "hand", "model1": "blaze_vitisai/models/blazepalm/"+dpu_arch+"/blazepalm.xmodel","model2": "blaze_vitisai/models/blazehandlandmark/"+dpu_arch+"/blazehandlandmark.xmodel" },
    { "blaze": "face", "model1": "blaze_tflite/models/face_detection_short_range.tflite",       "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "model1": "blaze_tflite/models/face_detection_full_range.tflite",        "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "model1": "blaze_tflite/models/face_detection_full_range_sparse.tflite", "model2": "blaze_tflite/models/face_landmark.tflite" },
    { "blaze": "face", "model1": "blaze_pytorch/models/blazeface.pth",                          "model2": "blaze_pytorch/models/blazeface_landmark.pth" },
    { "blaze": "pose", "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_lite.tflite" },
    { "blaze": "pose", "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_full.tflite" },
    { "blaze": "pose", "model1": "blaze_tflite/models/pose_detection.tflite",                   "model2": "blaze_tflite/models/pose_landmark_heavy.tflite" },
    { "blaze": "pose", "model1": "blaze_pytorch/models/blazepose.pth",                          "model2": "blaze_pytorch/models/blazepose_landmark.pth" }
]

hand_models = []
face_models = []
pose_models = []
for i in range(len(blaze_models)):
    blaze  = blaze_models[i]["blaze"]
    model1 = blaze_models[i]["model1"]
    model2 = blaze_models[i]["model2"]
    #print("[INFO] ",blaze,model1,model2)
   
    target1 = re.search('(.+?)/', model1).group(1) 
    target2 = re.search('(.+?)/', model1).group(1)
    if supported_targets[target1]==True and supported_targets[target2]==True:
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
        else:
            print("[ERROR] Invalid target : ",target1,".  MUST be a valid blaze_* directory.")
        blaze_landmark.load_model(model2)
       
        blaze_models[i]["detector"] = blaze_detector
        blaze_models[i]["landmark"] = blaze_landmark
        
        if blaze=="hand":
            hand_models.append(blaze_models[i])
        elif blaze=="face":
            face_models.append(blaze_models[i])
        elif blaze=="pose":
            pose_models.append(blaze_models[i])
 
        print("[INFO] ",blaze,model1,model2)

nb_hand_models = len(hand_models)
nb_face_models = len(face_models)
nb_pose_models = len(pose_models)

hand_model_id = 0
face_model_id = 0
pose_model_id = 0

def ignore(x):
    pass

blaze_list = { "hand", "face", "pose" }
for i,blaze in enumerate(blaze_list):

    if blaze in args.blaze:        

            if blaze == "hand":    
                blaze_detector_type = "blazepalm"
                blaze_landmark_type = "blazehandlandmark"
                blaze_title = "BlazeHandLandmark"
                blaze_detector = hand_models[hand_model_id]["detector"]
                blaze_landmark = hand_models[hand_model_id]["landmark"]
                
            elif blaze == "face":
                blaze_detector_type = "blazeface"
                blaze_landmark_type = "blazefacelandmark"
                blaze_title = "BlazeFaceLandmark"
                blaze_detector = face_models[face_model_id]["detector"]
                blaze_landmark = face_models[face_model_id]["landmark"]

            elif blaze == "pose":
                blaze_detector_type = "blazepose"
                blaze_landmark_type = "blazeposelandmark"
                blaze_title = "BlazePoseLandmark"
                blaze_detector = pose_models[pose_model_id]["detector"]
                blaze_landmark = pose_models[pose_model_id]["landmark"]
                 
            app_main_title = blaze_title+" Demo"
            app_ctrl_title = blaze_title+" Demo"
            cv2.namedWindow(app_main_title)

            thresh_min_score = blaze_detector.min_score_thresh
            thresh_min_score_prev = thresh_min_score
            cv2.createTrackbar('threshMinScore', app_ctrl_title, int(thresh_min_score*100), 100, ignore)


print("================================================================")
print(app_main_title)
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
print("\tPress 'z' to toggle profiling on/off")
print("================================================================")

bStep = False
bPause = False
bUseImage = False
bShowDebugImage = False
bShowScores = False
bShowFPS = False
bVerbose = args.debug
bProfile = args.profile

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
    #flag, frame = cap.retrieve()
    flag, frame = cap.read()
    if not flag:
        break
                
    blaze_list = { "hand", "face", "pose" }
    for i,blaze in enumerate(blaze_list):

        if blaze in args.blaze:        
            
            image = frame.copy()
            
            if blaze == "hand":    
                blaze_detector_type = "blazepalm"
                blaze_landmark_type = "blazehandlandmark"
                blaze_title = "BlazeHandLandmark"
                blaze_detector = hand_models[hand_model_id]["detector"]
                blaze_landmark = hand_models[hand_model_id]["landmark"]
                
            elif blaze == "face":
                blaze_detector_type = "blazeface"
                blaze_landmark_type = "blazefacelandmark"
                blaze_title = "BlazeFaceLandmark"
                blaze_detector = face_models[face_model_id]["detector"]
                blaze_landmark = face_models[face_model_id]["landmark"]

            elif blaze == "pose":
                blaze_detector_type = "blazepose"
                blaze_landmark_type = "blazeposelandmark"
                blaze_title = "BlazePoseLandmark"
                blaze_detector = pose_models[pose_model_id]["detector"]
                blaze_landmark = pose_models[pose_model_id]["landmark"]
             
            app_main_title = blaze_title+" Demo"
            app_ctrl_title = blaze_title+" Demo"
            app_debug_title = blaze_title+" Debug"
            
            # Get trackbar values
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
            profile_pre = timer()-start

            if bShowDebugImage:
                # show the resized input image
                debug_img = img1.astype(np.float32)/255.0
                debug_img = cv2.resize(debug_img,(blaze_landmark.resolution,blaze_landmark.resolution))
            
            normalized_detections = blaze_detector.predict_on_image(img1)
            if len(normalized_detections) > 0:
  
                start = timer()          
                palm_detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
                xc,yc,scale,theta = blaze_detector.detection2roi(palm_detections)
                hand_img,hand_affine,hand_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
                profile_extract = timer()-start

                if bShowDebugImage:
                    # show the ROIs
                    for i in range(hand_img.shape[0]):
                        debug_img = cv2.hconcat([debug_img,hand_img[i]])
                
                flags, normalized_landmarks = blaze_landmark.predict(hand_img)
                start = timer() 
                landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, hand_affine)

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
                   
                draw_roi(output,hand_box)
                draw_detections(output,palm_detections)
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
                
            # Profiling
            if bProfile:
               if len(normalized_detections) == 0:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)]"%(
                       profile_pre+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post
                       ))
               else:
                   print("[PROFILE] Detector[(%001.06f) (%001.06f) (%001.06f)] Extract[(%001.06f)] Landmark[(%001.06f) (%001.06f) (%001.06f)]  Annotate[(%001.06f)]"%(
                       profile_pre+blaze_detector.profile_pre, blaze_detector.profile_model, blaze_detector.profile_post,
                       profile_extract,                       
                       blaze_landmark.profile_pre, blaze_landmark.profile_model, blaze_landmark.profile_post,
                       profile_annotate
                       ))
            
            # show the output image
            cv2.imshow(app_main_title, output)

    if bStep == True:
        key = cv2.waitKey(0)
    elif bPause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(1)

    #print(key)
    
    if key == 119: # 'w'
        filename = ("%s_frame%04d_input.tif"%(blaze_landmark_type,frame_count))
        print("Capturing ",filename," ...")
        input_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir,filename),input_img)

        filename = ("%s_frame%04d_detection.tif"%(blaze_landmark_type,frame_count))
        print("Capturing ",filename," ...")
        cv2.imwrite(os.path.join(output_dir,filename),output)
        
        if bShowDebugImage:
            filename = ("%s_frame%04d_debug.tif"%(blaze_landmark_type,frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),debug_img)
      
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
            for i,blaze in enumerate(blaze_list):
                if blaze in args.blaze:        
                    if blaze == "hand":    
                        blaze_title = "BlazeHandLandmark"
                    elif blaze == "face":
                        blaze_title = "BlazeFaceLandmark"
                    elif blaze == "pose":
                        blaze_title = "BlazePoseLandmark"
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
        blaze_detector.set_debug(debug=bVerbose) 
        blaze_landmark.set_debug(debug=bVerbose)

    if key == 122: # 'z'
        bProfile = not bProfile 
        blaze_detector.set_profile(profile=bProfile) 
        blaze_landmark.set_profile(profile=bProfile)

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
