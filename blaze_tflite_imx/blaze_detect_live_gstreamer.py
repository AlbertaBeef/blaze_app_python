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

import numpy as np
import cv2
import os
from datetime import datetime
import itertools
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstBase, GstVideo

from ctypes import *
from typing import List
import pathlib
import time
import sys
import argparse
import glob
import subprocess
import re
import signal
from datetime import datetime

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

# Initialize GStreamer
Gst.init(None)

class GstDisplay:
    def __init__(self, width, height, title="Output"):
        self.width = width
        self.height = height
        self.title = title
        
        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline.new("display-pipeline")
        
        # Create elements
        self.appsrc = Gst.ElementFactory.make("appsrc", "source")
        self.videoconvert1 = Gst.ElementFactory.make("videoconvert", "convert1")
        self.rotate = Gst.ElementFactory.make("videoflip", "rotate")
        self.rotate.set_property("method", "clockwise")  # 90° left rotation
        self.videoconvert2 = Gst.ElementFactory.make("videoconvert", "convert2")
        self.autovideosink = Gst.ElementFactory.make("autovideosink", "sink")
        
        if not all([self.pipeline, self.appsrc, self.videoconvert1, self.rotate, 
                   self.videoconvert2, self.autovideosink]):
            print("ERROR: Could not create GStreamer elements")
            return
        
        # Configure appsrc
        self.appsrc.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height},framerate=30/1"))
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", True)
        
        # Add elements to pipeline
        self.pipeline.add(self.appsrc)
        self.pipeline.add(self.videoconvert1)
        self.pipeline.add(self.rotate)
        self.pipeline.add(self.videoconvert2)
        self.pipeline.add(self.autovideosink)
        
        # Link elements
        self.appsrc.link(self.videoconvert1)
        self.videoconvert1.link(self.rotate)
        self.rotate.link(self.videoconvert2)
        self.videoconvert2.link(self.autovideosink)
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline started")
    
    def display_frame(self, frame):
        # Convert frame to bytes
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
            
        buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        self.appsrc.emit("push-buffer", buffer)
    
    def close(self):
        self.pipeline.set_state(Gst.State.NULL)

# Parameters
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max(1, int(2*scale))
text_lineType = cv2.LINE_AA

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, default="", help="Video input device. Default is auto-detect (first usbcam)")
ap.add_argument('-I', '--image', default=False, action='store_true', help="Use 'woman_hands.jpg' image as input. Default is usbcam")
ap.add_argument('-b', '--blaze', type=str, default="hand", help="Application (hand, face, pose). Default is hand")
ap.add_argument('-m', '--model1', type=str, help='Path of blazepalm model. Default is models/palm_detection_without_custom_op.tflite')
ap.add_argument('-n', '--model2', type=str, help='Path of blazehandlardmark model. Default is models/hand_landmark.tflite')
ap.add_argument('-d', '--debug', default=False, action='store_true', help="Enable Debug mode. Default is off")
ap.add_argument('-w', '--withoutview', default=False, action='store_true', help="Disable Output viewing. Default is on")
ap.add_argument('-z', '--profilelog', default=False, action='store_true', help="Enable Profile Log (Latency). Default is off")
ap.add_argument('-Z', '--profileview', default=False, action='store_true', help="Enable Profile View (Latency). Default is off")
ap.add_argument('-f', '--fps', default=False, action='store_true', help="Enable FPS display. Default is off")
ap.add_argument('-N', '--npu', default=False, action='store_true', help="Enable NPU")
ap.add_argument('-q', '--quant', default=False, action='store_true', help="Select for quantized models")

args = ap.parse_args()

# Print command line options (commented out as in original)

nb_blaze_pipelines = 1

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
print("[INFO] Input Video : ", input_video)

output_dir = './captured-images'
profile_csv = './blaze_detect_live_gstreamer.csv'

if os.path.isfile(profile_csv):
    f_profile_csv = open(profile_csv, "a")
    print("[INFO] Appending to existing profiling results file :", profile_csv)
else:
    f_profile_csv = open(profile_csv, "w")
    print("[INFO] Creating new profiling results file :", profile_csv)
    f_profile_csv.write("time,user,hostname,pipeline,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n")

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)

# Global variables
cap = None
gst_display = None

def cleanup(signum, frame):
    print("\n[INFO] Caught interrupt signal. Cleaning up...")
    if cap is not None:
        cap.release()
    if gst_display is not None:
        gst_display.close()
    sys.exit(0)

# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Create GStreamer display
if not args.withoutview:
    gst_display = GstDisplay(frame_width, frame_height, "Blaze Detection Output")

# Model setup (same as original)
if args.blaze == "hand":
   blaze_detector_type = "blazepalm"
   blaze_landmark_type = "blazehandlandmark"
   blaze_title = "BlazeHandLandmark"
   default_detector_model='models/palm_detection_lite.tflite'
   default_landmark_model='models/hand_landmark_lite.tflite'
### Quantized models
#    default_detector_model='models/palm_detection_lite_quant.tflite'
#    default_landmark_model='models/hand_landmark_lite_quant.tflite'
# ### Vela converted models
#    default_detector_model='models/palm_detection_lite_quant_vela.tflite'
#    default_landmark_model='models/hand_landmark_lite_quant_vela.tflite'
elif args.blaze == "face":
   blaze_detector_type = "blazeface"
   blaze_landmark_type = "blazefacelandmark"
   blaze_title = "BlazeFaceLandmark"
   default_detector_model='models/face_detection_short_range.tflite'
   default_landmark_model='models/face_landmark.tflite'
### Quantized models
#    default_detector_model='models/face_detect_quant.tflite'
#    default_landmark_model='models/face_landmark_quant.tflite'
### Vela converted models
#    default_detector_model='models/output/face_detect_quant_vela.tflite'
#    default_landmark_model='models/output/face_landmark_quant_vela.tflite'
elif args.blaze == "pose":
   blaze_detector_type = "blazepose"
   blaze_landmark_type = "blazeposelandmark"
   blaze_title = "BlazePoseLandmark"
   default_detector_model = "models/pose_detection.tflite"
   default_landmark_model = "models/pose_landmark_lite.tflite"
### Quantized models
#    default_detector_model='models/pose_detection_quant.tflite'
#    default_landmark_model='models/pose_landmark_lite_quant.tflite'
### Vela converted models
#    default_detector_model='models/pose_detection_quant_vela.tflite'
#    default_landmark_model='models/pose_landmark_lite_quant_vela.tflite'
else:
   print("[ERROR] Invalid Blaze application : ", args.blaze, ". MUST be one of hand,face,pose.")

if args.model1 == None:
   args.model1 = default_detector_model
if args.model2 == None:
   args.model2 = default_landmark_model

DELEGATE_PATH = None

if args.npu == True:
   # Only use delegate for NON-Vela models
   if "_vela" not in args.model1 and "_vela" not in args.model2:
        DELEGATE_PATH = "/usr/lib/libethosu_delegate.so"
   print("[INFO] Delegate path is ", DELEGATE_PATH)

blaze_detector = BlazeDetector(blaze_detector_type, delegate_path=DELEGATE_PATH, quantized=args.quant)
blaze_detector.set_debug(debug=args.debug)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(args.model1)

blaze_landmark = BlazeLandmark(blaze_landmark_type, delegate_path=DELEGATE_PATH, quantized=args.quant)
blaze_landmark.set_debug(debug=args.debug)
blaze_landmark.load_model(args.model2)

print("================================================================")
print("Blaze Detect Live Demo (GStreamer)")
print(f"Detect model is: {default_detector_model}")
print(f"Landmark model is: {default_landmark_model}")
print("================================================================")

# Main loop variables
frame_count = 0
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)

# Register signal handler
signal.signal(signal.SIGINT, cleanup)

while True:
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    frame_count += 1

    if args.image:
        frame = cv2.imread('../woman_hands.jpg')
        if not isinstance(frame, np.ndarray):
            print("[ERROR] cv2.imread('woman_hands.jpg') FAILED!")
            break
    else:
        flag, frame = cap.read()
        if not flag:
            print("[ERROR] cap.read() FAILED!")
            break

    if args.profilelog or args.profileview:
        prof_title = [''] * nb_blaze_pipelines
        prof_resize = np.zeros(nb_blaze_pipelines)
        prof_detector_pre = np.zeros(nb_blaze_pipelines)
        prof_detector_model = np.zeros(nb_blaze_pipelines)
        prof_detector_post = np.zeros(nb_blaze_pipelines)
        prof_extract_roi = np.zeros(nb_blaze_pipelines)
        prof_landmark_pre = np.zeros(nb_blaze_pipelines)
        prof_landmark_model = np.zeros(nb_blaze_pipelines)
        prof_landmark_post = np.zeros(nb_blaze_pipelines)
        prof_annotate = np.zeros(nb_blaze_pipelines)
        prof_total = np.zeros(nb_blaze_pipelines)
        prof_fps = np.zeros(nb_blaze_pipelines)

    pipeline_id = 0
    image = frame.copy()
    output = image.copy()
    
    # Blaze detection pipeline
    start = timer()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img1, scale1, pad1 = blaze_detector.resize_pad(image)
    profile_resize = timer() - start

    normalized_detections = blaze_detector.predict_on_image(img1)
    if len(normalized_detections) > 0:
        #print("len(normalized_detections): ", len(normalized_detections))
        start = timer()          
        detections = blaze_detector.denormalize_detections(normalized_detections, scale1, pad1)

        xc, yc, scale, theta = blaze_detector.detection2roi(detections)
        roi_img, roi_affine, roi_box = blaze_landmark.extract_roi(image, xc, yc, theta, scale)
        profile_extract = timer() - start

        flags, normalized_landmarks = blaze_landmark.predict(roi_img)

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

        draw_roi(output, roi_box)
        draw_detections(output, detections)
        profile_annotate = timer() - start

    # Display FPS if enabled
    if rt_fps_valid and args.fps:
        cv2.putText(output, rt_fps_message, (rt_fps_x, rt_fps_y), 
                   text_fontType, text_fontSize, text_color, text_lineSize, text_lineType)

    # Display output using GStreamer
    if gst_display is not None:
        gst_display.display_frame(output)

    # Profiling logging
    if args.profilelog:            
        timestamp = datetime.now()
        pipeline_id = 0
        
        prof_resize[pipeline_id] = profile_resize
        prof_detector_pre[pipeline_id] = blaze_detector.profile_pre
        prof_detector_model[pipeline_id] = blaze_detector.profile_model
        prof_detector_post[pipeline_id] = blaze_detector.profile_post
        
        if len(normalized_detections) > 0:
            
            prof_extract_roi[pipeline_id] = profile_extract
            prof_landmark_pre[pipeline_id] = blaze_landmark.profile_pre
            prof_landmark_model[pipeline_id] = blaze_landmark.profile_model
            prof_landmark_post[pipeline_id] = blaze_landmark.profile_post
            prof_annotate[pipeline_id] = profile_annotate
        
        prof_total[pipeline_id] = profile_resize + blaze_detector.profile_pre + \
                                 blaze_detector.profile_model + blaze_detector.profile_post
        if len(normalized_detections) > 0:
            prof_total[pipeline_id] += profile_extract + blaze_landmark.profile_pre + \
                                      blaze_landmark.profile_model + blaze_landmark.profile_post + \
                                      profile_annotate
        prof_fps[pipeline_id] = 1.0 / prof_total[pipeline_id]
        
        csv_str = f"{timestamp},{user},{host},blaze_tflite," \
                  f"{prof_resize[pipeline_id]}," \
                  f"{prof_detector_pre[pipeline_id]}," \
                  f"{prof_detector_model[pipeline_id]}," \
                  f"{prof_detector_post[pipeline_id]}," \
                  f"{prof_extract_roi[pipeline_id]}," \
                  f"{prof_landmark_pre[pipeline_id]}," \
                  f"{prof_landmark_model[pipeline_id]}," \
                  f"{prof_landmark_post[pipeline_id]}," \
                  f"{prof_annotate[pipeline_id]}," \
                  f"{prof_total[pipeline_id]}," \
                  f"{prof_fps[pipeline_id]}\n"
        f_profile_csv.write(csv_str)
    
    # Update FPS counter
    rt_fps_count += 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = True
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        rt_fps_count = 0

# Cleanup
f_profile_csv.close()
if cap is not None:
    cap.release()
if gst_display is not None:
    gst_display.close()