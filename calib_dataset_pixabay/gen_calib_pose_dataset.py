import cv2
import os
import sys
import numpy as np

input_videos = []
input_videos.append("./videos/pixabay-liverpool-pier-head-england-uk-46098.mp4")
input_videos.append("./videos/pixabay-spring-walk-park-trees-flowers-15252.mp4")
input_videos.append("./videos/pixabay-pool-jump-throw-water-fun-46565.mp4")
input_videos.append("./videos/pixabay-yoga-yoga-studio-people-40401.mp4")
input_videos.append("./videos/pixabay-man-living-room-faces-expression-136253.mp4")
input_videos.append("./videos/pixabay-man-face-expression-irritated-182353.mp4")
input_videos.append("./videos/pixabay-girl-hug-cheerful-gesture-cute-129420.mp4")
video_id = 0
video_cnt = len(input_videos)

# Open first video
cap = cv2.VideoCapture(input_videos[video_id])
print("[INFO] Start of video ",input_videos[video_id])

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../blaze_common/'))
sys.path.append(os.path.abspath('../blaze_tflite/'))

from blaze_tflite.blazedetector import BlazeDetector
from blaze_tflite.blazelandmark import BlazeLandmark

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

blaze_detector_type = "blazepose"
blaze_landmark_type = "blazeposelandmark"
blaze_title = "BlazePoseLandmark"
default_detector_model='../blaze_tflite/models/pose_detection.tflite'
default_landmark_model='../blaze_tflite/models/pose_landmark_heavy.tflite'

app_main_title = blaze_title+" Calibration Dataset Generation"
cv2.namedWindow(app_main_title)

# 0.10 version
blaze_detector = BlazeDetector(blaze_detector_type)
blaze_detector.set_debug(debug=False)
blaze_detector.display_scores(debug=False)
blaze_detector.load_model(default_detector_model)

# 0.07 version (input resolution = 128x128)
# do not have model available, set input size explicitly
blaze_detector2 = BlazeDetector(blaze_detector_type)
blaze_detector2.h_scale = 128
blaze_detector2.w_scale = 128

# 0.10 version
blaze_landmark = BlazeLandmark(blaze_landmark_type)
blaze_landmark.set_debug(debug=False)
blaze_landmark.load_model(default_landmark_model)

calib_pose_detection_224_dataset = []
calib_pose_detection_128_dataset = []

calib_pose_landmark_256_dataset = []

while True:

        flag, frame = cap.read()
        if not flag:
            print("[INFO] End of video ",input_videos[video_id])

            print("")            
            print("[INFO] Collected ",len(calib_pose_detection_128_dataset)," images for calib_pose_detection_128_dataset")
            print("[INFO] Collected ",len(calib_pose_detection_224_dataset)," images for calib_pose_detection_224_dataset")
            print("[INFO] Collected ",len(calib_pose_landmark_256_dataset)," images for calib_pose_landmark_256_dataset")
            print("")            
            
            video_id += 1
            if video_id == video_cnt:
                break
            else:
                # Open next video
                cap = cv2.VideoCapture(input_videos[video_id])
                print("[INFO] Start of video ",input_videos[video_id])                    
                continue

        image = frame.copy()
        output = image.copy()
            
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img1,scale1,pad1=blaze_detector.resize_pad(image)
        
        img2,scale2,pad2=blaze_detector2.resize_pad(image)

        normalized_detections = blaze_detector.predict_on_image(img1)
        if len(normalized_detections) > 0:

            calib_pose_detection_128_dataset.append(img2)
            cv2.imshow("calib_pose_detection_128_dataset", cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))
            calib_pose_detection_224_dataset.append(img1)
            cv2.imshow("calib_pose_detection_224_dataset", cv2.cvtColor(img1,cv2.COLOR_RGB2BGR))

            detections = blaze_detector.denormalize_detections(normalized_detections,scale1,pad1)
                    
            xc,yc,scale,theta = blaze_detector.detection2roi(detections)
            roi_img,roi_affine,roi_box = blaze_landmark.extract_roi(image,xc,yc,theta,scale)
        
            flags, normalized_landmarks = blaze_landmark.predict(roi_img)
        
            landmarks = blaze_landmark.denormalize_landmarks(normalized_landmarks, roi_affine)

            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                if landmark.shape[1] > 33:
                    draw_landmarks(output, landmark[:,:2], POSE_FULL_BODY_CONNECTIONS, size=2)
                else:
                    draw_landmarks(output, landmark[:,:2], POSE_UPPER_BODY_CONNECTIONS, size=2)   
                
                img = roi_img[i] * 255.0
                img = img.astype(np.uint8)
                calib_pose_landmark_256_dataset.append(img)                                 
                cv2.imshow("calib_pose_landmark_256_dataset", cv2.cvtColor(img,cv2.COLOR_RGB2BGR))        
                   
            draw_roi(output,roi_box)
            draw_detections(output,detections)
        
        cv2.imshow(app_main_title, output)
        key = cv2.waitKey(1)


calib_pose_detection_128_dataset = np.array(calib_pose_detection_128_dataset)
calib_pose_detection_224_dataset = np.array(calib_pose_detection_224_dataset)
calib_pose_landmark_256_dataset = np.array(calib_pose_landmark_256_dataset)

print("[INFO] calib_pose_detection_128_dataset shape = ",
    calib_pose_detection_128_dataset.shape,calib_pose_detection_128_dataset.dtype,
    np.amin(calib_pose_detection_128_dataset),np.amax(calib_pose_detection_128_dataset))
print("[INFO] calib_pose_detection_224_dataset shape = ",
    calib_pose_detection_224_dataset.shape,calib_pose_detection_224_dataset.dtype,
    np.amin(calib_pose_detection_224_dataset),np.amax(calib_pose_detection_224_dataset))
print("[INFO] calib_pose_landmark_256_dataset shape = ",
    calib_pose_landmark_256_dataset.shape,calib_pose_landmark_256_dataset.dtype,
    np.amin(calib_pose_landmark_256_dataset),np.amax(calib_pose_landmark_256_dataset))

np.save("calib_pose_detection_128_dataset.npy", calib_pose_detection_128_dataset)
np.save("calib_pose_detection_224_dataset.npy", calib_pose_detection_224_dataset)
np.save("calib_pose_landmark_256_dataset.npy", calib_pose_landmark_256_dataset)



# Cleanup
cv2.destroyAllWindows()        
