import cv2
import os
import sys
import numpy as np

def capture_filenames(directory_name):
    subdirs = ['{}/{}'.format(directory_name,i) for i in sorted(os.listdir(directory_name)) ]
    cards = []
    for i,subdir in enumerate(subdirs):
        cards += ['{}/{}'.format(subdir,i) for i in os.listdir(subdir)]
    del subdirs
    return cards

image_location = "kaggle_hand_gestures_dataset"
image_filenames = capture_filenames(image_location)
#print(image_filenames)
 
image_cnt = len(image_filenames)
print("[INFO] ",image_cnt," images found in ",image_location)

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../blaze_common/'))
sys.path.append(os.path.abspath('../blaze_tflite/'))

from blaze_tflite.blazedetector import BlazeDetector
from blaze_tflite.blazelandmark import BlazeLandmark

from visualization import draw_detections, draw_landmarks, draw_roi
from visualization import HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_FULL_BODY_CONNECTIONS, POSE_UPPER_BODY_CONNECTIONS

blaze_detector_type = "blazepalm"
blaze_landmark_type = "blazehandlandmark"
blaze_title = "BlazeHandLandmark"
default_detector_model1='../blaze_tflite/models/palm_detection_full.tflite'
default_detector_model2='../blaze_tflite/models/palm_detection_without_custom_op.tflite' # palm_detection_v0_07.tflite
default_landmark_model1='../blaze_tflite/models/hand_landmark_full.tflite'
default_landmark_model2='../blaze_tflite/models/hand_landmark_v0_07.tflite'

app_main_title = blaze_title+" Calibration Dataset Generation"
cv2.namedWindow(app_main_title)

blaze_detector1 = BlazeDetector(blaze_detector_type)
blaze_detector1.set_debug(debug=False)
blaze_detector1.display_scores(debug=False)
blaze_detector1.load_model(default_detector_model1)

blaze_detector2 = BlazeDetector(blaze_detector_type)
blaze_detector2.set_debug(debug=False)
blaze_detector2.display_scores(debug=False)
blaze_detector2.load_model(default_detector_model2)

blaze_landmark1 = BlazeLandmark(blaze_landmark_type)
blaze_landmark1.set_debug(debug=False)
blaze_landmark1.load_model(default_landmark_model1)

blaze_landmark2 = BlazeLandmark(blaze_landmark_type)
blaze_landmark2.set_debug(debug=False)
blaze_landmark2.load_model(default_landmark_model2)

calib_palm_detection_192_dataset = []
calib_palm_detection_256_dataset = []
calib_hand_landmark_224_dataset = []
calib_hand_landmark_256_dataset = []


#while True:
for image_id in range(image_cnt):

        frame = cv2.imread(image_filenames[image_id])

        image = frame.copy()
        output = image.copy()
            
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img1,scale1,pad1=blaze_detector1.resize_pad(image)
        img2,scale2,pad2=blaze_detector2.resize_pad(image)

        normalized_detections = blaze_detector1.predict_on_image(img1)
        if len(normalized_detections) > 0:
        
            calib_palm_detection_192_dataset.append(img1)
            cv2.imshow("calib_palm_detection_192_dataset", cv2.cvtColor(img1,cv2.COLOR_RGB2BGR))

            calib_palm_detection_256_dataset.append(img2)
            cv2.imshow("calib_palm_detection_256_dataset", cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))
        
            detections = blaze_detector1.denormalize_detections(normalized_detections,scale1,pad1)
                    
            xc,yc,scale,theta = blaze_detector1.detection2roi(detections)
            roi_img,roi_affine,roi_box = blaze_landmark1.extract_roi(image,xc,yc,theta,scale)
            roi_img2,roi_affine2,roi_box2 = blaze_landmark2.extract_roi(image,xc,yc,theta,scale)
        
            flags, normalized_landmarks = blaze_landmark1.predict(roi_img)
        
            landmarks = blaze_landmark1.denormalize_landmarks(normalized_landmarks, roi_affine)

            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=1)   
                
                img = roi_img[i] * 255.0
                img = img.astype(np.uint8)
                calib_hand_landmark_224_dataset.append(img)                                 
                cv2.imshow("calib_hand_landmark_224_dataset", cv2.cvtColor(img,cv2.COLOR_RGB2BGR))        

                img = roi_img2[i] * 255.0
                img = img.astype(np.uint8)
                calib_hand_landmark_256_dataset.append(img)                                 
                cv2.imshow("calib_hand_landmark_256_dataset", cv2.cvtColor(img,cv2.COLOR_RGB2BGR))        
                   
            draw_roi(output,roi_box)
            draw_detections(output,detections)
        
        cv2.imshow(app_main_title, output)
        key = cv2.waitKey(1)


calib_palm_detection_192_dataset = np.array(calib_palm_detection_192_dataset)
calib_palm_detection_256_dataset = np.array(calib_palm_detection_256_dataset)
calib_hand_landmark_224_dataset = np.array(calib_hand_landmark_224_dataset)
calib_hand_landmark_256_dataset = np.array(calib_hand_landmark_256_dataset)

print("[INFO] calib_palm_detection_192_dataset shape = ",
    calib_palm_detection_192_dataset.shape,calib_palm_detection_192_dataset.dtype,
    np.amin(calib_palm_detection_192_dataset),np.amax(calib_palm_detection_192_dataset))
print("[INFO] calib_palm_detection_256_dataset shape = ",
    calib_palm_detection_256_dataset.shape,calib_palm_detection_256_dataset.dtype,
    np.amin(calib_palm_detection_256_dataset),np.amax(calib_palm_detection_256_dataset))
print("[INFO] calib_hand_landmark_224_dataset shape = ",
    calib_hand_landmark_224_dataset.shape,calib_hand_landmark_224_dataset.dtype,
    np.amin(calib_hand_landmark_224_dataset),np.amax(calib_hand_landmark_224_dataset))
print("[INFO] calib_hand_landmark_256_dataset shape = ",
    calib_hand_landmark_256_dataset.shape,calib_hand_landmark_256_dataset.dtype,
    np.amin(calib_hand_landmark_256_dataset),np.amax(calib_hand_landmark_256_dataset))

np.save("calib_palm_detection_192_dataset.npy", calib_palm_detection_192_dataset)
np.save("calib_palm_detection_256_dataset.npy", calib_palm_detection_256_dataset)
np.save("calib_hand_landmark_224_dataset.npy", calib_hand_landmark_224_dataset)
np.save("calib_hand_landmark_256_dataset.npy", calib_hand_landmark_256_dataset)



# Cleanup
cv2.destroyAllWindows()        