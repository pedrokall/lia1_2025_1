import mediapipe as mp
from mediapipe.tasks import python
import cv2
import numpy as np

file_name='Lucas.MOV'
file_name = file_name
model_path = 'pose_landmarker_full.task'

options = python.vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=python.vision.RunningMode.VIDEO)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  if pose_landmarks_list:
    for pose_landmarks in pose_landmarks_list:
      connections = [
        (11, 12), (12, 24), (24, 23), (23, 11),  
        
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  
        
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  
        
        (23, 25), (25, 27), (27, 29), (27, 31),  
        
        (24, 26), (26, 28), (28, 30), (28, 32),  
        
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        
        (11, 23), (12, 24),  
      ]
      
      for connection in connections:
        if connection[0] < len(pose_landmarks) and connection[1] < len(pose_landmarks):
          start_landmark = pose_landmarks[connection[0]]
          end_landmark = pose_landmarks[connection[1]]
          
          start_x = int(start_landmark.x * annotated_image.shape[1])
          start_y = int(start_landmark.y * annotated_image.shape[0])
          end_x = int(end_landmark.x * annotated_image.shape[1])
          end_y = int(end_landmark.y * annotated_image.shape[0])
          
          cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
      
      for i, landmark in enumerate(pose_landmarks):
        x = int(landmark.x * annotated_image.shape[1])
        y = int(landmark.y * annotated_image.shape[0])
        
        if i in [11, 12, 23, 24]:  
          color = (0, 0, 255)  
          radius = 8
        elif i in [13, 14, 15, 16]:  
          color = (255, 0, 0)  
          radius = 6
        elif i in [25, 26, 27, 28]:  
          color = (0, 255, 0)  
          radius = 6
        else:  
          color = (255, 255, 0)  
          radius = 4
          
        cv2.circle(annotated_image, (x, y), radius, color, -1)
  
  return annotated_image

with python.vision.PoseLandmarker.create_from_options(options) as landmarker:
  cap = cv2.VideoCapture(file_name)
  fps = cap.get(cv2.CAP_PROP_FPS)
  calc_timestamps = [0.0]

  if (cap.isOpened()== False): 
      print("Error opening video stream or file")

  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:    
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      calc_timestamps.append(int(calc_timestamps[-1] + 1000/fps))
      detection_result = landmarker.detect_for_video(mp_image, calc_timestamps[-1])
      
      annotated_image = draw_landmarks_on_image(frame, detection_result)
      cv2.imshow('Frame',annotated_image)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else: 
        break
  
  cap.release()
  cv2.destroyAllWindows()
