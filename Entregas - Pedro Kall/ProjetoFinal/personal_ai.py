import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import queue
import threading
import math

class PersonalAI:
  def __init__(self, video_source='Lucas.MOV'):
    self.video_source = video_source
    self.temp_q = queue.Queue()
    self.image_q = queue.Queue()

    model_path = 'pose_landmarker_full.task'
      
    self.options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)

  def draw_angle(self, frame, landmarks, p1, p2, pc):
    land = landmarks.pose_landmarks[0]
    h, w, c = frame.shape
    x1, y1 = (land[p1].x, land[p1].y)
    x2, y2 = (land[p2].x, land[p2].y)
    x3, y3 = (land[pc].x, land[pc].y)

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
    position = (int(x2 * w + 10), int(y2 * h +10))

    frame = cv2.putText(frame, str(int(angle)), position, 
                        cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 2)
    return frame, angle

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    if pose_landmarks_list:
      for pose_landmarks in pose_landmarks_list:
        connections = [
          (11, 12), (12, 24), (24, 23), (23, 11),
          (11, 13), (13, 15),
          (12, 14), (14, 16),
          (23, 25), (25, 27),
          (24, 26), (26, 28),
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
            
            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 4)
        
        important_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
        for i in important_points:
          if i < len(pose_landmarks):
            landmark = pose_landmarks[i]
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            
            if i in [11, 12]:
              color = (0, 0, 255)
              radius = 10
            elif i in [23, 24]:
              color = (255, 0, 0)
              radius = 10
            elif i in [25, 26]:
              color = (0, 255, 0)
              radius = 8
            elif i in [27, 28]:
              color = (255, 255, 0)
              radius = 8
            else:
              color = (255, 0, 255)
              radius = 6
              
            cv2.circle(annotated_image, (x, y), radius, color, -1)
            cv2.circle(annotated_image, (x, y), radius + 2, (255, 255, 255), 2)
    
    return annotated_image

  def process_video(self, display):
    with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
      cap = cv2.VideoCapture(self.video_source)
      # Se for webcam, definir FPS padrão
      if isinstance(self.video_source, int):
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1:
          self.fps = 30  # valor padrão para webcam
        use_webcam = True
      else:
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        use_webcam = False
      calc_timestamps = [0.0]
      if (cap.isOpened()== False): 
          print("Error opening video stream or file")
      start_time = None
      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:    
              mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
              if use_webcam:
                # Para webcam, usar tempo real
                if start_time is None:
                  start_time = cv2.getTickCount()
                  timestamp = 0.0
                else:
                  ticks = cv2.getTickCount() - start_time
                  timestamp = (ticks / cv2.getTickFrequency()) * 1000  # em ms
                calc_timestamps.append(timestamp)
              else:
                calc_timestamps.append(int(calc_timestamps[-1] + 1000 / self.fps))
              detection_result = landmarker.detect_for_video(mp_image, calc_timestamps[-1])              
              annotated_image = self.draw_landmarks_on_image(frame, detection_result)
              annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

              if display:
                cv2.imshow('Frame',annotated_image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break

              self.image_q.put((annotated_image, detection_result, calc_timestamps[-1] / 1000))
          else: 
              break
      self.image_q.put((1, 1, "done"))
      cap.release()
      cv2.destroyAllWindows()
  
  def run(self):
    t1 = threading.Thread(target=self.process_video, args=(False, ))
    t1.start()


if __name__ == "__main__":
  personalAI = PersonalAI()
  personalAI.process_video(True)