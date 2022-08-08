# import library and framework
import cv2
import csv
import time
import pickle
import numpy as np
import mediapipe as mp
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# create object for draw and get hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define global variable for hand keypoints data
hand_keypoint_data = np.array([])

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

# Load Model
with open("C:/Users/User/Documents/SIBI_Lang/Model/svm_model.sav", 'rb') as file:
    action_model = pickle.load(file)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  # Read until video is completed
  while cap.isOpened():
    # Capture frame-by-frame
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    try:
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      
      # Checking keypoints if complete will do this block of code
      if len(results.multi_hand_landmarks[0].landmark) >= 21:
        # define variable for centering and scaling process
        centering = np.array([])
        scaling = np.array([])

        # Centering X coordinate Process
        for indexPoint in range(21):
          centering = np.append(centering, (
            results.multi_hand_landmarks[0].landmark[0].x - results.multi_hand_landmarks[0].landmark[indexPoint].x))

        # Centering Y coordinate Process
        for indexPoint in range(21):
          centering = np.append(centering, (
            results.multi_hand_landmarks[0].landmark[0].y - results.multi_hand_landmarks[0].landmark[indexPoint].y))

        centering = centering.reshape(2, 21)
        
        # Scaling Process
        for indexIter in range(2):
          for jointIter in range(21):
            scaling = np.append(scaling, centering[indexIter][jointIter] / np.max(
              np.absolute(centering[indexIter])) * 320)
        
        # Normalization Process
        for jointIter in range(42):
          hand_keypoint_data = np.append(hand_keypoint_data, (scaling[jointIter] + 320))

        # Write spatiodata from hand keypoints coordinate
        if len(hand_keypoint_data) >= 210:
          # Write spatiodata to csv
          # Uncomment 3 lines below to write hand keypoint
          # with open('C:/Users/User/Documents/SIBI_Lang/C_Sibi_Lang.csv', 'a', newline='') as f:
          #   writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          #   writer.writerow(hand_keypoint_data)

          prediction = action_model.predict([hand_keypoint_data])

          cv2.putText(image,f'{prediction[0]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

          # deleted 42 old data 
          deletedIndex = np.arange(42)
          hand_keypoint_data = np.delete(hand_keypoint_data, deletedIndex)

    except Exception as e:
      continue

    finally:
      # font which we will be using to display FPS
      font = cv2.FONT_HERSHEY_SIMPLEX

      # time when we finish processing for this frame
      new_frame_time = time.time()

      fps = 1 / (new_frame_time - prev_frame_time)
      prev_frame_time = new_frame_time

      # converting the fps into integer
      fps = int(round(fps))

      # converting the fps to string so that we can display it on frame
      # by using putText function
      fps = str(fps)

      # puting the FPS count on the frame
      cv2.putText(image, fps, (550, 50), font, 2, (100, 255, 0), 3, cv2.LINE_AA)

      # Show the result
      cv2.imshow('Result', image)

      if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()