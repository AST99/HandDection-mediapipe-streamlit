from pyexpat import model
import pandas as pd
from pyexpat import model
import pandas as pd
import cv2
from ASL_Alphabet import predictMethodes
import mediapipe as mp
import numpy as np
import pickle

#########Begin predictions
###Load model
with open('ASL_model', 'rb') as f:
    clf = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_prediction(image):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = predictMethodes.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        image = ImageData['image']
        prediction = clf.predict([DistanceData])
        return prediction[0]


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    SpelledWord = ""
    while cap.isOpened():
            success, image = cap.read()
            if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

            try:
                SpelledWord = get_prediction(image)
                #cv2.putText(image,  SpelledWord, (50,50), 1, 2, 255)
                cv2.putText(image,SpelledWord,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

            cv2.imshow('frame', image)
                
            if cv2.waitKey(5) & 0xFF == 27: #press escape to break
                        break
    
    cap.release()
    cv2.destroyAllWindows()