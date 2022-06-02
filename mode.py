import cv2
from pyparsing import str_type
import streamlit as st
import numpy as np
import cv2 as cv
import pygame
import mediapipe as mp
import matplotlib.pyplot as plt

from ASL_Alphabet import prediction
import Hand_detection_func as hdf

def modeSimple():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,960)
    stframe = st.empty()

    while cap.isOpened():
        success, image = cap.read()
  
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
    
        # Perform Hands landmarks detection on the frame.
        image, results = hdf.detectHandsLandmarks(image, hdf.hands_videos, display=False)
    
        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            # Count the number of fingers up of each hand in the frame.
            image, fingers_statuses, count = hdf.countFingers(image, results, display=False)
        
        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        stframe.image(gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    



def modeCompliquer():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,960)

    stframe = st.empty()

    while cap.isOpened():
        success, image = cap.read()
  
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        image = cv2.flip(image, 1)
    
        # Perform Hands landmarks detection on the frame.
        image, results = hdf.detectHandsLandmarks(image, hdf.hands_videos, display=False)
    
        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            
            # Count the number of fingers up of each hand in the frame.
            image, fingers_statuses, count = hdf.countFingers(image, results, display=False)
    
        # Visualize the counted fingers.
        image = hdf.annotate(image, results, fingers_statuses, count, display=False)
        
        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        stframe.image(gray)

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def alphabet():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,960)
    stframe = st.empty()
    
    SpelledWord = ""
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        try:
            SpelledWord = prediction.get_prediction(image)
            #cv2.putText(image,  SpelledWord, (50,50), 1, 2, 255)
            cv2.putText(image,SpelledWord,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        stframe.image(gray)
        
                
        if cv2.waitKey(5) & 0xFF == 27: #press escape to break
            break




def selfi():
    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

    stframe = st.empty()

    # Read the filter image with its blue, green, red, and alpha channel.
    filter_imageBGRA = cv2.imread('media/filter.png', cv2.IMREAD_UNCHANGED)

    # Initialize a variable to store the status of the filter (i.e., whether to apply the filter or not).
    filter_on = False

    # Initialize the pygame modules and load the image-capture music file.
    pygame.init()
    pygame.mixer.music.load("media/cam.mp3")

    # Initialize the number of consecutive frames on which we want to check the hand gestures before triggering the events.
    num_of_frames = 5

    # Initialize a dictionary to store the counts of the consecutive frames with the hand gestures recognized.
    counter = {'V SIGN': 0, 'SPIDERMAN SIGN': 0, 'HIGH-FIVE SIGN': 0}

    # Initialize a variable to store the captured image.
    captured_image = None

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():
    
        # Read a frame.
        ok, frame = camera_video.read()
    
        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue
    
        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape
    
        # Resize the filter image to the size of the frame.
        filter_imageBGRA = cv2.resize(filter_imageBGRA, (frame_width, frame_height))
    
        # Get the three-channel (BGR) image version of the filter image.
        filter_imageBGR  = filter_imageBGRA[:,:,:-1]
    
        # Perform Hands landmarks detection on the frame.
        frame, results = hdf.detectHandsLandmarks(frame, hdf.hands_videos, draw=False, display=False)
    
        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            
            # Count the number of fingers up of each hand in the frame.
            frame, fingers_statuses, count = hdf.countFingers(frame, results, draw=False, display=False)
        
            # Perform the hand gesture recognition on the hands in the frame.
            _, hands_gestures = hdf.recognizeGestures(frame, fingers_statuses, count, draw=False, display=False)
        
         # Apply and Remove Image Filter Functionality.
            ####################################################################################################################
        
            # Check if any hand is making the SPIDERMAN hand gesture in the required number of consecutive frames.
            ####################################################################################################################
        
            # Check if the gesture of any hand in the frame is SPIDERMAN SIGN.
            if any(hand_gesture == "SPIDERMAN SIGN" for hand_gesture in hands_gestures.values()):
            
                # Increment the count of consecutive frames with SPIDERMAN hand gesture recognized.
                counter['SPIDERMAN SIGN'] += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                if counter['SPIDERMAN SIGN'] == num_of_frames:
            
                    # Turn on the filter by updating the value of the filter status variable to true.
                    filter_on = True
                
                    # Update the counter value to zero.
                    counter['SPIDERMAN SIGN'] = 0
                
            # Otherwise if the gesture of any hand in the frame is not SPIDERMAN SIGN.
            else:

                # Update the counter value to zero. As we are counting the consective frames with SPIDERMAN hand gesture.
                counter['SPIDERMAN SIGN'] = 0
        
            ####################################################################################################################
        
            # Check if any hand is making the HIGH-FIVE hand gesture in the required number of consecutive frames.
            ####################################################################################################################
        
            # Check if the gesture of any hand in the frame is HIGH-FIVE SIGN.
            if any(hand_gesture == "HIGH-FIVE SIGN" for hand_gesture in hands_gestures.values()):
            
                # Increment the count of consecutive frames with HIGH-FIVE hand gesture recognized.
                counter['HIGH-FIVE SIGN'] += 1

                # Check if the counter is equal to the required number of consecutive frames.  
                if counter['HIGH-FIVE SIGN'] == num_of_frames:
            
                     # Turn off the filter by updating the value of the filter status variable to False.
                    filter_on = False
                
                    # Update the counter value to zero.
                    counter['HIGH-FIVE SIGN'] = 0
                
            # Otherwise if the gesture of any hand in the frame is not HIGH-FIVE SIGN.
            else:

                # Update the counter value to zero. As we are counting the consective frames with HIGH-FIVE hand gesture.
                counter['HIGH-FIVE SIGN'] = 0
        
            ####################################################################################################################
        
        # Check if the filter is turned on.
        if filter_on:
        
            # Apply the filter by updating the pixel values of the frame at the indexes where the 
            # alpha channel of the filter image has the value 255.
            frame[filter_imageBGRA[:,:,-1]==255] = filter_imageBGR[filter_imageBGRA[:,:,-1]==255]
        
            ####################################################################################################################
    
        # Image Capture Functionality.
        ########################################################################################################################
    
        # Check if the hands landmarks are detected and the gesture of any hand in the frame is V SIGN.
        if results.multi_hand_landmarks and any(hand_gesture == "V SIGN" for hand_gesture in hands_gestures.values()):
        
            # Increment the count of consecutive frames with V hand gesture recognized.
            counter['V SIGN'] += 1
            
            # Check if the counter is equal to the required number of consecutive frames.  
            if counter['V SIGN'] == num_of_frames:
            
                # Make a border around a copy of the current frame.
                captured_image = cv2.copyMakeBorder(src=frame, top=10, bottom=10, left=10, right=10,
                                                borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
            
                # Capture an image and store it in the disk.
                cv2.imwrite('Image_Selfi/Captured_Image.png', captured_image)
                st.sidebar.image('Image_Selfi/Captured_Image.png')
            
                # Display a black image.
                cv2.imshow('Selfie-Capturing System', np.zeros((frame_height, frame_width)))

                # Play the image capture music to indicate the an image is captured and wait for 100 milliseconds.
                pygame.mixer.music.play()
                cv2.waitKey(100)
            
                # Update the counter value to zero.
                counter['V SIGN'] = 0
            
        # Otherwise if the gesture of any hand in the frame is not V SIGN.
        else:
        
            # Update the counter value to zero. As we are counting the consective frames with V hand gesture.
            counter['V SIGN'] = 0
    
        ########################################################################################################################
    
        # Check if we have captured an image.
        if captured_image is not None:
        
            # Resize the image to the 1/5th of its current width while keeping the aspect ratio constant.
            captured_image = cv2.resize(captured_image, (frame_width//5, int(((frame_width//5) / frame_width) * frame_height)))
        
            # Get the new height and width of the image.
            img_height, img_width, _ = captured_image.shape
        
            # Overlay the resized captured image over the frame by updating its pixel values in the region of interest.
            frame[10: 10+img_height, 10: 10+img_width] = captured_image
    
        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF
    
        # Check if 'ESC' is pressed and break the loop.
        if(k == 27):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(gray)
