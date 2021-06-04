##Gabriel de Sousa Gomes - RM: 79899
##Lucas Barreto Alencar de Moraes - RM: 80511
##Victor Nassif Marques da Cruz - RM: 80207

import cv2
import numpy as np
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_alt.xml"
eyePath = "haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
eyes_cascade = cv2.CascadeClassifier(eyePath)

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
Kernal = np.ones((3, 3), np.uint8)      #Declare kernal for morphology

croppedScale = 0.11 #

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,+1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.2, 1
    )

    #Desenha ROI face
    for (face_x, face_y, face_w, face_h) in faces:
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)

        faceROI = gray[face_y:face_y+face_h,face_x:face_x+face_w]
        
        #detecta 
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.2, 1)
        for (eye_x,eye_y,eye_w,eye_h) in eyes:
            #desenha círculo ROI eye
            eye_center = (face_x + eye_x + eye_w//2, face_y + eye_y + eye_h//2)
            radius = int(round((eye_w + eye_h)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            
            #encontra e desenha íris
            eyeROI = gray[face_y+eye_y:face_y+eye_y+eye_h, face_x+eye_x:face_x+eye_x+eye_w]
            ret, binary = cv2.threshold(eyeROI, 60, 255, cv2.THRESH_BINARY_INV)
            width, height = binary.shape


            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)  ##Opening Morphology
            dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)  ##Dilate Morphology

            binary = binary[int(croppedScale * height):height, :]    ##Crop top of the image
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                cnt = contours[0]
                M1 = cv2.moments(cnt)
                
                Cx1 = int(M1['m10'] / M1['m00'])##Find center of the contour
                Cy1 = int(M1['m01'] / M1['m00'])
                croppedImagePixelLength = int(croppedScale*height)## Number of pixels we cropped from the image
                center1 = (int(Cx1+face_x+eye_x), int(Cy1+face_y + eye_y + croppedImagePixelLength))    ##Center coordinates
                cv2.circle(frame, center1, 2, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            print(center1)


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()