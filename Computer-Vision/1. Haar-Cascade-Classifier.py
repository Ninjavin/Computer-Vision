# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:14:16 2019

@author: vinja
"""

import numpy as np
import cv2
#Open CV's Cascade Classifier function is pointed to where our classifier (XML) is stored
face_classifier = cv2.CascadeClassifier("C:/Users/vinja/Desktop/Computer-Vision/Haarcascades/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("C:/Users/vinja/Desktop/Computer-Vision/Haarcascades/haarcascade_eye.xml")
image = cv2.imread("C:/Users/vinja/Desktop/Computer-Vision/images.jfif")
#Convert Image to GrayScale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Easier to process a gray channel image

faces = face_classifier.detectMultiScale(image_gray,1.3,5) #Inbuilt feature of Haar Cascade Classifier

if faces is ():
    print("No faces found")


#x = x-coordinate
# y = y-coordinate
# w = width
# h = height
# (127,0,255) color of rectangle
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow("Face Detection", image)
    cv2.waitKey()
    roi_gray = image_gray[y:y+h, x:x+w] #crop only the face part
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow("Face Detection", image)
        cv2.waitKey()
cv2.destroyAllWindows()