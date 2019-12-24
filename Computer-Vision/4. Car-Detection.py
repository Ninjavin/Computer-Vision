# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:24:15 2019

@author: vinja
"""

import cv2
from time import sleep

car_classifier = cv2.CascadeClassifier("C:/Users/vinja/Desktop/Computer-Vision/Haarcascades/haarcascade_car.xml")

video_capture = cv2.VideoCapture("C:/Users/vinja/Desktop/Computer-Vision/cars.avi")

while video_capture.isOpened():
    sleep(0.05)
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray,1.2,3)
    
    for(x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.imshow('Cars', frame)
        
    if cv2.waitKey(1) == 13:
        break
    
video_capture.release()
cv2.destroyAllWindows()