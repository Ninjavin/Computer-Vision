# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:55:24 2019

@author: vinja
"""

import cv2
import numpy as np
#To Load An Image
image = cv2.imread("C:/Users/vinja/Desktop/Computer-Vision/images.jfif")
#To show the Image
cv2.imshow('Tony Stark',image)#first parameter- name of window that contains image

cv2.waitKey() #Pressing any button will close the image window
cv2.destroyAllWindows()

print(image.shape)

print("Height of image is:", image.shape[0], "pixels")
print("Width of image is:", image.shape[1], "pixels")