#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:34:55 2019

@author: manotr
"""

# import respectives libraries
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

theta = 45

def rotate_box(bb, cx, cy, h, w):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb


# making object of MTCNN Lib
detector = MTCNN()

# making object of Camera using cv2
cap = cv2.VideoCapture(0)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


# while condition true
while True:
    
    # read the frame from camera streaming
    ret, frame = cap.read()
    
    # if not matrix break the loop
    if not ret:
        break
    
    # conver BGR image to RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # find the faces and there bounding box from  frame
    boxes = detector.detect_faces(rgb)
    
    # if faces do the following
    if boxes != []:
        
        # find number of faces
        for person in boxes:
            
            # get face bounding box location
            bounding_box = person['box']
            
            # get the left eyes location
            left_eye = person['keypoints']['left_eye']
            
            # get the right eyes location
            right_eye = person['keypoints']['right_eye']
            
            # draw the face bounding box
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255), 2)
            
    # show the resultant frame        
    cv2.imshow("frame",frame)
    out.write(frame)
    # press the 'q' keyword to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release the camera object    
cap.release() 

out.release()

# close the open windows
cv2.destroyAllWindows() 