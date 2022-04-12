from turtle import color
from cv2 import sqrt
from math import acos, degrees
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os 

"""
Keypoint labels:
nose = 0
left eye = 1
right eye = 2
left ear = 3
right ear = 4
left shoulder = 5
right shoulder = 6
left elbow = 7
right elbow = 8
left wrist = 9
right wrist = 10
left hip = 11
right hip = 12
left knee = 13
right knee = 14
left ankle = 15
right ankle = 16
"""

def calcDistance(Xpoint1, Ypoint1, Xpoint2, Ypoint2):

    Xdifference = Xpoint1 - Xpoint2
    if (Xdifference < 0):
        Xdifference = Xdifference * -1
    
    Ydifference = Ypoint1 - Ypoint2
    if (Ydifference < 0):
        Ydifference = Ydifference * -1

    distance = (Xdifference ** 2) + (Ydifference ** 2)
    distance = sqrt(distance)

    return distance

def calcAngle(Xpoint1, Ypoint1, Xpoint2, Ypoint2, XpointJ, YpointJ):

    side1j = calcDistance(XpointJ, YpointJ, Xpoint1, Ypoint1)
    side2j = calcDistance(XpointJ, YpointJ, Xpoint2, Ypoint2)
    side12 = calcDistance(Xpoint1, Ypoint1, Xpoint2, Ypoint2)

    angle = degrees(acos((side1j * side1j + side2j * side2j - side12 * side12)/(2 * side1j * side2j)))
    
    return angle

def calcJoints(keyPoints_all_images):
    joints = np.array()
    """
    Joint labels:
    head (5,6,0) = 0
    left shoulder (7,11,5) = 1
    right shoulder (8,12,6) = 2
    left elbow (5,9,7) = 3
    right elbow (6,10,8) = 4
    left hip (5,13,11) = 5
    right hip (6,14,12) = 6
    left knee (11,15,13) = 7
    right knee (12,16,14) = 8
    """
    for i in range(len(keyPoints_all_images)):
        joints = calcAngle(5,6,0)
        joints.append(calcAngle(7,11,5)) 
        joints.append(calcAngle(8,12,6))
        joints.append(calcAngle(5,9,7))
        joints.append(calcAngle(6,10,8))
        joints.append(calcAngle(5,13,11))
        joints.append(calcAngle(6,14,12))
        joints.append(calcAngle(11,15,13))
        joints.append(calcAngle(12,16,14))

        if (i == 1):
            angles_all_images = np.array(joints)
        else:
            angles_all_images = np.vstack((angles_all_images, joints))

    return joints