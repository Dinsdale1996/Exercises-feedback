from turtle import color
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os 

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

def countReps(values, exercise):
    if exercise == "squats":
        j = 7
    
    lowest = 180
    count = 0
    reps = 0
    repPos = np.array()
    for i in range(len(values)):
        if values[i][j] < lowest:
            lowest = values[i][j]
            count = 0
        else:
            count += 1
            if count == 10:
                reps += 1
                repPos.append(i-10)
                lowest = 180

    return reps, repPos

def compTiming(valuesUser, ValuesTrainer, exercise):
    consistant = False
    timing = 0
    speed = ""
    
    repsUser, repPosUser = countReps(valuesUser, exercise)
    repsTrainer, RepPoseTrainer = countReps(ValuesTrainer, exercise)

    diffUser = np.array()
    for i in range (len(repPosUser)-1):
        diffUser.append(repPosUser[i+1] - repPosUser[i])
    
    averageUser = sum(diffUser)/len(diffUser)

    for i in range (len(diffUser)-1):
        if (diffUser[i] < averageUser +5) and (diffUser[i] > averageUser -5):
            timing +=1
    
    if timing >= (len(diffUser)/2):
        consistant = True
    
    diffTrainer = np.array()
    for i in range (len(RepPoseTrainer)-1):
        diffTrainer.append(RepPoseTrainer[i+1] - RepPoseTrainer[i])
    
    averageTrainer = sum(diffTrainer)/len(diffTrainer)

    if averageUser < averageTrainer +5 and averageUser > averageTrainer -5:
        speed = "Good"
    elif averageUser > averageTrainer +5:
        speed = "Slow"
    elif averageUser < averageTrainer -5:
        speed = "Fast"
    else:
        print("error")

    return consistant, speed

def compHead(valuesUser, valuesTrainer, consistant):
    j = 0
    return

def compShoulderL(valuesUser, valuesTrainer, consistant):
    j = 1
    return

def compShoulderR(valuesUser, valuesTrainer, consistant):
    j = 2
    return

def compElbowL(valuesUser, valuesTrainer, consistant):
    j = 3
    return

def compElbowR(valuesUser, valuesTrainer, consistant):
    j =  4
    return

def compHipL(valuesUser, valuesTrainer, consistant):
    j = 5
    return

def compHipR(valuesUser, valuesTrainer, consistant):
    good = 0
    j = 6
    if consistant == True:
        for i in range(len(valuesUser)):
            if valuesUser[i][j] < valuesTrainer[i][j] +5 and valuesUser[i][j] > valuesTrainer[i][j] -5:
                good +=1
            elif valuesUser[i][j] < valuesTrainer[i][j] +10 and valuesUser[i][j] > valuesTrainer[i][j] -10:
                ok +=1
            else:
                bad +=1
    return

def compKneeL(valuesUser, valuesTrainer, consistant):
    j = 7
    return

def compKneeR(valuesUser, valuesTrainer, consistant):
    j = 8
    return