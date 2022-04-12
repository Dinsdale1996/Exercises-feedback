from turtle import color
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os 

def read():
    values = np.empty()
    with open('data.txt', 'r') as f:
        for line in f:
            frame = [item.strip() for item in line.split(' ')]
            values.append(int(frame))
        return values

    

def write(saveData):
    f = open('data.txt', 'w')
    f.write(saveData)
    f.close()