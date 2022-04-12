from turtle import color
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os 

"""
Keypoint labels
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

EDGES = {
        (0,5): 'd', # left shoulder to nose
        (0,6): 'd', # right shoulder to nose
        (5,6): 'l', # left shoulder to right shoulder
        (5,7): 'l', # left shoulder to left elbow
        (6,8): 'l', # right shoulder to right elbow
        (7,9): 'l', # left elbow to left wrist
        (8,10): 'l', # right elbow to right wrist
        (5,11): 'l', # left shoulder to left hip
        (6,12): 'l', # right shoulder to right hip
        (11,13): 'l', # left hip to left knee
        (12,14): 'l', # right hip to right knee
        (13,15): 'l', # left knee to left ankle
        (14,16): 'l' # right knee to right ankle
    }

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

def draw_keypoints(frame, keypoints, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv.circle(frame, (int(ky), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence):

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence) & (c2 > confidence):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

#webacm/video detection 
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #scale image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    #input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)

    #demonstration output
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)

    draw_keypoints(frame, keypoints_with_scores, 0.4)

    #show feed
    cv.imshow('Movenet', frame)

    #close feed
    if cv.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()