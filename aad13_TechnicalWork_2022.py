import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from math import acos, degrees
from glob import glob
import cv2 as cv
import os 
import sys

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def drawKeypoints(frame, keypoints, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv.circle(frame, (int(ky), int(ky)), 4, (0,255,0), -1)

def drawConnections(frame, keypoints, edges, confidence):

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence) & (c2 > confidence):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

def saveFrame(videoPath, saveDir, userType, excersize):
    savePath = saveDir+'/'+excersize+'/'+excersize+"_"+userType
    
    cap = cv.VideoCapture(videoPath)
    assert cap.isOpened()
    fps_in = cap.get(cv.CAP_PROP_FPS)
    fps_out = 5

    index_in = -1
    index_out = -1
    idx = 1

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break
        
        index_in += 1
        out_due = int(index_in / fps_in * fps_out)

        if out_due > index_out:
            success, frame = cap.retrieve()

            if ret == False:
                cap.release()
                break

            cv.imwrite(f"{savePath}/{idx}.png", frame)
            index_out += 1
            idx += 1

def read(filePath, excersize, userType):
    #checks if file exists
    if (os.path.exists(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt')):
        #loads file to variable and returns
        values = np.loadtxt(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt', dtype=float)
        print("data loaded from "+filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt')
    else:
        #file does not exist
        print("Error: file not found")
    return values

def write(saveData, filePath, excersize, userType):
    #checks if file already exists
    if (os.path.exists(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt')):
        save = input("File already exists, Save over file y/n:")
        if save == "y":
            #saves data to file
            np.savetxt(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt', saveData, fmt='%1.2f')
        elif save == "n":
            print("file not saved")
        else:
            print("Error: incorrect input")
    else:
        #saves data to file
        np.savetxt(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt', saveData, fmt='%1.2f')

def checkConfidence(keyPoints_all_images, confidence):
    inaccurate = False
    confidenceFail = 0
    i = 0
    #checks all confidence values in array
    while i < len(keyPoints_all_images):
        ##if confidence value is below given threshold
        if keyPoints_all_images[i][2] < confidence:
            #set confidence value to 0 and counts number of uncertain values
            keyPoints_all_images[i][2] = 0
            confidenceFail += 1
        i += 1

    #if too many inaccurate values, inaccurate = True
    if confidenceFail > (len(keyPoints_all_images)/2):
        inaccurate = True
        
    return keyPoints_all_images, inaccurate

def calcDistance(Xpoint1, Ypoint1, Xpoint2, Ypoint2):
    #find distance between both X points and Y points
    Xdifference = Xpoint1 - Xpoint2
    if (Xdifference < 0):
        Xdifference = Xdifference * -1
    
    Ydifference = Ypoint1 - Ypoint2
    if (Ydifference < 0):
        Ydifference = Ydifference * -1

    #A squared + B squared = C squared
    distance = (Xdifference ** 2) + (Ydifference ** 2)
    distance = cv.sqrt(distance)
    #distance was considered as an array with [0] being the answer and 3 other values of 0.00
    #this just changes it so  distance only returns the single value
    distance = distance[0]
    #returns distance between the 2 points
    return distance

def calcAngle(Ypoint1, Xpoint1, Ypoint2, Xpoint2, YpointJ, XpointJ):
    #triganometry to calculate the angle if the length of all 3 sides are known
    side1j = calcDistance(XpointJ, YpointJ, Xpoint1, Ypoint1)
    side2j = calcDistance(XpointJ, YpointJ, Xpoint2, Ypoint2)
    side12 = calcDistance(Xpoint1, Ypoint1, Xpoint2, Ypoint2)
    a = (side1j * side1j + side2j * side2j - side12 * side12)
    b = (2 * side1j * side2j)
    c = a / b
    rad = acos(c)
    #convert to degrees
    angle = degrees(rad)
    return angle

def calcJoints(keyPoints_all_images):
    i = 0
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
    while i < len(keyPoints_all_images):
        badJoints = 0
        j = 0
        head = calcAngle(keyPoints_all_images[5+i][0],keyPoints_all_images[5+i][1],keyPoints_all_images[6+i][0],keyPoints_all_images[6+i][1],keyPoints_all_images[0+i][0],keyPoints_all_images[0+i][1])
        lShoulder = calcAngle(keyPoints_all_images[7+i][0],keyPoints_all_images[7+i][1],keyPoints_all_images[11+i][0],keyPoints_all_images[11+i][1],keyPoints_all_images[5+i][0],keyPoints_all_images[5+i][1])
        rShoulder = calcAngle(keyPoints_all_images[8+i][0],keyPoints_all_images[8+i][1],keyPoints_all_images[12+i][0],keyPoints_all_images[12+i][1],keyPoints_all_images[6+i][0],keyPoints_all_images[6+i][1])
        lElbow = calcAngle(keyPoints_all_images[5+i][0],keyPoints_all_images[5+i][1],keyPoints_all_images[9+i][0],keyPoints_all_images[9+i][1],keyPoints_all_images[7+i][0],keyPoints_all_images[7+i][1])
        rElbow = calcAngle(keyPoints_all_images[6+i][0],keyPoints_all_images[6+i][1],keyPoints_all_images[10+i][0],keyPoints_all_images[10+i][1],keyPoints_all_images[8+i][0],keyPoints_all_images[8+i][1])
        lHip = calcAngle(keyPoints_all_images[5+i][0],keyPoints_all_images[5+i][1],keyPoints_all_images[13+i][0],keyPoints_all_images[13+i][1],keyPoints_all_images[11+i][0],keyPoints_all_images[11+i][1])
        rHip = calcAngle(keyPoints_all_images[6+i][0],keyPoints_all_images[6+i][1],keyPoints_all_images[14+i][0],keyPoints_all_images[14+i][1],keyPoints_all_images[12+i][0],keyPoints_all_images[12+i][1])
        lKnee = calcAngle(keyPoints_all_images[11+i][0],keyPoints_all_images[11+i][1],keyPoints_all_images[15+i][0],keyPoints_all_images[15+i][1],keyPoints_all_images[13+i][0],keyPoints_all_images[13+i][1])
        rKnee = calcAngle(keyPoints_all_images[12+i][0],keyPoints_all_images[12+i][1],keyPoints_all_images[16+i][0],keyPoints_all_images[16+i][1],keyPoints_all_images[14+i][0],keyPoints_all_images[14+i][1])

        joints = [head, lShoulder, rShoulder, lElbow, rElbow, lHip, rHip, lKnee, rKnee]

        while j < 17:
            if keyPoints_all_images[i+j][2] == 0:
                badJoints += 1
            j += 1

        if (i == 0):
            angles_all_images = np.array(joints)
        elif badJoints > 8:
            print("too many uncertain joints in frame: frame removed")
        else:
            angles_all_images = np.vstack((angles_all_images, joints))

        i+=17

    return angles_all_images

def countReps(angles, exercise):
    i = 1
    
    if exercise == "squat":
        j = 7
    
    reps = 0
    repPos = np.array(1)
    while i < (len(angles)-1):
        if angles[i][j]<=angles[i-1][j]:
            if  angles[i][j]<=angles[i-1][j] and angles[i][j]<=angles[i+1][j]:
                reps += 0.5
                repPos = np.append(repPos,i)

        elif angles[i][j]>=angles[i-1][j]:
            if  angles[i][j]>=angles[i-1][j] and angles[i][j]>=angles[i+1][j]:
                reps += 0.5
                repPos = np.append(repPos,i)
        i+=1
    return repPos

def compTiming(repPosUser, RepPoseTrainer):
    consistant = False
    timing = 0
    speed = ""

    diffUser = np.array(1)
    for i in range (len(repPosUser)-1):
        diff = repPosUser[i+1] - repPosUser[i]
        diffUser = np.append(diffUser, diff)
    
    averageUser = sum(diffUser)/len(diffUser)

    for i in range (len(diffUser)-1):
        if (diffUser[i] < averageUser +5) and (diffUser[i] > averageUser -5):
            timing +=1
    
    if timing >= (len(diffUser)/2):
        consistant = True
    
    diffTrainer = np.array(1)
    for i in range (len(RepPoseTrainer)-1):
        diff = RepPoseTrainer[i+1] - RepPoseTrainer[i]
        diffTrainer = np.append(diffTrainer,diff)
    
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

def compJoint(repPosUser, anglesUser, RepPoseTrainer, anglesTrainer, joint):
    good, ok, bad = 0, 0, 0
    if joint == "head":
        j = 0
    elif joint == "shoulderL":
        j = 1
    elif joint == "shoulderR":
        j = 2
    elif joint == "elbowL":
        j = 3
    elif joint == "elbowR":
        j = 4 
    elif joint == "hipL":
        j = 5
    elif joint == "hipR":
        j = 6
    elif joint == "kneeL":
        j = 7
    elif joint == "kneeR":
        j = 8
    else:
        print("unrecognised joint default to head")
        j = 0

    if len(repPosUser) > len(RepPoseTrainer):
        checks = len(RepPoseTrainer)
    else:
        checks = len(repPosUser)

    for i in range(checks):
        if anglesUser[repPosUser[i]][j] < anglesTrainer[RepPoseTrainer[i]][j] +5 and anglesUser[repPosUser[i]][j] > anglesTrainer[RepPoseTrainer[i]][j] -5:
            good +=1
        elif anglesUser[repPosUser[i]][j] < anglesTrainer[RepPoseTrainer[i]][j] +10 and anglesUser[repPosUser[i]][j] > anglesTrainer[RepPoseTrainer[i]][j] -10:
            ok +=1
        else:
            bad +=1

    if good > bad and good > ok:
        feedback = "g"
    elif bad > good and bad > ok:
        feedback = "b"
    else:
        feedback = "k"

    return feedback

def compExercise(anglesUser, anglesTrainer, exercise):
    flags, head, shoulder, shoulderL, shoulderR, elbow, elbowL, elbowR, hip, hipL, hipR, knee, kneeL, kneeR = "","","","","","","","","","","","","","",

    repPosUser = countReps(anglesUser, exercise)
    RepPoseTrainer = countReps(anglesTrainer, exercise)
    consistant, speed = compTiming(repPosUser, RepPoseTrainer)

    if exercise == "squat":
        kneeL = compJoint(repPosUser, anglesUser, RepPoseTrainer, anglesTrainer, "kneeL")
        kneeR = compJoint(repPosUser, anglesUser, RepPoseTrainer, anglesTrainer, "kneeR")
        hipL = compJoint(repPosUser, anglesUser, RepPoseTrainer, anglesTrainer, "hipL")
        hipR = compJoint(repPosUser, anglesUser, RepPoseTrainer, anglesTrainer, "hipR")

        hip = hipL + hipR
        knee = kneeL + kneeR
    
    
    else:
        print("Error: No exercise selected")
    
    if knee == "bb" or knee =="bk" or knee =="kb" or knee =="bg" or knee =="gb":
        flags = flags + "k"

    if hip == "bb" or hip =="bk" or hip =="kb" or hip =="bg" or hip =="gb":
        flags = flags + "b"

    if shoulder == "bb" or shoulder =="bk" or shoulder =="kb" or shoulder =="bg" or shoulder =="gb":
        flags = flags + "a"
    
    if elbow == "bb" or elbow =="bk" or elbow =="kb" or elbow =="bg" or elbow =="gb":
        flags = flags + "e"
    
    if speed == "Fast":
        flags = flags + "s"
    elif speed == "Slow":
        flags = flags + "f"

    if consistant == False:
        flags = flags + "i"

    return flags

def responce(flags):
    print("Feedback:")
    if "h" in flags:
        print("Keep your head straight ")

    elif "a" in flags:
        print("Focus on your arms more")

    elif "e" in flags:
        print("Watch your elbows more")

    elif "k" in flags:
        print("Bend you knees more")

    elif "b" in flags:
        print("Keep your back straight")

    elif "f" in flags:
        print("Try to speed up")

    elif "s" in flags:
        print("Slow down each rep")

    elif "i" in flags:
        print("Try to keep your rep times consistent ")

    else:
        print("Great Job")

if __name__ == "__main__":
    
    model_path = "lite-model_movenet_singlepose_lightning_3.tflite"
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    filePath = '/home/aaron/Documents/major_project/data'
    excersize = "squat"
    userType = "trainer"
    videoP = '/home/aaron/Documents/major_project/data/videos/*'
    videoPath = glob(videoP)
    checkE = False
    checkC = False
    checkU = False
    menu = False

    #basic command line UI to select the intended use case and generate the required filepath
    while menu == False:
        checkE = False
        checkC = False
        while checkU == False:
            ans = input("Please choose function: \nproceed as triner - 1\nproceed as user - 2")
            if ans == "1":
                userType = "trainer"
                checkU = True
            elif ans == "2":
                userType = "user"
                checkU = True
            else:
                print("Error: incorrect input")
        
        while checkC == False:
            ans = input("do you need to conver a video: \ny/n")
            if ans == "y":
                checkC = True
                convert = True
            elif ans == "n":
                checkC = True
                convert = False
            else:
                print("Error: incorrect input")

        while checkE == False:
            ans = input("Please choose excersize: \nsquat - 1 \nback - q")
            if ans == "1":
                excersize = "squat"
                checkE = True
                menu = True
            elif ans == "q":
                checkU = False
                checkE = True
            else:
                print("Error: incorrect input")
    
    if convert == True:
        for path in videoPath:
            saveFrame(path, filePath, userType, excersize)

    for i in range(1, 300):
        #checks if file exists, if file doesnt exits all images have been processed
        if (os.path.exists(filePath+'/'+excersize+'/'+excersize+"_"+userType+'/'+str(i)+'.png')):
            #read image
            #image detection
            image_path = filePath+'/'+excersize+'/'+excersize+"_"+userType+'/'+str(i)+'.png'
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image)

            input_image = tf.cast(image, dtype=tf.float32)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
    
            # Reshape image
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, 192, 192)
            input_image = tf.cast(input_image, dtype=tf.float32)
    
            # Make predictions 
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

            width = 640
            height = 640
            #removes unnecessary dimentions in array
            #and converts X,Y values from decimal point values to coordinates on the image
            shaped = np.squeeze(np.multiply(keypoints_with_scores,[width,height,1]))
            #adds all values to an array
            if (i == 1):
                keyPoints_all_images = np.array(shaped)
            else:
                keyPoints_all_images = np.vstack((keyPoints_all_images, shaped))
        else:
            print(str(i-1) + " images found and processed")
            break

    keypointsChecked, inaccurate = checkConfidence(keyPoints_all_images, 0.3)

    if inaccurate == True:
        print("Error: too many uncertain keypoints, video unsuitable for comparison")
        sys.exit()

    else:
        angles_all_images = calcJoints(keypointsChecked)
    
    if userType == "trainer":
        write(angles_all_images, filePath, excersize, userType)
        print("data saved to "+filePath+'/'+excersize+'/'+excersize+"_"+userType+'/''test1.txt')
        sys.exit()
    else:
        trainer_angles = read(filePath, excersize, "trainer")
    
    flags = compExercise(angles_all_images, trainer_angles, excersize)

    responce(flags)