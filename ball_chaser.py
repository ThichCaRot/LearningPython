import cv2
import numpy as np
import time
from adafruit_motorkit import MotorKit
from PCA9685 import PCA9685

kit = MotorKit();
motors = [kit.motor1, kit.motor2, kit.motor3, kit.motor4]

motorMultiplier = [1.0, 1.0, 1.0, 1.0, 1.0]
motorSpeed = [0,0,0,0]
speedDef = 1.0
leftSpeed = speedDef
rightSpeed = speedDef
diff = 0
maxDiff = 0.5
turnTime = 0.5

cap = cv2. VideoCapture(0)
time.sleep(1)

kp = 1.0
ki = 1.0
kd = 1.0
ballX = 0.0
ballY = 0.0
x = {'axis':'X',
     'lastTime':int(round(time.time()*1000)),
     'lastEror':0.0,
     'error':0.0,
     'duration':0.0,
     'sumError':0.0,
     'dError':0.0,
     'PID':0.0}
y = {'axis':'Y',
     'lastTime':int(round(time.time()*1000)),
     'lastEror':0.0,
     'error':0.0,
     'duration':0.0,
     'sumError':0.0,
     'dError':0.0,
     'PID':0.0}

params = cv2.SimpleBlobDetector_Params()

params.filterByColor = False
params.filterByArea = True
params.minArea = 15000
params.maxArea = 40000
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

det = cv2.SimpleBlobDetector_create(params)

lower_blue = np.array([80,60,20])
upper_blue = np.array([130,255,255])

def driveMotors(leftChnl = speedDef, rightChnl = speedDef,
                duration = defTime):
    motorSpeed[0] = leftChnl * motorMultiplier[0]
    motorSpeed[1] = leftChnl * motorMultiplier[1]
    motorSpeed[2] = leftChnl * motorMultiplier[2]
    motorSpeed[3] = leftChnl * motorMultiplier[3]

    if(leftChnl < 0):
        motors[0].throttle(-motorSpeed[0])
        motors[1].throttle(-motorSpeed[1])
    else:
        motors[0].throttle(motorSpeed[0])
        motors[1].throttle(motorSpeed[1])

    if(rightChnl < 0):
        motors[2].throttle(-motorSpeed[2])
        motors[3].throttle(-motorSpeed[3])
    else:
        motors[2].throttle(motorSpeed[2])
        motors[3].throttle(motorSpeed[3])

    def PID(axis):
        lastTime = axis['lastTime']
        lastError = axis['lastError']

        now = int(round(time.time()*1000))
        duration = now-lastTime

        axis['sumError'] += axis['error'] * duration
        axis['dError'] = (axis['error'] - lastError)/duration

        if axis['sumError'] > 1:axis['sumError'] = 1
        if axis['sumError'] < -1:axis['sumError'] = -1

        axis['PID'] = kp * axis['error'] + ki * axis['sumError'] + kd * axis['dError']

        axis['lastError'] = axis['error']
        axis['lastTime'] = now

        return axis
    
    def killMotors():
        motors[0].throttle(0)
        motors[1].throttle(0)
        motors[2].throttle(0)
        motors[3].throttle(0)

    try:
        while True:
            ret, frame = cap.read()

            height, width, chan = np.share(frame)
            xMid = width/2 * 1.0
            yMid = height/2 * 1.0

            imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            blueMask = cv2.inRange(imgHSV, lower_blue, upper_blue)
            blue = cv2.blue(blueMask, (10,10))

            res = cv2.bitwise_and(frame, frame, mask=blue)

            keypoints = det.detect(blur)
            try:
                ballX = int(keypoints[0].pt[0])
                ballY = int(keypoints[0].pt[1])
            except:
                pass

            cv2.drawKeypoints(frame, keypoints, frame, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            xVariance = (ballX - xMid) / xMid
            yVariance = (yMid - ballY) / yMid

            x['error'] = xVariance/xMid
            y['error'] = yVariance/yMid

            x = PID(x)
            y = PID(y)

            leftSpeed = (speedDef * y['PID']) + (maxDiff * x['PID'])
            rightSpeed = (speedDef * y['PID']) + (maxDiff * x['PID'])

            if leftSpeed > (speedDef + maxDiff):
                leftSpeed = (speedDef + maxDiff)
            if leftSpeed < -(speedDef + maxDiff):
                leftSpeed = -(speedDef + maxDiff)
            if rightSpeed > (speedDef + maxDiff):
                rightSpeed = (speedDef + maxDiff)
            if rightSpeed < -(speedDef + maxDiff):
                rightSpeed = -(speedDef + maxDiff)

            driveMotors(leftSpeed, rightSpeed, driveTime)
    except KeyboardInterrupt:
        killMotors()
        cap.release()
        cv2.destroyAllWindows()