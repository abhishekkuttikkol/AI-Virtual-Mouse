import cv2
import time
import HandTrackingModule as htm 
import autopy 
import math
import numpy as np

###############################
wCam, hCam = 640, 480
##############################
CTime, pTime = 0, 0
cap = cv2.VideoCapture(0)
# cap.open('http://192.168.1.199:8080/video')
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = autopy.screen.size()

def main():
    ################################
    wCam, hCam = 640, 480
    frameR = 100 
    smoothening = 7
    ################################
    pLocX, pLocY = 0, 0
    cLocX, pLocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.open('http://192.168.1.199:8080/video') 
    cap.set(3, wCam)    
    cap.set(4, hCam)
    pTime = 0
    CTime = 0

    detector = htm.handDetector(detectionCon=0.7, maxHands=1)

    while cap.isOpened():
        success, image = cap.read()
        image = detector.findhands(image)
        lmLists, bbox= detector.findPosition(image, draw=True)
        fingers = detector.fingersUp()
        if len(lmLists) != 0:
            length, image, lineInfo = detector.findDistance(8, 12, image, draw=False)
            fingers = detector.fingersUp()
            
            if fingers[1] == 1 and fingers[2] == 0:
                cv2.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
                x3 = np.interp(lineInfo[0], (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(lineInfo[1], (frameR, hCam - frameR), (0, hScr))
                cLocX = pLocX +(x3 - pLocX) / smoothening
                cLocY = pLocY +(y3 - pLocY) / smoothening

                autopy.mouse.move(cLocX, cLocY)

                cv2.circle(image, (lineInfo[0], lineInfo[1]), 15, (255, 0, 255), cv2.FILLED)
                pLocX, pLocY = cLocX, cLocY

            if fingers[1] == 1 and fingers[2] == 1:
                # length, image, lineInfo = detector.findDistance(8, 12, image, draw=True)
                if length < 40:
                    cv2.circle(image, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        CTime = time.time()
        fps = 1/(CTime-pTime)
        pTime = CTime
        cv2.putText(image,str(int(fps)),(20,40),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
main()